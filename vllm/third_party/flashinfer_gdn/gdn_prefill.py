"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Gated Delta Net Chunked Prefill with state export - Blackwell SM100 Adapter
=============================================================================

FlashInfer 0.7.0.post1's adapter for the CuTe DSL chunked GDN kernel (gdn_kernels/blackwell/gdn_prefill.py), with
the stride checkpoints replaced by one state per sequence: ``checkpoint_offsets[b]`` (int32) asks for the state
after min(c, L) tokens of sequence b when c > 0, c % 16 == 0 and c <= 16 * ceil(L / 16); any other offset writes
nothing (FlashKDA's checkpoint ABI). The state goes to row ``checkpoint_rows[b]`` of ``checkpoint_state``
([R, H, V, K] fp32), or to row b without ``checkpoint_rows``.

Unlike FlashInfer's, the kernel takes the forget gate in log space and applies exp itself (the accurate expf that
torch.exp uses), and ``v_split=2`` runs each (sequence, value head) as two tiles that own 64 of the state's 128 rows
along V, for calls with few sequences. Every output element is computed as FlashInfer's non-context-parallel route
computes it.

Its CuTe DSL module name is its own, so its compiled kernels never share FlashInfer's on-disk cache directory, and
its arguments are not FlashInfer's: install it beside FlashInfer, never over it.

State layout: ``[N, H, V, K]``.
"""

import functools
from typing import Optional

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flashinfer.cute_dsl.utils import get_num_sm
from flashinfer.gdn_kernels.cute_dsl_cache_naming import make_kernel_name
from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

from .gated_delta_net_chunked import GatedDeltaNetChunkedKernel


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------


# Keyed on static kernel configuration. Head counts (HQ, HV) are part of
# the key because the tile scheduler and GQA reshape logic bake them in.
_CUTE_DSL_MODULE = "vllm_gdn_blackwell_prefill_export"


def _kernel_source_files() -> tuple:
    """Source files whose content invalidates the on-disk kernel cache."""
    from . import gated_delta_net_chunked, gated_delta_net_tile_scheduler

    return (
        __file__,
        gated_delta_net_chunked.__file__,
        gated_delta_net_tile_scheduler.__file__,
    )


def _prefill_kernel_name(
    io_dtype_str: str,
    state_dtype_str: str,
    HQ: int,
    HV: int,
    is_GQA: bool,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
    use_state_indices: bool,
    cu_seqlens_dtype_str: str,
    state_indices_dtype_str: str,
    checkpoint_offsets_dtype_str: str,
    use_checkpoint_rows: bool,
    v_split: int,
    initial_state_inner_strides,
    output_state_inner_strides,
    num_sm: int,
) -> str:
    """Specialization name within the vllm_gdn_blackwell_prefill_export module.

    Encodes every ``_get_compiled_cache`` key component plus ``num_sm``, which
    the compile below bakes in as ``max_active_clusters``.
    """
    return make_kernel_name(
        io_dtype_str,
        state_dtype_str,
        HQ,
        HV,
        is_GQA,
        use_initial_state,
        store_final_state,
        enable_checkpoints,
        use_state_indices,
        cu_seqlens_dtype_str,
        state_indices_dtype_str,
        checkpoint_offsets_dtype_str,
        use_checkpoint_rows,
        v_split,
        initial_state_inner_strides,
        output_state_inner_strides,
        num_sm,
    )


@functools.cache
def _get_compiled_cache(
    io_dtype_str: str,
    state_dtype_str: str,
    HQ: int,
    HV: int,
    is_GQA: bool,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
    use_state_indices: bool,
    cu_seqlens_dtype_str: str,
    state_indices_dtype_str: str,
    checkpoint_offsets_dtype_str: str,
    use_checkpoint_rows: bool,
    v_split: int,
    initial_state_inner_strides: tuple[int, ...] | None,
    output_state_inner_strides: tuple[int, ...] | None,
):
    """Return a mutable dict that lazily stores the compiled kernel.

    Holds only the compiled callable and static metadata (``num_sm``). Mutable
    execution state such as workspaces must not be stored here: the dict is
    process-global per specialization, so any buffer it holds is shared across
    streams and CUDA graphs.
    """
    return {}


def _cutlass_io_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.bfloat16:
        return cutlass.BFloat16
    elif torch_dtype == torch.float16:
        return cutlass.Float16
    else:
        raise ValueError(
            f"Unsupported dtype {torch_dtype}, expected bfloat16 or float16"
        )


def _cutlass_state_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.float32:
        return cutlass.Float32
    elif torch_dtype == torch.bfloat16:
        return cutlass.BFloat16
    elif torch_dtype == torch.float16:
        return cutlass.Float16
    elif torch_dtype == torch.float8_e4m3fn:
        return cutlass.Float8E4M3FN
    elif torch_dtype == torch.float8_e5m2:
        return cutlass.Float8E5M2
    else:
        raise ValueError(
            f"Unsupported state dtype {torch_dtype}, expected float32, bfloat16, "
            "float16, float8_e4m3fn, or float8_e5m2"
        )


def _mark_state_layout(s_cute, use_state_indices: bool, DK: int) -> None:
    """Mark the recurrent-state tensor layout for compilation (FlashInfer's own marking: a packed state keeps a
    divisibility hint on its stride-1 K dimension; a pool with a padded slot stride keeps every stride dynamic)."""
    if use_state_indices:
        s_cute.mark_layout_dynamic()
    else:
        s_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
            mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def chunk_gated_delta_rule_export_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    scale: float,
    checkpoint_offsets: Optional[torch.Tensor] = None,
    checkpoint_state: Optional[torch.Tensor] = None,
    state_indices: Optional[torch.Tensor] = None,
    checkpoint_rows: Optional[torch.Tensor] = None,
    v_split: int = 1,
) -> None:
    """Execute the Blackwell chunked GDN prefill kernel, exporting one state per sequence.

    All tensors must be contiguous and on the same CUDA device.

    Args:
        q: ``(total_tokens, HQ, DK)`` float16/bfloat16
        k: ``(total_tokens, HK, DK)`` float16/bfloat16
        v: ``(total_tokens, HV, DK)`` float16/bfloat16
        gate: ``(total_tokens, HO)`` float32, forget gate in log space (the kernel applies exp)
        beta: ``(total_tokens, HO)`` float32, update gate
        output: ``(total_tokens, HO, DK)`` float16/bfloat16, pre-allocated
        cu_seqlens: ``(num_seqs + 1,)`` int32
        initial_state: ``(num_seqs, HO, DK, DK)`` float32/bfloat16/float16/fp8, or None
        output_state: ``(num_seqs, HO, DK, DK)`` float32/bfloat16/float16/fp8, or None
        scale: attention scale factor (must not be 0)
        checkpoint_offsets: ``(num_seqs,)`` int32, one checkpoint offset per sequence, or None (no export)
        checkpoint_state: ``(num_rows, HO, DK, DK)`` float32, the exported states (with ``checkpoint_offsets``);
            ``num_rows == num_seqs`` without ``checkpoint_rows``
        checkpoint_rows: ``(num_seqs,)`` int32, the row of ``checkpoint_state`` each sequence's state goes to,
            read only for sequences whose offset exports, or None (row b for sequence b)
        v_split: 1, or 2 to run each (sequence, value head) as two tiles of 64 state rows along V
    """
    HQ = q.size(1)
    HV = v.size(1)
    DK = q.size(2)
    # The kernel derives its value split from the tile height, which assumes 128.
    if q.size(2) != 128 or k.size(2) != 128 or v.size(2) != 128:
        raise ValueError(
            f"head dimensions must be 128, got q {q.size(2)}, k {k.size(2)}, v {v.size(2)}"
        )
    is_GQA = HQ >= HV
    use_initial_state = initial_state is not None
    store_final_state = output_state is not None
    enable_checkpoints = checkpoint_offsets is not None
    io_dtype = _cutlass_io_dtype(q.dtype)
    B = cu_seqlens.size(0) - 1
    use_checkpoint_rows = checkpoint_rows is not None
    if enable_checkpoints:
        if initial_state is None or checkpoint_state is None:
            raise ValueError("state export needs an initial state and checkpoint_state")
        if checkpoint_offsets.dtype != torch.int32 or tuple(checkpoint_offsets.shape) != (B,):
            raise ValueError(f"checkpoint_offsets must be int32 [{B}], got {checkpoint_offsets.dtype} "
                             f"{list(checkpoint_offsets.shape)}")
        rows = checkpoint_state.shape[0] if use_checkpoint_rows else B
        if checkpoint_state.dtype != torch.float32 or tuple(checkpoint_state.shape) != (rows, HV, DK, DK):
            raise ValueError(f"checkpoint_state must be float32 [{rows}, {HV}, {DK}, {DK}], got "
                             f"{checkpoint_state.dtype} {list(checkpoint_state.shape)}")
        if use_checkpoint_rows and (
            checkpoint_rows.dtype != torch.int32 or tuple(checkpoint_rows.shape) != (B,)
        ):
            raise ValueError(f"checkpoint_rows must be int32 [{B}], got {checkpoint_rows.dtype} "
                             f"{list(checkpoint_rows.shape)}")
    elif use_checkpoint_rows:
        raise ValueError("checkpoint_rows needs checkpoint_offsets")
    if v_split not in (1, 2):
        raise ValueError(f"v_split must be 1 or 2, got {v_split}")

    # Auto-detect state dtype from initial_state, default to float32
    if initial_state is not None:
        state_torch_dtype = initial_state.dtype
    elif output_state is not None:
        state_torch_dtype = output_state.dtype
    else:
        state_torch_dtype = torch.float32
    state_dtype = _cutlass_state_dtype(state_torch_dtype)

    _initial_state = initial_state if use_initial_state else None
    _output_state = output_state if store_final_state else None
    use_state_indices = state_indices is not None
    _state_indices = state_indices if use_state_indices else None

    cache_key = (
        str(q.dtype),
        str(state_torch_dtype),
        HQ,
        HV,
        is_GQA,
        use_initial_state,
        store_final_state,
        enable_checkpoints,
        use_state_indices,
        str(cu_seqlens.dtype),
        str(state_indices.dtype) if state_indices is not None else "none",
        str(checkpoint_offsets.dtype) if checkpoint_offsets is not None else "none",
        use_checkpoint_rows,
        v_split,
        (
            tuple(initial_state.stride()[1:])
            if use_state_indices and initial_state is not None
            else None
        ),
        (
            tuple(output_state.stride()[1:])
            if use_state_indices and output_state is not None
            else None
        ),
    )
    cache = _get_compiled_cache(*cache_key)

    if "compiled" not in cache:
        # --- First call: compile the kernel ---
        num_sm = get_num_sm(q.device)
        max_active_clusters = num_sm

        gdn = GatedDeltaNetChunkedKernel(
            io_dtype=io_dtype,
            # The kernel requires the triangular-inverse dtype to match io_dtype
            # (asserted in its __init__); pass io_dtype so bf16 uses the same
            # validated path as fp16 instead of the default Float16.
            inverse_dtype=io_dtype,
            acc_dtype=cutlass.Float32,
            state_dtype=state_dtype,
            mma_tiler_qk=(64, 64, 128),
            mma_tiler_qs=(DK // v_split, 64, 128),
            mma_tiler_qkv=(DK // v_split, 64, 64),
            mma_tiler_kv=(DK // v_split, 128, 64),
            max_active_clusters=max_active_clusters,
            num_sm=num_sm,
            is_GQA=is_GQA,
            use_initial_state=use_initial_state,
            store_final_state=store_final_state,
            enable_checkpoints=enable_checkpoints,
            is_persistent=True,
            use_checkpoint_rows=use_checkpoint_rows,
        )

        # Convert PyTorch tensors to CuTe tensors for compilation.
        # Token dimension (dim 0) must be dynamic to handle varying seq lengths.
        # Head and head_dim dimensions stay static (part of cache key).
        q_cute = from_dlpack(q, assumed_align=16)
        q_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        k_cute = from_dlpack(k, assumed_align=16)
        k_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        v_cute = from_dlpack(v, assumed_align=16)
        v_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        gate_cute = from_dlpack(gate, assumed_align=16)
        gate_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
        beta_cute = from_dlpack(beta, assumed_align=16)
        beta_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
        o_cute = from_dlpack(output, assumed_align=16)
        o_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()

        s_in_cute = None
        if use_initial_state:
            s_in_cute = from_dlpack(_initial_state, assumed_align=16)
            _mark_state_layout(s_in_cute, use_state_indices, DK)

        s_out_cute = None
        if store_final_state:
            s_out_cute = from_dlpack(_output_state, assumed_align=16)
            _mark_state_layout(s_out_cute, use_state_indices, DK)

        s_indices_cute = None
        if use_state_indices:
            s_indices_cute = from_dlpack(
                _state_indices, assumed_align=4
            ).mark_layout_dynamic()

        s_checkpoints_cute = None
        checkpoint_offsets_cute = None
        checkpoint_rows_cute = None
        if enable_checkpoints:
            s_checkpoints_cute = from_dlpack(checkpoint_state, assumed_align=16)
            s_checkpoints_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
            )
            checkpoint_offsets_cute = from_dlpack(
                checkpoint_offsets, assumed_align=4
            ).mark_layout_dynamic()
            if use_checkpoint_rows:
                checkpoint_rows_cute = from_dlpack(
                    checkpoint_rows, assumed_align=4
                ).mark_layout_dynamic()

        workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(
            num_sm, B, HQ, HV, True
        )
        workspace = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
        workspace_cute = from_dlpack(workspace, assumed_align=16)

        stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)

        compiled = build_and_load_cute_dsl_kernel(
            _CUTE_DSL_MODULE,
            _prefill_kernel_name(*cache_key, num_sm),
            lambda: cute.compile(
                gdn,
                q_cute,
                k_cute,
                v_cute,
                gate_cute,
                beta_cute,
                o_cute,
                cu_seqlens_cute,
                s_in_cute,
                s_out_cute,
                s_indices_cute,
                s_checkpoints_cute,
                checkpoint_offsets_cute,
                checkpoint_rows_cute,
                0,
                scale,
                workspace_cute,
                stream,
                options="--enable-tvm-ffi --opt-level 3",
            ),
            extra_key_files=_kernel_source_files(),
        )

        cache["compiled"] = compiled
        cache["num_sm"] = num_sm

    # --- Execute ---
    compiled = cache["compiled"]
    num_sm = cache["num_sm"]

    workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(
        num_sm, B, HQ, HV, True
    )
    # Fresh per-call allocation: the kernel writes TMA tensormap descriptors into
    # the workspace on every launch, so a cached buffer shared across streams lets
    # concurrent same-specialization calls clobber each other's descriptors. The
    # caching allocator makes this near-free next to the prefill kernel itself.
    workspace = torch.empty(workspace_size, dtype=torch.int8, device=q.device)

    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    compiled(
        q,
        k,
        v,
        gate,
        beta,
        output,
        cu_seqlens,
        _initial_state,
        _output_state,
        _state_indices,
        checkpoint_state if enable_checkpoints else None,
        checkpoint_offsets,
        checkpoint_rows,
        0,
        scale,
        workspace,
        stream,
    )
