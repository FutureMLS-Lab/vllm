# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 fused DSA indexer prologues against the unfused reference path."""

import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
    indexer_k_norm_quant_insert,
    indexer_q_quant_fold_weights,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton indexer prologues need CUDA"
)

HEAD_DIM = 128
NUM_HEADS = 32
ENTRY = HEAD_DIM + 4  # fp8 values + one fp32 scale per token
BLOCK = 64  # K3.1 kernel block (merged-cache unit) size
MLA_BYTES = 656  # fp8_ds_mla bytes per token ahead of the indexer section
UNIT_STRIDE = 50512  # padded merged unit: 64 x (656 + 132), 656-aligned
TOPK = 2048


def _make_cache(merged: bool, num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (backing bytes, indexer cache view [num_blocks, BLOCK, ENTRY])."""
    if not merged:
        raw = torch.zeros(num_blocks, BLOCK, ENTRY, dtype=torch.uint8, device="cuda")
        return raw, raw
    # Same section view as split_merged_dsa_cache_views (after the squeeze).
    raw = torch.zeros(num_blocks, UNIT_STRIDE, dtype=torch.uint8, device="cuda")
    view = torch.as_strided(
        raw,
        size=(num_blocks, BLOCK, ENTRY),
        stride=(UNIT_STRIDE, ENTRY, 1),
        storage_offset=BLOCK * MLA_BYTES,
    )
    return raw, view


def _dequant(view: torch.Tensor) -> torch.Tensor:
    """Dequantize every token of an indexer cache view to [blocks*BLOCK, D]."""
    num_blocks = view.shape[0]
    rows = []
    for b in range(num_blocks):
        flat = view[b].reshape(-1)  # row-major within the block
        data = flat[: BLOCK * HEAD_DIM].view(current_platform.fp8_dtype())
        data = data.view(BLOCK, HEAD_DIM).float()
        scale_bytes = flat[BLOCK * HEAD_DIM : BLOCK * ENTRY].contiguous()
        if mqa_logits_use_mx_scales():
            # Four equal UE8M0 bytes: the exponent of the power-of-two scale.
            exp = scale_bytes.view(BLOCK, 4)[:, 0].to(torch.int32)
            scale = torch.pow(2.0, (exp - 127).float())
        else:
            scale = scale_bytes.view(torch.float32)
        rows.append(data * scale[:, None])
    return torch.cat(rows)


@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 7, 300])
@torch.inference_mode()
def test_k_norm_quant_insert_matches_unfused(merged: bool, num_tokens: int):
    torch.manual_seed(0)
    num_blocks = (num_tokens + 2 * BLOCK - 1) // BLOCK + 1
    # The merged wkv GEMM output is [k | weights]; k is a strided view of it.
    kw = torch.randn(
        num_tokens, HEAD_DIM + NUM_HEADS, dtype=torch.bfloat16, device="cuda"
    )
    k = kw[:, :HEAD_DIM]
    weight = torch.rand(HEAD_DIM, dtype=torch.float32, device="cuda") + 0.5
    bias = torch.randn(HEAD_DIM, dtype=torch.float32, device="cuda") * 0.1
    eps = 1e-6
    # Fewer slots than rows (cudagraph padding), plus a padded (-1) slot.
    num_slots = max(num_tokens - 1, 1)
    slots = torch.randperm(num_blocks * BLOCK, device="cuda")[:num_slots]
    if num_slots > 2:
        slots[1] = -1
    slots = slots.to(torch.int64)

    ref_raw, ref_view = _make_cache(merged, num_blocks)
    ref_k = F.layer_norm(k.float(), (HEAD_DIM,), weight, bias, eps).to(k.dtype)
    ops.indexer_k_quant_and_cache(
        ref_k[:num_slots].contiguous(),
        ref_view,
        slots,
        HEAD_DIM,
        "mxfp8" if mqa_logits_use_mx_scales() else "ue8m0",
    )

    out_raw, out_view = _make_cache(merged, num_blocks)
    topk = torch.full((num_tokens + 5, TOPK), 7, dtype=torch.int32, device="cuda")
    indexer_k_norm_quant_insert(k, weight, bias, eps, slots, out_view, topk, num_tokens)

    # Only LayerNorm reduction order differs, so bytes match except for rare
    # bf16 rounding ties; dequantized values agree to fp8 precision.
    mismatch = (ref_raw != out_raw).float().mean().item()
    assert mismatch < 1e-3, f"{mismatch:.2e} of cache bytes differ"
    torch.testing.assert_close(
        _dequant(out_view), _dequant(ref_view), atol=0.07, rtol=0.07
    )
    if merged:
        # The MLA section of each unit is untouched.
        mla = out_raw[:, : BLOCK * MLA_BYTES]
        assert torch.count_nonzero(mla) == 0
    assert torch.all(topk[:num_tokens] == -1)
    assert torch.all(topk[num_tokens:] == 7)


@pytest.mark.parametrize("num_heads", [NUM_HEADS, 20])
@pytest.mark.parametrize("num_tokens", [1, 33])
@torch.inference_mode()
def test_q_quant_fold_weights_matches_unfused(num_heads: int, num_tokens: int):
    torch.manual_seed(0)
    padded = 32
    q = torch.randn(
        num_tokens, num_heads * HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    ).view(num_tokens, num_heads, HEAD_DIM)
    q[0, 0].zero_()  # all-zero group: exercises the amax floor
    kw = torch.randn(
        num_tokens, HEAD_DIM + num_heads, dtype=torch.bfloat16, device="cuda"
    )
    w = kw[:, HEAD_DIM:]
    weight_scale = HEAD_DIM**-0.5 * num_heads**-0.5

    q_fp8, weights = indexer_q_quant_fold_weights(q, w, weight_scale, padded)

    ref_fp8, ref_scale = per_token_group_quant_fp8(
        q.reshape(-1, HEAD_DIM).contiguous(),
        HEAD_DIM,
        column_major_scales=False,
        use_ue8m0=True,
    )
    ref_fp8 = ref_fp8.view(num_tokens, num_heads, HEAD_DIM)
    ref_weights = w.float() * ref_scale.view(num_tokens, num_heads) * weight_scale

    assert q_fp8.shape == (num_tokens, padded, HEAD_DIM)
    assert weights.shape == (num_tokens, padded)
    assert torch.equal(
        q_fp8[:, :num_heads].view(torch.uint8), ref_fp8.view(torch.uint8)
    )
    torch.testing.assert_close(weights[:, :num_heads], ref_weights)
    if padded != num_heads:
        assert torch.count_nonzero(q_fp8[:, num_heads:].view(torch.uint8)) == 0
        assert torch.count_nonzero(weights[:, num_heads:]) == 0
