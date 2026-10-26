# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton prologues for the Kimi K3.1 DSA indexer, replacing the small
launch-bound kernels between the projection GEMMs and ``sparse_attn_indexer``.

The quant helpers are copied from ``deepseek_v32/common/kernels.py`` so this
model does not depend on another model's private kernels.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import mqa_logits_use_mx_scales
from vllm.v1.attention.ops.common import pack_ue8m0_x4

_FP8_DTYPE = current_platform.fp8_dtype()
_USE_FNUZ = torch.float8_e4m3fnuz == _FP8_DTYPE
_FP8_MAX = 224.0 if _USE_FNUZ else 448.0


@triton.jit
def _ue8m0_scale(amax, AMAX_FLOOR: tl.constexpr, FP8_MAX: tl.constexpr):
    scale = tl.div_rn(tl.maximum(amax, AMAX_FLOOR), FP8_MAX)
    return tl.math.exp2(tl.math.ceil(tl.math.log2(scale)))


@triton.jit
def _indexer_k_norm_quant_insert_kernel(
    k_ptr,
    k_stride,
    norm_w_ptr,
    norm_b_ptr,
    norm_eps,
    slot_mapping_ptr,
    num_slots,
    cache_ptr,
    cache_scale_ptr,
    cache_block_size,
    cache_block_stride,
    topk_ptr,
    topk_stride,
    HEAD_DIM: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    MX_SCALE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    if tl.program_id(1) == 1:
        # Padding rows included.
        for i in range(0, TOPK, TOPK_BLOCK):
            offs = i + tl.arange(0, TOPK_BLOCK)
            tl.store(topk_ptr + tok * topk_stride + offs, -1, mask=offs < TOPK)
        return

    # Padded rows and (under DCP) tokens owned by another rank have no slot.
    if tok >= num_slots:
        return
    slot = tl.load(slot_mapping_ptr + tok)
    if slot < 0:
        return

    offs = tl.arange(0, HEAD_DIM)
    x = tl.load(k_ptr + tok * k_stride + offs).to(tl.float32)
    mean = tl.sum(x, axis=0) / HEAD_DIM
    diff = x - mean
    rstd = tl.rsqrt(tl.sum(diff * diff, axis=0) / HEAD_DIM + norm_eps)
    w = tl.load(norm_w_ptr + offs).to(tl.float32)
    b = tl.load(norm_b_ptr + offs).to(tl.float32)
    # Round to bf16 like the unfused path so the cached bytes match.
    y = (diff * rstd * w + b).to(k_ptr.dtype.element_ty).to(tl.float32)

    # Same math as the CUDA indexer_k_quant_and_cache kernel.
    scale = _ue8m0_scale(tl.max(tl.abs(y), axis=0), 1e-4, FP8_MAX)
    fp8_dtype = tl.float8e4b8 if USE_FNUZ else tl.float8e4nv
    y_fp8 = tl.div_rn(y, scale).to(fp8_dtype)

    # Block layout: [block_size x HEAD_DIM fp8 values | block_size x 4B scale].
    block_start = (slot // cache_block_size) * cache_block_stride
    block_off = slot % cache_block_size
    tl.store(cache_ptr + block_start + block_off * HEAD_DIM + offs, y_fp8)
    scale_byte = block_start + cache_block_size * HEAD_DIM + block_off * 4
    if MX_SCALE:
        scale = pack_ue8m0_x4(scale)
    tl.store(cache_scale_ptr + scale_byte // 4, scale)


def indexer_k_norm_quant_insert(
    k: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    num_tokens: int,
) -> None:
    """LayerNorm + FP8 quant + cache insert of the indexer key, and the -1
    fill of the top-k rows ``[:num_tokens]``.

    ``kv_cache`` is ``[num_blocks, block_size, head_dim + 4]`` uint8, possibly
    a strided section of the merged cache. Rows past ``slot_mapping`` or with
    a negative slot are not inserted.
    """
    if num_tokens == 0:
        return
    head_dim = k.shape[-1]
    assert k.stride(-1) == 1
    assert kv_cache.dim() == 3 and kv_cache.shape[-1] == head_dim + 4
    assert triton.next_power_of_2(head_dim) == head_dim
    mx_scale = mqa_logits_use_mx_scales()
    u8_cache = kv_cache.view(torch.uint8)
    topk = topk_indices_buffer.shape[-1]
    _indexer_k_norm_quant_insert_kernel[(num_tokens, 2)](
        k,
        k.stride(0),
        norm_weight,
        norm_bias,
        norm_eps,
        slot_mapping,
        slot_mapping.shape[0],
        u8_cache.view(_FP8_DTYPE),
        u8_cache.view(torch.int32 if mx_scale else torch.float32),
        u8_cache.shape[1],
        u8_cache.stride(0),
        topk_indices_buffer,
        topk_indices_buffer.stride(0),
        HEAD_DIM=head_dim,
        TOPK=topk,
        TOPK_BLOCK=min(1024, triton.next_power_of_2(topk)),
        MX_SCALE=mx_scale,
        FP8_MAX=_FP8_MAX,
        USE_FNUZ=_USE_FNUZ,
        num_warps=1,
    )


@triton.jit
def _indexer_q_quant_fold_weights_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    w_ptr,
    w_stride,
    q_fp8_ptr,
    q_fp8_stride0,
    q_fp8_stride1,
    w_out_ptr,
    w_out_stride,
    weight_scale,
    HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + tok * q_stride0 + head * q_stride1 + offs).to(tl.float32)
    # Same math as per_token_group_quant_fp8(use_ue8m0=True) (eps=1e-10).
    scale = _ue8m0_scale(tl.max(tl.abs(q), axis=0), 1e-10, FP8_MAX)
    fp8_dtype = tl.float8e4b8 if USE_FNUZ else tl.float8e4nv
    q_fp8 = tl.clamp(tl.div_rn(q, scale), -FP8_MAX, FP8_MAX).to(fp8_dtype)
    tl.store(q_fp8_ptr + tok * q_fp8_stride0 + head * q_fp8_stride1 + offs, q_fp8)
    w = tl.load(w_ptr + tok * w_stride + head).to(tl.float32)
    tl.store(w_out_ptr + tok * w_out_stride + head, w * scale * weight_scale)


def indexer_q_quant_fold_weights(
    index_q: torch.Tensor,
    index_weights: torch.Tensor,
    weight_scale: float,
    padded_num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``index_q`` ``[T, H, D]`` to FP8 per head (ue8m0 scales) and
    fold each head's scale into ``index_weights`` ``[T, H]`` together with
    ``weight_scale``.

    Returns ``(q_fp8 [T, P, D], weights [T, P] fp32)`` where ``P =
    padded_num_heads``; heads ``[H, P)`` are zero.
    """
    num_tokens, num_heads, head_dim = index_q.shape
    assert index_q.stride(-1) == 1
    assert index_weights.shape == (num_tokens, num_heads)
    assert triton.next_power_of_2(head_dim) == head_dim
    alloc = torch.empty if padded_num_heads == num_heads else torch.zeros
    q_fp8 = alloc(
        (num_tokens, padded_num_heads, head_dim),
        dtype=_FP8_DTYPE,
        device=index_q.device,
    )
    weights = alloc(
        (num_tokens, padded_num_heads), dtype=torch.float32, device=index_q.device
    )
    if num_tokens == 0:
        return q_fp8, weights
    _indexer_q_quant_fold_weights_kernel[(num_tokens, num_heads)](
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_weights,
        index_weights.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weight_scale,
        HEAD_DIM=head_dim,
        FP8_MAX=_FP8_MAX,
        USE_FNUZ=_USE_FNUZ,
        num_warps=1,
    )
    return q_fp8, weights
