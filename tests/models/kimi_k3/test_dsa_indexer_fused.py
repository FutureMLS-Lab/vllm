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
    dsa_fused_prologue,
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
BLOCK = 64  # K3.1 kernel block size
MLA_BYTES = 656  # fp8_ds_mla bytes per token ahead of the indexer page
ROW_STRIDE = 50512  # packed block row: 64 x (656 + 132), 656-aligned
TOPK = 2048


def _make_cache(packed: bool, num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (backing bytes, indexer cache view [num_blocks, BLOCK, ENTRY])."""
    if not packed:
        raw = torch.zeros(num_blocks, BLOCK, ENTRY, dtype=torch.uint8, device="cuda")
        return raw, raw
    # One DSA layer's indexer page inside a packed block row.
    raw = torch.zeros(num_blocks, ROW_STRIDE, dtype=torch.uint8, device="cuda")
    view = torch.as_strided(
        raw,
        size=(num_blocks, BLOCK, ENTRY),
        stride=(ROW_STRIDE, ENTRY, 1),
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


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 7, 300])
@torch.inference_mode()
def test_k_norm_quant_insert_matches_unfused(packed: bool, num_tokens: int):
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

    ref_raw, ref_view = _make_cache(packed, num_blocks)
    ref_k = F.layer_norm(k.float(), (HEAD_DIM,), weight, bias, eps).to(k.dtype)
    ops.indexer_k_quant_and_cache(
        ref_k[:num_slots].contiguous(),
        ref_view,
        slots,
        HEAD_DIM,
        "mxfp8" if mqa_logits_use_mx_scales() else "ue8m0",
    )

    out_raw, out_view = _make_cache(packed, num_blocks)
    topk = torch.full((num_tokens + 5, TOPK), 7, dtype=torch.int32, device="cuda")
    indexer_k_norm_quant_insert(k, weight, bias, eps, slots, out_view, topk, num_tokens)

    # Only LayerNorm reduction order differs, so bytes match except for rare
    # bf16 rounding ties; dequantized values agree to fp8 precision.
    mismatch = (ref_raw != out_raw).float().mean().item()
    assert mismatch < 1e-3, f"{mismatch:.2e} of cache bytes differ"
    torch.testing.assert_close(
        _dequant(out_view), _dequant(ref_view), atol=0.07, rtol=0.07
    )
    if packed:
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


Q_LORA = 1536
KV_LORA = 512
ROPE = 64


def _rms_norm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return y.to(x.dtype)


def _dsa_caches(packed: bool, num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(fp8_ds_mla cache, indexer cache) views, both pages of one block row
    when packed."""
    if not packed:
        mla = torch.zeros(
            num_blocks, BLOCK, MLA_BYTES, dtype=torch.uint8, device="cuda"
        )
        return mla, _make_cache(False, num_blocks)[1]
    raw, index = _make_cache(True, num_blocks)
    return torch.as_strided(
        raw, (num_blocks, BLOCK, MLA_BYTES), (ROW_STRIDE, MLA_BYTES, 1)
    ), index


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 7, 300])
@torch.inference_mode()
def test_fused_prologue_matches_unfused(packed: bool, num_tokens: int):
    """dsa_fused_prologue == q/kv RMSNorm + fp8_ds_mla insert + indexer key insert
    + top-k fill, on strided views of the projection outputs."""
    torch.manual_seed(0)
    num_blocks = (num_tokens + 2 * BLOCK - 1) // BLOCK + 1
    qkv = torch.randn(
        num_tokens, Q_LORA + KV_LORA + ROPE, dtype=torch.bfloat16, device="cuda"
    )
    q_c, kv_c, k_pe = qkv.split([Q_LORA, KV_LORA, ROPE], dim=-1)
    k = torch.randn(
        num_tokens, HEAD_DIM + NUM_HEADS, dtype=torch.bfloat16, device="cuda"
    )[:, :HEAD_DIM]
    q_w = torch.rand(Q_LORA, dtype=torch.bfloat16, device="cuda") + 0.5
    kv_w = torch.rand(KV_LORA, dtype=torch.bfloat16, device="cuda") + 0.5
    k_w = torch.rand(HEAD_DIM, dtype=torch.float32, device="cuda") + 0.5
    k_b = torch.randn(HEAD_DIM, dtype=torch.float32, device="cuda") * 0.1
    eps = 1e-6
    # Fewer slots than rows (cudagraph padding), plus a padded (-1) slot.
    num_slots = max(num_tokens - 1, 1)
    slots = torch.randperm(num_blocks * BLOCK, device="cuda")[:num_slots]
    if num_slots > 2:
        slots[1] = -1
    slots = slots.to(torch.int64)

    ref_mla, ref_index = _dsa_caches(packed, num_blocks)
    ref_kv = _rms_norm_ref(kv_c, kv_w, eps)
    ops.concat_and_cache_mla(
        ref_kv[:num_slots].contiguous(),
        k_pe[:num_slots].contiguous(),
        ref_mla,
        slots,
        "fp8_ds_mla",
        torch.ones(1, dtype=torch.float32, device="cuda"),
    )
    ref_k = F.layer_norm(k.float(), (HEAD_DIM,), k_w, k_b, eps).to(k.dtype)
    ops.indexer_k_quant_and_cache(
        ref_k[:num_slots].contiguous(),
        ref_index,
        slots,
        HEAD_DIM,
        "mxfp8" if mqa_logits_use_mx_scales() else "ue8m0",
    )

    mla, index = _dsa_caches(packed, num_blocks)
    topk = torch.full((num_tokens + 5, TOPK), 7, dtype=torch.int32, device="cuda")
    q_out, kv_out = dsa_fused_prologue(
        q_c, kv_c, k_pe, k, q_w, kv_w, k_w, k_b, eps, topk, mla, slots, index, slots
    )

    # Only reduction order differs, so bytes match except for rare bf16 ties.
    torch.testing.assert_close(
        q_out, _rms_norm_ref(q_c, q_w, eps), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(kv_out, ref_kv, atol=2e-2, rtol=2e-2)
    mismatch = (mla != ref_mla).float().mean().item()
    assert mismatch < 2e-3, f"{mismatch:.2e} of MLA cache bytes differ"
    torch.testing.assert_close(
        _dequant(index), _dequant(ref_index), atol=0.07, rtol=0.07
    )
    assert torch.all(topk[:num_tokens] == -1)
    assert torch.all(topk[num_tokens:] == 7)


@torch.inference_mode()
def test_fused_prologue_without_caches():
    """Profile / dummy runs: norms only, no cache access."""
    qkv = torch.randn(3, Q_LORA + KV_LORA + ROPE, dtype=torch.bfloat16, device="cuda")
    q_c, kv_c, k_pe = qkv.split([Q_LORA, KV_LORA, ROPE], dim=-1)
    k = torch.randn(3, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(Q_LORA, dtype=torch.bfloat16, device="cuda")
    kn = torch.ones(HEAD_DIM, dtype=torch.float32, device="cuda")
    topk = torch.zeros(3, TOPK, dtype=torch.int32, device="cuda")
    q_out, kv_out = dsa_fused_prologue(
        q_c, kv_c, k_pe, k, w, w[:KV_LORA], kn, kn, 1e-6, topk,
        None, None, None, None,
    )  # fmt: skip
    torch.testing.assert_close(q_out, _rms_norm_ref(q_c, w, 1e-6))
    torch.testing.assert_close(kv_out, _rms_norm_ref(kv_c, w[:KV_LORA], 1e-6))
