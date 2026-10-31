# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA indexer inputs of one decode step, one 256-thread CTA per token:

- the indexer query [32, 128] (the BF16 wq_b projection of the normalized q_c) in FP8
  with one power-of-two scale per head, the scale folded into the head weights
  wv * scale / sqrt(32 * 128) as indexer_q_quant_fold_weights computes them, in the
  BF16 SM100's MQA logits take (mqa_logits_mx_inputs);
- the shared indexer key: LayerNorm (FP32) of the wk projection, BF16, then FP8 with
  a power-of-two scale into the indexer cache, as indexer_k_norm_quant_insert writes
  it: per block, the FP8 keys of its tokens, then their scales: FP32, or (``mx``: the
  SM100 MQA logits kernels' MX scales, mqa_logits_use_mx_scales) the scale's
  exponent as four equal packed UE8M0 bytes.

A half-warp owns one head (8 columns per lane); warp 0's low half also owns the key.
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32

from ..common.primitives import (
    at,
    float_bits,
    launch,
    load_bf16x8,
    pack_fp8,
    store_words,
)
from ..mla.query import half_warp_max, pow2_ceil

HEADS, DIM = 32, 128


@cute.jit
def half_warp_sum(value: Float32):
    for shift in cutlass.range_constexpr(3, -1, -1):
        value += cute.arch.shuffle_sync_bfly(value, 1 << shift)
    return value


class DsaIndex:
    """``key``/``weight``: column offsets of the wk and wv projections in the packed
    projection row; ``mx``: key scales as packed UE8M0 (MX) bytes."""

    def __init__(self, key: int, weight: int, mx: bool = False):
        self.key, self.weight, self.mx = key, weight, mx

    @cute.jit
    def __call__(self, *args):
        batch = args[1].shape[0]
        launch(self.kernel, args, (batch, 1, 1), 256)

    @cute.kernel
    def kernel(self, args):
        projected, iq, norm_w, norm_b, cache, slots, q_out, w_out, eps = args
        tid, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        half, column = lane // 16, (lane % 16) * 8
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        for p in cutlass.range_constexpr(2):
            head = p * 16 + warp * 2 + half
            values = load_bf16x8(at(iq, token, head * DIM + column), mutable=True)
            amax = Float32(1e-10)
            for i in cutlass.range_constexpr(8):
                amax = cute.arch.fmax(amax, abs(values[i]))
            amax = half_warp_max(amax)
            scale = pow2_ceil(amax / Float32(448))  # amax >= 1e-10
            inverse = Float32(1) / scale
            words = [
                pack_fp8(tuple(values[4 * w + i] * inverse for i in range(4)))
                for w in range(2)
            ]
            store_words(at(q_out, token, head, column), tuple(words))
            if lane % 16 == 0:
                wv = Float32(projected[token, self.weight + head])
                fold = (wv * scale) * Float32((HEADS * DIM) ** -0.5)
                w_out[token, head] = BFloat16(fold)
        if warp == 0:
            key = projected, self.key, norm_w, norm_b, cache, slots, token, eps
            insert_key(*key, self.mx, lane)


@cute.jit
def insert_key(
    projected,
    key: cutlass.Constexpr[int],
    norm_w,
    norm_b,
    cache,
    slots,
    token,
    eps,
    mx: cutlass.Constexpr[bool],
    lane,
):
    """One warp: LayerNorm (FP32) of row ``token``'s wk projection (at column
    ``key``), BF16, FP8 with a power-of-two scale into its indexer cache slot (the
    low half-warp writes; ``mx``: packed UE8M0 scales). Nothing for a slot < 0."""
    half, column = lane // 16, (lane % 16) * 8
    slot = slots[token]
    if slot >= 0:
        x = load_bf16x8(at(projected, token, key + column), mutable=True)
        total = Float32(0)
        for i in cutlass.range_constexpr(8):
            total += x[i]
        mean = half_warp_sum(total) / Float32(DIM)
        square = Float32(0)
        for i in cutlass.range_constexpr(8):
            square += (x[i] - mean) * (x[i] - mean)
        variance = half_warp_sum(square) / Float32(DIM)
        rstd = cute.math.rsqrt(variance + eps, fastmath=False)
        normed = cute.make_rmem_tensor(8, Float32)
        amax = Float32(0)
        for i in cutlass.range_constexpr(8):
            y = (x[i] - mean) * rstd * norm_w[column + i] + norm_b[column + i]
            normed[i] = Float32(BFloat16(y))
            amax = cute.arch.fmax(amax, abs(normed[i]))
        amax = half_warp_max(amax)
        scale = pow2_ceil(cute.arch.fmax(amax, Float32(1e-4)) / Float32(448))
        inverse = Float32(1) / scale
        tokens = cache.shape[1]
        page, offset = slot // tokens, slot % tokens
        base = at(cache, page, 0, 0)
        if half == 0:
            words = [
                pack_fp8(tuple(normed[4 * w + i] * inverse for i in range(4)))
                for w in range(2)
            ]
            store_words(base + offset * DIM + column, tuple(words))
            if lane == 0:
                scales = base + tokens * DIM
                word = float_bits(scale)
                if cutlass.const_expr(mx):
                    word = ((word >> 23) & 0xFF) * 0x01010101
                store_words(scales + offset * 4, (word,))
