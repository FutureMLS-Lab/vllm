# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA indexer inputs of one decode step, one 256-thread CTA per token:

- the indexer query [32, 128] (the BF16 wq_b projection of the normalized q_c) in FP8
  with one power-of-two scale per head (per_token_group_quant_fp8, ue8m0), the scale
  folded into the BF16 head weights SM100's MQA logits take:
  bf16(bf16(wv / sqrt(32)) * scale / sqrt(128));
- the shared indexer key: LayerNorm (FP32) of the wk projection, BF16, then FP8 with
  a power-of-two scale (indexer_k_quant_and_cache, ue8m0) into the indexer cache:
  per block, the FP8 keys of its tokens, then their FP32 scales.

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
    projection row."""

    def __init__(self, key: int, weight: int):
        self.key, self.weight = key, weight

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
            scale = pow2_ceil(cute.arch.fmax(amax / Float32(448), Float32(1e-10)))
            inverse = Float32(1) / scale
            words = [
                pack_fp8(tuple(values[4 * w + i] * inverse for i in range(4)))
                for w in range(2)
            ]
            store_words(at(q_out, token, head, column), tuple(words))
            if lane % 16 == 0:
                wv = Float32(projected[token, self.weight + head])
                scaled = Float32(BFloat16(wv * Float32(HEADS**-0.5)))
                w_out[token, head] = BFloat16((scaled * scale) * Float32(DIM**-0.5))
        slot = slots[token]
        if (warp == 0) & (slot >= 0):
            x = load_bf16x8(at(projected, token, self.key + column), mutable=True)
            total = Float32(0)
            for i in cutlass.range_constexpr(8):
                total += x[i]
            mean = half_warp_sum(total) / Float32(DIM)
            square = Float32(0)
            for i in cutlass.range_constexpr(8):
                square += (x[i] - mean) * (x[i] - mean)
            variance = half_warp_sum(square) / Float32(DIM)
            rstd = cute.math.rsqrt(variance + eps, fastmath=False)
            key = cute.make_rmem_tensor(8, Float32)
            amax = Float32(0)
            for i in cutlass.range_constexpr(8):
                y = (x[i] - mean) * rstd * norm_w[column + i] + norm_b[column + i]
                key[i] = Float32(BFloat16(y))
                amax = cute.arch.fmax(amax, abs(key[i]))
            amax = half_warp_max(amax)
            scale = pow2_ceil(cute.arch.fmax(amax, Float32(1e-4)) / Float32(448))
            inverse = Float32(1) / scale
            tokens = cache.shape[1]
            page, offset = slot // tokens, slot % tokens
            base = at(cache, page, 0, 0)
            if half == 0:
                words = [
                    pack_fp8(tuple(key[4 * w + i] * inverse for i in range(4)))
                    for w in range(2)
                ]
                store_words(base + offset * DIM + column, tuple(words))
                if lane == 0:
                    scales = base + tokens * DIM
                    store_words(scales + offset * 4, (float_bits(scale),))
