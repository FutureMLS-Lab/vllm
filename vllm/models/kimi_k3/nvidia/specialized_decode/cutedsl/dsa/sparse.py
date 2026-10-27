# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA decode over the fp8_ds_mla cache for the local heads only.

One cluster of ``splits`` CTAs per query row (token); CTA s attends top-k entries
[s * 2048 / splits, (s + 1) * 2048 / splits) of its row in chunks of 64:

  indices    logical top-k positions -> cache slots through the block table, once
             per CTA (row t belongs to request t // q; -1 entries are masked)
  KV         656 B rows bulk-copied two chunks ahead, dequantized to FP16 (FP8 x its
             power-of-two tile scale is exact in FP16; k_pe and the query are BF16
             values, exact in FP16 within its range)
  QK^T       FP16 mma.sync, the H heads padded to 16 rows; warp w owns tokens 8w..
  softmax    online, base 2 (scale * log2 e), one half-warp per head row
  PV         FP16 P x FP16 V (the dequantized latent), warp w owns columns 64w..

The splits merge in the cluster in one exchange: every CTA pushes its (max, sum)
rows to all peers and its unnormalized column block j to CTA j, which weights the
splits and stores BF16 [B, H, 512]. Rows without a valid entry are zero."""

import math

import cutlass
import cutlass.cute as cute
from cutlass import Float16, Float32, Int32, Uint32

from ..common.primitives import (
    alloc,
    asm,
    at,
    bf16x2,
    copy_g2s,
    launch,
    ldmatrix,
    load_words,
    mma_into,
    push_float,
    store_shared_words,
    store_words,
    unpack_bf16,
    wait,
    zeros,
)
from ..mla.query import half_warp_max

TOPK, CHUNK, ROW, PITCH = 2048, 64, 656, 584  # PITCH: BF16 KV / query row
UNIT = 64  # tokens per cache unit (the block table's granularity)
UNIT_BYTES = 50512
LOG2E = 1.4426950408889634


@cute.jit
def half_warp_sum(value: Float32):
    for shift in cutlass.range_constexpr(3, -1, -1):
        value += cute.arch.shuffle_sync_bfly(value, 1 << shift)
    return value


@asm((Uint32, Uint32), pure=True)
def fp8x4_to_f16(word: Uint32, scale: Uint32):
    """Four E4M3 bytes (low byte first) times the FP16x2 ``scale`` as two FP16x2
    words (a power-of-two scale keeps every product exact in FP16)."""
    code = (
        "{ .reg .b16 lo,hi; .reg .b32 x,y; mov.b32 {lo,hi},$2; "
        "cvt.rn.f16x2.e4m3x2 x,lo; cvt.rn.f16x2.e4m3x2 y,hi; "
        "mul.rn.f16x2 $0,x,$3; mul.rn.f16x2 $1,y,$3; }"
    )
    return code, "=r,=r,r,r", (word, scale)


@asm(Uint32, pure=True)
def f16x2(low, high):
    return "cvt.rn.f16x2.f32 $0,$2,$1;", "=r,f,f", (Float32(low), Float32(high))


def splits_for(batch):
    """Cluster size per query row: about one wave of CTAs."""
    return 16 if batch <= 8 else 8 if batch <= 16 else 4 if batch <= 32 else 2


class SparseMlaDecode:
    """``heads`` local heads (<= 16), ``batch`` query rows, ``q`` rows per request."""

    def __init__(self, heads: int, batch: int, q: int = 1, splits: int | None = None):
        self.heads, self.batch, self.q = heads, batch, q
        self.splits = splits or splits_for(batch)
        self.span = TOPK // self.splits
        self.chunks = self.span // CHUNK
        self.columns = 512 // self.splits

    @cute.jit
    def __call__(self, *args):
        grid = (self.splits, self.batch, 1)
        launch(self.kernel, args, grid, 256, cluster=(self.splits, 1, 1))

    @cute.kernel
    def kernel(self, args):
        query, topk, table, cache, out, scale = args
        tid, _, _ = cute.arch.thread_idx()
        split, row, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        S, H, C = self.splits, self.heads, self.columns
        smem = cutlass.utils.SmemAllocator()
        qs = alloc(smem, Float16, (16, PITCH), 128)
        raw = alloc(smem, cutlass.Uint8, (2, CHUNK, ROW), 128)
        kv = alloc(smem, Float16, (CHUNK, PITCH), 128)
        scores = alloc(smem, Float32, (16, CHUNK + 4))
        probs = alloc(smem, Float16, (16, CHUNK + 8), 128)
        alpha = alloc(smem, Float32, 16)
        slots = alloc(smem, Int32, self.span)
        receive = alloc(smem, Float32, (S, 16, C))
        stats = alloc(smem, Float32, (S, 32))
        bar = smem.allocate(cutlass.Uint64, 1)
        full = smem.allocate(cutlass.Uint64, 2)
        if tid == 0:
            for b in cutlass.range_constexpr(3):
                cute.arch.mbarrier_init(bar if b == 0 else full + b - 1, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.mbarrier_arrive_and_expect_tx(bar, S * (32 + 16 * C) * 4)
        cute.arch.cluster_arrive_relaxed()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        # Query rows (heads past H are zero rows of the 16-row MMA tile).
        for c in range(tid, 16 * 72, 256):
            h, j = c // 72, c % 72
            target = qs.iterator + h * PITCH + j * 8
            if h < H:
                words = load_words(at(query, row, h, j * 8), count=4)
                store_shared_words(target, self.to_f16(words))
            else:
                store_shared_words(target, (Uint32(0),) * 4)
        first, request = split * self.span, row // self.q
        for i in range(tid, self.span, 256):
            index = topk[row, first + i]
            slot = Int32(-1)
            if index >= 0:
                slot = table[request, index // UNIT] * UNIT + index % UNIT
            slots[i] = slot
        cute.arch.sync_threads()
        self.gather(cache, raw, slots, full, 0, tid)
        if cutlass.const_expr(self.chunks > 1):
            self.gather(cache, raw, slots, full, 1, tid)
        # Online softmax state of head row tid // 16 (one half-warp per row).
        m_run, l_run = Float32(-math.inf), Float32(0)
        acc = zeros((8, 4), Float32)
        log2_scale = scale * Float32(LOG2E)
        for c in range(self.chunks):
            buffer = c % 2
            wait(full + buffer, (c // 2) % 2, scope="cta")
            cute.arch.sync_threads()  # the previous chunk's PV is done with kv
            self.dequantize(raw, kv, slots, c, warp, lane)
            cute.arch.sync_threads()
            if c + 2 < self.chunks:
                self.gather(cache, raw, slots, full, c + 2, tid)
            # S = Q K^T over 576 columns: warp w, tokens 8w..8w+7 (two chains).
            s_acc = zeros((2, 4), Float32)
            for kk in cutlass.range_constexpr(18):
                k0 = kk * 32
                bq = ldmatrix(
                    kv.iterator + (8 * warp + lane % 8) * PITCH + k0 + 8 * (lane // 8)
                )
                for half in cutlass.range_constexpr(2):
                    a = ldmatrix(
                        qs.iterator
                        + (lane % 16) * PITCH
                        + k0
                        + 16 * half
                        + 8 * (lane // 16)
                    )
                    b = bq[2 * half : 2 * half + 2]
                    mma_into(s_acc[half, None], a, b)
            for e in cutlass.range_constexpr(4):
                h, t = group + 8 * (e // 2), 8 * warp + 2 * quarter + e % 2
                score = (s_acc[0, e] + s_acc[1, e]) * log2_scale
                if slots[c * CHUNK + t] < 0:
                    score = Float32(-math.inf)
                scores[h, t] = score
            cute.arch.sync_threads()
            # Online softmax: half-warp tid // 16 owns head row r, 4 tokens per lane.
            r, t0 = tid // 16, (tid % 16) * 4
            top = Float32(-math.inf)
            for i in cutlass.range_constexpr(4):
                top = cute.arch.fmax(top, scores[r, t0 + i])
            m_new = cute.arch.fmax(m_run, half_warp_max(top))
            base = m_new
            if m_new == Float32(-math.inf):
                base = Float32(0)
            total = Float32(0)
            p = cute.make_rmem_tensor(4, Float32)
            for i in cutlass.range_constexpr(4):
                p[i] = cute.math.exp2(scores[r, t0 + i] - base, fastmath=True)
                total += p[i]
            store_shared_words(
                probs.iterator + r * (CHUNK + 8) + t0,
                (f16x2(p[0], p[1]), f16x2(p[2], p[3])),
            )
            factor = cute.math.exp2(m_run - base, fastmath=True)
            l_run = l_run * factor + half_warp_sum(total)
            m_run = m_new
            if tid % 16 == 0:
                alpha[r] = factor
            cute.arch.sync_threads()
            # O = alpha O + P V: warp w owns latent columns 64w..64w+63.
            for n in cutlass.range_constexpr(8):
                acc[n, 0] *= alpha[group]
                acc[n, 1] *= alpha[group]
                acc[n, 2] *= alpha[group + 8]
                acc[n, 3] *= alpha[group + 8]
            for ks in cutlass.range_constexpr(CHUNK // 16):
                a = ldmatrix(
                    probs.iterator
                    + (lane % 16) * (CHUNK + 8)
                    + 16 * ks
                    + 8 * (lane // 16)
                )
                for n2 in cutlass.range_constexpr(4):
                    col = 64 * warp + 16 * n2 + 8 * (lane // 16)
                    b = ldmatrix(
                        kv.iterator + (16 * ks + lane % 16) * PITCH + col, trans=True
                    )
                    mma_into(acc[2 * n2, None], a, b[0:2])
                    mma_into(acc[2 * n2 + 1, None], a, b[2:4])
        # One exchange: (max, sum) rows to every peer, column blocks to their owner.
        cute.arch.cluster_wait()  # every peer has initialized its barrier
        r = tid // 16
        if tid % 16 == 0:
            for peer in cutlass.range_constexpr(S):
                push_float(stats.iterator + split * 32 + r, m_run, bar, peer)
                push_float(stats.iterator + split * 32 + 16 + r, l_run, bar, peer)
        for n in cutlass.range_constexpr(8):
            col = 64 * warp + 8 * n + 2 * quarter
            owner, local = col // C, col % C
            for half in cutlass.range_constexpr(2):
                pair = acc[n, 2 * half], acc[n, 2 * half + 1]
                target = at(receive, split, group + 8 * half, local)
                push_float(target, pair, bar, owner)
        cute.arch.cluster_arrive_relaxed()
        wait(bar)
        for c in range(tid, H * C // 2, 256):
            h, j = c // (C // 2), 2 * (c % (C // 2))
            top = Float32(-math.inf)
            for s in cutlass.range_constexpr(S):
                top = cute.arch.fmax(top, stats[s, h])
            lo, hi, total = Float32(0), Float32(0), Float32(0)
            if top != Float32(-math.inf):
                for s in cutlass.range_constexpr(S):
                    weight = cute.math.exp2(stats[s, h] - top, fastmath=True)
                    total += weight * stats[s, 16 + h]
                    lo += weight * receive[s, h, j]
                    hi += weight * receive[s, h, j + 1]
                lo, hi = lo / total, hi / total
            store_words(at(out, row, h, split * C + j), (bf16x2(lo, hi),))
        cute.arch.cluster_wait()  # no CTA exits while its pushes may be in flight

    @cute.jit
    def gather(self, cache, raw, slots, full, chunk, tid):
        """Bulk-copy the 64 KV rows of ``chunk`` into raw[chunk % 2] on its barrier
        (masked entries read slot 0)."""
        buf = chunk % 2
        if tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(full + buf, CHUNK * ROW)
        if tid < CHUNK:
            slot = slots[chunk * CHUNK + tid]
            if slot < 0:
                slot = Int32(0)
            source = at(cache, slot // UNIT, slot % UNIT, 0)
            target = raw.iterator + (buf * CHUNK + tid) * ROW
            copy_g2s(target, source, ROW, full + buf)

    @cute.jit
    def dequantize(self, raw, kv, slots, chunk, warp, lane):
        """Chunk ``chunk``'s raw rows (FP8 latent, tile scales, BF16 k_pe) -> kv FP16
        rows; warp w converts rows w, w + 8, ..; lane l the latent bytes 16l.. (tile
        l // 8). Masked entries become zero rows: the slot-0 row they were read from
        may hold any bytes (NaN codes included), which P = 0 would not cancel."""
        buf = chunk % 2
        for i in cutlass.range_constexpr(CHUNK // 8):
            t = warp + 8 * i
            keep = Uint32(~(slots[chunk * CHUNK + t] >> 31))  # all ones when valid
            source = raw.iterator + (buf * CHUNK + t) * ROW
            words = load_words(source + 16 * lane, count=4, shared=True, mutable=True)
            scale_word = load_words(
                source + 512 + 4 * (lane // 8),
                count=1,
                shared=True,
                mutable=True,
                result_type=Float32,
                word_type="f32",
            )
            tile_scale = f16x2(scale_word[0], scale_word[0])
            packed = []
            for w in cutlass.range_constexpr(4):
                packed += [x & keep for x in fp8x4_to_f16(words[w], tile_scale)]
            target = kv.iterator + t * PITCH + 16 * lane
            store_shared_words(target, tuple(packed[:4]))
            store_shared_words(target + 8, tuple(packed[4:]))
            if lane < 8:  # k_pe
                rope = load_words(
                    source + 528 + 16 * lane, count=4, shared=True, mutable=True
                )
                rope = tuple(x & keep for x in self.to_f16(rope))
                store_shared_words(kv.iterator + t * PITCH + 512 + 8 * lane, rope)

    @cute.jit
    def to_f16(self, words):
        """Four BF16x2 words as FP16x2."""
        converted = []
        for w in cutlass.range_constexpr(4):
            low, high = unpack_bf16(words[w])
            converted.append(f16x2(low, high))
        return tuple(converted)
