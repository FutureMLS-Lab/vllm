# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLASHINFER_MLA_SPARSE on FlashInfer's prims_ts sparse MLA kernel (SM100).

Kimi K3.1 DSA shapes: D576, top-k capacity 2048, 96 query heads (DCP8 gathered
heads, zero-padded to 128 for the 2-CTA schedule), per-tensor fp8 and
nvfp4_ds_mla caches written by the production cache writer into a packed
(BLHNC-strided) block layout. Every cache row no live index addresses (page 0,
the other layers' rows between blocks, unselected rows, rows named by entries
past a row's length) is NaN, so any stray read fails the comparison.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla

if not (
    current_platform.is_cuda()
    and current_platform.is_device_capability_family(100)
    and has_flashinfer_prims_ts_sparse_mla()
):
    pytest.skip(
        "needs SM100/SM103 and FlashInfer's prims_ts sparse MLA",
        allow_module_level=True,
    )

from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fims  # noqa: E402
from vllm.v1.attention.backends.mla.sparse_utils import (  # noqa: E402
    triton_filter_and_convert_dcp_index,
)
from vllm.v1.attention.ops.dcp import CPTritonContext, correct_attn_out  # noqa: E402

E4M3 = torch.float8_e4m3fn
KV_LORA_RANK, ROPE = 512, 64
TOPK = 2048
BLOCK_SIZE = 64
# Packed layout: other layers' pages sit between this layer's blocks.
BLOCK_STRIDE_ROWS = 3 * BLOCK_SIZE
ROW_BYTES = {"fp8": 576, "nvfp4_ds_mla": 352}
SCALE = 192**-0.5
# Row capacities as _prims_ts_row_buckets plans them: powers of two (the test
# shapes below have no CUDA graph sizes among them, so each row count below the
# largest capacity runs on a larger plan with packed-query offsets).
BUCKETS = tuple(1 << i for i in range(10))
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _impl(kv_cache_dtype: str, return_lse: bool = True, buckets=BUCKETS):
    impl = object.__new__(fims.FlashInferMLASparseImpl)
    impl.scale = SCALE
    impl.kv_lora_rank = KV_LORA_RANK
    impl.qk_rope_head_dim = ROPE
    impl.use_prims_ts = True
    impl.prims_ts_format = fims._PRIMS_TS_FORMATS[kv_cache_dtype]
    impl.need_to_return_lse_for_decode = return_lse
    impl.prims_ts_buckets = buckets
    return impl


def _cache(kv_cache_dtype: str, num_blocks: int, device) -> torch.Tensor:
    """[num_blocks, BLOCK_SIZE, row bytes] uint8 view of a NaN-filled packed
    buffer whose blocks are BLOCK_STRIDE_ROWS rows apart."""
    raw = torch.full(
        (num_blocks, BLOCK_STRIDE_ROWS, ROW_BYTES[kv_cache_dtype]),
        0xFF,
        dtype=torch.uint8,
        device=device,
    )
    return raw[:, :BLOCK_SIZE]


def _write(cache, kv_cache_dtype, kv_c, k_pe, slots, k_scale):
    """Production writer: token i into block slots[i] // BLOCK_SIZE, offset
    slots[i] % BLOCK_SIZE."""
    ops.concat_and_cache_mla(
        kv_c,
        k_pe,
        cache,
        slots.long(),
        kv_cache_dtype=kv_cache_dtype,
        scale=torch.tensor(k_scale, dtype=torch.float32, device=kv_c.device),
    )


def _dequant_rows(cache, kv_cache_dtype, k_scale) -> torch.Tensor:
    """fp32 [rows, 576] of every storage row (flat, packed stride), as the
    kernel loads them: per-tensor e4m3 times the descale, or NVFP4 NoPE
    expanded to e4m3(e2m1 * sf) (FlashInfer's loader) and unscaled e4m3 RoPE."""
    rows, _ = fims.flat_kv_row_view(cache, BLOCK_SIZE)
    if kv_cache_dtype == "fp8":
        return rows.view(E4M3).float() * k_scale
    table = torch.tensor(
        _E2M1 + [-v for v in _E2M1], dtype=torch.float32, device=rows.device
    )
    packed = rows[:, :256]
    nibbles = torch.stack([packed & 0xF, packed >> 4], -1).flatten(-2).long()
    sf = rows[:, 320:352].unflatten(-1, (4, 8)).transpose(-1, -2).flatten(-2)
    sf = sf.contiguous().view(E4M3).float()
    nope = (table[nibbles].unflatten(-1, (32, 16)) * sf[..., None]).flatten(-2)
    nope = nope.to(E4M3).float()
    rope = rows[:, 256:320].contiguous().view(E4M3).float()
    return torch.cat([nope, rope], -1)


def _quant_q(q: torch.Tensor, q_scale: float) -> torch.Tensor:
    """The layer's static per-tensor E4M3 query quantization."""
    t, h, d = q.shape
    q8, _ = ops.scaled_fp8_quant(
        q.reshape(t, h * d),
        torch.tensor([q_scale], dtype=torch.float32, device=q.device),
    )
    return q8.view(t, h, d)


def _reference(q8, q_scale, kv, indices, lengths):
    """fp32 sparse attention; natural-log LSE; empty rows -> (0, -inf)."""
    t, h, _ = q8.shape
    out = torch.zeros(t, h, KV_LORA_RANK, device=q8.device)
    lse = torch.full((t, h), -math.inf, device=q8.device)
    q = q8.float() * q_scale
    for i in range(t):
        sel = indices[i, : lengths[i]]
        sel = sel[sel >= 0].long()
        if sel.numel() == 0:
            continue
        k = kv[sel]
        s = q[i] @ k.T * SCALE
        lse[i] = torch.logsumexp(s, -1)
        out[i] = torch.softmax(s, -1) @ k[:, :KV_LORA_RANK]
    return out, lse


def _check(out, lse, ref_out, ref_lse, kv_cache_dtype):
    assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
    assert torch.isfinite(out).all(), "NaN/inf output: a masked row was read"
    empty = torch.isinf(ref_lse)
    assert torch.equal(torch.isinf(lse), empty)
    assert (lse[empty] < 0).all() and (out[empty.nonzero(as_tuple=True)] == 0).all()
    live = ~empty
    if live.any():
        o, r = out.float()[live], ref_out[live]
        rel = ((o - r).norm() / r.norm()).item()
        # E4M3 P (and the e4m3 Q/KV the reference already models).
        assert rel < (2e-2 if kv_cache_dtype == "fp8" else 3e-2), rel
        assert (lse[live] - ref_lse[live]).abs().max().item() < 2e-2


def _problem(
    kv_cache_dtype, counts, heads, k_scale, q_scale, seed, *, holes=False, blocks=None
):
    """Rows of local physical indices (compacted prefix, or with -1 holes
    inside the prefix when ``holes``) over live packed rows; entries past each
    row's length name poisoned rows."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = torch.device("cuda")
    num_blocks = blocks or 1 + math.ceil(max(max(counts), 1) * 2 / BLOCK_SIZE) + 4
    cache = _cache(kv_cache_dtype, num_blocks, dev)
    # Live tokens: every slot of blocks 1.. (page 0 stays poisoned).
    live_slots = torch.arange(BLOCK_SIZE, num_blocks * BLOCK_SIZE, device=dev)
    phys = (live_slots // BLOCK_SIZE) * BLOCK_STRIDE_ROWS + live_slots % BLOCK_SIZE
    t = len(counts)
    indices = torch.empty((t, TOPK), dtype=torch.int32, device=dev)
    lengths = torch.tensor(counts, dtype=torch.int32, device=dev)
    used = torch.zeros(phys.numel(), dtype=torch.bool, device=dev)
    for i, c in enumerate(counts):
        perm = torch.randperm(phys.numel(), generator=g, device=dev)
        row = phys[perm[:c]].to(torch.int32)
        if holes and c > 2:
            row[torch.randperm(c, generator=g, device=dev)[: c // 4]] = -1
        used[perm[:c]] |= row >= 0
        # Past the length: poisoned rows (page 0 and inter-block rows).
        tail = torch.randint(0, BLOCK_STRIDE_ROWS, (TOPK - c,), generator=g, device=dev)
        tail = torch.where(tail < BLOCK_SIZE, tail, tail % 7 + BLOCK_SIZE)
        indices[i] = torch.cat([row, tail.to(torch.int32)])
    # Write only the selected tokens; every other row stays NaN.
    kv_c = torch.randn(phys.numel(), KV_LORA_RANK, generator=g, device=dev) * 0.2
    k_pe = torch.randn(phys.numel(), ROPE, generator=g, device=dev) * 0.5
    sel = used.nonzero().squeeze(1)
    _write(
        cache,
        kv_cache_dtype,
        kv_c[sel].bfloat16(),
        k_pe[sel].bfloat16(),
        live_slots[sel],
        k_scale,
    )
    q = torch.randn(t, heads, KV_LORA_RANK + ROPE, generator=g, device=dev)
    q8 = _quant_q(q.bfloat16() * 4, q_scale)
    return cache, q8, indices, lengths


COUNTS = {
    # DCP8 local: about 1/8 of 2048, not multiples of 128, some empty.
    "dcp_local": [0, 1, 37, 128, 129, 251, 256, 300, 0, 255, 383, 512],
    "decode_b1": [251],
    "full": [2048, 2047, 1, 0, 1000],
}


@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "nvfp4_ds_mla"])
@pytest.mark.parametrize("heads", [96, 128, 16])
@pytest.mark.parametrize("case", list(COUNTS))
@pytest.mark.parametrize("holes", [False, True])
def test_prims_ts_matches_reference(kv_cache_dtype, heads, case, holes):
    torch.manual_seed(0)
    k_scale = 0.5 if kv_cache_dtype == "fp8" else 1.0
    q_scale = 2.0
    cache, q8, indices, lengths = _problem(
        kv_cache_dtype, COUNTS[case], heads, k_scale, q_scale, 1, holes=holes
    )
    layer = SimpleNamespace(_q_scale_float=q_scale, _k_scale_float=k_scale)
    out, lse = _impl(kv_cache_dtype)._forward_prims_ts(
        q8, cache, indices, lengths, layer, BLOCK_SIZE
    )
    kv = _dequant_rows(cache, kv_cache_dtype, k_scale)
    ref_out, ref_lse = _reference(q8, q_scale, kv, indices, lengths)
    _check(out, lse, ref_out, ref_lse, kv_cache_dtype)
    # 96 heads ran on the 128-head (2-CTA) plan.
    planned = {key[2] for key in fims._prims_ts_plans._plans}
    assert (128 if heads == 96 else heads) in planned


@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "nvfp4_ds_mla"])
def test_prims_ts_prefill_chunks(kv_cache_dtype):
    """Rows beyond the largest capacity run in chunks (with a ragged tail)."""
    counts = [(i * 37) % 300 for i in range(130)]
    cache, q8, indices, lengths = _problem(kv_cache_dtype, counts, 96, 1.0, 1.0, 2)
    layer = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=1.0)
    impl = _impl(kv_cache_dtype, buckets=(1, 2, 4, 8, 16, 32, 48))
    out, lse = impl._forward_prims_ts(q8, cache, indices, lengths, layer, BLOCK_SIZE)
    ref_out, ref_lse = _reference(
        q8, 1.0, _dequant_rows(cache, kv_cache_dtype, 1.0), indices, lengths
    )
    _check(out, lse, ref_out, ref_lse, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "nvfp4_ds_mla"])
def test_prims_ts_without_lse(kv_cache_dtype):
    counts = COUNTS["dcp_local"]
    cache, q8, indices, lengths = _problem(kv_cache_dtype, counts, 96, 1.0, 1.0, 3)
    layer = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=1.0)
    out, lse = _impl(kv_cache_dtype, return_lse=False)._forward_prims_ts(
        q8, cache, indices, lengths, layer, BLOCK_SIZE
    )
    assert lse is None
    ref_out, ref_lse = _reference(
        q8, 1.0, _dequant_rows(cache, kv_cache_dtype, 1.0), indices, lengths
    )
    _check(out, ref_lse, ref_out, ref_lse, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "nvfp4_ds_mla"])
@pytest.mark.parametrize("rows", [1, 5, 80, 320])
@pytest.mark.parametrize("exact", [False, True])
def test_prims_ts_cuda_graph_replay(kv_cache_dtype, rows, exact):
    """Plan in an eager warmup, capture, then replay on new inputs written in
    place (as decode graphs are replayed). ``exact``: the graph size is a
    planned capacity (as _prims_ts_row_buckets plans every graph size), else
    the call runs on the next larger capacity."""
    k_scale = 0.5 if kv_cache_dtype == "fp8" else 1.0
    layer = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=k_scale)
    impl = _impl(
        kv_cache_dtype, buckets=tuple(sorted({*BUCKETS, rows})) if exact else BUCKETS
    )

    def counts(seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, 400, (rows,), generator=g).tolist()

    blocks = 20
    cache, q8, indices, lengths = _problem(
        kv_cache_dtype, counts(10), 96, k_scale, 1.0, 10, blocks=blocks
    )
    impl._forward_prims_ts(q8, cache, indices, lengths, layer, BLOCK_SIZE)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = impl._forward_prims_ts(
            q8, cache, indices, lengths, layer, BLOCK_SIZE
        )
    for seed in (11, 12):
        cache2, q2, idx2, len2 = _problem(
            kv_cache_dtype, counts(seed), 96, k_scale, 1.0, seed, blocks=blocks
        )
        cache.copy_(cache2)
        q8.copy_(q2)
        indices.copy_(idx2)
        lengths.copy_(len2)
        graph.replay()
        torch.accelerator.synchronize()
        kv = _dequant_rows(cache, kv_cache_dtype, k_scale)
        ref_out, ref_lse = _reference(q8, 1.0, kv, indices, lengths)
        _check(out, lse, ref_out, ref_lse, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "nvfp4_ds_mla"])
@pytest.mark.parametrize("interleave", [1, 64])
@pytest.mark.parametrize("q_len", [1, 5])
def test_prims_ts_dcp8_merge(kv_cache_dtype, interleave, q_len):
    """DCP8: each rank attends its shard of the global top-k (converted and
    compacted by the production DCP index kernel) with the gathered 96 heads;
    the LSE-weighted merge (natural log) must equal attention over the whole
    top-k. Short rows leave ranks without entries (count 0)."""
    dcp, heads, k_scale = 8, 96, 0.5 if kv_cache_dtype == "fp8" else 1.0
    dev = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(7)
    seq_lens = [5000, 6, 700, 9000]
    num_reqs = len(seq_lens)
    tokens = num_reqs * q_len
    # Global per-request top-k (logical token ids), -1 padded.
    topk = torch.full((tokens, TOPK), -1, dtype=torch.int32, device=dev)
    for t in range(tokens):
        n_ctx = seq_lens[t // q_len] - (q_len - 1 - t % q_len)
        c = min(TOPK, n_ctx)
        topk[t, :c] = torch.randperm(n_ctx, generator=g, device=dev)[:c].int()
    req_id = torch.arange(num_reqs, device=dev, dtype=torch.int32).repeat_interleave(
        q_len
    )
    kv_c = torch.randn(sum(seq_lens), KV_LORA_RANK, generator=g, device=dev) * 0.2
    k_pe = torch.randn(sum(seq_lens), ROPE, generator=g, device=dev) * 0.5
    starts = [0]
    for s in seq_lens:
        starts.append(starts[-1] + s)
    q8 = _quant_q(
        torch.randn(tokens, heads, 576, generator=g, device=dev).bfloat16() * 4, 1.0
    )
    layer = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=k_scale)
    impl = _impl(kv_cache_dtype)

    outs, lses, ref_kv = [], [], torch.empty(sum(seq_lens), 576, device=dev)
    for rank in range(dcp):
        # This rank's local cache: block b of request r at physical block
        # 1 + r * blocks_per_req + b (page 0 unused).
        local_lens = [
            int(((torch.arange(s) // interleave) % dcp == rank).sum()) for s in seq_lens
        ]
        blocks_per_req = max(math.ceil(n / BLOCK_SIZE) for n in local_lens) + 1
        cache = _cache(kv_cache_dtype, 1 + num_reqs * blocks_per_req, dev)
        block_table = torch.tensor(
            [
                [1 + r * blocks_per_req + b for b in range(blocks_per_req)]
                for r in range(num_reqs)
            ],
            dtype=torch.int32,
            device=dev,
        )
        for r, s in enumerate(seq_lens):
            pos = torch.arange(s, device=dev)
            own = pos[(pos // interleave) % dcp == rank]
            local = (own // (interleave * dcp)) * interleave + own % interleave
            slots = block_table[r, local // BLOCK_SIZE].long() * BLOCK_SIZE + (
                local % BLOCK_SIZE
            )
            if own.numel():
                _write(
                    cache,
                    kv_cache_dtype,
                    kv_c[starts[r] + own].bfloat16(),
                    k_pe[starts[r] + own].bfloat16(),
                    slots,
                    k_scale,
                )
                rows = _dequant_rows(cache, kv_cache_dtype, k_scale)
                phys = (slots // BLOCK_SIZE) * BLOCK_STRIDE_ROWS + slots % BLOCK_SIZE
                ref_kv[starts[r] + own] = rows[phys]
        indices, counts = triton_filter_and_convert_dcp_index(
            req_id,
            block_table,
            topk,
            dcp_size=dcp,
            dcp_rank=rank,
            cp_kv_cache_interleave_size=interleave,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_STRIDE_ROWS=BLOCK_STRIDE_ROWS,
            NUM_TOPK_TOKENS=TOPK,
            return_valid_counts=True,
        )
        out, lse = impl._forward_prims_ts(q8, cache, indices, counts, layer, BLOCK_SIZE)
        outs.append(out)
        lses.append(lse)
    assert any((lse == -math.inf).all(-1).any() for lse in lses), "no empty rank row"
    stacked = torch.stack(lses)
    merged = sum(
        correct_attn_out(o.clone(), stacked, r, CPTritonContext(), True)[0].float()
        for r, o in enumerate(outs)
    )
    flat = topk.clone()
    for t in range(tokens):
        sel = flat[t] >= 0
        flat[t, sel] += starts[t // q_len]
    lengths = (flat >= 0).sum(-1).int()
    flat = torch.sort(flat, dim=-1, descending=True).values
    ref_out, _ = _reference(q8, 1.0, ref_kv, flat, lengths)
    rel = ((merged - ref_out).norm() / ref_out.norm()).item()
    assert rel < (2e-2 if kv_cache_dtype == "fp8" else 3e-2), rel


def test_prims_ts_planned_up_front():
    """Every capacity is planned at once (no planning while serving), and a
    geometry first seen during CUDA graph capture is an error, not a JIT."""
    impl = _impl("nvfp4_ds_mla", buckets=(1, 2, 4, 8, 37))
    plans = impl._get_prims_ts_plans(torch.device("cuda", 0), 96, TOPK)
    assert sorted(plans) == [1, 2, 4, 8, 37]
    num_geometries = len(fims._prims_ts_plans._plans)
    cache, q8, indices, lengths = _problem("nvfp4_ds_mla", [5, 0, 77], 96, 1, 1, 4)
    layer = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=1.0)
    impl._forward_prims_ts(q8, cache, indices, lengths, layer, BLOCK_SIZE)
    assert len(fims._prims_ts_plans._plans) == num_geometries
    with (
        torch.cuda.graph(torch.cuda.CUDAGraph()),
        pytest.raises(RuntimeError, match="not planned before CUDA graph capture"),
    ):
        _impl("nvfp4_ds_mla", buckets=(3,))._forward_prims_ts(
            q8, cache, indices, lengths, layer, BLOCK_SIZE
        )


def test_row_buckets():
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16384),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[1, 2, 5, 80, 320, 6000]
        ),
    )
    buckets = fims._prims_ts_row_buckets(config)
    assert buckets == tuple(sorted({1 << i for i in range(13)} | {5, 80, 320}))
    config.scheduler_config.max_num_batched_tokens = 100
    config.compilation_config.cudagraph_capture_sizes = None
    assert fims._prims_ts_row_buckets(config) == (1, 2, 4, 8, 16, 32, 64, 100)


def test_select_kernel():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        parallel_config=SimpleNamespace(
            use_ubatching=False, decode_context_parallel_size=8
        ),
        kernel_config=SimpleNamespace(flashinfer_sparse_mla_kernel="auto"),
    )
    # auto: prims_ts under DCP when it can run, else trtllm-gen (no error).
    assert fims.use_prims_ts(config, "fp8")
    assert fims.use_prims_ts(config, "nvfp4_ds_mla")
    assert not fims.use_prims_ts(config, "auto")
    config.parallel_config.use_ubatching = True
    assert not fims.use_prims_ts(config, "fp8")
    config.parallel_config.use_ubatching = False
    # auto without DCP: trtllm-gen for fp8; nvfp4 still needs prims_ts.
    config.parallel_config.decode_context_parallel_size = 1
    assert not fims.use_prims_ts(config, "fp8")
    assert fims.use_prims_ts(config, "nvfp4_ds_mla")
    config.parallel_config.decode_context_parallel_size = 8
    # trtllm: per-tensor fp8 stays on trtllm-gen; nvfp4 needs prims_ts.
    config.kernel_config.flashinfer_sparse_mla_kernel = "trtllm"
    assert not fims.use_prims_ts(config, "fp8")
    assert not fims.use_prims_ts(config, "auto")
    assert fims.use_prims_ts(config, "nvfp4_ds_mla")
    config.kernel_config.flashinfer_sparse_mla_kernel = "prims_ts"
    assert fims.use_prims_ts(config, "fp8")
    assert fims.use_prims_ts(config, "fp8_e4m3")
    with pytest.raises(ValueError, match="kv-cache dtype"):
        fims.use_prims_ts(config, "auto")
    config.parallel_config.use_ubatching = True
    with pytest.raises(ValueError, match="micro-batching"):
        fims.use_prims_ts(config, "nvfp4_ds_mla")
    config.parallel_config.use_ubatching = False
    config.model_config.hf_text_config.qk_rope_head_dim = 0
    with pytest.raises(ValueError, match="qk_rope_head_dim"):
        fims.use_prims_ts(config, "nvfp4_ds_mla")
