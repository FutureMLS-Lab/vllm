# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared DSA core (Kimi K3.1 sparse MLA layers): the orchestration of one layer's
stages over the packed MLA cache and the indexer cache (64-entry pages):

  query stage      MlaQuery (dsa) -> wq_b -> DsaIndex (query_stage.py)
  indexer stage    TP on FlashInfer: the fused decode indexer (indexer.py: scores,
                   exact top-k and the physical rows in two PDL-chained kernels);
                   DCP with the #63 merge: its scorer and local top-N select
                   feed the merge (indexer_stage.py); else the stock decode
                   scoring and top-k (indexer_stage.py)
  sparse core      FlashInfer sparse MLA on a per-tensor FP8 cache (attn_fi.py:
                   trtllm-gen or prims_ts per batch, called directly into the value
                   stage's rows), else FlashMLA sparse on fp8_ds_mla (attn_ds.py)
  MlaValue         W_UV and the sigmoid gate of the BF16 latent

Query overlap (overlap.py, per batch of rows): the indexer path needs only q_c, so
QueryNorm writes it (or IndexFront fuses it with wq_b and DsaIndex, front.py) on the
main stream while MlaQuery (and, under DCP, the q mailbox poll) runs on a side
stream that the main stream joins before the sparse core; or (TP, "late") MlaQuery
runs on the main stream right after the fused indexer's select, beside it.

Under DCP (= TP, dcp_stage.py) each rank's cache holds 1 / DCP of every context: the
query and DsaIndex insert only the owned slots, the query publishes its heads into
every rank's q mailbox, the indexer merges the ranks' local top-k into global token
ids, which are filtered to this rank's slots (on FlashInfer with the #63 merge: into
this rank's slots directly) for the sparse decode of every rank's gathered heads, and
DcpValue merges this rank's heads by LSE.

The packed projection row is [q_a 1536 | kv_a 576 | gate H*128 | wk 128 | wv 32]."""

from functools import cache

import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake, layout, stream, to_cute
from ..mla.dcp import PROJECTED
from ..mla.prepared import pad_requests
from ..mla.value import MlaValue
from .attn_ds import flashmla_core, flashmla_dcp_core
from .attn_fi import FlashInferCore, dsa_on_flashinfer, fi_kernels, kernel_for
from .dcp_stage import compile_dcp_value, local_slots, publish_args, run_value
from .indexer import PAGE as INDEX_PAGE
from .indexer import FusedIndexer
from .indexer import reserve as reserve_fused_indexer
from .indexer import static_unsupported as fused_indexer_static_unsupported
from .indexer import supports as fused_indexer_unsupported
from .indexer_stage import IndexerStage
from .overlap import (
    FRONT,
    LATE,
    MODES,
    NORM,
    QUERY_OVERLAP_UNFUSED,
    overlap_for,
    side_stream,
)
from .query_stage import (
    QUERY_HEADS,
    QueryStage,
    compile_front,
    compile_index,
    compile_norm,
    compile_query,
    compile_wait,
    gate_columns,
)

__all__ = [
    "QUERY_HEADS",
    "PreparedDsa",
    "compile_core",
    "dsa_on_flashinfer",
    "gate_columns",
]


# FlashInfer sparse MLA kernels that read their inputs (prims_ts: query and rows;
# trtllm-gen: the counts) before their PDL wait: the kernel before them is held.
HELD_KERNELS = ("prims_ts", "trtllm")
# On FlashInfer with a per-tensor FP8 query the fused decode indexer always runs
# (where fused_indexer_unsupported allows): at TP instead of the stock op and the
# conversion; under DCP feeding the #63 merge its scores, plus (up to one row per
# SM, FusedIndexer.packs) its local top-N select as B' rows. The exact merge
# re-reads a saturated rank's logits (more than N of a row's global top-k), so it
# always gets the DeepGEMM-layout logits, unmodified until it returns; the B' rows
# meet its producer contract (bit-exact #63's pack of the exact local top-N, ties
# to the lowest key: test_fused_indexer_dcp).
# serial query overlap: the fused index front from MlaQuery's q_c rows
# (IndexFrontMma, normalized) instead of the split-K wq_b and DsaIndex, up to this
# many rows (0: never). Under fork it measured no gain (QueryNorm stays a launch);
# under DCP8 wait it lost 1-8 us (it would wait on MlaQuery's q mailbox poll).
FUSED_INDEX = 64


@cache
def compile_core(
    batch,
    heads,
    pitch,
    specs,
    device,
    q=1,
    dcp=0,
    q_heads=0,
    sms=148,
    fp8=False,
    lse_base_e=True,
    nvfp4=False,
    overlap="serial",
    partials="bf16",
    cap=64,
    release=True,
):
    """CuTe kernels (query, index, value) of one batch of rows (``q`` per request).
    ``specs``: the layouts of the weights, the MLA cache and the indexer cache.
    Under DCP (``dcp``: its size, ``q_heads``: the gathered query's padded heads,
    ``cap``: the DcpBox rows) the query also publishes into the q mailbox and the
    value is DcpValue. ``fp8``: a per-tensor FP8 query [B, H, 576] and cache, and
    the value reads FlashInfer's [B, H, 512] latent (under DCP: an FP8 q mailbox,
    and an LSE in base e if
    ``lse_base_e``, else base 2). ``nvfp4``: that FP8 query over an nvfp4_ds_mla
    cache (FlashInfer prims_ts). ``overlap``: the query overlap mode (overlap.py).
    ``partials``: the DcpBox's partial rows. ``release``: the query launches the
    next kernel early (PDL)."""
    qnorm, kvnorm, qb, uk, uv, kv, index_kv = (
        fake(shape, stride, dtype, 4 if dtype == torch.float32 else 16)
        for shape, stride, dtype in specs
    )
    query_heads = heads if fp8 else QUERY_HEADS
    with torch.accelerator.device_index(device):
        fakes = qnorm, kvnorm, qb, uk, kv
        query_kernel = compile_query(
            batch,
            heads,
            pitch,
            fakes,
            device,
            dcp,
            q_heads,
            fp8,
            nvfp4=nvfp4,
            overlap=overlap,
            cap=cap,
            release=release,
        )
        index_kernel = compile_index(batch, heads, pitch, index_kv)
        if dcp:
            args = heads, dcp, batch, specs[4], pitch, device, sms, q, q_heads
            value_kernel = compile_dcp_value(*args, fp8, lse_base_e, partials, cap)
        else:
            latent = fake((batch, heads, 512), (query_heads * 512, 512, 1))
            gate = fake((batch, heads * 128), (pitch, 1))
            lengths = fake((batch // q,), (1,), torch.int32, 4)
            output = fake((batch, heads * 128), (heads * 128, 1))
            value = MlaValue(heads, batch, q)
            v_args = (latent, uv, gate, lengths, output, make_fake_stream())
            value_kernel = cute.compile(value, *v_args)
    return query_kernel, index_kernel, value_kernel


def kv_cache_blocks():
    """The 64-token blocks the KV cache holds, if known before it is allocated (a
    num_gpu_blocks override, or a cache already sized): a request's context fits in
    it, so it bounds the fused indexer's logits with the longest context. None if
    unknown (the KV cache is sized after the specialized decode is built)."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is None:
        return None
    cache = config.cache_config
    blocks = cache.num_gpu_blocks or cache.num_gpu_blocks_override
    if not blocks or not cache.block_size:
        return None
    return -(-blocks * cache.block_size // INDEX_PAGE)


class PreparedDsa:
    """One K3.1 DSA layer. ``attention``: the KimiK3DSAAttention; ``pitches``: the
    packed projection pitch per batch; ``queries``: the query lengths q (1 + the
    draft length under DSpark verify) prepared for every batch divisible by q;
    ``kernels``: the FlashInfer sparse MLA kernels to pick from per batch (default:
    attn_fi.fi_kernels); ``prims_ts_rows``: the prims_ts plans' row capacities
    (default: every batch on its own plan);
    ``overlap``: a query overlap mode or "rows:mode" table (overlap.QUERY_OVERLAP).
    """

    def __init__(
        self,
        attention,
        pitches,
        scratch,
        *,
        queries=(1,),
        dcp=None,
        parity=0,
        kernels=None,
        overlap=None,
        prims_ts_rows=None,
    ):
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl

        wrapper = attention.mla_attn
        self.layer = layer = wrapper.mla_attn
        self.impl, self.indexer = layer.impl, attention.dsa_indexer
        # The KV format: fp8_ds_mla (FlashMLA sparse), else FlashInfer
        # sparse's per-tensor FP8 query over a per-tensor FP8 cache (trtllm-gen or
        # prims_ts) or an nvfp4_ds_mla one (prims_ts).
        self.fp8 = dsa_on_flashinfer(attention)
        self.nvfp4 = layer.kv_cache_dtype == "nvfp4_ds_mla"
        if not (self.fp8 or isinstance(self.impl, FlashMLASparseImpl)):
            raise ValueError(
                "Kimi K3.1 specialized decode needs the FLASHMLA_SPARSE or "
                f"FLASHINFER_MLA_SPARSE backend, got {type(self.impl).__name__}"
            )
        # The index projection runs as a BF16 split-K GEMM (stock: BF16 wq_index).
        if not isinstance(self.indexer.wq_index.quant_method, UnquantizedLinearMethod):
            raise ValueError(
                "Kimi K3.1 specialized decode needs an unquantized (BF16) wq_index"
            )
        self.box, self.parity = dcp, parity
        # The DcpBox mailboxes hold cap rows: a wider batch would store past them.
        assert dcp is None or max(pitches) <= dcp.cap, (max(pitches), dcp.cap)
        self.dcp = dcp.size if dcp is not None else 1
        if dcp is None and not self.fp8:
            assert self.impl.fp8_decode_padded_heads == QUERY_HEADS
        self.heads = layer.W_UK_T.shape[0]
        self.device, self.pitches = layer.W_UK_T.device, pitches
        *shared, fi_buffers = scratch
        self.query_stage = QueryStage(attention, pitches, shared, self.fp8, self.nvfp4)
        # Under DCP on FlashInfer: the #63 merge's owned-row outputs (indexer_stage).
        merge_rows = max(pitches) if self.dcp > 1 and self.fp8 else 0
        self.indexer_stage = IndexerStage(self.indexer, merge_rows)
        self.shapes = [(b, q) for b in pitches for q in queries if b % q == 0]
        self.fi = None
        self.overlaps = {
            (b, q): overlap_for(b, overlap, self.dcp > 1) for b, q in self.shapes
        }
        if self.fp8:
            if kernels is None:
                kernels = fi_kernels(layer.kv_cache_dtype)
            dcp_size = self.dcp
            rows = {b: kernel_for(b, dcp_size > 1, kernels) for b in pitches}
            self.cores = {(b, q): rows[b] for b, q in self.shapes}
            heads = dcp.q_heads if dcp is not None else self.heads
            sharding = dcp_size, self.indexer.dcp_rank
            # The merge's rows are narrower than the top-k: plan both widths.
            merged = self.indexer_stage.local_topk
            widths = (self.indexer_stage.dcp_rows.shape[1],) if merged else ()
            self.fi = FlashInferCore(
                self.impl,
                layer,
                heads,
                fi_buffers,
                rows,
                *sharding,
                widths=widths,
                capacities=prims_ts_rows,
            )
        elif self.dcp == 1:
            self.cores = dict.fromkeys(self.shapes, "flashmla")
        else:
            self.cores = dict.fromkeys(self.shapes, "dcp")
            rows, device = max(pitches), self.device
            padded = dcp.q_heads
            self.rows = torch.zeros(
                (rows, 1, padded, 512), dtype=torch.bfloat16, device=device
            )
            self.dummy_table = torch.zeros((rows, 1), dtype=torch.int32, device=device)
            self.cache_lens = torch.full(
                (rows,), self.indexer.max_model_len, dtype=torch.int32, device=device
            )
        self.kernels, self.overlap_kernels = {}, {}
        local_topk = self.indexer_stage.local_topk
        static = (
            fused_indexer_static_unsupported(self.indexer, *s) for s in self.shapes
        )
        if self.fp8 and (self.dcp == 1 or local_topk) and not any(static):
            # The fused indexer's buffers (prepare_cache), shared by every layer,
            # sized for the longest context the KV cache can hold: reserved now, as
            # the specialized decode is built before vLLM profiles its memory for
            # the KV cache (only where prepare_cache can run it: its cache layout
            # aside, the same checks).
            args = self.indexer, self.device, max(pitches), local_topk
            reserve_fused_indexer(*args, blocks=kv_cache_blocks())
        # DCP "proj" partials: every gathered head's W_UV (collective).
        self.uv_gathered = dcp.value_weights(layer.W_UV) if dcp else None
        # TP on FlashInfer: the fused decode indexer (prepare_cache; None: stock).
        self.fused: FusedIndexer | None = None

    @property
    def query(self):
        """This layer's query rows (FlashInfer: FP8 [B, H, 576] bytes)."""
        return self.query_stage.query

    def prepare_cache(self, kv):
        caches = kv.view(torch.uint8), self.indexer.k_cache.kv_cache
        weights = self.query_stage.weights
        specs = tuple(layout(w) for w in (*weights, *caches))
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        unsupported = [fused_indexer_unsupported(self.indexer, *s) for s in self.shapes]
        fused = self.box is None and self.fp8
        fused = fused and not any(unsupported)
        if not fused:  # the late query follows the fused indexer's select
            # Without it (fp8_ds_mla, unsupported shapes) the TP default is the
            # stack's table from before the late modes (QUERY_OVERLAP_UNFUSED).
            for (batch, q), mode in self.overlaps.items():
                if mode in LATE:
                    mode = overlap_for(batch, QUERY_OVERLAP_UNFUSED)
                self.overlaps[batch, q] = mode
        for batch, q in self.shapes:
            args: tuple = (batch, self.heads, self.pitches[batch], specs)
            args += (self.device.index, q)
            partials = "bf16"
            if self.box is not None:
                args += (self.dcp, self.box.q_heads, sms)
                partials = self.box.mailbox(batch).format
            lse_base_e = self.cores[batch, q] != "trtllm"  # trtllm-gen: base 2
            overlap = self.overlaps[batch, q]
            cap = self.box.cap if self.box is not None else 64
            self.kernels[batch, q] = compile_core(
                *args,
                fp8=self.fp8,
                lse_base_e=lse_base_e,
                nvfp4=self.nvfp4,
                overlap=overlap,
                partials=partials,
                cap=cap,
                release=(batch, q) not in self.held(LATE),
            )
            if overlap != "serial" or batch <= FUSED_INDEX:
                self.overlap_kernels[batch, q] = self.compile_overlap(
                    batch, q, specs[6]
                )
        rows = max(self.pitches)
        if fused:
            # Every mode but the late ones has the select right before the decode
            # (wait too: QueryWait runs only under DCP).
            hold = self.held(tuple(m for m in MODES if m not in LATE))
            # The select maps keys to rows of the MLA cache's pages (kv.shape[1]
            # tokens: 64, or the hybrid model's attention page, e.g. 2048).
            args = self.indexer, self.shapes, self.device, rows
            self.fused = FusedIndexer(*args, hold=hold, page=kv.shape[1])
        stage = self.indexer_stage
        if stage.local_topk and not any(unsupported):
            stage.fused = FusedIndexer(
                self.indexer,
                self.shapes,
                self.device,
                rows,
                local_topk=stage.local_topk,
                packed=True,
            )
        return self

    def held(self, modes):
        """The shapes on a FlashInfer kernel (HELD_KERNELS) whose overlap mode is
        one of ``modes``: the kernel right before their sparse decode (TP: the
        fused indexer's select, or the late query) must not launch it early.
        prims_ts' split-KV decode launches with PDL but reads its query and rows
        before its PDL wait, so it would read the previous layer's; trtllm-gen
        reads its per-row counts early too (a CUDA graph with only the counts
        poisoned before each layer hangs at 60 rows)."""
        return {
            shape
            for shape, core in self.cores.items()
            if core in HELD_KERNELS and self.overlaps[shape] in modes
        }

    def indexer_core(self, projected, batch):
        """Top-k of the current rows into the shared buffer (stock decode scoring)."""
        stage = self.query_stage
        assert projected.shape[0] == batch
        return self.indexer_stage.run(projected, stage.q_fp8, stage.head_w)

    def fused_core(self, cache, metadata, batch, q, join=None):
        """TP on FlashInfer: the fused decode indexer's physical rows, then the
        sparse decode -> the [B, H, 512] latent. ``join``: what makes the sparse
        decode wait for the query (run_query)."""
        from vllm.forward_context import get_forward_context
        from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

        assert self.fused is not None and metadata.block_size == self.fused.page, (
            metadata.block_size,
            self.fused.page,
        )
        decode = get_forward_context().attn_metadata[self.indexer.k_cache.prefix]
        _, block_rows = flat_kv_row_view(cache, metadata.block_size)
        stage = self.query_stage
        args = stage.q_fp8, stage.head_w, decode.decode, metadata.block_table
        rows, valid = self.fused.run(*args, block_rows, batch, q)
        assert self.fi is not None
        if join is not None:
            join()
        query = self.query[:batch]
        return self.fi.decode(query, cache, rows, valid, metadata.block_size)[0]

    def sparse_core(self, cache, topk, metadata, batch, q):
        """Sparse decode of the query rows -> the latent ([B, 64, 512]-strided on
        fp8_ds_mla, [B, H, 512] on FlashInfer)."""
        query = self.query[:batch]
        if self.fi is not None:
            return self.fi.run(query, cache, topk[:batch], metadata)[0]
        return flashmla_core(self.impl, query, cache, topk[:batch], metadata)

    def compile_overlap(self, batch, q, index_spec):
        """The overlap kernels of a batch: (front, wait, norm). ``front``: the fused
        index front (IndexFront / IndexFrontMma from the packed rows under front /
        late_front, IndexFrontMma from q_c under serial up to FUSED_INDEX
        rows) or None; ``wait``: QueryWait under DCP (else None); ``norm``:
        QueryNorm under fork / late (else None). Each kernel is compiled once per
        key and shared by every DSA layer (query_stage memoizes them)."""
        mode, pitch, device = self.overlaps[batch, q], self.pitches[batch], self.device
        first = wait = None
        with torch.accelerator.device_index(device.index):
            if self.box is not None:
                args = batch, self.heads, self.dcp, self.box.q_heads, self.fp8
                wait = compile_wait(*args, self.box.cap)
            norm = None
            if mode in NORM:
                norm = compile_norm(batch, self.heads, pitch)
            fused = mode == "serial" and batch <= FUSED_INDEX
            if mode in FRONT or fused:
                weight = self.query_stage.index_projection.weight
                specs = layout(weight), index_spec
                normalized = mode not in FRONT
                args = batch, self.heads, pitch, *specs, device.index, normalized
                first = compile_front(*args)
        return first, wait, norm

    def run_query(self, projected, cache, slots, q, publish=()):
        """MlaQuery and the q_c rows per the overlap mode. Returns what makes the
        sparse core wait for the query (None: it is already in stream order)."""
        stage, batch = self.query_stage, projected.shape[0]
        query, mode = self.kernels[batch, q][0], self.overlaps[batch, q]
        if mode == "serial":
            stage.run_query(query, projected, cache, slots, publish)
            return None
        first, wait, norm = self.overlap_kernels[batch, q]
        if mode in LATE:  # MlaQuery inside the fused indexer's chain (fused_core)
            if mode in NORM:
                stage.run_norm(norm, projected)
            return lambda: stage.run_query(query, projected, cache, slots)
        mailbox = publish[:1]
        if mode == "wait":
            stage.run_query(query, projected, cache, slots, publish)
            if wait is None:
                return None
            return lambda: wait(*mailbox, stream(self.device))
        main = torch.cuda.current_stream(self.device)
        side = side_stream(self.device.index)
        side.wait_stream(main)
        if mode == "fork":
            stage.run_norm(norm, projected)
        with torch.cuda.stream(side):
            stage.run_query(query, projected, cache, slots, publish)
            if wait is not None:
                wait(*mailbox, stream(self.device))
        return lambda: main.wait_stream(side)

    def run_index(self, kernel, projected, index_slots, q):
        """The indexer inputs: wq_b and DsaIndex (``kernel``), or the fused index
        front (from the packed rows, or under serial from q_c)."""
        shape = projected.shape[0], q
        mode, stage = self.overlaps[shape], self.query_stage
        front = self.overlap_kernels.get(shape, (None,))[0]
        if mode in FRONT:
            return stage.run_front(front, projected, index_slots)
        if front is not None:  # serial: from MlaQuery's q_c rows
            return stage.run_front(front, projected, index_slots, normalized=True)
        return stage.run_index(kernel, projected, index_slots)

    def dcp_core(self, cache, topk, metadata, batch, projected=None, join=None, q=1):
        """Sparse decode of every rank's query heads (the gathered q mailbox) over the
        local shard into the DcpValue rows, LSE and per-row valid counts: on a
        per-tensor FP8 cache FlashInfer over this rank's flat rows, on fp8_ds_mla
        FlashMLA as the stock decode. ``topk``: the global ids, or None with
        ``projected``: on FlashInfer the indexer stage's DCP merge (#63) gives this
        rank's rows directly, else the stock op's global ids are filtered to them.
        ``join``: what makes the sparse decode wait for the query (run_query)."""
        from vllm.v1.attention.backends.mla.flashmla_sparse import _fp8_kv_pages

        indexer, md = self.indexer, metadata
        query = self.box.q[self.parity, :batch]
        if self.fi is not None:
            stride_rows = self.fi.rows(cache, md.block_size)[1]
            merged = None
            if topk is None and self.indexer_stage.local_topk:
                stage = self.query_stage
                args = stage.q_fp8, stage.head_w, md, stride_rows, batch
                merged = self.indexer_stage.merge_rows(*args, q)
            if merged is None:
                if topk is None:
                    topk = self.indexer_core(projected, batch)
                interleave = indexer.cp_kv_cache_interleave_size
                args = topk, md, batch, stride_rows, interleave
                merged = self.fi.convert(*args)
            if join is not None:
                join()
            latent, lse = self.fi.decode(query, cache, *merged, md.block_size)
            return latent, lse, merged[1]
        if topk is None:
            topk = self.indexer_core(projected, batch)
        pages, stride_rows = _fp8_kv_pages(cache, md.block_size)
        slots, valid = local_slots(
            md,
            topk[:batch],
            self.dcp,
            indexer.dcp_rank,
            indexer.cp_kv_cache_interleave_size,
            stride_rows,
        )
        if join is not None:
            join()
        out, table, lens = (
            t[:batch] for t in (self.rows, self.dummy_table, self.cache_lens)
        )
        return flashmla_dcp_core(
            self.impl, query, pages, slots, valid, out, table, lens
        )

    def run_dcp(self, projected, out, cache, slots, index_slots, metadata, q):
        """``run`` under DCP: the query publishes into the q mailbox and DcpValue
        merges every rank's partial rows of this rank's heads (global lengths)."""
        batch = projected.shape[0]
        reqs, box, p = batch // q, self.box, self.parity
        _, lengths, _ = pad_requests(reqs, metadata.block_table, metadata.seq_lens)
        _, index, value = self.kernels[batch, q]
        stage = self.query_stage
        publish = publish_args(box, p)
        join = self.run_query(projected, cache, slots, q, publish)
        self.run_index(index, projected, index_slots, q)
        core = self.dcp_core(cache, None, metadata, batch, projected, join, q)
        uv = stage.views[4]
        if box.mailbox(batch).format == PROJECTED:
            uv = self.uv_gathered[1]
        args = value, box, p, publish, self.heads, core, lengths[:reqs], uv
        return run_value(*args, projected, out)

    def run(self, projected, out, *, cache, slots, index_slots, metadata, q=1):
        """``projected``: the packed projection rows; ``out`` [B, H * 128] gets the
        gated attention output; ``slots`` / ``index_slots``: the MLA and indexer
        cache slots. Rows beyond the requests run as empty requests."""
        if self.box is not None:
            args = cache, slots, index_slots, metadata, q
            return self.run_dcp(projected, out, *args)
        batch, s = projected.shape[0], stream(self.device)
        reqs = batch // q
        _, lengths, _ = pad_requests(reqs, metadata.block_table, metadata.seq_lens)
        _, index, value = self.kernels[batch, q]
        stage = self.query_stage
        join = self.run_query(projected, cache, slots, q)
        self.run_index(index, projected, index_slots, q)
        if self.fused is not None:
            latent = self.fused_core(cache, metadata, batch, q, join)
        else:
            topk = self.indexer_core(projected, batch)
            if join is not None:
                join()
            latent = self.sparse_core(cache, topk, metadata, batch, q)
        gate = to_cute(projected[:, slice(*gate_columns(self.heads))])
        lens, uv = to_cute(lengths[:reqs], 4), stage.views[4]
        value(to_cute(latent[:, : self.heads]), uv, gate, lens, to_cute(out), s)
        return out
