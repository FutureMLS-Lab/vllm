# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared DSA core (Kimi K3.1 sparse MLA layers) over the packed fp8_ds_mla and
indexer caches (FlashMLA sparse reads whole blocks, the indexer 64-entry pages):

  MlaQuery (dsa)   BF16 query rows [B, 64 (padded heads), 576], the fp8_ds_mla KV
                   insert and the normalized q_c rows
  wq_b             split-K projection of q_c -> [B, 32 * 128]
  DsaIndex         FP8 indexer query + head weights, indexer key insert
  indexer core     the stock decode scoring and top-k (sparse_attn_indexer, key
                   insert skipped)
  sparse core      SparseMlaDecode over the local heads (core "cute"), or logical ->
                   physical top-k and FlashMLA sparse decode (core "flashmla"); on a
                   per-tensor FP8 cache (FLASHINFER_MLA_SPARSE) the query and KV row
                   are FP8 and FlashInfer's trtllm-gen sparse decode reads the flat
                   rows of the packed blocks (core "flashinfer")
  MlaValue         W_UV and the sigmoid gate of the BF16 latent

Under DCP (= TP) each rank's cache holds 1 / DCP of every context: the query and
DsaIndex insert only the owned slots, the indexer merges the ranks' local top-k
into global token ids, every rank's BF16 query heads are gathered for FlashMLA
sparse decode over the local shard, and the stock LSE combine returns this rank's
heads to MlaValue.

The packed projection row is [q_a 1536 | kv_a 576 | gate H*128 | wk 128 | wv 32]."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from vllm.utils.torch_utils import _encode_layer_name

from ..common.host import (
    dynamic_view,
    fake,
    layout,
    prepare_shared_carveout,
    stream,
    to_cute,
)
from ..common.policy import batch_table
from ..gemm.projection import PreparedProjection
from ..mla.dcp import compile_value
from ..mla.prepared import pad_requests
from ..mla.query import MlaQuery
from ..mla.value import MlaValue
from .index import DIM, HEADS, DsaIndex
from .sparse import TOPK, SparseMlaDecode

QUERY_HEADS = 64  # FlashMLA sparse FP8 decode reads 64 or 128 query heads
# Sparse core per decode batch (GB300, TP8, 2048 of a 32k context; us per layer,
# cute vs FlashMLA: B1 11.8/21.4, B4 12.1/21.9, B8 16.6/21.1, B16 27.4/24.4,
# B32 27.9/34.9, B64 52.0/52.8).
SPARSE_CORE = "1-8:cute,9-16:flashmla,17-32:cute,33-64:flashmla"
CORES = ("cute", "flashmla", "flashinfer")


def core_for(batch, core="auto"):
    if core == "auto":
        core = batch_table(SPARSE_CORE).get(batch, "flashmla")
    if core not in CORES:
        raise ValueError(
            f"Kimi K3.1 DSA core {core!r}: expected auto, cute or flashmla"
        )
    return core


def gate_columns(heads):
    return 2112, 2112 + heads * 128


@cache
def compile_core(
    batch, heads, pitch, specs, device, q=1, dcp=0, q_heads=0, sms=148, fp8=False
):
    """CuTe kernels (query, index, value) of one batch of rows (``q`` per request).
    ``specs``: the layouts of the weights, the MLA cache and the indexer cache.
    Under DCP (``dcp``: its size, ``q_heads``: the gathered query's padded heads) the
    query also publishes into the q mailbox and the value is DcpValue. ``fp8``: a
    per-tensor FP8 query [B, H, 576] and cache, and the value reads FlashInfer's
    [B, H, 512] latent."""
    qnorm, kvnorm, qb, uk, uv, kv, index_kv = (
        fake(shape, stride, dtype, 4 if dtype == torch.float32 else 16)
        for shape, stride, dtype in specs
    )
    scalar = fake((1,), (1,), torch.float32, 4)
    query_heads = heads if fp8 else QUERY_HEADS
    if fp8:
        query = fake((batch, heads, 576), (heads * 576, 576, 1), torch.uint8)
    else:
        query = fake((batch, QUERY_HEADS, 576), (QUERY_HEADS * 576, 576, 1))
    qc = fake((batch, 1536), (1536, 1))
    projected = fake((batch, pitch), (pitch, 1))
    slots = fake((batch,), (1,), torch.int64, 8)
    f32 = cutlass.Float32
    s = make_fake_stream()
    mla_query = MlaQuery(
        heads,
        batch,
        q_b_cp_async=heads == 12 and batch == 1,
        dsa=True,
        fp8=fp8,
        dcp=dcp,
        q_heads=q_heads,
    )
    q_args = (projected, qnorm, kvnorm, qb, uk, kv, slots, scalar, scalar, f32(1e-6))
    publish: tuple = ()
    if dcp:  # this parity's q mailbox (BF16 bytes), its multicast address, rank
        mailbox = fake((64, q_heads, 1152), None, torch.uint8)
        publish = (mailbox, cutlass.Int64(0), cutlass.Int32(0))
    _, key = gate_columns(heads)
    index = DsaIndex(key, key + DIM)
    iq = fake((batch, HEADS * DIM), (HEADS * DIM, 1))
    norm = fake((DIM,), (1,), torch.float32, 4)
    q_out = fake((batch, HEADS, DIM), None, torch.uint8)
    w_out = fake((batch, HEADS), None, torch.float32, 4)
    i_args = (projected, iq, norm, norm, index_kv, slots, q_out, w_out, f32(1e-6))
    latent = fake((batch, heads, 512), (query_heads * 512, 512, 1))
    gate = fake((batch, heads * 128), (pitch, 1))
    lengths = fake((batch // q,), (1,), torch.int32, 4)
    output = fake((batch, heads * 128), (heads * 128, 1))
    with torch.accelerator.device_index(device):
        query_kernel = cute.compile(mla_query, *q_args, query, qc, *publish, s)
        if batch > 1:
            query_kernel = prepare_shared_carveout(query_kernel, device, 100)
        index_kernel = cute.compile(index, *i_args, s)
        if dcp:
            value_kernel = compile_value(
                heads, dcp, batch, 64, specs[4], pitch, device, sms, q, q_heads
            )
        else:
            value = MlaValue(heads, batch, q)
            value_kernel = cute.compile(value, latent, uv, gate, lengths, output, s)
    return query_kernel, index_kernel, value_kernel


@cache
def compile_sparse(batch, heads, kv_spec, device, q=1):
    """SparseMlaDecode of one batch of rows into the [B, 64, 512]-strided latent."""
    query = fake((batch, QUERY_HEADS, 576), (QUERY_HEADS * 576, 576, 1))
    topk = fake((batch, TOPK), (TOPK, 1), torch.int32, 4)
    rows = cute.sym_int(), cute.sym_int()
    table = fake(rows, (cute.sym_int(), 1), torch.int32, 4)  # any block table pitch
    latent = fake((batch, heads, 512), (QUERY_HEADS * 512, 512, 1))
    kv = fake(*kv_spec)
    args = (query, topk, table, kv, latent, cutlass.Float32(1.0), make_fake_stream())
    with torch.accelerator.device_index(device):
        return cute.compile(SparseMlaDecode(heads, batch, q), *args)


def index_policies(batches):
    """Split-K tactics of the replicated wq_b projection (N 4096, K 1536) per padded
    batch (GB300 us, previous / these: B1-8 4.9/2.9, B32 3.7/3.4, B64 4.9/3.9)."""
    tactics = {8: (128, 8, 3, 4), 16: (64, 16, 2, 6), 32: (64, 32, 1, 8)}
    tactics[64] = (64, 64, 1, 4)
    return {b: tactics[next(n for n in tactics if b <= n)] for b in batches}


class PreparedDsa:
    """One K3.1 DSA layer. ``attention``: the KimiK3DSAAttention; ``pitches``: the
    packed projection pitch per batch; ``queries``: the query lengths q (1 + the
    draft length under DSpark verify) prepared for every batch divisible by q."""

    def __init__(
        self,
        attention,
        pitches,
        scratch,
        *,
        queries=(1,),
        core="auto",
        dcp=None,
        parity=0,
    ):
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
            FlashInferMLASparseImpl,
        )
        from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl

        wrapper = attention.mla_attn
        self.layer = layer = wrapper.mla_attn
        self.impl, self.indexer = layer.impl, attention.dsa_indexer
        self.fp8 = isinstance(self.impl, FlashInferMLASparseImpl)
        if not (self.fp8 or isinstance(self.impl, FlashMLASparseImpl)):
            raise ValueError(
                "Kimi K3.1 specialized decode needs the FLASHMLA_SPARSE or "
                f"FLASHINFER_MLA_SPARSE backend, got {type(self.impl).__name__}"
            )
        if self.fp8 and dcp is not None:
            raise ValueError("Kimi K3.1 DSA on FlashInfer: no DCP specialized decode")
        self.box, self.parity = dcp, parity
        self.dcp = dcp.size if dcp is not None else 1
        if dcp is None and not self.fp8:
            assert self.impl.fp8_decode_padded_heads == QUERY_HEADS
        self.heads = layer.W_UK_T.shape[0]
        norms = wrapper.q_a_layernorm.weight, wrapper.kv_a_layernorm.weight
        self.weights: tuple = (*norms, wrapper.q_b_proj.weight, layer.W_UK_T)
        self.weights += (layer.W_UV,)
        self.device, self.pitches = layer.W_UK_T.device, pitches
        self.eps = cutlass.Float32(wrapper.q_a_layernorm.variance_epsilon)
        knorm = self.indexer.k_index_norm
        self.key_eps = cutlass.Float32(knorm.eps)
        self.key_norm = to_cute(knorm.weight, 4), to_cute(knorm.bias, 4)
        self.views = tuple(
            to_cute(w, 4 if w.dtype == torch.float32 else 16) for w in self.weights
        )
        self.unit = torch.ones(1, dtype=torch.float32, device=self.device)
        self.query, self.qc, self.iq, self.q_fp8, self.head_w, self.latent = scratch
        self.shapes = [(b, q) for b in pitches for q in queries if b % q == 0]
        self.cores = {(b, q): core_for(b, core) for b, q in self.shapes}
        # Inverse per-tensor scales of the FP8 query and KV row (FlashMLA: unused).
        self.scales_inv = self.unit, self.unit
        if self.fp8:
            self.cores = dict.fromkeys(self.shapes, "flashinfer")
            self.scales_inv = tuple(
                (1 / scale.float()).reshape(1).to(self.device)
                for scale in (layer._q_scale, layer._k_scale)
            )
            shape = (max(pitches), self.heads, 576)
            self.query = torch.zeros(shape, dtype=torch.uint8, device=self.device)
        if self.dcp > 1:
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
        self.kernels, self.sparse = {}, {}
        policies = index_policies(sorted(pitches))
        wq = self.indexer.wq_index.weight
        self.index_projection = PreparedProjection(wq, policies).prepare()
        self.k_cache_name = _encode_layer_name(self.indexer.k_cache.prefix)
        self.max_model_len = self.indexer.max_model_len
        self.max_total = self.indexer.max_total_seq_len
        self.topk = self.indexer.topk_tokens

    def prepare_cache(self, kv):
        caches = kv.view(torch.uint8), self.indexer.k_cache.kv_cache
        specs = tuple(layout(w) for w in (*self.weights, *caches))
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        for batch, q in self.shapes:
            args: tuple = (batch, self.heads, self.pitches[batch], specs)
            args += (self.device.index, q)
            if self.box is not None:
                args += (self.dcp, self.box.q_heads, sms)
            self.kernels[batch, q] = compile_core(*args, fp8=self.fp8)
            if self.cores[batch, q] == "cute":
                sparse = batch, self.heads, specs[5], self.device.index, q
                self.sparse[batch, q] = compile_sparse(*sparse)
        return self

    def indexer_core(self, projected, batch):
        """Top-k of the current rows into the shared buffer (stock decode scoring)."""
        indexer = self.indexer
        return torch.ops.vllm.sparse_attn_indexer(
            projected,
            self.k_cache_name,
            indexer.k_cache.kv_cache,
            self.q_fp8[:batch].view(torch.float8_e4m3fn),
            None,
            None,
            self.head_w[:batch],
            DIM,
            indexer.scale_fmt,
            self.topk,
            DIM,
            self.max_model_len,
            self.max_total,
            indexer.topk_indices_buffer,
            True,  # the key is inserted by DsaIndex
            False,
            False,
            _encode_layer_name(""),
            False,
            indexer.dcp_rank,
            indexer.dcp_world_size,
            indexer.cp_kv_cache_interleave_size,
            topk_backend=indexer.topk_backend,
        )

    def sparse_core(self, cache, topk, metadata, batch, q):
        """Sparse decode of the padded BF16 query rows -> [B, 64, 512]-strided."""
        from vllm.v1.attention.backends.mla.flashmla_sparse import _fp8_kv_pages
        from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache

        impl = self.impl
        if self.cores[batch, q] == "flashinfer":
            # As the stock decode: top-k -> flat rows of the packed blocks.
            from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view

            impl.record_logical_topk_ready()
            _, block_rows = flat_kv_row_view(cache, metadata.block_size)
            physical, valid = impl._convert_logical_to_physical_topk(
                topk[:batch],
                metadata,
                block_stride_rows=block_rows,
                return_valid_counts=True,
            )
            latent, _ = impl._forward_mqa_kernel(
                self.query[:batch].view(torch.float8_e4m3fn),
                cache,
                physical,
                valid,
                layer=self.layer,
                block_size=metadata.block_size,
                is_decode=True,
            )
            return latent
        if self.cores[batch, q] == "cute":
            reqs = batch // q
            table, _, _ = pad_requests(reqs, metadata.block_table, metadata.seq_lens)
            query, rows = to_cute(self.query[:batch]), to_cute(topk[:batch], 4)
            kv = to_cute(cache.view(torch.uint8))
            latent = to_cute(self.latent[:batch, : self.heads])
            scale = cutlass.Float32(impl.softmax_scale)
            table = dynamic_view(table[:reqs], 4)
            self.sparse[batch, q](
                query, rows, table, kv, latent, scale, stream(self.device)
            )
            return self.latent[:batch]
        # As the stock decode: the packed blocks as 64-row pages of flat rows.
        pages, stride_rows = _fp8_kv_pages(cache, metadata.block_size)
        impl.record_logical_topk_ready()
        physical = impl._convert_logical_to_physical_topk(
            topk[:batch],
            metadata,
            block_stride_rows=stride_rows,
            return_valid_counts=False,
        )
        kernel_metadata = metadata.fp8_extra_metadata
        latent, _ = flash_mla_with_kvcache(
            q=self.query[:batch].unsqueeze(0),
            k_cache=pages.view(torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=kernel_metadata.cache_lens,
            tile_scheduler_metadata=kernel_metadata.scheduler_metadata,
            is_fp8_kvcache=True,
            indices=physical.unsqueeze(0),
            softmax_scale=impl.softmax_scale,
        )
        return latent[0]

    def dcp_core(self, cache, topk, metadata, batch):
        """FlashMLA sparse decode of every rank's query heads (the gathered q mailbox)
        over the local shard into the DcpValue rows and (natural-log) LSE. The global
        top-k ids are filtered to this rank's slots and compacted to a prefix whose
        length bounds FlashMLA's work (about 1 / DCP of the top-k); a fresh scheduler
        metadata per call follows those lengths."""
        from vllm.v1.attention.backends.mla.flashmla_sparse import _fp8_kv_pages
        from vllm.v1.attention.backends.mla.sparse_utils import (
            triton_filter_and_convert_dcp_index,
        )
        from vllm.v1.attention.ops.flashmla import (
            flash_mla_with_kvcache,
            get_mla_metadata,
        )

        indexer, md = self.indexer, metadata
        pages, stride_rows = _fp8_kv_pages(cache, md.block_size)
        slots, valid = triton_filter_and_convert_dcp_index(
            md.req_id_per_token[:batch],
            md.block_table,
            topk[:batch],
            dcp_size=self.dcp,
            dcp_rank=indexer.dcp_rank,
            cp_kv_cache_interleave_size=indexer.cp_kv_cache_interleave_size,
            BLOCK_SIZE=md.block_size,
            BLOCK_STRIDE_ROWS=stride_rows,
            NUM_TOPK_TOKENS=topk.shape[1],
            return_valid_counts=True,
        )
        rows, lse = flash_mla_with_kvcache(
            q=self.box.q[self.parity, :batch].unsqueeze(1),
            k_cache=pages.view(torch.uint8).unsqueeze(-2),
            block_table=self.dummy_table[:batch],
            head_dim_v=512,
            cache_seqlens=self.cache_lens[:batch],
            tile_scheduler_metadata=get_mla_metadata()[0],
            is_fp8_kvcache=True,
            indices=slots.unsqueeze(1),
            topk_length=valid,
            softmax_scale=self.impl.softmax_scale,
            out=self.rows[:batch],
        )
        # A rank owning none of a row's entries: DcpValue drops it (LSE -inf).
        return rows, lse.masked_fill((valid == 0).view(-1, 1, 1), float("-inf"))

    def run_dcp(self, projected, out, cache, slots, index_slots, metadata, q):
        """``run`` under DCP: the query publishes into the q mailbox and DcpValue
        merges every rank's partial rows of this rank's heads (global lengths)."""
        batch, s = projected.shape[0], stream(self.device)
        reqs, box, p = batch // q, self.box, self.parity
        _, lengths, _ = pad_requests(reqs, metadata.block_table, metadata.seq_lens)
        mq, index, value = self.kernels[batch, q]
        qnorm, kvnorm, qb, uk, uv = self.views
        x, q_slots = to_cute(projected), to_cute(slots[:batch], 8)
        kv = to_cute(cache.view(torch.uint8))
        unit = to_cute(self.unit, 4)
        mailbox = to_cute(box.q[p].view(torch.uint8))
        q_args = (x, qnorm, kvnorm, qb, uk, kv, q_slots, unit, unit, self.eps)
        publish = (mailbox, box.mc(p), cutlass.Int32(box.rank))
        mq(*q_args, to_cute(self.query[:batch]), to_cute(self.qc[:batch]), *publish, s)
        self.run_index(x, index_slots, batch, index)
        topk = self.indexer_core(projected, batch)
        latent, lse = self.dcp_core(cache, topk, metadata, batch)
        lens = to_cute(lengths[:reqs], 4)
        boxes = to_cute(box.partial[p]), to_cute(box.lse[p], 4)
        peers = to_cute(box.peers, 8), to_cute(box.lse_peers, 8)
        gate = to_cute(projected[:, 2112:])
        # DcpValue's [B, 1, DCP * H] views of FlashMLA's padded-head rows and LSE.
        heads, padded = self.heads * self.dcp, box.q_heads
        latent = latent.as_strided((batch, 1, heads, 512), (padded * 512, 0, 512, 1))
        lse = lse.as_strided((batch, 1, heads), (padded, 0, 1))
        v_args = (to_cute(latent), to_cute(lse, 4), lens, *boxes, *peers, mailbox, uv)
        value(*v_args, gate, lens, to_cute(out), cutlass.Int32(p), publish[2], s)
        return out

    def run_index(self, x, index_slots, batch, index):
        """FP8 indexer query and head weights; the key into the indexer cache."""
        iq = self.index_projection.run(self.qc[:batch], self.iq[:batch])
        rows = to_cute(self.q_fp8[:batch]), to_cute(self.head_w[:batch], 4)
        key = to_cute(self.indexer.k_cache.kv_cache), to_cute(index_slots[:batch], 8)
        s = stream(self.device)
        index(x, to_cute(iq), *self.key_norm, *key, *rows, self.key_eps, s)

    def run(self, projected, out, *, cache, slots, index_slots, metadata, q=1):
        """``projected``: the packed projection rows; ``out`` [B, H * 128] gets the
        gated attention output; ``slots`` / ``index_slots``: the MLA and indexer
        cache slots. Rows beyond the requests run as empty requests."""
        if self.box is not None:
            args = cache, slots, index_slots, metadata, q
            return self.run_dcp(projected, out, *args)
        batch, s = projected.shape[0], stream(self.device)
        reqs = batch // q
        lengths = metadata.seq_lens
        _, lengths, _ = pad_requests(reqs, metadata.block_table, lengths)
        mq, index, value = self.kernels[batch, q]
        qnorm, kvnorm, qb, uk, uv = self.views
        kv = to_cute(cache.view(torch.uint8))
        q_scale_inv, k_scale_inv = (to_cute(t, 4) for t in self.scales_inv)
        x, q_slots = to_cute(projected), to_cute(slots[:batch], 8)
        scales = q_scale_inv, k_scale_inv, self.eps
        q_args = (x, qnorm, kvnorm, qb, uk, kv, q_slots, *scales)
        mq(*q_args, to_cute(self.query[:batch]), to_cute(self.qc[:batch]), s)
        self.run_index(x, index_slots, batch, index)
        topk = self.indexer_core(projected, batch)
        latent = self.sparse_core(cache, topk, metadata, batch, q)
        gate = to_cute(projected[:, slice(*gate_columns(self.heads))])
        lens = to_cute(lengths[:reqs], 4)
        value(to_cute(latent[:, : self.heads]), uv, gate, lens, to_cute(out), s)
        return out
