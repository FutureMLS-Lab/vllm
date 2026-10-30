# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer attention providers (input projection, KDA/MLA core, o-projection) and
the helpers ``SpecializedDecode.build_providers`` uses."""

from typing import Any

import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

from ..cutedsl.dsa.index import DIM, HEADS
from ..cutedsl.dsa.prepared import QUERY_HEADS, PreparedDsa, dsa_on_flashinfer
from ..cutedsl.gemm.projection import PreparedProjection
from ..cutedsl.kda.prepared import PreparedKda
from ..cutedsl.mla.dcp import DcpBox
from ..cutedsl.mla.prepared import PreparedMla
from ..cutedsl.oproj.prepared import PreparedOproj

# Decode batches (tokens) served by the prepared providers: every batch up to 8, the
# padded CUDA-graph batches and each further captured size; others run stock.
BATCHES = (*range(1, 9), 16, 32, 64)
PREPARED_MAX = BATCHES[-1]


def projection_policies(tp, rows, batches):
    """Split-K (M tile, N tile, split, stages) per execution rows' next tested bin."""
    policies = {}
    for batch in batches:
        n = next(b for b in (8, 16, 32, 64) if batch <= b)
        if rows == 6288 and (tp == 8 or n > 8):
            tactic = (64, 64, 1, 6) if n == 64 else (128, n, 2, 5 if n == 32 else 6)
        else:  # TP16 KDA B64: a sixth stage, -0.7 us per layer in the KDA chain
            stages = {8: 8, 16: 10, 32: 8, 64: 6 if rows == 3216 else 5}[n]
            tactic = (64, n, 4 if n == 8 else 2, stages)
        policies[batch] = tactic
    return policies


class SkinnyProjection:
    """B1/B2 skinny GEMM; static K loads two weight tiles before the PDL wait."""

    def __init__(self, weight, fallback):
        self.weight, self.fallback, k = weight, fallback, weight.shape[1]
        tactics = {1: (128, 4, 1), 2: (64, 4, 2)}
        self.configs = {m: SkinnyGemmConfig(m, *t, 8, k) for m, t in tactics.items()}
        for m, cfg in self.configs.items():  # compile before capture
            shape_dynamic_skinny_gemm(weight.new_zeros(m, k), weight, cfg, None)

    def run(self, x, out):
        cfg = self.configs.get(x.shape[0])
        if cfg is None:
            return self.fallback.run(x, out)
        return shape_dynamic_skinny_gemm(x, self.weight, cfg, None)


def scratch(runtime, name, shape, dtype=torch.bfloat16):
    if name not in runtime.buffers:
        runtime.buffers[name] = torch.empty(shape, device=runtime.device, dtype=dtype)
    return runtime.buffers[name]


def pack_dsa_projection(attention):
    """One [q_a | kv_a | gate | wk | wv] weight for a K3.1 DSA layer; the stock
    projections become views of it."""
    packed = getattr(attention, "packed_projection", None)
    if packed is not None:
        return packed
    linears = (
        attention.fused_qkv_a_proj,
        attention.g_proj,
        attention.dsa_indexer.wkv_index,
    )
    packed = torch.cat([linear.weight.data for linear in linears])
    row = 0
    for linear in linears:
        rows = linear.weight.shape[0]
        linear.weight.data = packed[row : row + rows]
        row += rows
    attention.packed_projection = packed
    return packed


def attention_kind(attention):
    if hasattr(attention, "in_proj_qkvgfab"):
        return "kda"
    return "dsa" if hasattr(attention, "dsa_indexer") else "mla"


class Attention:
    """Input projection -> KDA/MLA/DSA core -> o-projection with its fused AttnRes."""

    def __init__(self, decoder, runtime, prefetch, mailbox, small_batch, twoshot=None):
        attention, tp = decoder.self_attn, runtime.tp
        self.kind = kind = attention_kind(attention)
        if kind == "dsa":
            weight = pack_dsa_projection(attention)
        else:
            name = "in_proj_qkvgfab" if kind == "kda" else "fused_qkv_a_g_proj"
            weight = getattr(attention, name).weight
        pitch = weight.shape[0]
        policies = projection_policies(tp, pitch, runtime.batches)
        split_k = PreparedProjection(weight, policies).prepare()
        self.projection = SkinnyProjection(weight, split_k)
        self.core: Any  # PreparedKda, PreparedMla or PreparedDsa
        if kind == "dsa":
            pitches = {b: pitch for b in runtime.batches}
            rows = dsa_scratch(runtime)
            policy = dict(queries=runtime.queries, core=runtime.dsa_core)
            if runtime.dcp > 1:
                heads = attention.num_local_heads
                policy.update(
                    dcp=dcp_box(
                        runtime,
                        heads,
                        dsa=True,
                        flashinfer=dsa_on_flashinfer(attention),
                    ),
                    parity=runtime.dsa_layers % 2,
                )
            runtime.dsa_layers += 1
            self.core = PreparedDsa(attention, pitches, rows, **policy)
        elif kind == "kda":
            mode = lambda b: (1 if b == 1 else 2 if b <= 8 else 0) if tp == 8 else 0
            policies = {b: (pitch, mode(b)) for b in runtime.batches}
            q = runtime.q  # verify steps: B requests x q tokens
            verify = [b // q for b in runtime.batches if q > 1 and b % q == 0]
            policies.update({(b, q): (pitch, int(tp == 8)) for b in verify})
            self.core = PreparedKda(attention, policies, is_conv_state_dim_first())
        else:
            heads = attention.num_local_heads
            query = scratch(runtime, "mla.query", (64, heads, 576), torch.uint8)
            policy = dict(max_context=runtime.max_context, core=runtime.mla_core)
            policy.update(queries=runtime.queries)
            if runtime.dcp > 1:
                policy.update(
                    dcp=dcp_box(runtime, heads), parity=runtime.mla_layers % 2
                )
            runtime.mla_layers += 1
            pitches = {b: pitch for b in runtime.batches}
            self.core = PreparedMla(attention, pitches, query, **policy)
        parity = decoder.layer_idx % 2
        front: tuple | None = None
        if decoder.is_moe_layer:
            front = small_batch.front_views[parity], small_batch.hint_views[parity]
            front += (small_batch.batches,)
        self.output = PreparedOproj(
            decoder, mailbox, runtime.batches, kind != "kda", front, twoshot
        )
        projected = scratch(runtime, kind + ".projected", (64, pitch))
        local = scratch(runtime, "attention.local", (64, 12288 // tp))
        self.prefetch, self.layer = prefetch, decoder
        self.storage = {b: (projected[:b], local[:b]) for b in range(1, 65)}

    def prepare_layout(self, caches):
        if self.kind == "kda":
            self.core.prepare_caches(caches)
        else:
            self.core.prepare_cache(caches[0])

    def run_dsa(self, projected, local, caches, metadata, slot_mapping, q):
        from vllm.forward_context import get_forward_context
        from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

        attention = self.layer.self_attn
        slots = slot_mapping[attention.mla_attn.mla_attn.layer_name]
        # The packed indexer splits into kernel pages; its metadata builder
        # derives the write slots from their block table.
        indexer = get_forward_context().attn_metadata[
            attention.dsa_indexer.k_cache.prefix
        ]
        assert isinstance(indexer, DeepseekV32IndexerMetadata)
        index_slots = indexer.slot_mapping
        self.core.run(
            projected,
            local,
            cache=caches[0],
            slots=slots,
            index_slots=index_slots,
            metadata=metadata,
            q=q,
        )
        self.prefetch.fork("attention_moe", self.layer.layer_idx, projected.shape[0])

    def run(self, x, out, prefix, blocks, caches, metadata, slot_mapping, q=1):
        b = x.shape[0]
        projected, local = self.storage[b]
        projected = self.projection.run(x, projected)
        if self.kind == "dsa":
            self.run_dsa(projected, local, caches, metadata, slot_mapping, q)
        elif self.kind == "kda":
            self.prefetch.fork("attention_moe", self.layer.layer_idx, b)
            if q > 1:  # uniform q-token verify rows
                slots = metadata.spec_state_indices_tensor
                accepted = metadata.num_accepted_tokens
                self.core.run(
                    projected,
                    local,
                    caches=caches,
                    slots=slots,
                    accepted=accepted,
                    q=q,
                )
            else:
                slots = metadata.non_spec_state_indices_tensor
                self.core.run(projected, local, caches=caches, slots=slots)
        else:
            decode = metadata.decode
            tables, lengths = decode.block_table, decode.seq_lens
            # Request metadata shorter than b: the core runs the prepared batch b
            # with the missing rows as empty requests.
            rows = min(b, q * min(lengths.numel(), tables.shape[0]))
            if rows < b:
                local[rows:].zero_()
            slots = slot_mapping[self.layer.self_attn.layer_name]
            mla = dict(cache=caches[0], block_table=tables, lengths=lengths)
            # DCP: lengths are the local shard's, totals the full sequences'.
            mla.update(totals=decode.dcp_tot_seq_lens)
            self.core.run(projected, local, slots=slots, q=q, **mla)
            self.prefetch.fork("attention_moe", self.layer.layer_idx, b)
        self.output.run(local, prefix, blocks, out)
        self.prefetch.fork("oproj_down", self.layer.layer_idx, b)
        return out


def dsa_scratch(runtime):
    """The DSA layers' shared rows: the BF16 query (heads past the local ones stay
    zero), the normalized q_c, the indexer projection, FP8 query and head weights,
    and the sparse core's latent."""
    if "dsa.query" not in runtime.buffers:
        shape = (PREPARED_MAX, QUERY_HEADS, 576)
        runtime.buffers["dsa.query"] = torch.zeros(
            shape, device=runtime.device, dtype=torch.bfloat16
        )
    return (
        runtime.buffers["dsa.query"],
        scratch(runtime, "dsa.qc", (PREPARED_MAX, 1536)),
        scratch(runtime, "dsa.iq", (PREPARED_MAX, HEADS * DIM)),
        scratch(runtime, "dsa.q_fp8", (PREPARED_MAX, HEADS, DIM), torch.uint8),
        scratch(runtime, "dsa.weights", (PREPARED_MAX, HEADS)),
        scratch(runtime, "dsa.latent", (PREPARED_MAX, QUERY_HEADS, 512)),
    )


def dcp_box(runtime, heads, dsa=False, flashinfer=False):
    """The model's DCP mailboxes (one set for every MLA layer, one for every DSA
    layer: a q mailbox padded to the sparse decode's heads, BF16 for FlashMLA
    sparse's 64 or 128, FP8 for FlashInfer's trtllm-gen / prims_ts query heads)."""
    name = "dsa_dcp_box" if dsa else "dcp_box"
    if getattr(runtime, name) is None:
        from vllm.distributed import get_dcp_group

        group = get_dcp_group()
        assert group.world_size == runtime.dcp == runtime.tp
        from vllm.config import get_current_vllm_config

        args = group.device_group, runtime.device, group.world_size, group.rank_in_group
        parallel = get_current_vllm_config().parallel_config
        interleave = parallel.cp_kv_cache_interleave_size
        q_heads = None
        if dsa and flashinfer:
            from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
                trtllm_gen_query_heads,
            )

            q_heads = trtllm_gen_query_heads(group.world_size * heads)
        elif dsa:
            q_heads = 64 if group.world_size * heads <= 64 else 128
        box = DcpBox(
            *args,
            heads,
            interleave=interleave,
            q_heads=q_heads,
            q_bf16=dsa and not flashinfer,
        )
        setattr(runtime, name, box)
    return getattr(runtime, name)


def pack_front(moe, rank, tp):
    """[this rank's router rows (padded to 16); this rank's latent down shard]."""
    router, down = moe.gate.weight, moe.routed_expert_down_proj.weight
    rows, shard = 896 // tp, 3584 // tp
    router = router[rank * rows : (rank + 1) * rows]
    down = down[rank * shard : (rank + 1) * shard]
    return torch.cat((router, router.new_zeros((-rows % 16, 7168)), down)).detach()
