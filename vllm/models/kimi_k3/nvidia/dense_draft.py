# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense (non-DSA) MTP draft support for Kimi K3.1 (DSA target).

The draft layer has no indexer weights, and no SM100 dense MLA kernel can
read the globally enforced fp8_ds_mla cache. Instead of a real dense
kernel, the draft keeps the sparse chain and substitutes this fake indexer
that fills every query row's candidate set with the full logical range
(``0..len-1``). The existing ``triton_convert_req_index_to_global_index``
then yields physical slots plus ``topk_length == len``, which the sparse
kernels honor, so the result is mathematically exact dense attention.

The fill runs eagerly (the indexer op is eager-break during CUDA graph
capture) and only depends on device-side metadata (per-request seq lens and
request-id mapping from the MLA layer's FlashMLASparseMetadata).

Dense EAGLE3 and DSpark drafts take another route: they keep plain fp8 MLA rows
in the target's merged pages and decode with FLASHINFER_MLA
(``configure_dense_draft``, ``place_dense_draft_in_merged_cache``).
"""

from dataclasses import replace
from types import MethodType, SimpleNamespace
from typing import Any

import torch
from torch import nn

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import MergedMLAIndexerSpec

logger = init_logger(__name__)


def _fill_full_range_(rows: torch.Tensor, valid_lens: torch.Tensor) -> None:
    """Fill each row with arange up to its valid length, -1 past the tail.

    The tail must be invalidated because the buffer is persistent and rows
    are reused across requests.
    """
    if rows.shape[0] == 0:
        return
    width = rows.shape[1]
    ar = torch.arange(width, device=rows.device, dtype=torch.int32)
    valid = valid_lens.reshape(-1, 1).to(torch.int32)
    rows.copy_(torch.where(ar.unsqueeze(0) < valid, ar.unsqueeze(0), -1))


class KimiK3DenseDraftIndexer(nn.Module):
    """Indexer replacement for a dense MTP draft layer on a DSA target.

    Duck-typed to the ``MLAModules`` indexer contract (``topk_tokens``) and
    to the impl's ``dense_draft`` probe. Never runs the sparse_attn_indexer
    op: its ``forward`` just fills the shared top-k buffer with full-range
    candidates derived from the draft MLA layer's attention metadata.
    """

    # FlashMLASparseImpl probes this to take dense-aware branches (duck type,
    # same convention as the old line's DenseDraftIndexer).
    dense_draft: bool = True

    def __init__(
        self,
        *,
        max_model_len: int,
        max_num_batched_tokens: int,
        dcp_kg_slack: int = 0,
        index_head_dim: int,
        prefix: str,
        device: torch.device | str,
        use_merged_cache: bool = False,
    ) -> None:
        super().__init__()
        # MLAModules contract: MultiHeadLatentAttention asserts topk_tokens.
        self.topk_tokens = max_model_len
        self.max_model_len = max_model_len
        # K31 hybrid: the draft's MLA layer is marked with the fake k_cache's
        # geometry so it registers the merged spec (same layout as the target).
        self.use_merged_cache = use_merged_cache
        self.topk_indices_buffer = torch.full(
            (max_num_batched_tokens + dcp_kg_slack, max_model_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        # Shape-compat facade of the real indexer's k_cache: lets callers
        # that only read head_dim / prefix (e.g. merged-spec marker setup in
        # hybrid models) work unchanged. The dense draft never inserts or
        # reads indexer cache data, and nothing is registered under this
        # prefix (it is not an nn.Module).
        self.k_cache = SimpleNamespace(
            head_dim=index_head_dim + 4,
            index_head_dim=index_head_dim,
            prefix=f"{prefix}.k_cache",
        )
        # The draft MLA layer's registered layer name: its FlashMLASparseMetadata
        # entry carries seq_lens / req_id_per_token used for the fill.
        assert prefix.endswith(".dsa_indexer"), prefix
        self._attn_layer_name = prefix[: -len(".dsa_indexer")] + ".attn"

    def forward(
        self,
        hidden_states: torch.Tensor,
        index_q_input: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        rotary_emb: nn.Module | None = None,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            # Profiling/dummy run: metadata is absent; nothing to fill.
            return self.topk_indices_buffer

        metadata = attn_metadata[self._attn_layer_name]
        assert isinstance(metadata, FlashMLASparseMetadata)

        buf = self.topk_indices_buffer
        num_actual_tokens = metadata.num_actual_tokens
        num_decode_tokens = metadata.num_decode_tokens
        seq_lens = metadata.seq_lens
        assert seq_lens is not None
        req_ids = metadata.req_id_per_token[:num_actual_tokens]

        if num_decode_tokens > 0:
            # ends[b,j] = seq_lens[b] - decode_len[b] + j + 1: each query row
            # sees only its own causal prefix (strict dense causal attention).
            # Filling seq_lens for every row would leak later draft tokens of
            # the same verify batch into earlier rows' attention.
            qsl = metadata.query_start_loc
            req_ids_d = req_ids[:num_decode_tokens]
            decode_lens = qsl[1:] - qsl[:-1]
            positions = (
                torch.arange(num_decode_tokens, device=buf.device) - qsl[:-1][req_ids_d]
            )
            ends = seq_lens[req_ids_d] - decode_lens[req_ids_d] + positions + 1
            _fill_full_range_(buf[:num_decode_tokens], ends)
        if num_actual_tokens > num_decode_tokens:
            # Prefill rows: candidates = the token's causal context (cached
            # prefix + position within the request + 1).
            qsl = metadata.query_start_loc
            req_ids_p = req_ids[num_decode_tokens:]
            query_len = qsl[1:] - qsl[:-1]
            cached_ctx = seq_lens - query_len
            positions = (
                torch.arange(
                    num_decode_tokens,
                    num_actual_tokens,
                    device=buf.device,
                )
                - qsl[:-1][req_ids_p]
            )
            _fill_full_range_(
                buf[num_decode_tokens:num_actual_tokens],
                cached_ctx[req_ids_p] + positions + 1,
            )
        return buf


def configure_dense_draft(spec_config: SpeculativeConfig, text_config: Any) -> None:
    """Default a dense MLA EAGLE3 or DSpark draft of a KDA-hybrid DSA target
    (``text_config``) to FLASHINFER_MLA on a plain fp8 cache, the setup that shares
    the target's merged pages, and reject anything else. MTP keeps the sparse path."""
    if (
        spec_config.method not in ("eagle3", "dspark")
        or getattr(text_config, "linear_attn_config", None) is None
    ):
        return
    backend = AttentionBackendEnum.FLASHINFER_MLA
    spec_config.attention_backend = spec_config.attention_backend or backend
    spec_config.kv_cache_dtype = spec_config.kv_cache_dtype or "fp8"
    fp8_draft_cache = spec_config.kv_cache_dtype in ("fp8", "fp8_e4m3")
    if spec_config.attention_backend != backend or not fp8_draft_cache:
        raise ValueError(
            "A dense MLA draft on a hybrid (KDA + MLA) DSA target needs "
            f"FLASHINFER_MLA on an fp8 KV cache, got {spec_config.attention_backend} "
            f"and {spec_config.kv_cache_dtype!r}. Leave both unset in "
            "--speculative-config."
        )


def _merged_kv_cache_spec(layer: Any, vllm_config: VllmConfig) -> MergedMLAIndexerSpec:
    # vllm_config is the runner's, so this is the target's cache format.
    cache_dtype = vllm_config.cache_config.cache_dtype
    dtype = kv_cache_dtype_str_to_dtype(cache_dtype, vllm_config.model_config)
    index_dim = vllm_config.model_config.hf_text_config.dsa_config.index_head_dim
    spec = MergedMLAIndexerSpec.from_mla_and_indexer(
        block_size=vllm_config.cache_config.block_size,
        mla_head_size=layer.head_size,
        mla_dtype=dtype,
        cache_dtype_str=cache_dtype,
        # KimiK3DSAIndexer keys: fp8 values plus one fp32 scale.
        index_head_size=index_dim + 4,
        index_head_dim=index_dim,
        kernel_block_size=64,
    )
    return replace(
        spec,
        non_causal_multi_token_decode=layer.non_causal_multi_token_decode,
    )


def _bind_plain_rows(layer: Any, kv_cache: torch.Tensor) -> None:
    # kv_cache: this layer's merged pages, (units, 1, 64, merged row bytes) at the
    # unit stride. Plain fp8 MLA rows, one byte per element.
    row = layer.head_size
    layer.kv_cache = torch.as_strided(
        kv_cache,
        size=(kv_cache.shape[0], kv_cache.shape[2], row),
        stride=(kv_cache.stride(0), row, 1),
    )


def place_dense_draft_in_merged_cache(layer: Any, vllm_config: VllmConfig) -> None:
    """Let a dense Kimi-K3 draft's MLA layer (EAGLE3, DSpark) share a KDA-hybrid DSA
    target's merged pages: it registers the target's spec, joining its MLA KV cache
    group, and binds plain fp8 MLA rows at the head of each 64-token unit. Both
    methods are replaced on this layer only, so the shared MLA layer is unchanged."""
    text_config = vllm_config.model_config.hf_text_config
    if (
        getattr(text_config, "dsa_config", None) is not None
        and getattr(text_config, "linear_attn_config", None) is not None
    ):
        layer.get_kv_cache_spec = MethodType(_merged_kv_cache_spec, layer)
        layer.bind_kv_cache = MethodType(_bind_plain_rows, layer)
