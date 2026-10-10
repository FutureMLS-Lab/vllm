# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense (non-DSA) MTP draft support for Kimi K3 Flash (DSA target).

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
"""

from types import SimpleNamespace

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata

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


class KimiK3FlashDenseDraftIndexer(nn.Module):
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
