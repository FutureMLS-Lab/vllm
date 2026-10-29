# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3.1 DSA indexer cache, shared by the NVIDIA and ROCm stacks."""

from dataclasses import replace

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_compressed_slot_mapping,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.kv_cache_layout import KVCacheLayout


class KimiK3IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """Writes the indexer keys at slots derived from the block table, as the
    compressed indexers are written: in a packed block row the block table
    addresses 64-token kernel pages, which manager slots do not."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.use_pcp:
            raise NotImplementedError("Kimi-K3.1 packed DSA caches with PCP")
        self.slot_mapping_buffer = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        slot_mapping = get_compressed_slot_mapping(
            common_attn_metadata.num_actual_tokens,
            common_attn_metadata.slot_mapping,
            common_attn_metadata.query_start_loc,
            common_attn_metadata.seq_lens,
            common_attn_metadata.block_table_tensor,
            self.kv_cache_spec.block_size,
            compress_ratio=1,
            out=self.slot_mapping_buffer,
            dcp_world_size=self.dcp_world_size,
            cp_interleave=self.cp_kv_cache_interleave_size,
        )
        return super().build(
            common_prefix_len,
            replace(common_attn_metadata, slot_mapping=slot_mapping),
            fast_build,
        )


class KimiK3IndexerBackend(DeepseekV32IndexerBackend):
    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # Pack the indexer pages beside the MLA pages inside each block, so a
        # KDA state page sizes the whole block row rather than one MLA page.
        return (KVCacheLayout.BLHNC,)

    @classmethod
    def writes_kv_cache_from_block_table(cls) -> bool:
        return True

    @staticmethod
    def get_builder_cls() -> type[KimiK3IndexerMetadataBuilder]:
        return KimiK3IndexerMetadataBuilder


class KimiK3IndexerCache(DeepseekV32IndexerCache):
    def get_attn_backend(self) -> type[AttentionBackend]:
        return KimiK3IndexerBackend


def packed_dsa_block_size(
    vllm_config: VllmConfig,
    mla_bytes_per_token: int,
    state_bytes: int,
    alignment: int,
) -> int | None:
    """Block size of the packed cache of a KDA-hybrid DSA model, else None.

    A block row holds every DSA layer's MLA and indexer pages; size it to fit
    as many KDA states as there are DSA layers. A power-of-two multiple of
    ``alignment`` keeps the block sizes of different TP sizes divisible, so
    P/D transfers can pair blocks.
    """
    text_config = vllm_config.model_config.hf_text_config
    dsa_config = getattr(text_config, "dsa_config", None)
    linear_attn_config = getattr(text_config, "linear_attn_config", None)
    if dsa_config is None or linear_attn_config is None:
        return None
    num_dsa = len(linear_attn_config["full_attn_layers"])
    num_states = len(linear_attn_config["kda_layers"])
    states_per_row = cdiv(num_states, cdiv(num_states, num_dsa))
    # The indexer keeps fp8 values plus one fp32 scale per token.
    row_bytes_per_token = num_dsa * (
        mla_bytes_per_token + dsa_config.index_head_dim + 4
    )
    return alignment * next_power_of_2(
        cdiv(states_per_row * state_bytes, alignment * row_bytes_per_token)
    )
