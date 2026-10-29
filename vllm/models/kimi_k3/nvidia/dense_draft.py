# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense MLA draft support for Kimi K3.1 (DSA target).

Dense EAGLE3 and DSpark drafts keep plain fp8 MLA rows in the target's merged
pages and decode with FLASHINFER_MLA (``configure_dense_draft``,
``place_dense_draft_in_merged_cache``).
"""

from dataclasses import replace
from types import MethodType
from typing import Any

import torch

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import MergedMLAIndexerSpec


def configure_dense_draft(spec_config: SpeculativeConfig, text_config: Any) -> None:
    """Default an EAGLE3/DSpark draft's attention backend and KV dtype on a
    KDA-hybrid DSA target: FLASHINFER_MLA + fp8 for MLA drafts (in the target's
    pages), FLASH_ATTN + model dtype for GQA drafts (own cache)."""
    if (
        spec_config.method not in ("eagle3", "dspark")
        or getattr(text_config, "linear_attn_config", None) is None
    ):
        return
    draft_text_config = spec_config.draft_model_config.hf_text_config
    is_mla_draft = getattr(draft_text_config, "kv_lora_rank", None) is not None
    if is_mla_draft:
        # MLA draft: fp8 rows inside the target's merged pages; bf16 would not fit.
        backend, kv_cache_dtype = AttentionBackendEnum.FLASHINFER_MLA, "fp8"
    else:
        # GQA draft: its own cache, not the target's sparse backend or fp8_ds_mla.
        backend, kv_cache_dtype = AttentionBackendEnum.FLASH_ATTN, "auto"
    spec_config.attention_backend = spec_config.attention_backend or backend
    spec_config.kv_cache_dtype = spec_config.kv_cache_dtype or kv_cache_dtype
    fp8_draft_cache = spec_config.kv_cache_dtype in ("fp8", "fp8_e4m3")
    if is_mla_draft and (
        spec_config.attention_backend != backend or not fp8_draft_cache
    ):
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
    assert kv_cache.shape[2] * row <= kv_cache.stride(0), (
        f"dense draft rows ({kv_cache.shape[2]} x {row}B) overflow the "
        f"{kv_cache.stride(0)}B merged unit"
    )
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
