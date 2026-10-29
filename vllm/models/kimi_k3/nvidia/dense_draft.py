# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense MLA draft support for Kimi K3.1 (DSA target).

Dense EAGLE3 and DSpark drafts keep their own fp8 MLA cache and decode with
FLASHINFER_MLA (``configure_dense_draft``).
"""

from typing import Any

from vllm.config import SpeculativeConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def configure_dense_draft(spec_config: SpeculativeConfig, text_config: Any) -> None:
    """Default an EAGLE3/DSpark draft's attention backend and KV dtype on a
    KDA-hybrid DSA target: FLASHINFER_MLA + fp8 for MLA drafts, FLASH_ATTN +
    model dtype for GQA drafts. MTP is unchanged."""
    if (
        spec_config.method not in ("eagle3", "dspark")
        or getattr(text_config, "linear_attn_config", None) is None
    ):
        return
    draft_text_config = spec_config.draft_model_config.hf_text_config
    is_mla_draft = getattr(draft_text_config, "kv_lora_rank", None) is not None
    if is_mla_draft:
        # MLA draft: no SM100 dense MLA kernel reads the target's fp8_ds_mla.
        backend, kv_cache_dtype = AttentionBackendEnum.FLASHINFER_MLA, "fp8"
    else:
        # GQA draft: its own cache, not the target's sparse backend or fp8_ds_mla.
        backend, kv_cache_dtype = AttentionBackendEnum.FLASH_ATTN, "auto"
    spec_config.attention_backend = spec_config.attention_backend or backend
    spec_config.kv_cache_dtype = spec_config.kv_cache_dtype or kv_cache_dtype
