# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 drafts on Kimi-K3.1: dense MLA drafts (EAGLE3, DSpark) on
FLASHINFER_MLA, GQA drafts (Qwen3 DSpark) in a KV cache group of their own."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.config import KimiK3ForConditionalGenerationConfig
from vllm.models.kimi_k3.nvidia.gqa_draft import GQADraftKVCacheGrouping
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.kv_cache_planning import get_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

# Kimi-K3.1's text config: DSA with 128-dim indexer keys, on a KDA hybrid.
K31_TEXT_CONFIG = SimpleNamespace(
    dsa_config=SimpleNamespace(index_head_dim=128), linear_attn_config={}
)


MLA_DRAFT = SimpleNamespace(kv_lora_rank=512)
GQA_DRAFT = SimpleNamespace(num_key_value_heads=16)


@pytest.mark.parametrize(
    ("method", "draft_text_config", "expected"),
    [
        # A dense Kimi-K3 draft reads its own fp8 MLA cache.
        ("eagle3", MLA_DRAFT, (AttentionBackendEnum.FLASHINFER_MLA, "fp8")),
        ("dspark", MLA_DRAFT, (AttentionBackendEnum.FLASHINFER_MLA, "fp8")),
        # A GQA draft (Qwen3 DSpark) keeps its own cache, off the sparse MLA path.
        ("dspark", GQA_DRAFT, (AttentionBackendEnum.FLASH_ATTN, "auto")),
        # Other methods are left alone: n-gram has no draft layers.
        ("ngram", None, (None, None)),
    ],
)
def test_kimi_k31_defaults_draft_attention(method, draft_text_config, expected):
    spec_config = SimpleNamespace(
        method=method,
        attention_backend=None,
        kv_cache_dtype=None,
        draft_model_config=SimpleNamespace(hf_text_config=draft_text_config),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=K31_TEXT_CONFIG),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        attention_config=SimpleNamespace(backend=None),
        speculative_config=spec_config,
    )

    KimiK3ForConditionalGenerationConfig.verify_and_update_config(vllm_config)

    assert (spec_config.attention_backend, spec_config.kv_cache_dtype) == expected


def _k31_cache_dtype(cache_dtype, backend=None):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=K31_TEXT_CONFIG),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        attention_config=SimpleNamespace(backend=backend),
        speculative_config=None,
    )
    KimiK3ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
    return vllm_config.cache_config.cache_dtype


@pytest.mark.parametrize("cache_dtype", ["fp8_ds_mla", "nvfp4_ds_mla"])
def test_kimi_k31_accepts_ds_mla_cache_dtypes(cache_dtype):
    assert _k31_cache_dtype(cache_dtype) == cache_dtype


@pytest.mark.parametrize(
    ("sm100", "backend", "cache_dtype", "expected"),
    [
        # SM10x defaults to FlashInfer sparse MLA, which reads per-tensor fp8.
        (True, None, "auto", "fp8_e4m3"),
        (True, None, "fp8", "fp8"),
        (True, AttentionBackendEnum.FLASHINFER_MLA_SPARSE, "auto", "fp8_e4m3"),
        # FlashMLA sparse, explicit or the default elsewhere, reads fp8_ds_mla.
        (True, AttentionBackendEnum.FLASHMLA_SPARSE, "auto", "fp8_ds_mla"),
        (True, AttentionBackendEnum.FLASHMLA_SPARSE, "fp8", "fp8_ds_mla"),
        (False, None, "auto", "fp8_ds_mla"),
    ],
)
def test_kimi_k31_defaults_kv_cache_to_its_sparse_backend(
    monkeypatch, sm100, backend, cache_dtype, expected
):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: sm100
    )
    assert _k31_cache_dtype(cache_dtype, backend) == expected


@pytest.mark.parametrize(
    ("block_size", "draft_kv_heads", "draft_block_size"),
    [
        # Attention TP1 (DP): 16 KV heads need 4096 B/token, over the MLA
        # page's 576 B/token, so the draft gets the largest block that fits.
        (8960, 16, 896),
        # Attention TP8: 2 KV heads per rank fit at the target's block size.
        (1280, 2, 1280),
    ],
)
def test_gqa_draft_gets_its_own_group_and_a_fitting_block(
    block_size, draft_kv_heads, draft_block_size
):
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(
                is_block_outermost=False
            ),
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    mla = MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=576, dtype=torch.uint8
    )
    kda = MambaSpec(
        block_size=block_size,
        shapes=((2, 512), (3, 32, 32)),
        dtypes=(torch.float32, torch.float32),
        page_size_padded=mla.page_size_bytes,
        mamba_cache_mode="align",
    )
    target = {
        f"model.layers.{i}": mla if i % 4 == 3 or i == 92 else kda for i in range(93)
    }
    draft = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=draft_kv_heads,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=2048,
    )
    specs = target | {f"model.layers.{93 + i}.attn": draft for i in range(5)}

    groups = GQADraftKVCacheGrouping.get_kv_cache_groups(vllm_config, specs)

    # The target keeps its 4 groups; the 5 draft layers form a fifth.
    assert [len(g.layer_names) for g in groups] == [23, 23, 23, 24, 5]
    assert groups[-1].kv_cache_spec.block_size == draft_block_size
    # The draft's pages are padded to the target's (one page size for all groups).
    assert {g.kv_cache_spec.page_size_bytes for g in groups} == {mla.page_size_bytes}
    # Without a draft, the default grouping is unchanged.
    assert GQADraftKVCacheGrouping.get_kv_cache_groups(
        vllm_config, target
    ) == get_kv_cache_groups(vllm_config, target)
