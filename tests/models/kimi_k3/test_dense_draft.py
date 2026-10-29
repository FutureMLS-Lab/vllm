# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 drafts on Kimi-K3.1: dense MLA drafts (EAGLE3, DSpark) in the merged
DSA pages, GQA drafts (Qwen3 DSpark) in a KV cache group of their own."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.config import KimiK3ForConditionalGenerationConfig
from vllm.models.kimi_k3.nvidia.dense_draft import place_dense_draft_in_merged_cache
from vllm.models.kimi_k3.nvidia.gqa_draft import GQADraftKVCacheGrouping
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.kv_cache_planning import get_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MambaSpec,
    MergedMLAIndexerSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.utils import split_merged_dsa_cache_views

# Kimi-K3.1's text config: DSA with 128-dim indexer keys, on a KDA hybrid.
K31_TEXT_CONFIG = SimpleNamespace(
    dsa_config=SimpleNamespace(index_head_dim=128), linear_attn_config={}
)


MLA_DRAFT = SimpleNamespace(kv_lora_rank=512)
GQA_DRAFT = SimpleNamespace(num_key_value_heads=16)


@pytest.mark.parametrize(
    ("method", "draft_text_config", "expected"),
    [
        # A dense Kimi-K3 draft shares K3.1's merged pages.
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
        speculative_config=spec_config,
    )

    KimiK3ForConditionalGenerationConfig.verify_and_update_config(vllm_config)

    assert (spec_config.attention_backend, spec_config.kv_cache_dtype) == expected


@pytest.mark.parametrize("cache_dtype", ["fp8_ds_mla", "nvfp4_ds_mla"])
def test_kimi_k31_accepts_ds_mla_cache_dtypes(cache_dtype):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=K31_TEXT_CONFIG),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        speculative_config=None,
    )
    KimiK3ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
    assert vllm_config.cache_config.cache_dtype == cache_dtype

    vllm_config.cache_config.cache_dtype = "bfloat16"
    with pytest.raises(ValueError, match="fp8_ds_mla and nvfp4_ds_mla"):
        KimiK3ForConditionalGenerationConfig.verify_and_update_config(vllm_config)


@pytest.mark.parametrize(
    ("cache_dtype", "row_bytes", "unit_bytes"),
    [
        # 64 x (656 + 132) = 50432, padded to a multiple of the 656B row.
        ("fp8_ds_mla", 656, 50512),
        # 64 x (352 + 132) = 30976 cannot hold the draft's 64 x 576B rows;
        # padded to 36864 rounded up to the 352B row.
        ("nvfp4_ds_mla", 352, 36960),
    ],
)
def test_dense_draft_shares_the_target_mla_group_and_pages(
    cache_dtype, row_bytes, unit_bytes
):
    # Kimi-K3.1 (69 KDA + 24 merged MLA + indexer layers) plus an EAGLE3 draft.
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="kimi_k31"),
            hf_text_config=K31_TEXT_CONFIG,
        ),
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            block_size=1088,
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(
                is_block_outermost=True
            ),
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=SimpleNamespace(
            method="eagle3", use_eagle=lambda: True, use_eagle_block_drop=lambda: True
        ),
    )
    draft = SimpleNamespace(head_size=576, non_causal_multi_token_decode=False)
    place_dense_draft_in_merged_cache(draft, vllm_config)
    draft_spec = draft.get_kv_cache_spec(vllm_config)

    # The draft registers the target's exact spec.
    mla = MergedMLAIndexerSpec.from_mla_and_indexer(
        block_size=1088,
        mla_head_size=576,
        mla_dtype=torch.uint8,
        cache_dtype_str=cache_dtype,
        index_head_size=132,
        index_head_dim=128,
        kernel_block_size=64,
    )
    assert draft_spec == mla
    assert mla.padded_unit_bytes == unit_bytes

    # So it joins the target's MLA group: K3.1 keeps its 4 groups. (That group
    # takes the EAGLE block drop through the draft layer's is_draft tag.)
    kda = MambaSpec(
        block_size=1088,
        shapes=((2, 512), (3, 32, 32)),
        dtypes=(torch.float32, torch.float32),
        page_size_padded=mla.page_size_bytes,
        mamba_cache_mode="align",
        num_speculative_blocks=2,
    )
    specs = {
        f"model.layers.{i}": mla if i % 4 == 3 or i == 92 else kda for i in range(93)
    }
    specs["draft.layers.0"] = draft_spec
    groups = get_kv_cache_groups(vllm_config, specs)
    assert len(groups) == 4
    (draft_group,) = [g for g in groups if "draft.layers.0" in g.layer_names]
    assert len(draft_group.layer_names) == 25

    # In its own pages it binds 64 plain fp8 MLA rows of 576 B at the head of
    # each unit, at the unit stride; the merged-view split leaves it alone.
    unit, num_units, row = mla.padded_unit_bytes, 6, row_bytes + 132
    buf = torch.zeros(num_units, unit, dtype=torch.uint8)
    kv_caches = {
        "draft": torch.as_strided(buf, (num_units, 1, 64, row), (unit, unit, row, 1))
    }
    split_merged_dsa_cache_views(
        kv_caches, {"draft": draft}, [KVCacheGroupSpec(["draft"], draft_spec)]
    )
    draft.bind_kv_cache(kv_caches["draft"])
    view = draft.kv_cache
    assert view.shape == (num_units, 64, 576)
    assert view.stride() == (unit, 576, 1)
    view.fill_(7)
    assert bool((buf[:, : 64 * 576] == 7).all())
    assert bool((buf[:, 64 * 576 :] == 0).all())


@pytest.mark.parametrize(
    ("block_size", "draft_kv_heads", "draft_block_size"),
    [
        # Attention TP1 (DP): 16 KV heads need 4096 B/token, over the merged MLA
        # page's ~789 B/token, so the draft gets the largest block that fits.
        (8960, 16, 1280),
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
    mla = MergedMLAIndexerSpec.from_mla_and_indexer(
        block_size=block_size,
        mla_head_size=576,
        mla_dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        index_head_size=132,
        index_head_dim=128,
        kernel_block_size=64,
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
