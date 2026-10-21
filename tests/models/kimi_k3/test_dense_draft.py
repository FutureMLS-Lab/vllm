# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense Kimi-K3 drafts (EAGLE3, DSpark) in Kimi-K3.1's merged DSA pages."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.config import KimiK3ForConditionalGenerationConfig
from vllm.models.kimi_k3.nvidia.dense_draft import place_dense_draft_in_merged_cache
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.kv_cache_planning import get_kv_cache_groups
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec, MergedMLAIndexerSpec
from vllm.v1.worker.utils import split_merged_dsa_cache_views

# Kimi-K3.1's text config: DSA with 128-dim indexer keys, on a KDA hybrid.
K31_TEXT_CONFIG = SimpleNamespace(
    dsa_config=SimpleNamespace(index_head_dim=128), linear_attn_config={}
)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        # A dense Kimi-K3 draft shares K3.1's merged pages.
        ("eagle3", (AttentionBackendEnum.FLASHINFER_MLA, "fp8")),
        ("dspark", (AttentionBackendEnum.FLASHINFER_MLA, "fp8")),
        # Other methods are left alone: K3.1's MTP layers keep the sparse path, and
        # n-gram has no draft layers.
        ("mtp", (None, None)),
        ("ngram", (None, None)),
    ],
)
def test_kimi_k31_defaults_dense_draft_attention(method, expected):
    spec_config = SimpleNamespace(
        method=method, attention_backend=None, kv_cache_dtype=None
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=K31_TEXT_CONFIG),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        speculative_config=spec_config,
    )

    KimiK3ForConditionalGenerationConfig.verify_and_update_config(vllm_config)

    assert (spec_config.attention_backend, spec_config.kv_cache_dtype) == expected


def test_dense_draft_shares_the_target_mla_group_and_pages():
    # Kimi-K3.1 (69 KDA + 24 merged MLA + indexer layers) plus an EAGLE3 draft.
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="kimi_k31"),
            hf_text_config=K31_TEXT_CONFIG,
        ),
        cache_config=SimpleNamespace(
            cache_dtype="fp8_ds_mla",
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
        cache_dtype_str="fp8_ds_mla",
        index_head_size=132,
        index_head_dim=128,
        kernel_block_size=64,
    )
    assert draft_spec == mla

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
    unit, num_units = mla.padded_unit_bytes, 6
    buf = torch.zeros(num_units, unit, dtype=torch.uint8)
    kv_caches = {
        "draft": torch.as_strided(buf, (num_units, 1, 64, 788), (unit, unit, 788, 1))
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
