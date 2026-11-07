# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``mamba_spec_state_layout`` "compact" is refused where it cannot work."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.cpu_test


def make_config(
    speculative: bool, kv_transfer_config: KVTransferConfig | None = None
) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", skip_tokenizer_init=True),
        cache_config=CacheConfig(mamba_spec_state_layout="compact"),
        speculative_config=(
            SpeculativeConfig(method="ngram", num_speculative_tokens=2)
            if speculative
            else None
        ),
        kv_transfer_config=kv_transfer_config,
    )


@pytest.fixture
def supporting_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ModelConfig, "supports_compact_spec_state", property(lambda self: True)
    )


def test_compact_layout_requires_speculative_decoding():
    with pytest.raises(ValueError, match="requires speculative decoding"):
        make_config(speculative=False)


def test_compact_layout_requires_a_supporting_model():
    with pytest.raises(
        ValueError, match="not supported for architecture 'OPTForCausalLM'"
    ):
        make_config(speculative=True)


@pytest.mark.usefixtures("supporting_model")
def test_compact_layout_requires_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA"):
        make_config(speculative=True)


@pytest.mark.usefixtures("supporting_model")
def test_compact_layout_excludes_other_kv_connectors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    connector = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both")
    with pytest.raises(
        ValueError, match="does not support KV connectors other than MooncakeStore"
    ):
        make_config(speculative=True, kv_transfer_config=connector)


@pytest.mark.usefixtures("supporting_model")
def test_compact_layout_accepts_the_mooncake_store_connector(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    connector = KVTransferConfig(
        kv_connector="MooncakeStoreConnector", kv_role="kv_both"
    )
    config = make_config(speculative=True, kv_transfer_config=connector)
    assert config.cache_config.use_kda_recoverssm


class _Layer(MambaBase):
    mamba_type = "linear_attention"

    def get_state_shape(self):
        return ((4,),)

    def get_state_dtype(self):
        return (torch.float32,)


class _CompactLayer(_Layer):
    supports_compact_spec_state = True


def _spec_config(use_replayssm: bool, use_kda_recoverssm: bool):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            use_replayssm=use_replayssm,
            use_kda_recoverssm=use_kda_recoverssm,
            mamba_block_size=16,
            mamba_page_size_padded=None,
            mamba_cache_mode="align",
        ),
        num_speculative_tokens=3,
    )


def _spec_blocks(layer: MambaBase, config) -> int:
    spec = layer.get_kv_cache_spec(config)
    assert isinstance(spec, MambaSpec)
    return spec.num_speculative_blocks


def test_every_mamba_layer_of_a_compact_model_must_support_it():
    compact = _spec_config(use_replayssm=False, use_kda_recoverssm=True)
    assert _spec_blocks(_CompactLayer(), compact) == 0
    with pytest.raises(ValueError, match="_Layer does not support"):
        _Layer().get_kv_cache_spec(compact)
    # Under ReplaySSM (Kimi-K3's RecoverSSM included) every layer verifies
    # without per-position states, as before.
    replay = _spec_config(use_replayssm=True, use_kda_recoverssm=True)
    assert _spec_blocks(_Layer(), replay) == 0
    per_position = _spec_config(use_replayssm=False, use_kda_recoverssm=False)
    assert _spec_blocks(_Layer(), per_position) == 3
