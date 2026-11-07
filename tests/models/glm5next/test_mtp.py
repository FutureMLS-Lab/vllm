# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash MTP layer stored unquantized in a quantized checkpoint.

NVIDIA's NVFP4 checkpoint of GLM-5.3-Flash keeps its MTP layer in BF16 while
its quantization config does not exclude the layer, so building that layer
with the NVFP4 config creates parameters the checkpoint does not have. Under
NVFP4 the MTP layer is built unquantized when the checkpoint stores it without
scales; other quantization methods build it as their config says.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn
from transformers import Glm5NextTextConfig

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.models.glm5next.common import model as glm_model
from vllm.models.glm5next.common import mtp as glm_mtp

pytestmark = pytest.mark.cpu_test

NUM_HIDDEN_LAYERS = 2
HIDDEN, INTERMEDIATE = 64, 128


def _write_index(directory: Path, names: list[str]) -> None:
    weight_map = dict.fromkeys(names, "model-00001-of-00001.safetensors")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def _nvfp4_checkpoint(directory: Path) -> None:
    """Base layer 1 quantized, MTP layer 2 in BF16."""
    proj = "mlp.gate_up_proj"
    _write_index(
        directory,
        [
            f"model.layers.1.{proj}.weight",
            f"model.layers.1.{proj}.weight_scale",
            f"model.layers.1.{proj}.weight_scale_2",
            f"model.layers.1.{proj}.input_scale",
            f"model.layers.2.{proj}.weight",
            "model.layers.2.enorm.weight",
            "model.layers.2.eh_proj.weight",
        ],
    )


def _fp8_block_checkpoint(directory: Path) -> None:
    """Block FP8: every quantized weight with its ``weight_scale_inv``."""
    names = []
    for layer in (1, 2):
        for proj in ("mlp.gate_up_proj", "self_attn.o_proj"):
            names += [
                f"model.layers.{layer}.{proj}.weight",
                f"model.layers.{layer}.{proj}.weight_scale_inv",
            ]
    _write_index(directory, names)


def test_checkpoint_stores_layer_unquantized(tmp_path: Path):
    _nvfp4_checkpoint(tmp_path)
    model, nvfp4 = str(tmp_path), _nvfp4_config()
    assert glm_mtp.checkpoint_stores_layer_unquantized(nvfp4, model, None, 2)
    assert not glm_mtp.checkpoint_stores_layer_unquantized(nvfp4, model, None, 1)
    # A layer the checkpoint does not have is not called unquantized.
    assert not glm_mtp.checkpoint_stores_layer_unquantized(nvfp4, model, None, 3)


def test_without_an_index_no_layer_is_unquantized(tmp_path: Path):
    assert not glm_mtp.checkpoint_stores_layer_unquantized(
        _nvfp4_config(), str(tmp_path), None, 2
    )


def test_a_block_fp8_checkpoint_keeps_its_mtp_layer_quantized(tmp_path: Path):
    """Its scales have other names than NVFP4's, and its config is taken as
    is: the layer is not called unquantized."""
    _fp8_block_checkpoint(tmp_path)
    fp8 = Fp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
    )
    for config in (fp8, None):
        assert not glm_mtp.checkpoint_stores_layer_unquantized(
            config, str(tmp_path), None, 2
        )


class _Attention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.o_proj = SimpleNamespace(reduce_results=True)


def _nvfp4_config() -> ModelOptNvFp4Config:
    return ModelOptNvFp4Config.from_config(
        {
            "quant_method": "modelopt",
            "quantization": {
                "quant_algo": "NVFP4",
                "kv_cache_quant_algo": None,
                "group_size": 16,
                "exclude_modules": ["lm_head"],
            },
        }
    )


def test_mtp_layer_stored_unquantized_is_built_unquantized(
    monkeypatch, tmp_path: Path, dist_init, default_vllm_config
):
    _nvfp4_checkpoint(tmp_path)
    monkeypatch.setattr(glm_model, "Glm5NextMLAAttention", _Attention)
    config = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        first_k_dense_replace=NUM_HIDDEN_LAYERS + 1,
        n_routed_experts=4,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mhc=False,
        index_topk=16,
        index_kpool=4,
        # A checkpoint config field the MLA layer reads.
        mla_use_nope=False,
    )
    draft_model_config = SimpleNamespace(
        model=str(tmp_path), revision=None, hf_text_config=config
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=draft_model_config),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        cache_config=None,
        quant_config=_nvfp4_config(),
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
        kernel_config=SimpleNamespace(enable_jit_warmup=False),
    )

    layer = glm_mtp.Glm5NextMultiTokenPredictorLayer(
        vllm_config, f"model.layers.{NUM_HIDDEN_LAYERS}"
    )

    mlp = layer.mtp_block.mlp
    assert isinstance(mlp, glm_model.Glm5NextMLP)
    for proj in (mlp.gate_up_proj, mlp.down_proj):
        assert isinstance(proj.quant_method, UnquantizedLinearMethod)
