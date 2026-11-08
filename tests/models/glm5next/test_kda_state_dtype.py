# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--mamba-ssm-cache-dtype`` must reach GLM-5.3-Flash's KDA recurrent state
(it used to be dropped, leaving the state float32 unconditionally)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.interfaces import supports_compact_spec_state
from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
from vllm.models.glm5next.common.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
)


@pytest.mark.parametrize(
    ("ssm_dtype", "expected"),
    [("auto", torch.float32), ("float32", torch.float32), ("bfloat16", torch.bfloat16)],
)
def test_kda_recurrent_state_dtype_follows_ssm_cache_dtype(ssm_dtype, expected):
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    cache_config = SimpleNamespace(
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype=ssm_dtype,
        use_kda_recoverssm=False,
    )
    vllm_config = SimpleNamespace(model_config=model_config, cache_config=cache_config)
    want = (torch.bfloat16, expected)

    assert Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config) == want
    assert (
        Glm5NextForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)
        == want
    )
    # The layer's own dtype must agree with the model-level one, since the
    # KV-cache spec and the kernels read them separately.
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    layer.model_config, layer.cache_config = model_config, cache_config
    assert layer.get_state_dtype() == want


def test_compact_spec_state_keeps_the_state_page():
    """``mamba_spec_state_layout`` "compact" (RecoverSSM verify) leaves the
    state page as it is: the verify records live outside the KV cache."""
    hf_config = SimpleNamespace(
        linear_num_heads=4, linear_head_dim=32, linear_conv_kernel_dim=4
    )

    def vllm_config(use_kda_recoverssm: bool) -> SimpleNamespace:
        return SimpleNamespace(
            model_config=SimpleNamespace(
                dtype=torch.bfloat16, hf_text_config=hf_config
            ),
            cache_config=SimpleNamespace(
                mamba_cache_dtype="auto",
                mamba_ssm_cache_dtype="auto",
                use_kda_recoverssm=use_kda_recoverssm,
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            speculative_config=SimpleNamespace(num_speculative_tokens=3),
        )

    for model_cls in (Glm5NextForCausalLM, Glm5NextForConditionalGeneration):
        assert supports_compact_spec_state(model_cls)
        for get in (
            model_cls.get_mamba_state_dtype_from_config,
            model_cls.get_mamba_state_shape_from_config,
        ):
            assert get(vllm_config(True)) == get(vllm_config(False))
        assert model_cls.get_mamba_state_dtype_from_config(vllm_config(True)) == (
            torch.bfloat16,
            torch.float32,
        )
