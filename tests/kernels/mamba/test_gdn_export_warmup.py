# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""With in-chunk export on FlashInfer's SM100 kernel, every exporting prefill
runs the per-sequence export kernel: an error compiling it at warmup fails
startup instead of passing for the warmup's autotuner warning. The layer's
real warmup methods run on stand-in kernels."""

import types
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

pytestmark = pytest.mark.cpu_test

HEADS, DIM = 2, 128


def _layer() -> SimpleNamespace:
    layer = SimpleNamespace(
        _prefill_kernels_warmed_up=False,
        _exports_checkpoints=True,
        _export_per_sequence=True,
        _checkpoint_buffer_spec=((1, HEADS, DIM, DIM), torch.float32),
        num_k_heads=HEADS,
        num_v_heads=HEADS,
        tp_size=1,
        head_k_dim=DIM,
        head_v_dim=DIM,
        A_log=None,
        dt_bias=None,
        gdn_prefill_backend="flashinfer",
        prefix="model.layers.0.linear_attn",
        chunk_gated_delta_rule=lambda **kwargs: None,
        get_state_dtype=lambda: (torch.float32, torch.float32),
    )
    for name in (
        "_warmup_prefill_kernels",
        "_warmup_flashinfer_prefill_variants",
        "_warmup_export_kernel",
    ):
        if hasattr(gdn.QwenGatedDeltaNetAttention, name):
            method = getattr(gdn.QwenGatedDeltaNetAttention, name)
            setattr(layer, name, types.MethodType(method, layer))
    return layer


def test_an_export_kernel_that_fails_to_compile_fails_the_warmup(
    monkeypatch: pytest.MonkeyPatch,
):
    def prep(conv_output, num_k_heads, head_k_dim, head_v_dim, **kwargs):
        tokens = conv_output.shape[0]
        return (
            torch.zeros(tokens, num_k_heads, head_k_dim),
            torch.zeros(tokens, num_k_heads, head_k_dim),
            torch.zeros(tokens, HEADS, head_v_dim),
            torch.zeros(tokens, HEADS),
            torch.zeros(tokens, HEADS),
        )

    def export(**kwargs):
        raise RuntimeError("export kernel failed to compile")

    workspace = SimpleNamespace(
        get_simultaneous=lambda *specs: [torch.zeros(spec[0]) for spec in specs]
    )
    monkeypatch.setattr(gdn, "fused_post_conv_prep", prep)
    monkeypatch.setattr(gdn, "fi_chunk_gated_delta_rule", lambda **kwargs: None)
    monkeypatch.setattr(gdn, "fi_chunk_gated_delta_rule_export", export)
    monkeypatch.setattr(gdn, "current_workspace_manager", lambda: workspace)
    qkv = torch.zeros(1, 3 * HEADS * DIM)
    with pytest.raises(RuntimeError, match="failed to compile"):
        _layer()._warmup_prefill_kernels(qkv, HEADS * DIM)
