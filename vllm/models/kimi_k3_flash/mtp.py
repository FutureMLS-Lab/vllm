# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 Flash Multi-Token-Prediction (MTP) draft model.

The converted checkpoint keeps the MTP (nextn) layer as a full DSA decoder
block plus the DeepSeek-V3-style MTP accessories under
``model.layers.{num_hidden_layers}.*``: ``enorm``/``hnorm``/``eh_proj``
(concat order already swapped to the dsv3 ``[enorm(x_emb), hnorm(hidden)]``
layout by the converter), an independent ``embed_tokens`` copy, and a
``shared_head`` (norm + lm head). The block itself reuses
``KimiK3FlashDecoderLayer``, so the MTP layer runs the same MLA + DSA
indexer + MoE path as the backbone.
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.kimi_k3.common.mtp import fused_mtp_input
from vllm.sequence import IntermediateTensors

from .configs import KimiK3FlashTextConfig
from .model import (
    KIMI_K3_FLASH_DSA_RENAME_PARAMS_MAPPING,
    KimiK3FlashDecoderLayer,
    get_spec_layer_idx_from_weight_name,
    maybe_pad_weight_with_param,
)

logger = init_logger(__name__)


class SharedHead(nn.Module):
    def __init__(
        self,
        config: KimiK3FlashTextConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class KimiK3FlashMultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        config: KimiK3FlashTextConfig,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, "eh_proj"),
            return_bias=False,
        )
        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )

        # The MTP block holds a single DSA layer; give it a top-k buffer with
        # the same sizing as the backbone model-level one (KimiK3FlashModel):
        # [max_num_batched_tokens + dcp allgather_k slack, index_topk].
        assert config.dsa_config is not None, (
            "KimiK3Flash MTP layer must be a DSA layer "
            "(dsa_config missing from the text config)"
        )
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        dcp_kg_slack = dcp_world_size - 1 if dcp_world_size > 1 else 0
        topk_indices_buffer = torch.empty(
            (
                vllm_config.scheduler_config.max_num_batched_tokens + dcp_kg_slack,
                config.dsa_config.index_topk,
            ),
            dtype=torch.int32,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # The prefix must end in the numeric spec-layer index so the decoder
        # layer parses layer_idx and takes the DSA attention branch.
        self.mtp_block = KimiK3FlashDecoderLayer(
            config,
            vllm_config.cache_config,
            quant_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert inputs_embeds is not None
        hidden_states = self.eh_proj(
            fused_mtp_input(
                positions,
                inputs_embeds,
                previous_hidden_states,
                self.enorm.weight,
                self.hnorm.weight,
                self.enorm.variance_epsilon,
            )
        )
        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        # Produce the normalized logits input and the pre-norm recurrent state
        # in one fused add-RMSNorm launch.
        logits_hidden_states, hidden_states = self.shared_head.norm(
            hidden_states, residual
        )
        return logits_hidden_states, hidden_states


class KimiK3FlashMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: KimiK3FlashTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers

        self.layers = torch.nn.ModuleDict(
            {
                str(idx): KimiK3FlashMultiTokenPredictorLayer(
                    config, vllm_config, f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = self.logits_processor(mtp_layer.shared_head.head, hidden_states)
        return logits


class KimiK3FlashMTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        self.model = KimiK3FlashMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Mirror KimiK3FlashForCausalLM.load_weights naming: stacked QKV-A and
        # gate/up fusion, HF-style expert names, and the shared DSA indexer
        # rename table, scoped to the spec-layer weights only.
        stacked_params_mapping: list[tuple[str, str, Any]] = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        if self.config.is_mla and self.config.mla_config.q_lora_rank is not None:
            stacked_params_mapping += [
                (".fused_qkv_a_proj", ".q_a_proj", 0),
                (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            ]

        if self.config.is_moe:
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts,
            )
        else:
            expert_params_mapping = []

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            # The multimodal checkpoint prefixes text weights with
            # ``language_model.``; strip it so names match this draft model's
            # parameter paths (``model.layers.{i}.``). Non-text weights
            # (vision_tower, mm_projector, ...) never match a spec layer below.
            if name.startswith("language_model."):
                name = name[len("language_model.") :]
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)

            # The MoE submodule keeps the internal `block_sparse_moe`
            # attribute name while HF-style checkpoints name it `mlp` (same
            # rule as the main model).
            if (
                ".mlp.experts." in name
                or ".mlp.gate." in name
                or ".mlp.shared_experts." in name
            ):
                name = name.replace(".mlp.", ".block_sparse_moe.", 1)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Routed experts are handled by the expert mapping below;
                # skip them here before the name rewrite (".experts." with a
                # leading dot does not match ".shared_experts.").
                if ".experts." in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # Only take this mapping if the fused destination actually
                # exists (QKV-A fusion is conditional on q_lora_rank).
                if name_mapped not in params_dict:
                    continue
                name = name_mapped
                param = params_dict[name]
                loaded_weight = maybe_pad_weight_with_param(name, loaded_weight, param)
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for (
                    expert_param_name,
                    expert_weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if expert_weight_name not in name:
                        continue
                    name = name.replace(expert_weight_name, expert_param_name)
                    # Mxfp4 checkpoints use .weight_packed: try both namings,
                    # with and without the _packed suffix.
                    if name not in params_dict and name.endswith("_packed"):
                        name = name.removesuffix("_packed")
                    if name not in params_dict and f"{name}_packed" in params_dict:
                        name = f"{name}_packed"
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=expert_shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # DSA indexer renames. Must run before the bias skip
                    # below: the indexer k_norm.bias only exists under its
                    # checkpoint name at this point and would otherwise be
                    # dropped as an "extra bias".
                    for (
                        param_name,
                        weight_name,
                    ) in KIMI_K3_FLASH_DSA_RENAME_PARAMS_MAPPING:
                        if weight_name in name:
                            name = name.replace(weight_name, param_name)
                            break

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name

                    # The embedding is shared across MTP layers; only the
                    # first spec layer carries the hoisted (non-".layers")
                    # copy.
                    if spec_layer != self.model.mtp_start_layer_idx and (
                        ".layers" not in name
                    ):
                        continue
                    # Weights without a matching parameter (e.g. the unused
                    # eagle3_fc) are skipped.
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    loaded_weight = maybe_pad_weight_with_param(
                        name, loaded_weight, param
                    )
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # Validate that weights were loaded for each expected MTP layer.
        loaded_layers: set[int] = set()
        for param_name in loaded_params:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
            if spec_layer is not None:
                loaded_layers.add(spec_layer)
        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if layer_idx not in loaded_layers:
                raise ValueError(
                    f"MTP speculative decoding layer {layer_idx} weights "
                    f"missing from checkpoint. The checkpoint may not include "
                    f"the MTP layer weights. Use a checkpoint that includes "
                    f"MTP layer weights, or disable speculative decoding."
                )

        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """Rewrite a checkpoint weight name to this module's parameter path.

        Top-level MTP submodules (enorm/hnorm/eh_proj/shared_head) stay under
        ``model.layers.{spec_layer}.*``; the shared ``embed_tokens`` is
        hoisted to ``model.*``; everything else is a transformer-block weight
        and gets ``.mtp_block`` inserted.
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            name = name.replace(
                f"model.layers.{spec_layer}.",
                f"model.layers.{spec_layer}.mtp_block.",
            )
        elif shared_weight:
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name
