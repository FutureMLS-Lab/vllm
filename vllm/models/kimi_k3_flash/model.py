# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 Flash text model: MLA + MoE backbone with DSA (DeepSeek Sparse
Attention).

The DSA path is wired to the native upstream DSA facilities: the indexer
holds a ``DeepseekV32IndexerCache`` and drives
``torch.ops.vllm.sparse_attn_indexer`` directly, and the sparse MLA layers
are consumed by the FlashMLA sparse backend. Index share ("F"/"S" layers,
``index_topk_freq`` > 1) follows the upstream deepseek_v2 / deepseek_v32
formula: "F" layers build their own indexer and write the model-level shared
topk_indices_buffer, "S" layers reuse the indices written by the previous
"F" layer.
"""

import math
from collections.abc import Iterable
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    get_dcp_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SituAndMul
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backends.mla import indexer as v32_indexer

from .configs import KimiK3FlashDSAConfig, KimiK3FlashTextConfig
from .dense_draft import KimiK3FlashDenseDraftIndexer

# Importing registers the torch.ops.vllm.index_rope custom op (side effect)
# and binds the directly-called index_rope python entry point.
from .index_rope import index_rope

logger = init_logger(__name__)


def maybe_pad_weight_with_param(
    name: str, loaded_weight: torch.Tensor, param: torch.Tensor
) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    # (param_name, shard_dim, is_replicated_linear)
    pad_name_dim = [
        # MLA projection related
        (".self_attn.q_b_proj.weight", 0, False),
        (".self_attn.q_proj.weight", 0, False),
        (".self_attn.kv_b_proj.weight", 0, False),
        (".self_attn.o_proj.weight", 1, False),
        (".self_attn.g_proj.weight", 0, False),
    ]
    shard_dim = None
    is_replicated = None
    is_weight_scale = False
    for param_name, dim, tp_sharded in pad_name_dim:
        if name.endswith(param_name):
            shard_dim = dim
            is_replicated = tp_sharded
            break
        elif name.endswith(param_name + "_scale_inv"):
            is_weight_scale = True
            break
    if shard_dim is None:
        return loaded_weight

    expected_full_size = (
        param.shape[shard_dim] if is_replicated else param.shape[shard_dim] * tp_size
    )
    actual_size = loaded_weight.shape[shard_dim]
    pad_size = expected_full_size - actual_size
    assert pad_size >= 0, (
        f"Pad {name} dim {shard_dim} from {actual_size} to "
        f"{expected_full_size} is negative"
    )
    if pad_size == 0:
        return loaded_weight
    if is_weight_scale:
        # do not support weight scale auto padding
        raise NotImplementedError(f"Padding for {name} is not implemented yet")
    padding = [0] * (2 * len(loaded_weight.shape))
    padding[-(2 * shard_dim + 1)] = pad_size
    loaded_weight = F.pad(loaded_weight, tuple(padding))
    logger.warning_once(
        "Pad %s dim %d from %d to %d",
        name,
        shard_dim,
        actual_size,
        expected_full_size,
    )

    return loaded_weight


def may_ar_or_rs(x: torch.Tensor, enable_sp: bool) -> torch.Tensor:
    """Maybe all-reduce or reduce-scatter to combine partial results.

    When sequence parallelism is enabled, the result is reduced and scattered
    across ranks along the sequence dim (dim=0) via reduce-scatter, so each
    rank keeps its own shard. Otherwise a regular all-reduce is performed and
    the full tensor is replicated on every rank.
    """
    if enable_sp:
        return tensor_model_parallel_reduce_scatter(x, dim=0)
    return tensor_model_parallel_all_reduce(x)


class KimiK3FlashMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
    ) -> None:
        super().__init__()
        self.reduce_results = reduce_results
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.down_proj",
        )
        assert hidden_act in ["silu", "situ", "situv2"], (
            f"unsupported hidden_act {hidden_act!r}, "
            f"expected one of 'silu', 'situ', 'situv2'."
        )
        self.hidden_act = hidden_act
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        elif hidden_act == "situ":
            # situglu; the config guarantees situ always has beta
            assert activation_situ_beta is not None
            self.act_fn = SituAndMul(beta=activation_situ_beta)
        elif hidden_act == "situv2":
            # situgluv2; the config guarantees situv2 always has beta/linear_beta
            assert activation_situ_beta is not None
            assert activation_situ_linear_beta is not None
            self.act_fn = SituAndMul(
                beta=activation_situ_beta, linear_beta=activation_situ_linear_beta
            )
        else:
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )

    def postprocess(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.reduce_results:
            hidden_states = may_ar_or_rs(hidden_states, False)
        return hidden_states

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        x = self.postprocess(x)
        return x


class KimiK3FlashMoE(nn.Module):
    def __init__(
        self,
        config: KimiK3FlashTextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = config.hidden_size
        # MoE configs always carry these two fields (non-MoE uses KimiK3FlashMLP)
        moe_intermediate_size = cast(int, config.moe_intermediate_size)
        num_experts = cast(int, config.num_experts)
        moe_renormalize = config.moe_renormalize
        self.use_latent_moe: bool = (
            getattr(config, "routed_expert_hidden_size", None) is not None
        )
        self.moe_hidden_size: int = (
            config.routed_expert_hidden_size
            if self.use_latent_moe
            else config.hidden_size
        )
        self.latent_moe_use_norm = config.latent_moe_use_norm
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_shared_experts = config.num_shared_experts

        assert config.hidden_act in ["silu", "situ", "situv2"], (
            f"unsupported hidden_act {config.hidden_act!r}, "
            f"expected one of 'silu', 'situ', 'situv2'."
        )
        self.hidden_act = config.hidden_act
        self.activation_situ_beta = (
            config.moonshot_activation_situ_beta
            if self.hidden_act in ["situ", "situv2"]
            else None
        )
        self.activation_situ_linear_beta = (
            config.moonshot_activation_situ_linear_beta
            if self.hidden_act == "situv2"
            else None
        )

        logger.warning_once("MoE renormalize is %s.", moe_renormalize)
        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        if config.use_auxfree_balance is not None:
            logger.warning_once(
                "use_auxfree_balance is deprecated and will be replaced by "
                "topk_method str if exists."
            )
            if config.use_auxfree_balance:
                self.gate.e_score_correction_bias = nn.Parameter(
                    torch.empty(num_experts)
                )
            else:
                self.gate.e_score_correction_bias = None
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(torch.empty(num_experts))
        else:
            self.gate.e_score_correction_bias = None

        self.experts = FusedMoEFactory(
            num_experts=num_experts,
            top_k=cast(int, config.num_experts_per_token),
            hidden_size=self.moe_hidden_size,
            intermediate_size=moe_intermediate_size,
            # NOTE: MoEActivation has no situv2; the situ kernel with
            # activation_situ_linear_beta set covers situv2 semantics.
            activation=("situ" if self.hidden_act == "situv2" else self.hidden_act),
            activation_situ_beta=self.activation_situ_beta,
            activation_situ_linear_beta=self.activation_situ_linear_beta,
            reduce_results=False,
            renormalize=moe_renormalize,
            quant_config=quant_config,
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.moe_router_activation_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            # NOTE: the routed_scaling_factor of the factory applies to
            # topk_weights; it is linearly equivalent to scaling the experts
            # output by hand. Do not apply both.
            routed_scaling_factor=config.routed_scaling_factor,
            router_logits_dtype=torch.float32,  # we use the fp32 gate
        )

        if self.use_latent_moe:
            # TODO: check if using ReplicatedLinear is better than
            # ColumnParallelLinear + all_gather
            self.routed_expert_down_proj: ReplicatedLinear | None = ReplicatedLinear(
                input_size=config.hidden_size,
                output_size=self.moe_hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            self.routed_expert_up_proj: ReplicatedLinear | None = ReplicatedLinear(
                input_size=self.moe_hidden_size,
                output_size=config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
            if self.latent_moe_use_norm:
                self.routed_expert_norm: RMSNorm | None = RMSNorm(
                    self.moe_hidden_size, eps=config.rms_norm_eps
                )

        else:
            self.routed_expert_down_proj = None
            self.routed_expert_up_proj = None
            self.routed_expert_norm = None

        if self.num_shared_experts is not None:
            intermediate_size = moe_intermediate_size * self.num_shared_experts
            self.shared_experts = KimiK3FlashMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                activation_situ_beta=self.activation_situ_beta,
                activation_situ_linear_beta=self.activation_situ_linear_beta,
            )

    def _compute_router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute router logits in fp32 (num_tokens, n_experts)."""
        return torch.mm(hidden_states, self.gate.weight.t(), out_dtype=torch.float32)

    def postprocess(
        self,
        final_hidden_states: torch.Tensor,
        shared_output: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_latent_moe:
            if self.tp_size > 1 and shared_output is not None:
                concat_tensor = torch.cat([final_hidden_states, shared_output], dim=-1)
                concat_tensor = may_ar_or_rs(concat_tensor, False)
                final_hidden_states, shared_output = concat_tensor.split(
                    [
                        final_hidden_states.shape[-1],
                        shared_output.shape[-1],
                    ],
                    dim=-1,
                )
            elif self.tp_size > 1:
                final_hidden_states = may_ar_or_rs(final_hidden_states, False)
            if self.latent_moe_use_norm:
                assert self.routed_expert_norm is not None
                final_hidden_states = self.routed_expert_norm(final_hidden_states)
            assert self.routed_expert_up_proj is not None
            final_hidden_states, _ = self.routed_expert_up_proj(final_hidden_states)
            if shared_output is not None:
                final_hidden_states = final_hidden_states + shared_output
        else:
            if shared_output is not None:
                final_hidden_states = final_hidden_states + shared_output
            final_hidden_states = may_ar_or_rs(final_hidden_states, False)

        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        # fp32 gate, router_logits: (num_tokens, n_experts)
        router_logits = self._compute_router_logits(hidden_states)
        shared_output = None
        if self.num_shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
        if self.use_latent_moe:
            assert self.routed_expert_down_proj is not None
            hidden_states, _ = self.routed_expert_down_proj(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = self.postprocess(final_hidden_states, shared_output)

        return final_hidden_states.view(num_tokens, hidden_size)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class KimiK3FlashMLAAttention(nn.Module):
    """Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and
    https://github.com/flashinfer-ai/flashinfer/pull/551).
    """

    def __init__(
        self,
        config: KimiK3FlashTextConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        reduce_results: bool = True,
        o_bias: bool = False,
    ) -> None:
        super().__init__()
        # NOTE: the LayerCommunicator breaks the allreduce + o_bias ordering
        # (ar+bias -> bias+ar); assert o_bias features away for now.
        assert not o_bias
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size > 0:
            num_padded_heads = (num_heads + tp_size - 1) // tp_size * tp_size
            logger.warning_once(
                "MLA num_heads %d is not divisible by tp_size %d, will pad to %d heads",
                num_heads,
                tp_size,
                num_padded_heads,
            )
            self.num_heads = num_padded_heads
        self.num_local_heads = self.num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )

        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rope_parameters: dict[str, Any] = {"rope_theta": rope_theta}
        if rope_scaling:
            rope_parameters.update(rope_scaling)
            # get_rope takes the rope type from the "rope_type" key.
            rope_parameters["rope_type"] = "deepseek_yarn"
        self.use_nope = config.mla_use_nope
        if config.mla_use_nope:
            self.rotary_emb = None
        else:
            # The checkpoint conversion (moonshot-k2-opensource) stores MLA
            # rope dims in GPT-J interleaved layout; pair the rope the same
            # way (freqs are identical, only the dim pairing differs).
            self.rotary_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                is_neox_style=False,
                rope_parameters=rope_parameters,
            )
        if rope_scaling:
            mscale_all_dim = rope_parameters.get("mscale_all_dim", False)
            scaling_factor = rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        assert config.mla_config is not None
        if config.mla_config.use_output_gate:
            assert config.mla_config.use_full_rank_gate
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                self.v_head_dim * self.num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
            g_proj=self.g_proj if (hasattr(self, "g_proj")) else None,
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)


class KimiK3FlashDSAIndexerKNorm(nn.Module):
    """Bias-full fp32 LayerNorm for the DSA indexer shared key.

    ``F.layer_norm(x.float(), (dim,), weight.float(), bias.float(),
    eps).type_as(x)`` -- fp32 compute with a cast back to the input dtype.
    weight/bias are stored in fp32 directly (numerically equivalent to
    casting per forward), keeping only the input/output dtype casts.
    Selected via ``dsa_config.index_k_norm_type == "layernorm"``; legacy
    checkpoints keep the RMSNorm path untouched.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.float(), self.normalized_shape, self.weight, self.bias, self.eps
        ).type_as(x)


class KimiK3FlashDSAIndexer(nn.Module):
    """DSA indexer wired to this tree's native upstream DSA facilities.

    Holds the wq/wk/wv indexer projections (same parameter names as the
    checkpoint conversion) and a native ``DeepseekV32IndexerCache``
    (AttentionLayerBase, registers itself in static_forward_context);
    forward quantizes the indexer query, folds the scales into the per-head
    weights and drives ``torch.ops.vllm.sparse_attn_indexer`` directly --
    the same pattern as the upstream deepseek_v2.Indexer. ``topk_tokens``
    is part of the MLAModules indexer contract (asserted by
    MultiHeadLatentAttention).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        hidden_size: int,
        index_q_input_size: int,
        index_head_dim: int,
        index_num_heads: int,
        index_topk: int,
        rms_norm_eps: float,
        index_rope_dim: int = 0,
        index_k_norm_type: str = "rmsnorm",
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.index_head_dim = index_head_dim
        self.index_num_heads = index_num_heads
        self.indexer_weight_scale = self.index_num_heads**-0.5
        # MLAModules contract: MultiHeadLatentAttention asserts the indexer
        # exposes topk_tokens.
        self.topk_tokens = index_topk
        # Leading dims of every indexer head that receive RoPE.  Weights are
        # pair-permuted at checkpoint conversion (interleaved -> half-split),
        # so serving applies neox-style rope with the main-attention
        # cos_sin_cache (YaRN scaling therefore carries over automatically).
        # 0 = pre-rope checkpoints; the forward path is then unchanged.
        if index_rope_dim < 0 or index_rope_dim % 2 != 0:
            raise ValueError(
                f"dsa_config.index_rope_dim must be even and >= 0, got {index_rope_dim}"
            )
        if index_rope_dim > index_head_dim:
            raise ValueError(
                f"dsa_config.index_rope_dim ({index_rope_dim}) must not "
                f"exceed index_head_dim ({index_head_dim})"
            )
        self.index_rope_dim = index_rope_dim
        self.wq_index = ReplicatedLinear(
            index_q_input_size,
            self.index_num_heads * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_index",
        )
        # The indexer query projection (wq_index) may be quantized in fp8
        # checkpoints, while the shared-key (wk_index) and per-head weights
        # (wv_index) projections are stored in bf16.
        self.wk_index = ReplicatedLinear(
            hidden_size,
            self.index_head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wk_index",
        )
        self.wv_index = ReplicatedLinear(
            hidden_size,
            self.index_num_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wv_index",
        )
        self.k_index_norm: nn.Module
        if index_k_norm_type == "layernorm":
            self.k_index_norm = KimiK3FlashDSAIndexerKNorm(
                self.index_head_dim, eps=rms_norm_eps
            )
        elif index_k_norm_type == "rmsnorm":
            self.k_index_norm = RMSNorm(self.index_head_dim, eps=rms_norm_eps)
        else:
            raise ValueError(
                f"dsa_config.index_k_norm_type must be 'rmsnorm' or "
                f"'layernorm', got {index_k_norm_type!r}"
            )

        # Index cache scales use ue8m0 (power-of-2 rounding at write), the
        # convention of upstream DeepSeek V3.2; the DeepGEMM scoring kernels
        # read the scales as full fp32, which is exact for power-of-2 values.
        self.scale_fmt = "ue8m0"
        self.quant_block_size = self.index_head_dim
        self.index_softmax_scale = self.index_head_dim**-0.5
        # DeepGEMM indexer kernels support at most 64 index heads; smaller
        # head counts are zero-padded to the next supported width.
        if self.index_num_heads <= 32:
            self.padded_index_num_heads = 32
        elif self.index_num_heads <= 64:
            self.padded_index_num_heads = 64
        else:
            raise ValueError(
                "Official DeepGEMM DSA indexer supports at most 64 index "
                f"heads, got {self.index_num_heads}."
            )
        # NOTE: fp8 naive index cache: value in fp8 plus one fp32
        # scale per quant_block_size elements (derived, never hardcoded).
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.index_head_dim
            + self.index_head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            # Always non-None at runtime (set by the DecoderLayer constructor)
            cache_config=cast(CacheConfig, cache_config),
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_total_seq_len = v32_indexer.get_max_prefill_buffer_size(vllm_config)
        self.topk_indices_buffer = topk_indices_buffer
        # Same derivation as the upstream SparseAttnIndexer
        # (sparse_attn_indexer.py): enable the allgather-k path when prefill
        # context parallelism is configured.
        self.use_pcp = vllm_config.parallel_config.prefill_context_parallel_size > 1
        # DCP wiring, same as the upstream SparseAttnIndexer: the op merges
        # each rank's local top-K into the global top-K only when these are
        # passed through (sparse_attn_indexer.py:_merge_dcp_topk_global).
        self._parallel_config = vllm_config.parallel_config
        self.dcp_world_size = self._parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self._cp_kv_cache_interleave_size: int | None = None

    @property
    def cp_kv_cache_interleave_size(self) -> int:
        """Same lazy resolution as upstream SparseAttnIndexer: the real value
        is finalized after block_size setup, ahead of warmup/serving."""
        if self._cp_kv_cache_interleave_size is None:
            value = self._parallel_config.cp_kv_cache_interleave_size
            if isinstance(get_forward_context().attn_metadata, dict):
                self._cp_kv_cache_interleave_size = value
            return value
        return self._cp_kv_cache_interleave_size

    def _apply_index_rope(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE on the leading index_rope_dim dims of each head.

        Slices the leading index_rope_dim//2 frequencies from the main
        attention's cos_sin_cache (matching training, which reuses the
        leading freqs of the main freqs_cis for the indexer), computes in
        fp32 and casts back. index_q: [T, H, D]; index_k: [T, D] (already
        k_index_norm-ed, matching the training order norm -> rope).

        Uses the neox/half-split pairing (dim i pairs with dim i+half),
        matching the pair-permuted checkpoint conversion.
        """
        # index_rope (./index_rope.py) is a single in-place kernel (replacing
        # the former handwritten torch composition of ~20 small kernels).
        if torch.compiler.is_compiling():
            # Raw Triton launches are not traceable: go through the
            # registered custom op when compiling.
            torch.ops.vllm.index_rope(
                index_q,
                index_k,
                positions.view(-1),
                rotary_emb.cos_sin_cache,
                self.index_rope_dim,
                False,  # interleave: neox/half-split
            )
        else:
            # Non-compiled path: call directly, saving custom-op dispatch.
            index_rope(
                index_q,
                index_k,
                positions.view(-1),
                rotary_emb.cos_sin_cache,
                self.index_rope_dim,
                False,  # interleave: neox/half-split
            )
        return index_q, index_k

    def _quantize_index_q_and_weights(
        self,
        index_q: torch.Tensor,
        index_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FP8-quantize the indexer query and fold scales into the weights.

        Same quant layout as the upstream Indexer.forward (ue8m0 per-group
        scales folded into fp32 weights with the softmax scale; index_v
        already carries the 1/sqrt(num_heads) factor from forward) and the
        same head padding policy (DeepGEMM supports 32/64 index heads).
        """
        q_flat = index_q.reshape(-1, self.index_head_dim).contiguous()
        q_fp8, q_scale = per_token_group_quant_fp8(
            q_flat,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=True,
        )
        q_fp8 = q_fp8.view(index_q.shape)
        q_scale = q_scale.view(index_q.shape[0], self.index_num_heads, -1)
        if q_scale.shape[-1] != 1:
            raise RuntimeError(
                "Kimi K3 Flash DSA indexer expects one Q scale per index head."
            )
        weights = (
            (index_weights.float().unsqueeze(-1) * q_scale * self.index_softmax_scale)
            .squeeze(-1)
            .contiguous()
        )
        if self.padded_index_num_heads != self.index_num_heads:
            padded_q = q_fp8.new_zeros(
                (index_q.shape[0], self.padded_index_num_heads, self.index_head_dim)
            )
            padded_weights = weights.new_zeros(
                (index_q.shape[0], self.padded_index_num_heads)
            )
            padded_q[:, : self.index_num_heads, :] = q_fp8
            padded_weights[:, : self.index_num_heads] = weights
            q_fp8 = padded_q
            weights = padded_weights
        return q_fp8, weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        index_q_input: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        rotary_emb: nn.Module | None = None,
    ) -> torch.Tensor:
        if index_q_input is None:
            # q_lora_rank is None: MultiHeadLatentAttention.forward_native
            # passes q_c=None, so the indexer query reads hidden_states.
            index_q_input = hidden_states
        index_q = self.wq_index(index_q_input)[0].view(
            -1, self.index_num_heads, self.index_head_dim
        )
        index_k = self.k_index_norm(self.wk_index(hidden_states)[0])
        if self.index_rope_dim > 0:
            assert positions is not None and rotary_emb is not None, (
                "index_rope_dim > 0 requires positions and the main "
                "attention rotary_emb (mla_use_nope models cannot enable "
                "indexer rope)"
            )
            index_q, index_k = self._apply_index_rope(
                index_q, index_k, positions, rotary_emb
            )
        index_v = self.wv_index(hidden_states)[0] * self.indexer_weight_scale
        q_fp8, weights = self._quantize_index_q_and_weights(index_q, index_v)
        # Writes the fp8 index-k cache and the top-k indices into the shared
        # model-level buffer (same call as SparseAttnIndexer.forward_cuda:
        # under a dummy run kv_cache is an empty placeholder and the op takes
        # the workspace-reservation path internally, so kv_cache[0] indexing
        # is not allowed here).
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_fp8,
            None,  # q_scale: the scale is folded into per-head weights
            index_k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.index_head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            False,  # skip_k_cache_insert
            self.use_pcp,
            self._parallel_config.pcp_shard_decode_requests,
            _encode_layer_name(""),  # dense_mha_metadata_layer_name
            False,  # use_fp4_cache
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
        )


def _kimi_k3_flash_dsa_build_indexer(
    config: "KimiK3FlashTextConfig", dsa_config: "KimiK3FlashDSAConfig", layer_id: int
) -> bool:
    """Whether this DSA layer is an "F" layer (builds its own indexer).

    Index-share F/S classification with the freq/offset periodic rule; the
    formula matches upstream deepseek_v2.py and deepseek_v32 attention.py
    verbatim. Layers at or beyond num_hidden_layers always build an indexer.

    "S" layers (returns False) do not build an indexer; they reuse the
    indices the previous "F" layer wrote into the model-level shared
    topk_indices_buffer for sparse attention.
    """
    _index_topk_freq = dsa_config.index_topk_freq
    _index_skip_topk_offset = dsa_config.index_skip_topk_offset
    _skip_topk = max(layer_id - _index_skip_topk_offset + 1, 0) % _index_topk_freq != 0
    # The skip pattern only governs backbone layers.
    _num_hidden_layers = getattr(config, "num_hidden_layers", None)
    _is_mtp_layer = _num_hidden_layers is not None and layer_id >= _num_hidden_layers
    return not _skip_topk or _is_mtp_layer


class KimiK3FlashDSAAttention(nn.Module):
    """Kimi K3 Flash DSA (DeepSeek Sparse Attention) inference path.

    Structurally identical to KimiK3FlashMLAAttention (same MLA projections,
    rope, output gate and forward around MultiHeadLatentAttention); the
    difference is the DSA indexer wired through
    ``MLAModules(indexer=..., indexer_rotary_emb=..., is_sparse=True,
    topk_indices_buffer=...)`` so the MLA layer computes top-k indices via
    the native ``sparse_attn_indexer`` custom op and the FlashMLA sparse
    backend consumes them. The MLA projection weights intentionally keep the
    same parameter names as the dense MLA path, so existing checkpoint
    conversion and TP loading semantics stay unchanged. DSA-specific indexer
    weights live under ``self_attn.dsa_indexer.*``.
    """

    def __init__(
        self,
        config: KimiK3FlashTextConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        reduce_results: bool = True,
        o_bias: bool = False,
        topk_indices_buffer: torch.Tensor | None = None,
        dense_draft: bool = False,
    ) -> None:
        super().__init__()
        assert not o_bias
        if config.dsa_config is None:
            raise ValueError("dsa_config is required for KimiK3FlashDSAAttention")
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size > 0:
            num_padded_heads = (num_heads + tp_size - 1) // tp_size * tp_size
            logger.warning_once(
                f"DSA num_heads {num_heads} is not divisible by tp_size "
                f"{tp_size}, will pad to {num_padded_heads} heads"
            )
            self.num_heads = num_padded_heads
        self.num_local_heads = self.num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # get_rope takes the rope parameters packed in a dict
        rope_parameters: dict[str, Any] = {"rope_theta": rope_theta}
        if rope_scaling:
            rope_parameters.update(rope_scaling)
            rope_parameters["rope_type"] = "deepseek_yarn"
        self.use_nope = config.mla_use_nope
        if config.mla_use_nope:
            self.rotary_emb = None
        else:
            # The checkpoint conversion (moonshot-k2-opensource) stores MLA
            # rope dims in GPT-J interleaved layout; pair the rope the same
            # way (freqs are identical, only the dim pairing differs).
            self.rotary_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                is_neox_style=False,
                rope_parameters=rope_parameters,
            )
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        assert config.mla_config is not None
        if config.mla_config.use_output_gate:
            assert config.mla_config.use_full_rank_gate
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                self.v_head_dim * self.num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )

        dsa_config = config.dsa_config
        self.index_topk = dsa_config.index_topk
        self.index_head_dim = dsa_config.index_head_dim
        self.index_num_heads = dsa_config.index_num_heads
        self.index_rope_dim = getattr(dsa_config, "index_rope_dim", 0) or 0
        if self.index_rope_dim > 0 and self.use_nope:
            raise ValueError(
                "dsa_config.index_rope_dim > 0 is incompatible with "
                "mla_use_nope (there is no rotary cache to share)"
            )
        if self.index_rope_dim > min(self.qk_rope_head_dim, self.index_head_dim):
            raise ValueError(
                f"dsa_config.index_rope_dim ({self.index_rope_dim}) must "
                f"not exceed min(qk_rope_head_dim={self.qk_rope_head_dim}, "
                f"index_head_dim={self.index_head_dim})"
            )

        vllm_config = get_current_vllm_config()
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        dcp_kg_slack = dcp_world_size - 1 if dcp_world_size > 1 else 0
        self.dsa_indexer: KimiK3FlashDSAIndexer | KimiK3FlashDenseDraftIndexer | None
        if dense_draft:
            # Dense (non-DSA) MTP draft on a DSA target: no indexer weights,
            # and no SM100 dense MLA kernel reads the globally enforced
            # fp8_ds_mla cache. Keep the sparse chain but substitute a fake
            # indexer that fills full-range candidates (mathematically exact
            # dense attention), with a max_model_len-wide buffer instead of
            # the index_topk-wide one.
            self.skip_topk = False
            self.dsa_indexer = KimiK3FlashDenseDraftIndexer(
                max_model_len=vllm_config.model_config.max_model_len,
                max_num_batched_tokens=(
                    vllm_config.scheduler_config.max_num_batched_tokens
                ),
                dcp_kg_slack=dcp_kg_slack,
                index_head_dim=self.index_head_dim,
                prefix=f"{prefix}.dsa_indexer",
                device=("cuda" if torch.cuda.is_available() else "cpu"),
            )
            self.topk_indices_buffer = self.dsa_indexer.topk_indices_buffer
        elif topk_indices_buffer is None:
            # Fallback for hosts that build KimiK3FlashDecoderLayer directly:
            # allocate a private buffer. The main model path always passes
            # the single model-level buffer created in KimiK3FlashModel.__init__.
            logger.warning_once(
                "KimiK3FlashDSAAttention did not receive a model-level "
                "topk_indices_buffer; allocating a per-layer fallback."
            )
            topk_buffer_device = "cuda" if torch.cuda.is_available() else "cpu"
            # Same guard as KimiK3FlashModel.__init__'s model-level buffer: in
            # allgather_k mode prefill rows are padded to ceil(P/W) per rank,
            # so the all-gathered output takes num_decode_tokens +
            # W*ceil(P/W) rows, at most W-1 more than num_actual_tokens.
            topk_indices_buffer = torch.empty(
                (
                    vllm_config.scheduler_config.max_num_batched_tokens + dcp_kg_slack,
                    self.index_topk,
                ),
                dtype=torch.int32,
                device=topk_buffer_device,
            )
            self.topk_indices_buffer = topk_indices_buffer
        else:
            self.topk_indices_buffer = topk_indices_buffer

        if not dense_draft:
            # Index-share F/S role classification (formula matches upstream
            # deepseek_v2.py / deepseek_v32 attention.py verbatim, see
            # _kimi_k3_flash_dsa_build_indexer). "S" layers build no indexer/index
            # cache spec and do not call the sparse_attn_indexer op in forward;
            # the main attention reads the indices written by the previous "F"
            # layer into the shared buffer as usual. With freq=1 (default) all
            # layers are "F".
            self.skip_topk = not _kimi_k3_flash_dsa_build_indexer(
                config, dsa_config, extract_layer_index(prefix)
            )
            # Surface the index-share config in the startup log
            # (freq/offset/this layer's role).
            if get_tensor_model_parallel_rank() == 0:
                logger.info_once(
                    "Kimi K3 Flash DSA index share: index_topk_freq=%s, "
                    "index_skip_topk_offset=%s; this layer "
                    "(layer_id=%d) role=%s",
                    dsa_config.index_topk_freq,
                    dsa_config.index_skip_topk_offset,
                    extract_layer_index(prefix),
                    "S(skip_topk)" if self.skip_topk else "F(with indexer)",
                )

        if dense_draft:
            pass  # indexer already substituted above
        elif not self.skip_topk:
            index_q_input_size = (
                self.q_lora_rank if self.q_lora_rank is not None else self.hidden_size
            )
            self.dsa_indexer = KimiK3FlashDSAIndexer(
                vllm_config=vllm_config,
                hidden_size=self.hidden_size,
                index_q_input_size=index_q_input_size,
                index_head_dim=self.index_head_dim,
                index_num_heads=self.index_num_heads,
                index_topk=self.index_topk,
                rms_norm_eps=config.rms_norm_eps,
                index_rope_dim=self.index_rope_dim,
                index_k_norm_type=getattr(dsa_config, "index_k_norm_type", "rmsnorm")
                or "rmsnorm",
                cache_config=cache_config,
                quant_config=quant_config,
                topk_indices_buffer=self.topk_indices_buffer,
                prefix=f"{prefix}.dsa_indexer",
            )
        else:
            # "S" layer: no indexer (same shape as upstream
            # self.indexer = None, deepseek_v2.py).
            self.dsa_indexer = None

        # The fp8_ds_mla normalization happens at config-verification time
        # (see the KimiK3Flash verifier in vllm/model_executor/models/config.py):
        # the runner resolves kv_cache_dtype from cache_config.cache_dtype
        # before the model is built; changing it inside the model would make
        # the spec dtype (bf16) inconsistent with the fp8_ds_mla page size.

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            g_proj=self.g_proj if (hasattr(self, "g_proj")) else None,
            indexer=self.dsa_indexer,
            indexer_rotary_emb=self.rotary_emb,
            is_sparse=True,
            topk_indices_buffer=self.topk_indices_buffer,
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
            skip_topk=self.skip_topk,
        )

        self.prefix = prefix
        self.debug_layer_idx = int(self.prefix.split(".")[-2])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        output = self.mla_attn(positions, hidden_states)
        return output


class KimiK3FlashLayerCommunicator:
    """Handles ops between attention and MLP layers.
    like communication, normalization, residual connections, etc.
    """

    def __init__(
        self,
        input_layernorm: RMSNorm,
        post_attention_layernorm: RMSNorm,
    ) -> None:
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        return hidden_states, residual

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states = may_ar_or_rs(hidden_states, False)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return hidden_states, residual


class KimiK3FlashDecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiK3FlashTextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.is_mla = config.is_mla
        self.is_moe = config.is_moe
        attn_quant_config = quant_config
        if (
            config.dsa_config is not None
            and layer_idx >= config.num_hidden_layers
            and not config.is_dsa_layer(layer_idx)
        ):
            # Dense (non-DSA) MTP draft of a DSA target model: it has no
            # indexer weights and no SM100 dense MLA kernel reads the
            # globally enforced fp8_ds_mla cache. Build the sparse form
            # with a DenseDraftIndexer (full-range fill == exact dense).
            assert config.mla_config is not None
            self.self_attn = KimiK3FlashDSAAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.mla_config.qk_nope_head_dim,
                qk_rope_head_dim=config.mla_config.qk_rope_head_dim,
                v_head_dim=config.mla_config.v_head_dim,
                q_lora_rank=config.mla_config.q_lora_rank,
                kv_lora_rank=config.mla_config.kv_lora_rank,
                rope_theta=getattr(config, "rope_theta", 10000),
                rope_scaling=getattr(config, "rope_scaling", None),
                max_position_embeddings=config.pretraining_sequence_length,
                quant_config=attn_quant_config,
                o_bias=config.attention_bias,
                reduce_results=False,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
                dense_draft=True,
            )
        elif config.is_dsa_layer(layer_idx):
            # DSA layers always use the MLA config
            assert config.mla_config is not None
            self.self_attn = KimiK3FlashDSAAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.mla_config.qk_nope_head_dim,
                qk_rope_head_dim=config.mla_config.qk_rope_head_dim,
                v_head_dim=config.mla_config.v_head_dim,
                q_lora_rank=config.mla_config.q_lora_rank,
                kv_lora_rank=config.mla_config.kv_lora_rank,
                rope_theta=getattr(config, "rope_theta", 10000),
                rope_scaling=getattr(config, "rope_scaling", None),
                max_position_embeddings=config.pretraining_sequence_length,
                quant_config=attn_quant_config,
                o_bias=config.attention_bias,
                reduce_results=False,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
            )
        elif self.is_mla:
            assert config.mla_config is not None
            self.self_attn = KimiK3FlashMLAAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.mla_config.qk_nope_head_dim,
                qk_rope_head_dim=config.mla_config.qk_rope_head_dim,
                v_head_dim=config.mla_config.v_head_dim,
                q_lora_rank=config.mla_config.q_lora_rank,
                kv_lora_rank=config.mla_config.kv_lora_rank,
                rope_theta=getattr(config, "rope_theta", 10000),
                rope_scaling=getattr(config, "rope_scaling", None),
                max_position_embeddings=config.pretraining_sequence_length,
                quant_config=attn_quant_config,
                o_bias=config.attention_bias,
                reduce_results=False,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(
                "Kimi K3 Flash requires an MLA config; the GQA attention path "
                "is not supported."
            )
        if (
            self.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.block_sparse_moe = KimiK3FlashMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = KimiK3FlashMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                activation_situ_beta=config.moonshot_activation_situ_beta,
                activation_situ_linear_beta=(
                    config.moonshot_activation_situ_linear_beta
                ),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_idx = layer_idx

        self.layer_communicator = KimiK3FlashLayerCommunicator(
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual
        )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class KimiK3FlashModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.config = config

        assert not getattr(parallel_config, "enable_sequence_parallelism", False), (
            "sequence parallelism is not supported by KimiK3FlashModel yet"
        )

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Single model-level top-k buffer shared by all DSA layers (upstream
        # DeepseekV2Model layout): each layer's indexer op writes it and the
        # same layer's sparse MLA attention reads it, in sequence.
        # [max_num_batched_tokens + dcp_kg_slack, index_topk] int32.
        self.topk_indices_buffer: torch.Tensor | None = None
        if config.dsa_config is not None and any(
            config.is_dsa_layer(layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ):
            topk_buffer_device = "cuda" if torch.cuda.is_available() else "cpu"
            # In allgather_k mode prefill rows are padded to ceil(P/W) per
            # rank, so the all-gathered output takes num_decode_tokens +
            # W*ceil(P/W) rows, at most W-1 more than num_actual_tokens.
            dcp_kg_slack = (
                parallel_config.decode_context_parallel_size - 1
                if parallel_config.decode_context_parallel_size > 1
                else 0
            )
            self.topk_indices_buffer = torch.empty(
                (
                    vllm_config.scheduler_config.max_num_batched_tokens + dcp_kg_slack,
                    config.dsa_config.index_topk,
                ),
                dtype=torch.int32,
                device=topk_buffer_device,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: KimiK3FlashDecoderLayer(
                config,
                cache_config,
                quant_config,
                prefix,
                topk_indices_buffer=self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        # "S" layers reuse the indices written by the previous "F" layer on
        # the same rank into the shared buffer. Two situations would leave
        # the first "S" layer reading a never-written (torch.empty) buffer:
        # 1) index_skip_topk_offset=0 (or an unlucky dsa_layers start
        #    parity) makes the very first DSA layer of the model an "S"
        #    layer, hit even with PP=1;
        # 2) a PP split point lands between an "F" layer and its "S"
        #    consumers, making the downstream rank's first local DSA layer
        #    an "S" layer.
        # Fail closed: every rank's first local DSA layer must be an "F"
        # layer.
        if config.dsa_config is not None and config.dsa_config.index_topk_freq > 1:
            first_dsa_layer = next(
                (
                    i
                    for i in range(self.start_layer, self.end_layer)
                    if config.is_dsa_layer(i)
                ),
                None,
            )
            if first_dsa_layer is not None:
                assert _kimi_k3_flash_dsa_build_indexer(
                    config, config.dsa_config, first_dsa_layer
                ), (
                    "invalid index share config: this rank's first local DSA "
                    f"layer (layer {first_dsa_layer}) is an 'S' layer with no "
                    "'F' layer before it to write the shared indices buffer "
                    "(index_skip_topk_offset=0 or a PP boundary lands "
                    "between 'F' and 'S'). Increase index_skip_topk_offset, "
                    "adjust the PP boundary, or set index_topk_freq=1."
                )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                key: torch.zeros(
                    (batch_size, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                )
                for key in ("hidden_states", "residual")
            }
        )

    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: do not use += (it would become an in-place aten.add_), so
        # the returned tensor does not keep aliasing the input buffer.
        return hidden_states + residual

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states = self.postprocess_layer(hidden_states, residual)
        return hidden_states


# DSA checkpoint weight rename mapping (the DSA indexer module
# names follow the checkpoint conversion). Specific rules must come
# before the generic ".self_attn.indexer." rule: the loop breaks on
# the first match, and the generic rule would otherwise shadow them.
# Shared by the main model and the MTP draft model.
KIMI_K3_FLASH_DSA_RENAME_PARAMS_MAPPING = [
    # Checkpoint-native indexer names (HF opensource layout)
    (".self_attn.dsa_indexer.wq_index.", ".self_attn.indexer.wq_b."),
    (".self_attn.dsa_indexer.wk_index.", ".self_attn.indexer.wk."),
    (".self_attn.dsa_indexer.wv_index.", ".self_attn.indexer.weights_proj."),
    (".self_attn.dsa_indexer.k_index_norm.", ".self_attn.indexer.k_norm."),
    # Converted-checkpoint names (indexer projections at self_attn level)
    (".self_attn.dsa_indexer.wq_index.", ".self_attn.wq_index."),
    (".self_attn.dsa_indexer.wk_index.", ".self_attn.wk_index."),
    (".self_attn.dsa_indexer.wv_index.", ".self_attn.wv_index."),
    (".self_attn.dsa_indexer.k_index_norm.", ".self_attn.k_index_norm."),
    # Generic fallback for already-nested dsa_indexer-style names
    (".self_attn.dsa_indexer.", ".self_attn.indexer."),
]


class KimiK3FlashForCausalLM(nn.Module, SupportsPP):
    # XXX: we dont support lora now, no need for modules below
    # `packed_modules_mapping`, `embedding_modules` and
    # `embedding_padding_modules`

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        model_config = vllm_config.model_config
        self.config = model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = KimiK3FlashModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        hidden_states = self.model.norm(hidden_states, None)
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # shard_id is int in the stacked mapping and str in the expert mapping
        stacked_params_mapping: list[tuple[str, str, Any]] = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        rename_params_mapping = KIMI_K3_FLASH_DSA_RENAME_PARAMS_MAPPING
        # TODO: use AutoWeightsLoader?
        if self.config.is_mla and self.config.mla_config.q_lora_rank is not None:
            stacked_params_mapping += [
                (".fused_qkv_a_proj", ".q_a_proj", 0),
                (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            ]

        if self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
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
        for args in weights:
            name, loaded_weight = args[:2]

            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors
                # in the checkpoint. Skip them.
                continue

            # The MoE submodule keeps the internal `block_sparse_moe`
            # attribute name (parameters live in that namespace), while
            # HF-style checkpoints name it `mlp`. Rename only the
            # MoE-owned subtrees; dense layers keep their real `mlp` module.
            if (
                ".mlp.experts." in name
                or ".mlp.gate." in name
                or ".mlp.shared_experts." in name
            ):
                name = name.replace(".mlp.", ".block_sparse_moe.", 1)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                # (".experts." with a leading dot does not match
                # ".shared_experts."; both mlp./block_sparse_moe. expert
                # namings are covered after the rename above.)
                if (".experts." in name) and name not in params_dict:
                    continue

                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                # if go with fusion option, then update name
                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                loaded_weight = maybe_pad_weight_with_param(name, loaded_weight, param)
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    # Mxfp4 checkpoints use .weight_packed: try both namings,
                    # with and without the _packed suffix.
                    if name not in params_dict and name.endswith("_packed"):
                        name = name.removesuffix("_packed")
                    if name not in params_dict and f"{name}_packed" in params_dict:
                        name = f"{name}_packed"

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=shard_id,
                    )
                    loaded_params.add(name)
                    break
                else:
                    # DSA checkpoint weight renaming. Must run before the
                    # bias skip below: the indexer k_norm.bias only exists
                    # under its checkpoint name at this point and would
                    # otherwise be dropped as an "extra bias".
                    for param_name, weight_name in rename_params_mapping:
                        if weight_name in name:
                            logger.warning_once(
                                "rename weight %s to param %s",
                                name,
                                name.replace(weight_name, param_name),
                            )
                            name = name.replace(weight_name, param_name)
                            break

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # Remapping the name of FP8 kv-scale.
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name

                    if is_pp_missing_parameter(name, self):
                        continue

                    if name not in params_dict:
                        # Log only once (warning_once dedupes per process and
                        # only on TP rank 0), avoiding repeated messages on
                        # multiple ranks / hot updates.
                        if get_tensor_model_parallel_rank() == 0:
                            logger.warning_once(
                                "ckpt weight %s has no matching model param, skipped",
                                name,
                            )
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

        return loaded_params


def get_spec_layer_idx_from_weight_name(
    config: KimiK3FlashTextConfig, weight_name: str
) -> int | None:
    if (
        hasattr(config, "num_nextn_predict_layers")
        and config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(
                f"model.layers.{layer_idx + i}."
            ) or weight_name.startswith(
                f"language_model.model.layers.{layer_idx + i}."
            ):
                return layer_idx + i
    return None
