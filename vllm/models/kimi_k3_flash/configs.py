# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 Flash (kimi_k3_flash) model configurations.

`KimiK3FlashConfig` is the composite multimodal config (text + vision towers);
`KimiK3FlashTextConfig` is the text backbone config with MLA and DSA (DeepSeek
Sparse Attention) sub-structures.
"""

import inspect
from dataclasses import asdict, dataclass
from typing import Any

from transformers.configuration_utils import PretrainedConfig

from vllm.logger import init_logger
from vllm.transformers_utils.configs.kimi_k25 import KimiK25VisionConfig

logger = init_logger(__name__)


@dataclass
class KimiK3FlashMLAConfig:
    """Configuration class for MLA (Multi-head Latent Attention)."""

    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    q_lora_rank: int | None
    kv_lora_rank: int
    use_nope: bool = False
    use_output_gate: bool = False
    use_full_rank_gate: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "KimiK3FlashMLAConfig":
        allowed_keys = inspect.signature(cls).parameters
        for key in config_dict:
            if key not in allowed_keys:
                logger.warning("%s is set but does not support. Ignore it.", key)
        return cls(**{k: v for k, v in config_dict.items() if k in allowed_keys})


@dataclass
class KimiK3FlashDSAConfig:
    """Configuration class for DSA (DeepSeek Sparse Attention)."""

    dsa_layers: list[int]
    index_topk: int = 2048
    index_head_dim: int = 64
    index_num_heads: int = 16
    # Leading dims of every indexer head that receive RoPE (neox/half-split
    # after checkpoint conversion; shares the main-attention rotary cache,
    # YaRN included). 0 = no indexer RoPE (all pre-rope checkpoints).
    index_rope_dim: int = 0
    # Norm on the indexer shared key: "rmsnorm" (legacy checkpoints) or
    # "layernorm" (bias-full fp32 LayerNorm matching official DeepSeek V3.2;
    # the converter detects it from the presence of k_index_norm.bias).
    index_k_norm_type: str = "rmsnorm"
    indexer_use_fp8: bool = False
    # Index share (index cache): every index_topk_freq-th DSA layer builds its
    # own indexer ("F" layer) and writes the model-wide shared
    # topk_indices_buffer; layers in between ("S" layers) reuse it, skipping
    # indexer QK scoring and top-k. Semantics follow upstream vLLM's
    # index_topk_freq / index_skip_topk_offset; the F/S classification formula
    # matches upstream deepseek_v2.py / deepseek_v32 attention.py, and the
    # consumption lives in model.py (_kimi_k3_flash_dsa_build_indexer and
    # KimiK3FlashDSAAttention). freq=1 makes every layer an F layer.
    index_topk_freq: int = 1
    # The first index_skip_topk_offset DSA layers are forced Full (own indexer).
    index_skip_topk_offset: int = 1

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "KimiK3FlashDSAConfig":
        config_dict = dict(config_dict)
        if "dsa_layers" not in config_dict and "layers" in config_dict:
            config_dict["dsa_layers"] = config_dict["layers"]
        config_dict.pop("layers", None)
        allowed_keys = inspect.signature(cls).parameters
        for key in config_dict:
            if key not in allowed_keys:
                logger.warning("%s is set but does not support. Ignore it.", key)
        cfg = cls(**{k: v for k, v in config_dict.items() if k in allowed_keys})
        if cfg.index_topk_freq < 1:
            raise ValueError(
                f"index_topk_freq must be >= 1, got {cfg.index_topk_freq}; "
                "0 would ZeroDivisionError in the F/S modulo check"
            )
        return cfg


class KimiK3FlashTextConfig(PretrainedConfig):
    r"""Configuration class for the Kimi K3 Flash text backbone.

    The MLA parameters live in the `mla_config` sub-structure and the DSA
    parameters in `dsa_config`; both are parsed into dataclasses.
    """

    model_type = "kimi_k3_flash_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type="kimi_k3_flash_text",
        vocab_size=163840,
        hidden_size=4096,
        head_dim=None,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        pretraining_sequence_length=4096,
        max_position_embeddings=None,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        q_proj_bias: bool = False,
        k_proj_bias: bool = False,
        v_proj_bias: bool = False,
        tie_word_embeddings=False,
        # moe config
        moe_intermediate_size: int | None = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        routed_expert_hidden_size=None,
        latent_moe_use_norm: bool = False,
        num_experts: int | None = None,
        num_experts_per_token: int | None = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        topk_method: str = "noaux_tc",
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        use_auxfree_balance: bool | None = None,
        # mla config
        mla_config: dict | None = None,
        dsa_config: dict | None = None,
        dsa_layers: list[int] | None = None,
        index_topk: int | None = None,
        index_head_dim: int | None = None,
        index_num_heads: int | None = None,
        indexer_use_fp8: bool | None = None,
        moonshot_activation_situ_beta: float | None = None,
        moonshot_activation_situ_linear_beta: float | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.pretraining_sequence_length = pretraining_sequence_length
        # transformers v5 rope standardization (PretrainedConfig.__post_init__)
        # reads self.max_position_embeddings; fall back to the pretraining
        # sequence length when the checkpoint does not set it.
        self.max_position_embeddings = (
            max_position_embeddings
            if max_position_embeddings is not None
            else pretraining_sequence_length
        )
        self.hidden_size = hidden_size
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.q_proj_bias = q_proj_bias
        self.k_proj_bias = k_proj_bias
        self.v_proj_bias = v_proj_bias
        if isinstance(mla_config, dict):
            self.mla_config: KimiK3FlashMLAConfig | None = (
                KimiK3FlashMLAConfig.from_dict(mla_config)
            )
        else:
            self.mla_config = mla_config
        if dsa_config is None and dsa_layers is not None:
            dsa_config = {
                "dsa_layers": dsa_layers,
            }
            if index_topk is not None:
                dsa_config["index_topk"] = index_topk
            if index_head_dim is not None:
                dsa_config["index_head_dim"] = index_head_dim
            if index_num_heads is not None:
                dsa_config["index_num_heads"] = index_num_heads
            if indexer_use_fp8 is not None:
                dsa_config["indexer_use_fp8"] = indexer_use_fp8
        if isinstance(dsa_config, dict):
            self.dsa_config: KimiK3FlashDSAConfig | None = (
                KimiK3FlashDSAConfig.from_dict(dsa_config)
            )
        else:
            self.dsa_config = dsa_config
        # FlashMLASparseMetadataBuilder reads top-level hf_config.index_topk
        # like upstream DeepSeek; expose it at the config level instead of
        # injecting it implicitly at model construction time.
        if self.dsa_config is not None:
            self.index_topk = self.dsa_config.index_topk
        # moe config
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_renormalize = moe_renormalize
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_router_activation_func = moe_router_activation_func
        assert self.moe_router_activation_func in ("softmax", "sigmoid")
        self.moe_intermediate_size = moe_intermediate_size
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.latent_moe_use_norm = latent_moe_use_norm
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.use_auxfree_balance = use_auxfree_balance
        self.topk_method = topk_method
        assert self.hidden_act in ["silu", "situ", "situv2"], (
            f"unsupported hidden_act {self.hidden_act!r}, "
            f"expected one of 'silu', 'situ', 'situv2'."
        )
        if self.hidden_act in ["situ", "situv2"]:
            assert moonshot_activation_situ_beta is not None, (
                "moonshot_activation_situ_beta must be set"
                " when hidden_act is 'situ' or 'situv2'."
            )
        if self.hidden_act == "situv2":
            assert moonshot_activation_situ_linear_beta is not None, (
                "moonshot_activation_situ_linear_beta must be set"
                " when hidden_act is 'situv2'."
            )
        self.moonshot_activation_situ_beta = moonshot_activation_situ_beta
        self.moonshot_activation_situ_linear_beta = moonshot_activation_situ_linear_beta
        if self.mla_config is None:
            self.mla_use_nope: bool = False
        else:
            self.mla_use_nope = getattr(self.mla_config, "use_nope", False)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self):
        return self.mla_config is not None

    # NOTE: some upstream code (e.g. the MLA prefill backend selector) reads
    # qk_nope_head_dim etc. from the top level; pass the mla_config fields
    # through here to avoid patching each call site. Raises AttributeError
    # when mla_config is None so getattr(config, name, default) keeps working.
    def _get_mla_attr(self, name: str):
        if self.mla_config is None:
            raise AttributeError(name)
        return getattr(self.mla_config, name)

    @property
    def q_lora_rank(self):
        return self._get_mla_attr("q_lora_rank")

    @property
    def kv_lora_rank(self):
        return self._get_mla_attr("kv_lora_rank")

    @property
    def qk_nope_head_dim(self):
        return self._get_mla_attr("qk_nope_head_dim")

    @property
    def qk_rope_head_dim(self):
        return self._get_mla_attr("qk_rope_head_dim")

    @property
    def v_head_dim(self):
        return self._get_mla_attr("v_head_dim")

    @property
    def is_moe(self):
        return self.num_experts is not None

    def is_dsa_layer(self, layer_idx: int):
        return (
            self.dsa_config is not None
            and (layer_idx + 1) in self.dsa_config.dsa_layers
        )

    def _rope_scaling_validation(self):
        """Validate the `rope_scaling` configuration."""
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict):
            raise ValueError(
                f"`rope_scaling` must be a dictionary, got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in [
            "linear",
            "dynamic",
            "yarn",
            "deepseekyarn",
        ]:
            raise ValueError(
                "`rope_scaling` must have key 'type', and `rope_scaling`'s name "
                "field must be one of ['linear', 'dynamic', 'yarn', "
                f"'deepseekyarn'], got {rope_scaling_type}"
            )
        if isinstance(rope_scaling_factor, int):
            rope_scaling_factor = float(rope_scaling_factor)
            self.rope_scaling["factor"] = rope_scaling_factor

        if (
            rope_scaling_factor is None
            or not isinstance(rope_scaling_factor, float)
            or rope_scaling_factor <= 1.0
        ):
            raise ValueError(
                "`rope_scaling`'s factor field must be an float > 1, "
                f"got {rope_scaling_factor}"
            )

    @classmethod
    def from_dict(
        cls, config_dict: dict[str, Any], **kwargs
    ) -> "KimiK3FlashTextConfig":
        result = super().from_dict(config_dict, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        if isinstance(config.mla_config, dict):
            config.mla_config = KimiK3FlashMLAConfig.from_dict(config.mla_config)
        if isinstance(config.dsa_config, dict):
            config.dsa_config = KimiK3FlashDSAConfig.from_dict(config.dsa_config)
        return result

    def to_dict(self) -> dict[str, Any]:
        ret = super().to_dict()
        if isinstance(ret["mla_config"], KimiK3FlashMLAConfig):
            ret["mla_config"] = asdict(ret["mla_config"])
        if isinstance(ret.get("dsa_config"), KimiK3FlashDSAConfig):
            ret["dsa_config"] = asdict(ret["dsa_config"])
        return ret


class KimiK3FlashConfig(PretrainedConfig):
    """Kimi K3 Flash multimodal model configuration.

    Args:
        text_config (dict | KimiK3FlashTextConfig): Configuration for the text model.
        vision_config (dict | KimiK25VisionConfig): Configuration for the
            vision tower and projector (same MoonViT3d family as Kimi-K2.5).
        ignore_index (int): The ignore index for the loss function.
        media_placeholder_token_id (int): The token ID used for media placeholders.
        image_placeholder (str): The per-image placeholder carried in the
            prompt, expanded into a resolution-aware media block by the
            model's ``_get_prompt_updates``.
        pad_token_id: The token ID to use for padding.

    """

    model_type = "kimi_k3_flash"

    def __init__(
        self,
        text_config: dict | KimiK3FlashTextConfig | None = None,
        vision_config: dict | KimiK25VisionConfig | None = None,
        projector_hidden_act: str | None = "gelu",
        projector_ln_eps: float = 1e-5,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 151664,
        image_placeholder: str = "<|kimi_image_placeholder|>",
        pad_token_id: int = 0,
        **kwargs,
    ):
        if text_config is None:
            text_config = KimiK3FlashTextConfig()
        elif isinstance(text_config, dict):
            text_config = KimiK3FlashTextConfig(**text_config)
        if vision_config is None:
            vision_config = KimiK25VisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = KimiK25VisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config

        # Set the projector output size to the text hidden size if not
        # explicitly set.
        if self.vision_config.mm_hidden_size == self.vision_config.hidden_size:
            self.vision_config.mm_hidden_size = self.text_config.hidden_size

        # FlashMLASparseMetadataBuilder reads top-level hf_config.index_topk
        # like upstream DeepSeek; expose it at the config level instead of
        # injecting it implicitly at model construction time.
        dsa_config = getattr(self.text_config, "dsa_config", None)
        if dsa_config is not None:
            self.index_topk = dsa_config.index_topk
        # used for the projector head
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.image_placeholder = image_placeholder

        # Propagate quantization config from the text model.
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)
