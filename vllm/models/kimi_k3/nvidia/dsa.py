# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA (DeepSeek Sparse Attention) for the kimi_k3 stack.

The DSA indexer uses the native upstream facilities
(``DeepseekV32IndexerCache`` + ``torch.ops.vllm.sparse_attn_indexer``); the
sparse MLA layers go through ``MultiHeadLatentAttentionWrapper`` with
``MLAModules(is_sparse=True)``. Checkpoints name indexer weights
``self_attn.indexer.{wq_b,wk,weights_proj,k_norm}``; the module tree keeps
``self_attn.dsa_indexer.*`` and load_weights renames into it.
"""

import math
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_dcp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.models.kimi_k3.nvidia.index_rope import index_rope
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backends.mla import indexer as v32_indexer

logger = init_logger(__name__)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class KimiK3DSAIndexerKNorm(nn.Module):
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


class KimiK3DSAIndexer(nn.Module):
    """wq/wk/wv indexer projections + native ``DeepseekV32IndexerCache``.

    Forward quantizes the indexer query, folds the scales into the per-head
    weights and drives ``torch.ops.vllm.sparse_attn_indexer`` directly (the
    upstream deepseek_v2.Indexer pattern).
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
        use_merged_cache: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.index_head_dim = index_head_dim
        self.index_num_heads = index_num_heads
        self.indexer_weight_scale = self.index_num_heads**-0.5
        # MLAModules contract: MultiHeadLatentAttention asserts the indexer
        # exposes topk_tokens.
        self.topk_tokens = index_topk
        # Leading dims of every indexer head that receive RoPE. Weights are
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
            self.k_index_norm = KimiK3DSAIndexerKNorm(
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
        self.use_merged_cache = use_merged_cache
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.index_head_dim
            + self.index_head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cast(CacheConfig, cache_config),
        )
        if self.use_merged_cache:
            self.k_cache.merged_into_mla_cache = True
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


class KimiK3DSAAttention(nn.Module):
    """Kimi K3.1 DSA (DeepSeek Sparse Attention) inference path.

    Same MLA projections as the dense Kimi-K3 MLA path, plus the DSA indexer
    wired through ``MLAModules(is_sparse=True)`` so the FlashMLA sparse
    backend consumes the top-k indices.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        reduce_results: bool = True,
        topk_indices_buffer: torch.Tensor | None = None,
        dense_draft: bool = False,
    ) -> None:
        super().__init__()
        if config.dsa_config is None:
            raise ValueError("dsa_config is required for KimiK3DSAAttention")
        if current_platform.is_rocm():
            raise NotImplementedError(
                "Kimi-K3.1 DSA requires the CUDA FlashMLA sparse / DeepGEMM "
                "indexer kernels; ROCm is not supported."
            )
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
        self.max_position_embeddings = config.max_position_embeddings

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

        # Full-attention layers are NoPE; the rope path below only runs for
        # checkpoints whose MLA uses positional encoding (GPT-J interleaved
        # rope dims; freqs are identical, only the dim pairing differs).
        rope_parameters: dict = dict(getattr(config, "rope_parameters", None) or {})
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = 10000.0
        rope_scaling = (
            rope_parameters
            if rope_parameters.get("rope_type") not in (None, "default")
            else None
        )
        self.use_nope = bool(config.mla_use_nope)
        if self.use_nope:
            self.rotary_emb = None
        else:
            if rope_scaling:
                rope_parameters["rope_type"] = "deepseek_yarn"
            self.rotary_emb = get_rope(
                qk_rope_head_dim,
                max_position=self.max_position_embeddings,
                is_neox_style=False,
                rope_parameters=rope_parameters,
            )
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # The output gate is full-rank (v_head_dim * num_heads), same shape
        # as the dense Kimi-K3 MLA's g_proj.
        if config.mla_use_output_gate:
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
        self.index_rope_dim = dsa_config.index_rope_dim
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
        if topk_indices_buffer is None:
            # Fallback for hosts that build KimiDecoderLayer directly:
            # allocate a private buffer. The main model path always passes
            # the single model-level buffer created in KimiLinearModel.
            logger.warning_once(
                "KimiK3DSAAttention did not receive a model-level "
                "topk_indices_buffer; allocating a per-layer fallback."
            )
            topk_buffer_device = "cuda" if torch.cuda.is_available() else "cpu"
            # Same guard as KimiLinearModel's model-level buffer: in
            # allgather_k mode prefill rows are padded to ceil(P/W) per rank,
            # so the all-gathered output takes num_decode_tokens +
            # W*ceil(P/W) rows, at most W-1 more than num_actual_tokens.
            dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
            dcp_kg_slack = dcp_world_size - 1 if dcp_world_size > 1 else 0
            topk_indices_buffer = torch.empty(
                (
                    vllm_config.scheduler_config.max_num_batched_tokens + dcp_kg_slack,
                    self.index_topk,
                ),
                dtype=torch.int32,
                device=topk_buffer_device,
            )
        self.topk_indices_buffer = topk_indices_buffer

        # Kimi-K3.1 has no index-share mechanism: every DSA layer builds
        # its own indexer; fail loudly on configs that ask for it.
        if getattr(dsa_config, "index_topk_freq", 1) not in (None, 1):
            raise NotImplementedError(
                "Kimi-K3.1 DSA does not support index_topk_freq != 1 "
                f"(index share), got {dsa_config.index_topk_freq}."
            )

        # KDA-hybrid models merge the indexer cache into the MLA layer's
        # merged per-layer cache (MergedMLAIndexerSpec); non-hybrid DSA keeps the
        # stock standalone indexer cache.
        use_merged_cache = getattr(config, "linear_attn_config", None) is not None

        if dense_draft:
            # Dense (non-DSA) MTP draft on a DSA target: no indexer weights,
            # and no SM100 dense MLA kernel reads the enforced fp8_ds_mla
            # cache. Substitute a fake indexer that fills full-range
            # candidates (mathematically exact dense attention); its k_cache
            # facade keeps the merged-cache spec registration byte-identical
            # to the target's (the indexer section is simply never touched).
            from vllm.models.kimi_k3.nvidia.dense_draft import (
                KimiK3DenseDraftIndexer,
            )

            dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
            self.dsa_indexer = KimiK3DenseDraftIndexer(
                max_model_len=vllm_config.model_config.max_model_len,
                max_num_batched_tokens=(
                    vllm_config.scheduler_config.max_num_batched_tokens
                ),
                dcp_kg_slack=(dcp_world_size - 1 if dcp_world_size > 1 else 0),
                index_head_dim=self.index_head_dim,
                prefix=f"{prefix}.dsa_indexer",
                device=("cuda" if torch.cuda.is_available() else "cpu"),
                use_merged_cache=use_merged_cache,
            )
            self.topk_indices_buffer = self.dsa_indexer.topk_indices_buffer
        else:
            index_q_input_size = (
                self.q_lora_rank if self.q_lora_rank is not None else self.hidden_size
            )
            self.dsa_indexer = KimiK3DSAIndexer(
                vllm_config=vllm_config,
                hidden_size=self.hidden_size,
                index_q_input_size=index_q_input_size,
                index_head_dim=self.index_head_dim,
                index_num_heads=self.index_num_heads,
                index_topk=self.index_topk,
                rms_norm_eps=config.rms_norm_eps,
                index_rope_dim=self.index_rope_dim,
                index_k_norm_type=dsa_config.index_k_norm_type,
                cache_config=cache_config,
                quant_config=quant_config,
                topk_indices_buffer=self.topk_indices_buffer,
                use_merged_cache=use_merged_cache,
                prefix=f"{prefix}.dsa_indexer",
            )

        # The fp8_ds_mla normalization happens at config-verification time
        # (see the KimiK3 verifier in vllm/model_executor/models/config.py):
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
        )

        # Merged cache mode (MergedMLAIndexerSpec): the registered MLA layer is
        # the wrapper's inner mla_attn; mark it with the indexer geometry so
        # its get_kv_cache_spec returns the merged spec and the runner binds
        # the indexer's section view of the merged cache to it.
        if self.dsa_indexer.use_merged_cache:
            mla_layer = cast(Any, self.mla_attn.mla_attn)
            mla_layer.ds_index_head_size = self.dsa_indexer.k_cache.head_dim
            mla_layer.ds_index_head_dim = self.index_head_dim
            mla_layer.ds_indexer_layer_name = self.dsa_indexer.k_cache.prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)
