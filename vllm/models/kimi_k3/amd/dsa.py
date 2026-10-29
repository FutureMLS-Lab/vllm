# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA (DeepSeek Sparse Attention) for the kimi_k3 stack on ROCm.

Ports ``nvidia/dsa.py``: the indexer runs through
``SparseAttnIndexer.forward_hip`` (the DeepSeek V3.2 ROCm pattern) and the
sparse MLA layers are served by ROCM_AITER_MLA_SPARSE over the fp8_ds_mla
cache. The DeepGEMM-only 32/64 index-head zero padding is dropped: the ROCm
indexer kernels take the native head count.
"""

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
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
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.models.kimi_k3.common.dsa_cache import KimiK3IndexerCache
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.attention.backends.mla import indexer as v32_indexer

logger = init_logger(__name__)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class KimiK3DSAIndexerKNorm(nn.Module):
    """Bias-full fp32 LayerNorm for the DSA indexer shared key.

    weight/bias are stored in fp32 directly (numerically equivalent to
    casting per forward), keeping only the input/output dtype casts.
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
    """wq/wk/wv indexer projections + ``SparseAttnIndexer`` (ROCm dispatch).

    Forward quantizes the indexer query, folds the scales into the per-head
    weights and drives ``torch.ops.vllm.rocm_aiter_sparse_attn_indexer`` via
    ``SparseAttnIndexer.forward_hip`` with ``cache_layout="NORMAL"``: the
    indexer pages are block-flat, which the Triton insert wrapper cannot infer
    from the cache shape alone.
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
        # convention of upstream DeepSeek V3.2; the scoring kernels read the
        # scales as full fp32, which is exact for power-of-2 values.
        self.scale_fmt = "ue8m0"
        self.quant_block_size = self.index_head_dim
        self.index_softmax_scale = self.index_head_dim**-0.5
        # fp8 naive index cache: fp8 value bytes plus one fp32 scale per head
        # (quant_block_size == index_head_dim by contract), hence +4 bytes.
        self.k_cache = KimiK3IndexerCache(
            head_dim=self.index_head_dim + 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cast(CacheConfig, cache_config),
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_total_seq_len = v32_indexer.get_max_prefill_buffer_size(vllm_config)
        self.topk_indices_buffer = topk_indices_buffer
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.index_head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            topk_indices_buffer,
            cache_layout="NORMAL",
        )

    def _quantize_index_q_and_weights(
        self,
        index_q: torch.Tensor,
        index_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FP8-quantize the indexer query and fold scales into the weights.

        Same quant layout as the upstream Indexer.forward (ue8m0 per-group
        scales folded into fp32 weights with the softmax scale; index_v
        already carries the 1/sqrt(num_heads) factor from forward).
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
        index_v = self.wv_index(hidden_states)[0] * self.indexer_weight_scale
        q_fp8, weights = self._quantize_index_q_and_weights(index_q, index_v)
        # Under a dummy run kv_cache is an empty placeholder and the op takes
        # its workspace-reservation path.
        return self.indexer_op.forward_hip(hidden_states, q_fp8, index_k, weights)


class KimiK3DSAAttention(nn.Module):
    """Kimi K3.1 DSA (DeepSeek Sparse Attention) inference path on ROCm.

    Same MLA projections as the dense Kimi-K3 MLA path, plus the DSA indexer
    wired through ``MLAModules(is_sparse=True)`` so the ROCM_AITER_MLA_SPARSE
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
    ) -> None:
        super().__init__()
        if config.dsa_config is None:
            raise ValueError("dsa_config is required for KimiK3DSAAttention")
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Kimi-K3.1 DSA on ROCm does not support prefill/decode "
                "context parallelism yet."
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

        # NoPE-only: the production caller asserts mla_use_nope before
        # building this class; re-assert for direct construction.
        if not config.mla_use_nope:
            raise NotImplementedError(
                "Kimi-K3.1 DSA on ROCm is NoPE-only (requires mla_use_nope)"
            )
        self.rotary_emb = None
        # The YaRN mscale adjustment also runs under NoPE (it rescales the
        # attention softmax, independent of rotary application); nvidia/dsa.py
        # applies it identically.
        rope_parameters: dict = dict(getattr(config, "rope_parameters", None) or {})
        rope_scaling = (
            rope_parameters
            if rope_parameters.get("rope_type") not in (None, "default")
            else None
        )
        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

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
        if dsa_config.index_rope_dim != 0:
            raise NotImplementedError(
                "Kimi-K3.1 DSA on ROCm requires index_rope_dim == 0 "
                f"(got {dsa_config.index_rope_dim}); the index_rope path is not "
                "ported yet."
            )

        if topk_indices_buffer is None:
            raise ValueError(
                "KimiK3DSAAttention requires the model-level "
                "topk_indices_buffer created by KimiLinearModel"
            )
        self.topk_indices_buffer = topk_indices_buffer

        # Kimi-K3.1 has no index-share mechanism: every DSA layer builds
        # its own indexer; fail loudly on configs that ask for it.
        if getattr(dsa_config, "index_topk_freq", 1) not in (None, 1):
            raise NotImplementedError(
                "Kimi-K3.1 DSA does not support index_topk_freq != 1 "
                f"(index share), got {dsa_config.index_topk_freq}."
            )

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
            index_k_norm_type=dsa_config.index_k_norm_type,
            cache_config=cache_config,
            quant_config=quant_config,
            topk_indices_buffer=self.topk_indices_buffer,
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
        # The plain wrapper, not the AMD KimiK3MultiHeadLatentAttentionWrapper:
        # its fused decode Q-prep path writes dense 576-wide rows, which the
        # 656-byte fp8_ds_mla records cannot take.
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
