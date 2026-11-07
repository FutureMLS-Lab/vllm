# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A hybrid scheduler (an attention group, align-mode Mamba groups and an
indexer ring) and a step driver for CPU tests of hybrid state checkpoints."""

from typing import Any, Literal

import pytest
import torch

from tests.v1.kv_connector.unit.utils import MockKVConnector
from vllm.config import (
    CacheConfig,
    KVEventsConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.kv_transfer import KVRole
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.hybrid_checkpoint.frontier import FrontierCheckpoint
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

EOS = 2


class HybridMockKVConnector(MockKVConnector, SupportsHMA):
    """``MockKVConnector`` (a fixed number of matched tokens, loads nothing)
    on several KV cache groups."""

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None


class FrontierRecordingKVConnector(HybridMockKVConnector):
    """``HybridMockKVConnector`` that, with ``stores_frontiers``, records each
    finished request's frontier checkpoint and the request's block ids then."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.frontiers: list[tuple[str, FrontierCheckpoint, tuple[list[int], ...]]] = []

    @property
    def stores_frontier_checkpoints(self) -> bool:
        extra_config = self._kv_transfer_config.kv_connector_extra_config
        return bool(extra_config.get("stores_frontiers"))

    def register_finished_frontier(
        self, request: Request, frontier: FrontierCheckpoint
    ) -> None:
        block_ids = self._kv_cache_manager.get_block_ids(request.request_id)
        self.frontiers.append((request.request_id, frontier, block_ids))


class FrontierLoadKVConnector(HybridMockKVConnector):
    """A connector whose store holds a frontier checkpoint of
    ``frontier_loads[request_id]`` tokens for some requests: it matches up to
    that length and reports the load as a frontier load, recording where the
    scheduler moves its start and the tokens it then loads."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        extra_config = self._kv_transfer_config.kv_connector_extra_config
        self.lengths: dict[str, int] = extra_config["frontier_loads"]
        self.defers: dict[str, int] = dict(extra_config.get("defer_lookups", {}))
        self.moved: list[tuple[str, int]] = []
        self.loaded: dict[str, int] = {}
        self.deferred: list[tuple[str, int]] = []

    def defer_lookup(self, request: Request, num_computed_tokens: int) -> bool:
        self.deferred.append((request.request_id, num_computed_tokens))
        left = self.defers.get(request.request_id, 0)
        self.defers[request.request_id] = left - 1
        return left > 0

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        length = self.lengths.get(request.request_id, 0)
        return max(0, length - num_computed_tokens), False

    def frontier_load_length(self, request: Request) -> int | None:
        return self.lengths.get(request.request_id)

    def move_frontier_load_start(self, request: Request, num_tokens: int) -> None:
        self.moved.append((request.request_id, num_tokens))

    def update_state_after_alloc(
        self, request: Request, blocks: Any, num_external_tokens: int
    ) -> None:
        if num_external_tokens:
            self.loaded[request.request_id] = num_external_tokens


KVConnectorFactory.register_connector(
    "HybridMockKVConnector", __name__, HybridMockKVConnector.__name__
)
KVConnectorFactory.register_connector(
    "FrontierLoadKVConnector", __name__, FrontierLoadKVConnector.__name__
)
KVConnectorFactory.register_connector(
    "FrontierRecordingKVConnector", __name__, FrontierRecordingKVConnector.__name__
)


def mock_connector(
    matched_tokens: int,
    kv_role: KVRole = "kv_both",
    kv_load_failure_policy: Literal["recompute", "fail"] = "recompute",
) -> KVTransferConfig:
    """A connector that matches ``matched_tokens`` more tokens of every request
    and loads them synchronously."""
    return KVTransferConfig(
        kv_connector="HybridMockKVConnector",
        kv_role=kv_role,
        kv_load_failure_policy=kv_load_failure_policy,
        kv_connector_extra_config={"matched_tokens": matched_tokens, "is_async": False},
    )


def frontier_load_connector(
    frontier_loads: dict[str, int], defer_lookups: dict[str, int] | None = None
) -> KVTransferConfig:
    """A connector holding frontier checkpoints of the given lengths, which
    defers the lookups of some requests for the given number of steps."""
    return KVTransferConfig(
        kv_connector="FrontierLoadKVConnector",
        kv_role="kv_both",
        kv_load_failure_policy="recompute",
        kv_connector_extra_config={
            "matched_tokens": 0,
            "is_async": False,
            "frontier_loads": frontier_loads,
            "defer_lookups": defer_lookups or {},
        },
    )


def frontier_recording_connector(stores_frontiers: bool) -> KVTransferConfig:
    """A connector that matches nothing and, with ``stores_frontiers``,
    records the frontier checkpoints the scheduler hands it."""
    return KVTransferConfig(
        kv_connector="FrontierRecordingKVConnector",
        kv_role="kv_both",
        kv_load_failure_policy="recompute",
        kv_connector_extra_config={
            "matched_tokens": 0,
            "is_async": False,
            "stores_frontiers": stores_frontiers,
        },
    )


def make_hybrid_scheduler(
    block_size: int = 64,
    num_blocks: int = 200,
    max_num_batched_tokens: int = 1024,
    long_prefill_token_threshold: int = 0,
    checkpoint_alignment: int | None = 16,
    ring: int = 4,
    resume_catchup: str | None = "decode",
    hash_block_size: int | None = None,
    kv_transfer_config: KVTransferConfig | None = None,
    speculative_config: SpeculativeConfig | None = None,
    num_speculative_blocks: int = 0,
    ring_compression_ratio: int | None = None,
    async_scheduling: bool = False,
    compact_layout: bool = False,
    kv_events_config: KVEventsConfig | None = None,
    **cache_kwargs: Any,
) -> Scheduler:
    """``cache_kwargs`` are CacheConfig fields; ``ring`` 0 builds no ring.
    ``num_speculative_blocks``: the Mamba groups' state columns per draft
    position (0: one column). ``compact_layout`` sets
    ``mamba_spec_state_layout`` "compact" after the config's validation, which
    accepts it only on a supporting model and with no KV connector but
    MooncakeStoreConnector, so that a test connector can stand in."""
    vllm_config = VllmConfig(
        kv_transfer_config=kv_transfer_config,
        kv_events_config=kv_events_config,
        speculative_config=speculative_config,
        model_config=ModelConfig(
            model="facebook/opt-125m",
            dtype="float16",
            seed=42,
            skip_tokenizer_init=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=8192,
            long_prefill_token_threshold=long_prefill_token_threshold,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=async_scheduling,
        ),
        cache_config=CacheConfig(
            block_size=block_size,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            **cache_kwargs,
        ),
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    if compact_layout:
        vllm_config.cache_config.mamba_spec_state_layout = "compact"
    groups = [
        KVCacheGroupSpec(
            ["attn"],
            FullAttentionSpec(
                block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        *(
            KVCacheGroupSpec(
                [f"mamba{i}"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_prefill_checkpoint_blocks=int(checkpoint_alignment is not None),
                    prefill_checkpoint_alignment=checkpoint_alignment,
                    num_speculative_blocks=num_speculative_blocks,
                ),
            )
            for i in range(2)
        ),
    ]
    if ring:
        groups.append(
            KVCacheGroupSpec(
                ["ring"],
                CircularBufferSpec(
                    block_size=ring,
                    num_kv_heads=2,
                    head_size=1,
                    head_size_v=0,
                    dtype=torch.bfloat16,
                    dcp_sharded=False,
                    resume_catchup=resume_catchup,
                    compression_ratio=ring_compression_ratio,
                ),
            )
        )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )
    register_all_kvcache_specs(vllm_config)
    # The checkpoint features need Model Runner V2, which CPU builds select
    # only when asked to.
    scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
    with pytest.MonkeyPatch.context() as m:
        m.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        return scheduler_cls(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=block_size,
            hash_block_size=hash_block_size or block_size,
            log_stats=True,
        )


def prompt(seed: int, num_tokens: int) -> list[int]:
    return [3 + (seed * 7919 + i * 104729) % 30000 for i in range(num_tokens)]


def add_request(
    scheduler: Scheduler,
    request_id: str,
    tokens: list[int],
    max_tokens: int,
    session_id: str | None = None,
) -> Request:
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, EOS)
    request = Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(scheduler.hash_block_size, sha256),
        session_id=session_id,
    )
    scheduler.add_request(request)
    return request


def step(scheduler: Scheduler) -> SchedulerOutput:
    """Schedule one step and return its output; requests that finished their
    prefill sample one token."""
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens)
    sampled = []
    for request_id in req_ids:
        request = scheduler.requests[request_id]
        position = request.num_computed_tokens
        sampled.append([] if request.is_prefill_chunk else [3 + position % 997])
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return output


def run(scheduler: Scheduler, *requests: Request) -> list[SchedulerOutput]:
    outputs = []
    while not all(r.is_finished() for r in requests):
        outputs.append(step(scheduler))
    return outputs


def admitted_at(scheduler: Scheduler, request: Request) -> int:
    """Step until ``request`` is admitted; returns the tokens its prefix-cache
    hit covered."""
    while request.status == RequestStatus.WAITING:
        output = step(scheduler)
        for new in output.scheduled_new_reqs:
            if new.req_id == request.request_id:
                return new.num_computed_tokens
    raise AssertionError(f"{request.request_id} was not admitted")
