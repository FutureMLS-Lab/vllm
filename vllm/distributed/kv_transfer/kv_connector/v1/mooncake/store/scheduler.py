# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-project/vllm-ascend
# (vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/).
"""Scheduler-side logic for MooncakeStoreConnector."""

import time
from collections.abc import Sequence

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
    BoundaryPut,
    FrontierLoad,
    FrontierPut,
    LoadSpec,
    MooncakeLookupResult,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreWorkerMetadata,
    ReqMeta,
    RequestTracker,
    StoreUnit,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.frontier import (  # noqa: E501
    FrontierLookup,
    FrontierStoreConfig,
    ring_group_ids,
    ring_unit,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (  # noqa: E501
    LookupKeyClient,
)
from vllm.logger import init_logger
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.hybrid_checkpoint.frontier import FrontierCheckpoint
from vllm.v1.core.hybrid_checkpoint.keys import (
    FRONTIER_LOCATOR_TOKENS,
    frontier_locator_key,
    frontier_locator_unit,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)

# frontier_supersede_lag: chains of frontiers kept, the oldest dropped first.
_SUPERSEDE_CHAINS = 4096


def _new_req_prefill_tokens(request: NewRequestData) -> list[int]:
    """Tokens this prefill will compute KV for.

    Under the v2 model runner, resumed-from-preemption requests appear in
    ``scheduled_new_reqs`` with ``prefill_token_ids`` set to the request's full
    token list (prompt + previously-generated). For all other cases this falls
    back to the original prompt.
    """
    if request.prefill_token_ids is not None:
        return request.prefill_token_ids
    assert request.prompt_token_ids is not None
    return request.prompt_token_ids


def _partial_tail_non_mamba_puts(
    coord: MooncakeStoreCoordinator,
    req_meta: ReqMeta,
    block_sizes: Sequence[int],
) -> list[BoundaryPut]:
    """Locate the non-Mamba blocks a partial-tail save publishes.

    A later request resumes at the prompt's Mamba checkpoint (``boundary``) only
    if both pieces of KV are stored:

    - Mamba groups: the state at ``boundary``, handed off by the core as a
      boundary put of its own, so Mamba groups are not listed here.
    - Other groups: the KV from the last LCM-aligned normal save up to
      ``boundary`` plus the group's EAGLE proof margin (``proof_end``). Each
      block is keyed by the hash at its end, clipped to ``proof_end``.

    Groups whose proof is not computed yet are omitted. For example, with LCM
    16, blocks of 4 tokens, and a checkpoint at 44, the full-attention blocks
    are 8-10, keyed at 36, 40 and 44.

    A Mamba state handed off at any other position, i.e. a shared-prefix
    junction (``--enable-mamba-shared-prefix-checkpoint``), adds no puts here.

    ``block_sizes`` are the per-group block sizes, indexed like
    ``req_meta.block_ids``.
    """
    mamba_tails = [
        put.num_tokens
        for put in req_meta.boundary_puts or []
        if put.group_id in coord.mamba_group_ids
        and put.num_tokens % block_sizes[put.group_id]
    ]
    # The tail is due on the save that first covers the prompt, or when Mamba
    # hands off its checkpoint state.
    if not mamba_tails and not req_meta.publish_partial_tail:
        return []
    prompt_tokens = req_meta.num_prompt_tokens
    completed = req_meta.completed_token_len
    assert prompt_tokens is not None and completed is not None
    hash_block_size = coord.hash_block_size
    # Core drops the Mamba checkpoint whenever EAGLE block drop is on, whichever
    # groups carry the eagle flag.
    boundary = get_mamba_prefill_checkpoint_position(
        prompt_tokens, hash_block_size, coord.use_eagle
    )
    # Any other sub-block Mamba hand-off is a shared-prefix junction. Its
    # attention KV, EAGLE proof included, ends on an attention block end no later
    # than this checkpoint's proof, so the normal and prompt-tail saves store it.
    if not req_meta.publish_partial_tail and boundary not in mamba_tails:
        return []
    num_hashes = len(req_meta.block_hashes)
    if boundary == 0 or boundary // hash_block_size > num_hashes:
        return []
    start = boundary // coord.lcm_block_size * coord.lcm_block_size
    puts: list[BoundaryPut] = []
    for group_id, block_size in enumerate(block_sizes):
        if group_id in coord.mamba_group_ids:
            continue
        proof_end = boundary + coord.eagle_proof_margin_by_group.get(group_id, 0)
        if proof_end > completed or proof_end // hash_block_size > num_hashes:
            continue
        group_blocks = req_meta.block_ids[group_id]
        for block_idx in range(start // block_size, cdiv(proof_end, block_size)):
            if block_idx >= len(group_blocks):
                break
            block_id = group_blocks[block_idx]
            if block_id == NULL_BLOCK_ID:
                continue
            num_tokens = min((block_idx + 1) * block_size, proof_end)
            puts.append(BoundaryPut(group_id, block_id, num_tokens))
    return puts


class MooncakeStoreScheduler:
    """Scheduler-side component for MooncakeStoreConnector."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        kvc_extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.load_async = kvc_extra_config.get("load_async", True)
        self.lookup_async = kvc_extra_config.get("lookup_async", False)
        # Skips lookup CPU cost on instances that never load KV from the store.
        self.enable_lookup = kvc_extra_config.get("enable_lookup", True)
        self.save_decode_cache = kvc_extra_config.get("save_decode_cache", False)
        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = bool(
            kv_event_config and kv_event_config.enable_kv_cache_events
        )
        self.client = LookupKeyClient(vllm_config)
        self.kv_cache_config = kv_cache_config
        self._store_group_ids = kv_cache_config.prefix_cacheable_group_ids
        # Map scheduler group IDs to store group indices. Groups outside the
        # store projection, such as the QSA ring, map to None and are skipped.
        self._store_group_id_by_kv_cache_group_id = {
            group_id: store_group_id
            for store_group_id, group_id in enumerate(self._store_group_ids)
        }
        store_groups = kv_cache_config.prefix_cacheable_groups

        # Align with the engine's own scheduler_block_size and hash_block_size.
        self._block_size, self._hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        mamba_groups = {
            group_id: group.kv_cache_spec
            for group_id, group in enumerate(store_groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
        }
        self._boundary_state_group_ids = frozenset(mamba_groups)

        # Mirrors MooncakeStoreConnector._capacity_only.
        self._capacity_only = (
            self.kv_role == "kv_consumer"
            and not self.enable_lookup
            and not self.save_decode_cache
        )
        self._store_coord: MooncakeStoreCoordinator | None = None
        self.enable_partial_hash_hits = False
        if not self._capacity_only:
            assert all(
                spec.mamba_cache_mode == "align" for spec in mamba_groups.values()
            ), "MooncakeStoreScheduler requires mamba_cache_mode='align'"
            self._store_coord = MooncakeStoreCoordinator.from_kv_cache_config(
                kv_cache_config, vllm_config, self._block_size, self._hash_block_size
            )
            self.enable_partial_hash_hits = self._store_coord.enable_partial_hash_hits

        self._gpu_block_pool: BlockPool | None = None
        self._num_workers = vllm_config.parallel_config.world_size
        self._next_store_job_id = 0
        # store_job_id -> (referenced block ids, ranks yet to report completion)
        self._pinned_saves: dict[int, tuple[list[int], int]] = {}
        # staging_buffer_mib: store_job_id -> ranks yet to report the
        # job's blocks released, and the jobs whose blocks are already free.
        self._unreleased: dict[int, int] = {}
        self._released_jobs: set[int] = set()

        # Per-request state
        self.load_specs: dict[str, LoadSpec] = {}  # to be loaded
        self._request_trackers: dict[str, RequestTracker] = {}  # scheduled new requests
        self._unfinished_requests: dict[str, tuple[Request, tuple[list[int], ...]]] = {}
        self._unfinished_request_ids: set[str] = set()
        self._finished_partial_tail_metas: dict[str, ReqMeta] = {}

        # frontier_write_through: lookups also find stored Mamba frontier
        # checkpoints, and finished requests' frontiers become store jobs.
        self._frontiers: FrontierLookup | None = None
        self._ring_group_ids = ring_group_ids(kv_cache_config)
        self._ring_unit = ring_unit(kv_cache_config)
        self._finished_frontier_metas: list[ReqMeta] = []
        self._frontier_config = FrontierStoreConfig.from_vllm_config(vllm_config)
        # frontier_supersede_remove: request id -> (key, length) of the
        # frontier it resumed from, and (store group id, tokens) of the
        # boundary states it stored.
        self._supersede = False
        self._resumed_frontiers: dict[str, tuple[bytes, int]] = {}
        self._boundary_tokens: dict[str, set[tuple[int, int]]] = {}
        if self._frontier_config.write_through and not self._capacity_only:
            self._frontiers = FrontierLookup(
                self.client,
                self._block_size,
                self._hash_block_size,
                get_hash_fn_by_name(vllm_config.cache_config.prefix_caching_hash_algo),
                memo_s=self._frontier_config.lookup_memo_s,
                reuse_s=self._frontier_config.lookup_reuse_s,
            )
            self._supersede = self._frontier_config.supersede_remove
        # frontier_load_local_pages: a stored frontier's load skips the pages
        # the GPU caches.
        self._load_local_pages = (
            self._frontiers is not None and self._frontier_config.load_local_pages
        )
        # unit_policy_url: request id -> (key, position) of the deepest store
        # unit on its path (the one it resumed at, then each unit it wrote),
        # its last lookup's (local hit, store hit, frontier key), and the keys
        # of the units admitted requests resumed at.
        self._store_units = (
            self._frontiers is not None
            and self._frontier_config.unit_policy_url is not None
        )
        self._unit_frontier: dict[str, tuple[bytes, int]] = {}
        self._unit_lookups: dict[str, tuple[int, int, bytes | None]] = {}
        self._unit_touches: list[bytes] = []
        # frontier_write_wait_s: the frontiers whose write job has not
        # completed on every rank, key -> (length, the writer's hash at the
        # frontier's anchor unit, finish time), the job writing each, and the
        # requests that waited for one and have not been looked up since.
        self._write_wait_s = (
            self._frontier_config.write_wait_s if self._frontiers is not None else 0.0
        )
        self._writing_frontiers: dict[bytes, tuple[int, bytes | None, float]] = {}
        self._frontier_jobs: dict[int, bytes] = {}
        self._frontier_waiters: set[str] = set()
        # frontier_supersede_lag: by the key of a session's newest frontier,
        # what its chain's last finishes would have superseded (oldest first),
        # each emitted ``lag`` finishes later.
        self._supersede_lag = (
            self._frontier_config.supersede_lag if self._supersede else 0
        )
        self._supersede_queue: dict[
            bytes,
            list[tuple[tuple[tuple[bytes, int], ...], tuple[tuple[int, int], ...]]],
        ] = {}

    @property
    def stores_frontiers(self) -> bool:
        return self._frontiers is not None

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        self._gpu_block_pool = gpu_block_pool

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Check for external KV cache hit.

        Returns ``(None, False)`` when an async lookup is still in flight,
        signaling the scheduler to retry this request on a later step.
        """
        if not self.enable_lookup:
            return 0, False

        # Fine-grained hits may land on a hash boundary inside a block; without
        # partial hits, prefixes shorter than one physical block are skipped.
        align = (
            self._hash_block_size if self.enable_partial_hash_hits else self._block_size
        )
        if request.num_tokens < align:
            return 0, False

        if self._frontiers is not None:
            lookup_result = self._frontiers.lookup(
                request, num_computed_tokens, non_block=self.lookup_async
            )
        else:
            lookup_result = self.client.lookup(
                request.request_id,
                request.num_tokens,
                request.block_hashes,
                non_block=self.lookup_async,
            )
        if lookup_result is None:
            # Lookup not ready yet; scheduler will retry on a later step.
            return None, False
        num_external_hit_tokens = lookup_result.hit_length
        if self._frontiers is not None and self._frontiers.record_paths:
            path, elapsed_ms = self._frontiers.paths.pop(
                request.request_id, ("none", 0.0)
            )
            logger.debug(
                "Mooncake frontier lookup: req=%s tokens=%d local=%d path=%s "
                "ms=%.1f hit=%d frontier=%d",
                request.request_id,
                request.num_tokens,
                num_computed_tokens,
                path,
                elapsed_ms,
                num_external_hit_tokens,
                lookup_result.frontier is not None,
            )
        if request.request_id in self._frontier_waiters:
            self._frontier_waiters.discard(request.request_id)
            assert self._frontiers is not None
            hit = (
                lookup_result.frontier is not None
                and num_external_hit_tokens > num_computed_tokens
            )
            self._frontiers.counts["waited_hits" if hit else "waited_misses"] += 1
        if self._supersede:
            self._note_resumed_frontier(request, num_computed_tokens, lookup_result)
        if self._store_units:
            self._unit_lookups[request.request_id] = (
                num_computed_tokens,
                num_external_hit_tokens,
                lookup_result.frontier,
            )

        if num_external_hit_tokens < num_computed_tokens:
            need_to_allocate = 0
        else:
            need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.debug(
            "Reqid: %s, Total tokens %d, kvpool hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        if need_to_allocate <= 0:
            return 0, False

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=num_external_hit_tokens,
            can_load=False,
            tail_key_boundaries=lookup_result.tail_key_boundaries,
            frontier=(
                None
                if lookup_result.frontier is None
                else FrontierLoad(lookup_result.frontier, num_external_hit_tokens)
            ),
        )

        return need_to_allocate, self.load_async

    def _note_resumed_frontier(
        self, request: Request, local: int, result: MooncakeLookupResult
    ) -> None:
        """``frontier_supersede_remove``: the stored frontier a request loads
        and resumes from. A local hit reaches the connector block-aligned, so
        a frontier the request resumes from on the GPU is not seen here."""
        if result.frontier is not None and result.hit_length > local:
            self._resumed_frontiers[request.request_id] = (
                bytes(result.frontier),
                result.hit_length,
            )
        else:
            self._resumed_frontiers.pop(request.request_id, None)

    def _note_resume_unit(self, request: Request, position: int) -> None:
        """``unit_policy_url``: the request starts computing at ``position``,
        its resume point: the frontier its lookup found in the store or on the
        GPU (inside a hash unit only a frontier), else the state at that hash
        unit. A use of that unit, and the parent of the first unit the request
        writes. The connector's lookup sees only the block-aligned local hit,
        so the position comes from the schedule."""
        assert self._frontiers is not None
        req_id = request.request_id
        lookup = self._unit_lookups.pop(req_id, None)
        self._unit_frontier.pop(req_id, None)
        if position == 0:
            return
        unit = self._hash_block_size
        if lookup is not None and lookup[1] == position and lookup[2] is not None:
            key = lookup[2]
        elif position % unit:
            key = self._frontiers.key_at(request, position)
        else:
            key = request.block_hashes[position // unit - 1]
        self._unit_frontier[req_id] = (bytes(key), position)
        self._unit_touches.append(bytes(key))

    def _next_unit(
        self, req_id: str, key: bytes, position: int, frontier: bool
    ) -> StoreUnit:
        """``unit_policy_url``: the unit a job of ``req_id`` writes at
        ``position``, under the request's deepest store unit before it."""
        deepest = self._unit_frontier.get(req_id)
        parent = deepest[0] if deepest is not None and deepest[1] < position else None
        self._unit_frontier[req_id] = (key, position)
        return StoreUnit(key, position, parent, frontier)

    def frontier_load_length(self, request: Request) -> int | None:
        """``frontier_load_local_pages``: the length of the stored frontier the
        request's pending load resumes at, or None."""
        load_spec = self.load_specs.get(request.request_id)
        if (
            not self._load_local_pages
            or load_spec is None
            or load_spec.frontier is None
        ):
            return None
        return load_spec.kvpool_cached_tokens

    def move_frontier_load_start(self, request: Request, num_tokens: int) -> None:
        """``frontier_load_local_pages``: the request takes the full pages
        before ``num_tokens`` (on the block grid, past its lookup's local hit,
        at most the frontier's block start) from the GPU's cache, so its
        pending load reads only the pages from there and the frontier."""
        load_spec = self.load_specs[request.request_id]
        assert load_spec.frontier is not None
        block_start = load_spec.frontier.length // self._block_size * self._block_size
        assert num_tokens % self._block_size == 0, num_tokens
        assert load_spec.vllm_cached_tokens < num_tokens <= block_start, (
            load_spec.vllm_cached_tokens,
            num_tokens,
            block_start,
        )
        load_spec.vllm_cached_tokens = num_tokens
        logger.debug(
            "Mooncake frontier load: req=%s starts at %d of %d (the pages before "
            "it are cached)",
            request.request_id,
            num_tokens,
            load_spec.frontier.length,
        )

    def frontier_ring_groups(self, request: Request) -> tuple[int, ...]:
        """The indexer-ring groups a pending load resuming at a stored
        frontier writes: every ring, unless the frontier ends on a pool
        boundary."""
        load_spec = self.load_specs.get(request.request_id)
        if (
            load_spec is None
            or load_spec.frontier is None
            or load_spec.frontier.length % self._ring_unit == 0
        ):
            return ()
        return self._ring_group_ids

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ):
        """Update state after block allocation."""
        local_block_ids: tuple[list[int], ...] = ()
        if num_external_tokens > 0:
            local_block_ids = blocks.get_block_ids(
                group_ids=self._store_group_ids,
            )

        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        self._unfinished_request_ids.add(request.request_id)

        if request.request_id not in self.load_specs:
            return

        if num_external_tokens == 0:
            self.load_specs[request.request_id].can_load = False
            return

        assert (
            num_external_tokens > 0
            and num_external_tokens
            == self.load_specs[request.request_id].kvpool_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
        ), (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].kvpool_cached_tokens} - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens}"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True
        frontier = self.load_specs[request.request_id].frontier
        if frontier is not None and (rings := self.frontier_ring_groups(request)):
            all_block_ids = blocks.get_block_ids()
            frontier.rings = tuple(
                (group_id, all_block_ids[group_id][0]) for group_id in rings
            )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Build connector metadata for this scheduler step."""
        is_consumer = self.kv_role == "kv_consumer"
        can_process_cached = not is_consumer or self.save_decode_cache

        for finished_req_id in scheduler_output.finished_req_ids:
            self.client.discard(finished_req_id)
            self.load_specs.pop(finished_req_id, None)
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._unfinished_request_ids.discard(finished_req_id)
            self._resumed_frontiers.pop(finished_req_id, None)
            self._boundary_tokens.pop(finished_req_id, None)
            self._unit_frontier.pop(finished_req_id, None)
            self._unit_lookups.pop(finished_req_id, None)
            self._frontier_waiters.discard(finished_req_id)
            if self._frontiers is not None:
                self._frontiers.forget(finished_req_id)
                self._frontiers.paths.pop(finished_req_id, None)

        preempted_ids = scheduler_output.preempted_req_ids or set()
        for req_id in preempted_ids:
            self.load_specs.pop(req_id, None)
            if request_tracker := self._request_trackers.get(req_id):
                request_tracker.reset()
            self._unfinished_requests.pop(req_id, None)
            self._unit_frontier.pop(req_id, None)

        meta = MooncakeStoreConnectorMetadata(
            self._unfinished_request_ids,
            preempted_ids,
        )
        if self._unit_touches:
            meta.unit_touches, self._unit_touches = self._unit_touches, []

        # Handle new requests
        for request in scheduler_output.scheduled_new_reqs:
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            assert request.req_id in self._unfinished_requests
            request_tuple = self._unfinished_requests.get(request.req_id)
            request_real = request_tuple[0]  # type: ignore[index]
            if self._store_units:
                self._note_resume_unit(request_real, request.num_computed_tokens)

            unfolded_block_ids = tuple(
                request.block_ids[group_id].copy() for group_id in self._store_group_ids
            )

            prefill_tokens = _new_req_prefill_tokens(request)
            request_tracker = RequestTracker(
                req_id=request.req_id,
                token_len=num_tokens_to_compute,
                allocated_block_ids=unfolded_block_ids,
                num_saved_tokens=0,
                token_ids=(
                    prefill_tokens[:num_tokens_to_compute]
                    if self.enable_kv_events
                    else None
                ),
                prefill_end_tokens=len(prefill_tokens),
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                load_spec=load_spec,
                # A consumer may write decode KV without becoming a prefill
                # producer. Loads are still carried by the same metadata.
                skip_save=is_consumer,
                block_hashes=request_real.block_hashes,
                num_prompt_tokens=request_real.num_prompt_tokens,
                save_partial_tail=self.enable_partial_hash_hits,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        # Handle cached (running, or MRV1 resumed-from-preemption) requests
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if can_process_cached:
            for i, req_id in enumerate(cached_reqs.req_ids):
                new_block_ids = cached_reqs.new_block_ids[i]
                if new_block_ids:
                    new_block_ids = tuple(
                        new_block_ids[group_id] for group_id in self._store_group_ids
                    )

                req_meta = None
                if req_id in cached_reqs.resumed_req_ids:
                    # Resumed after preemption
                    if not new_block_ids:
                        continue
                    new_block_ids = tuple(b.copy() for b in new_block_ids)
                    load_spec = self.load_specs.pop(req_id, None)
                    request_tuple = self._unfinished_requests.get(req_id)
                    request_real = request_tuple[0]  # type: ignore[index]
                    if self._store_units:
                        self._note_resume_unit(
                            request_real, request_real.num_computed_tokens
                        )
                    num_tokens_to_compute = (
                        request_real.num_computed_tokens
                        + scheduler_output.num_scheduled_tokens[req_id]
                    )
                    # On resume, the request re-prefills prompt + previously
                    # generated tokens (all_token_ids).
                    prefill_tokens = list(request_real.all_token_ids)
                    request_tracker = RequestTracker(
                        req_id=req_id,
                        token_len=num_tokens_to_compute,
                        allocated_block_ids=new_block_ids,
                        num_saved_tokens=0,
                        token_ids=(
                            prefill_tokens[:num_tokens_to_compute]
                            if self.enable_kv_events
                            else None
                        ),
                        prefill_end_tokens=len(prefill_tokens),
                    )
                    self._request_trackers[req_id] = request_tracker

                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=load_spec,
                        skip_save=is_consumer,
                        block_hashes=request_real.block_hashes,
                        num_prompt_tokens=request_real.num_prompt_tokens,
                        save_partial_tail=self.enable_partial_hash_hits,
                    )
                else:
                    # Decode/chunked request
                    request_tracker = self._request_trackers[req_id]
                    num_computed_token = cached_reqs.num_computed_tokens[i]
                    # Use the tracker's snapshot of the prefill range so resumed
                    # requests keep saving past the original prompt boundary.
                    prefill_end = request_tracker.prefill_end_tokens
                    is_decode = num_computed_token >= prefill_end
                    if is_decode and not self.save_decode_cache:
                        continue

                    num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
                    req_tuple = self._unfinished_requests.get(req_id)
                    if req_tuple:
                        unfinished_req = req_tuple[0]
                        num_current_tokens = request_tracker.token_len
                        new_token_ids = unfinished_req.all_token_ids[
                            num_current_tokens : num_current_tokens + num_new_tokens
                        ]
                        request_tracker.token_len += len(new_token_ids)
                        if request_tracker.token_ids is not None:
                            request_tracker.token_ids.extend(new_token_ids)
                    else:
                        raise ValueError(
                            f"Request {req_id} is not in _unfinished_requests"
                        )
                    # A block is usually allocated before the step that fills
                    # it, so reaching a save boundary does not imply that this
                    # step has new block ids.
                    if new_block_ids:
                        request_tracker.update(new_block_ids)
                    if is_consumer and not is_decode:
                        continue

                    req_meta = ReqMeta.from_request_tracker(
                        request_tracker,
                        self._block_size,
                        load_spec=None,
                        skip_save=False,
                        block_hashes=unfinished_req.block_hashes,
                        num_prompt_tokens=unfinished_req.num_prompt_tokens,
                        save_partial_tail=self.enable_partial_hash_hits,
                    )

                if req_meta is not None:
                    meta.add_request(req_meta)

        # Handle requests with pending load specs not yet scheduled
        request_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        for request_id, (
            unfinished_req,
            block_ids,
        ) in self._unfinished_requests.items():
            if request_id not in request_ids and request_id not in cached_reqs.req_ids:
                load_spec = self.load_specs.pop(request_id, None)
                # A load spec may have been proposed by this connector's
                # lookup but rejected by MultiConnector in favor of another
                # connector. Only the chosen connector may issue the pending
                # load; the normal store path gets its blocks later from
                # SchedulerOutput once the request is actually scheduled.
                if load_spec is None or not load_spec.can_load:
                    continue
                num_tokens_to_compute = load_spec.kvpool_cached_tokens
                request_tracker = RequestTracker(
                    req_id=request_id,
                    token_len=num_tokens_to_compute,
                    allocated_block_ids=block_ids,
                    num_saved_tokens=0,
                )
                self._request_trackers[request_id] = request_tracker
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    load_spec=load_spec,
                    skip_save=None,
                    block_hashes=unfinished_req.block_hashes,
                    num_prompt_tokens=unfinished_req.num_prompt_tokens,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)

        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        if (
            block_state is not None
            and block_state.boundary_state_offloads
            and not is_consumer
        ):
            self._handle_boundary_state_offloads(
                block_state.boundary_state_offloads, meta
            )

        self._apply_current_save_block_ids(meta, scheduler_output)

        # Finish-time handoffs arrive after the producing step's metadata was
        # built. Their exact blocks were pinned when they were registered, so
        # they remain valid after request cleanup and need no current snapshot.
        for req_meta in self._finished_partial_tail_metas.values():
            meta.add_request(req_meta)
        self._finished_partial_tail_metas.clear()
        for req_meta in self._finished_frontier_metas:
            meta.add_request(req_meta)
        self._finished_frontier_metas.clear()

        if not self._capacity_only:
            self._reference_save_blocks(meta)
        return meta

    def _apply_current_save_block_ids(
        self,
        meta: MooncakeStoreConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Replace append-only mirrors with the core's current block tables."""
        save_metas = [req_meta for req_meta in meta.requests if req_meta.can_save]
        if not save_metas:
            return

        block_state = scheduler_output.kv_connector_block_state
        assert block_state is not None, (
            "Current block tables are required for Mooncake store jobs"
        )
        for req_meta in save_metas:
            block_ids = block_state.get_block_ids(req_meta.req_id)
            assert block_ids is not None, (
                f"Missing current block table for store request {req_meta.req_id}"
            )
            req_meta.block_ids = tuple(
                block_ids[group_id] for group_id in self._store_group_ids
            )

    def _reference_save_blocks(self, meta: MooncakeStoreConnectorMetadata) -> None:
        """Take a GPU block reference for every store job this step emits.

        The worker DMAs out of these blocks after the step that scheduled them,
        so a reference keeps them out of the free queue even once the request
        itself is freed, until every rank reports the job done.
        """
        pool = self._gpu_block_pool
        coord = self._store_coord
        assert coord is not None
        block_sizes = [g.kv_cache_spec.block_size for g in coord.kv_cache_groups]
        for req_meta in meta.requests:
            if not req_meta.can_save:
                continue
            assert pool is not None, (
                "GPU block pool must be bound before any store job is emitted"
            )
            if req_meta.store_job_id is not None:
                assert req_meta.store_job_id in self._pinned_saves
                continue
            if tail := _partial_tail_non_mamba_puts(coord, req_meta, block_sizes):
                req_meta.boundary_puts = [*tail, *(req_meta.boundary_puts or [])]
            req_meta.store_job_id = store_job_id = self._next_store_job_id
            self._next_store_job_id += 1
            block_ids = [put.block_id for put in req_meta.boundary_puts or []]
            assert NULL_BLOCK_ID not in block_ids, (
                "A null block cannot back a boundary put"
            )
            if req_meta.token_len_chunk:
                # Normal prefix save: pin all non-Mamba sources for retries.
                # Every allocated block is referenced, not just the ones covering
                # this job's token range: a rank resumes from its own last
                # successful offset, which lags the scheduler's whenever a save was
                # skipped or failed, so it may read anywhere below the range.
                block_ids.extend(
                    block_id
                    for group_id, group in enumerate(req_meta.block_ids)
                    if group_id not in self._boundary_state_group_ids
                    for block_id in group
                    if block_id != NULL_BLOCK_ID
                )
            # An aligned boundary block may also be present in the request's
            # block table. Take and release exactly one reference per block.
            block_ids = list(dict.fromkeys(block_ids))
            assert NULL_BLOCK_ID not in block_ids
            if not block_ids:
                continue
            self._pinned_saves[store_job_id] = (block_ids, self._num_workers)
            pool.touch([pool.blocks[block_id] for block_id in block_ids])

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        """Queue and pin a finish-time tail for the next connector step."""
        if self.kv_role == "kv_consumer" or not partial_tail_offloads:
            return False
        tracker = self._request_trackers.get(request.request_id)
        if tracker is None or not any(block_ids):
            return False
        boundaries = {boundary for _, _, boundary in partial_tail_offloads}
        if len(boundaries) != 1:
            raise ValueError(
                "Partial-tail offloads for one request must share a boundary"
            )
        boundary_tokens = next(iter(boundaries))
        if boundary_tokens > tracker.prefill_end_tokens:
            return False

        mamba_puts: list[BoundaryPut] = []
        for group_id, block_id, boundary in partial_tail_offloads:
            store_group_id = self._store_group_id_by_kv_cache_group_id.get(group_id)
            if store_group_id not in self._boundary_state_group_ids:
                return False
            if block_id == NULL_BLOCK_ID:
                return False
            mamba_puts.append(BoundaryPut(store_group_id, block_id, boundary))
        req_meta = ReqMeta(
            req_id=request.request_id,
            token_len_chunk=0,
            block_ids=tuple(
                block_ids[group_id].copy() for group_id in self._store_group_ids
            ),
            block_hashes=list(request.block_hashes),
            can_save=True,
            num_prompt_tokens=request.num_prompt_tokens,
            prefill_end_tokens=tracker.prefill_end_tokens,
            boundary_puts=mamba_puts,
            completed_token_len=request.num_computed_tokens,
        )
        if self._store_units:
            key = bytes(
                request.block_hashes[boundary_tokens // self._hash_block_size - 1]
            )
            req_meta.store_units = (
                self._next_unit(
                    request.request_id, key, boundary_tokens, frontier=False
                ),
            )
        coord = self._store_coord
        assert coord is not None
        req_meta.boundary_puts = [
            *_partial_tail_non_mamba_puts(
                coord,
                req_meta,
                [g.kv_cache_spec.block_size for g in coord.kv_cache_groups],
            ),
            *mamba_puts,
        ]
        pinned_block_ids = list(
            dict.fromkeys(put.block_id for put in req_meta.boundary_puts)
        )

        pool = self._gpu_block_pool
        assert pool is not None, (
            "GPU block pool must be bound before a finish-time handoff"
        )
        assert request.request_id not in self._finished_partial_tail_metas
        store_job_id = self._next_store_job_id
        self._next_store_job_id += 1
        self._pinned_saves[store_job_id] = (pinned_block_ids, self._num_workers)
        pool.touch([pool.blocks[block_id] for block_id in pinned_block_ids])

        req_meta.store_job_id = store_job_id
        self._finished_partial_tail_metas[request.request_id] = req_meta
        tracker.has_pending_offload = True
        # The store job owns exact block refs, so request cleanup need not wait.
        return False

    def register_finished_frontier(
        self, request: Request, frontier: FrontierCheckpoint
    ) -> None:
        """``frontier_write_through``: queue and pin a finished request's
        frontier checkpoint for the next connector step: its full pages from
        the first block its normal saves did not cover, under their block
        hashes, then the frontier's blocks under its key. A frontier shorter
        than ``FRONTIER_LOCATOR_TOKENS`` has no locator and is not stored."""
        if self._frontiers is None or frontier.num_tokens < FRONTIER_LOCATOR_TOKENS:
            return
        length = frontier.num_tokens
        tracker = self._request_trackers.get(request.request_id)
        first = tracker.num_saved_tokens // self._block_size if tracker else 0
        k = length // self._block_size
        store_group = self._store_group_id_by_kv_cache_group_id
        pages = [
            BoundaryPut(store_group[group_id], block_ids[i], (i + 1) * self._block_size)
            for group_id, block_ids in frontier.pages.items()
            for i in range(first, k)
        ]
        blocks = tuple(
            (store_group[group_id], block_id)
            for group_id, block_id in frontier.blocks.items()
            if group_id in store_group
        )
        rings = tuple(
            (group_id, block_id)
            for group_id, block_id in frontier.blocks.items()
            if group_id not in store_group
        )
        # frontier_supersede_remove: everything the request stored past the
        # frontier it resumed from follows the session's own frontier.
        supersedes: tuple[tuple[bytes, int], ...] = ()
        states: tuple[tuple[int, int], ...] = ()
        resumed = (
            self._resumed_frontiers.get(request.request_id) if self._supersede else None
        )
        if resumed is not None and resumed[1] < length:
            supersedes = (resumed,)
            states = tuple(
                sorted(
                    (group_id, tokens)
                    for group_id, tokens in self._boundary_tokens.get(
                        request.request_id, ()
                    )
                    if resumed[1] < tokens <= length
                )
            )
        if self._supersede_lag:
            supersedes, states = self._lag_supersede(
                resumed, bytes(frontier.key), supersedes, states
            )
        locator = frontier_locator_key(
            self._frontiers.hash_fn,
            request,
            frontier_locator_unit(length, self._hash_block_size),
            self._hash_block_size,
        )
        req_meta = ReqMeta(
            req_id=request.request_id,
            token_len_chunk=0,
            block_ids=tuple([] for _ in self._store_group_ids),
            block_hashes=list(request.block_hashes),
            can_save=True,
            num_prompt_tokens=request.num_prompt_tokens,
            boundary_puts=pages,
            completed_token_len=length,
            frontier=FrontierPut(
                key=bytes(frontier.key),
                length=length,
                locator=bytes(locator),
                blocks=blocks,
                rings=rings,
                supersedes=supersedes,
                superseded_states=states,
            ),
        )
        if self._store_units:
            req_meta.store_units = (
                self._next_unit(
                    request.request_id, bytes(frontier.key), length, frontier=True
                ),
            )
        pinned = list(
            dict.fromkeys(
                [put.block_id for put in pages]
                + [block_id for _, block_id in blocks + rings]
            )
        )
        pool = self._gpu_block_pool
        assert pool is not None, (
            "GPU block pool must be bound before a finish-time handoff"
        )
        req_meta.store_job_id = store_job_id = self._next_store_job_id
        self._next_store_job_id += 1
        self._pinned_saves[store_job_id] = (pinned, self._num_workers)
        pool.touch([pool.blocks[block_id] for block_id in pinned])
        self._finished_frontier_metas.append(req_meta)
        if self._write_wait_s:
            anchor_unit = length // self._hash_block_size
            anchor = (
                bytes(request.block_hashes[anchor_unit - 1]) if anchor_unit else None
            )
            key = bytes(frontier.key)
            self._writing_frontiers[key] = (length, anchor, time.monotonic())
            self._frontier_jobs[store_job_id] = key

    def _lag_supersede(
        self,
        resumed: tuple[bytes, int] | None,
        key: bytes,
        supersedes: tuple[tuple[bytes, int], ...],
        states: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[tuple[bytes, int], ...], tuple[tuple[int, int], ...]]:
        """``frontier_supersede_lag``: what this finish supersedes, the list
        the finish ``lag`` turns earlier on its chain would have (none while
        the chain is shorter). The chain continues from the frontier the
        request resumed from (its queue moves to the new frontier ``key``); a
        request that resumed from none starts a chain. The positions of the
        older lists lie on the request's own prefix, so its block hashes key
        them. At most ``_SUPERSEDE_CHAINS`` chains are kept, the oldest
        dropped."""
        queue = self._supersede_queue.pop(resumed[0], []) if resumed is not None else []
        queue.append((supersedes, states))
        emitted: tuple[tuple[tuple[bytes, int], ...], tuple[tuple[int, int], ...]] = (
            (),
            (),
        )
        if len(queue) > self._supersede_lag:
            emitted = queue.pop(0)
        self._supersede_queue[key] = queue
        while len(self._supersede_queue) > _SUPERSEDE_CHAINS:
            del self._supersede_queue[next(iter(self._supersede_queue))]
        return emitted

    def defer_lookup(self, request: Request, num_computed_tokens: int) -> bool:
        """``frontier_write_wait_s``: True while ``request`` extends past
        ``num_computed_tokens`` (its whole local hit) a frontier of its own
        session whose write job has not completed on every rank, finished
        less than the wait ago: the scheduler tries it again at its next step
        rather than resume it short of the frontier. Its lookup memo is
        dropped, so its next lookup probes afresh."""
        if not self._writing_frontiers:
            return False
        assert self._frontiers is not None
        unit = self._hash_block_size
        hashes = request.block_hashes
        req_id = request.request_id
        counts = self._frontiers.counts
        now = time.monotonic()
        expired = []
        found = False
        for key, (length, anchor, finished) in self._writing_frontiers.items():
            late = now - finished >= self._write_wait_s
            if late:
                # Waited for long enough: it defers no lookup any more.
                expired.append(key)
                if req_id not in self._frontier_waiters:
                    continue
            if not num_computed_tokens < length < request.num_tokens:
                continue
            anchor_unit = length // unit
            if anchor_unit > len(hashes):
                continue
            if (bytes(hashes[anchor_unit - 1]) if anchor_unit else None) != anchor:
                continue
            if self._frontiers.key_at(request, length) != key:
                continue
            if late:
                counts["wait_timeouts"] += 1
                continue
            found = True
            break
        for key in expired:
            del self._writing_frontiers[key]
        if found:
            if req_id not in self._frontier_waiters:
                self._frontier_waiters.add(req_id)
                counts["waited"] += 1
            counts["wait_steps"] += 1
            logger.debug(
                "Mooncake frontier lookup deferred: req=%s local=%d frontier=%d "
                "(being written)",
                req_id,
                num_computed_tokens,
                length,
            )
            self._frontiers.forget(req_id)
            return True
        return False

    def _handle_boundary_state_offloads(
        self,
        offloads: dict[str, list[tuple[int, int, int]]],
        meta: MooncakeStoreConnectorMetadata,
    ) -> None:
        """Attach exact boundary-state blocks to store jobs for this step.

        Flushed in the step they arrive: the CoW copy is enqueued before the
        connector event records, so this step's event fences the exact block.
        Entries ride the request's save meta when present, else an
        offload-only ReqMeta (``token_len_chunk=0`` skips the normal save,
        ``can_save=True`` takes the normal enqueue and store-job pinning path).
        """
        save_metas = {m.req_id: m for m in meta.requests if m.can_save}
        for req_id, entries in offloads.items():
            tracker = self._request_trackers.get(req_id)
            req_tuple = self._unfinished_requests.get(req_id)
            if tracker is None or req_tuple is None:
                # Request finished/preempted within this step; its blocks are
                # going away, so the offload is conservatively dropped.
                logger.debug("Dropping boundary-state offload for request %s", req_id)
                continue
            accepted: list[BoundaryPut] = []
            for group_id, block_id, boundary_tokens in entries:
                # Every other group stops saving at the end of this prefill, so
                # a mamba-only key past it can never complete a joint hybrid
                # hit. `prefill_end_tokens` — not the original prompt length —
                # is the boundary: a resumed request re-prefills and re-saves
                # its previously generated tokens for every group.
                if boundary_tokens > tracker.prefill_end_tokens:
                    continue
                if block_id == NULL_BLOCK_ID:
                    continue
                store_group_id = self._store_group_id_by_kv_cache_group_id.get(group_id)
                if store_group_id not in self._boundary_state_group_ids:
                    continue
                accepted.append(BoundaryPut(store_group_id, block_id, boundary_tokens))
            if not accepted:
                continue
            if self._supersede:
                self._boundary_tokens.setdefault(req_id, set()).update(
                    (put.group_id, put.num_tokens) for put in accepted
                )
            units: tuple[StoreUnit, ...] = ()
            if self._store_units:
                block_hashes = req_tuple[0].block_hashes
                units = tuple(
                    self._next_unit(
                        req_id,
                        bytes(block_hashes[tokens // self._hash_block_size - 1]),
                        tokens,
                        frontier=False,
                    )
                    for tokens in sorted({put.num_tokens for put in accepted})
                )
            tracker.has_pending_offload = True
            if (req_meta := save_metas.get(req_id)) is not None:
                req_meta.boundary_puts = accepted
                req_meta.store_units = units
                continue
            meta.add_request(
                ReqMeta(
                    req_id=req_id,
                    token_len_chunk=0,
                    block_ids=tracker.allocated_block_ids,
                    block_hashes=req_tuple[0].block_hashes,
                    can_save=True,
                    num_prompt_tokens=req_tuple[0].num_prompt_tokens,
                    prefill_end_tokens=tracker.prefill_end_tokens,
                    boundary_puts=accepted,
                    completed_token_len=tracker.token_len,
                    store_units=units,
                )
            )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Drop the block references of store jobs every rank has finished,
        or (``staging_buffer_mib``) every rank has copied out."""
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, MooncakeStoreWorkerMetadata):
            return
        pool = self._gpu_block_pool
        assert pool is not None
        for store_job_id, count in meta.released_saves.items():
            pinned = self._pinned_saves.get(store_job_id)
            if pinned is None or store_job_id in self._released_jobs:
                continue
            remaining = self._unreleased.get(store_job_id, self._num_workers) - count
            assert remaining >= 0, (
                f"store job {store_job_id} released by too many ranks"
            )
            if remaining > 0:
                self._unreleased[store_job_id] = remaining
                continue
            self._unreleased.pop(store_job_id, None)
            self._released_jobs.add(store_job_id)
            pool.free_blocks(pool.blocks[bid] for bid in reversed(pinned[0]))
        for store_job_id, count in meta.completed_saves.items():
            pinned = self._pinned_saves.get(store_job_id)
            if pinned is None:
                # The job referenced no blocks, so nothing was recorded for it.
                continue
            block_ids, remaining = pinned
            remaining -= count
            if remaining > 0:
                self._pinned_saves[store_job_id] = (block_ids, remaining)
                continue
            assert remaining == 0, (
                f"store job {store_job_id} reported by too many ranks"
            )
            del self._pinned_saves[store_job_id]
            self._unreleased.pop(store_job_id, None)
            if (key := self._frontier_jobs.pop(store_job_id, None)) is not None:
                self._writing_frontiers.pop(key, None)
            if store_job_id in self._released_jobs:
                self._released_jobs.discard(store_job_id)
                continue
            # Tail-first, as elsewhere, so the shared prefix is evicted last.
            pool.free_blocks(pool.blocks[bid] for bid in reversed(block_ids))

    def has_pending_push_work(self) -> bool:
        """Keep the engine stepping while any store job still holds block refs.

        Completions only reach the scheduler as worker metadata on a step, so an
        engine that quiesced with jobs in flight would leave those references
        held indefinitely. Nothing else keeps it alive now that a finishing
        request no longer defers its own free.
        """
        return bool(self._pinned_saves)

    def reset_store(self) -> bool:
        """Trigger a global ``remove_all(force=True)`` on the Mooncake master.

        Routes through the existing LookupKey ZMQ admin channel to worker
        rank 0, which owns the ``MooncakeDistributedStore`` handle.

        Ordering assumption: caller (typically
        ``Scheduler.reset_connector_cache``, invoked via
        ``reset_prefix_cache(reset_connector=True)``) MUST ensure no
        in-flight Mooncake lookups or transfers. For RL workflows this is
        satisfied at the step boundary after weight updates and rollout
        drain. Violating this can allow stale KV to be served on the next
        request, defeating the hard-reset guarantee.

        Returns True on ACK from worker, False on NACK or RPC error.
        """
        try:
            ok = self.client.reset()
            if ok:
                logger.info("Mooncake store reset via remove_all succeeded.")
            else:
                logger.warning("Mooncake store reset returned NACK from worker.")
            return ok
        except Exception as e:
            logger.error("Mooncake reset_store RPC failed: %s", e)
            return False
