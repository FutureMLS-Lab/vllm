# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MooncakeStoreConnector with ``frontier_write_through``: Mamba frontier
checkpoints written to the store and found there by another engine."""

import dataclasses
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    frontier as fr,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    protocol as pr,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BoundaryPut,
    ChunkedTokenDatabase,
    FrontierLoad,
    FrontierProbe,
    FrontierPut,
    KeyMetadata,
    LoadSpec,
    MooncakeLookupResult,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreWorkerMetadata,
    PoolKey,
    ReqMeta,
    RequestTracker,
    TailKeyBoundary,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (  # noqa: E501
    MooncakeStoreScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (  # noqa: E501
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    MooncakeStoreWorker,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.hybrid_checkpoint.frontier import FrontierCheckpoint
from vllm.v1.core.hybrid_checkpoint.keys import (
    FRONTIER_LOCATOR_TOKENS,
    frontier_key,
    frontier_locator_key,
    frontier_locator_unit,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    MambaStateGather,
    generate_scheduler_kv_cache_config,
    hash_block_tokens,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

B, U = 64, 16


class _Req:
    """What frontier and locator keys read of a request: its tokens and block
    hashes at the hash unit, with no extra keys."""

    def __init__(self, tokens: list[int], request_id: str = "r") -> None:
        init_none_hash(sha256)
        self.request_id = request_id
        self.all_token_ids = list(tokens)
        self.num_tokens = len(tokens)
        self.num_prompt_tokens = len(tokens)
        self.mm_features: list = []
        self.lora_request = None
        self.cache_salt = None
        self.prompt_embeds = None
        self.block_hashes: list[BlockHash] = []
        parent = None
        for i in range(len(tokens) // U):
            parent = hash_block_tokens(
                sha256, parent, self.all_token_ids[i * U : (i + 1) * U], None
            )
            self.block_hashes.append(parent)


def _tokens(n: int, seed: int = 0) -> list[int]:
    return [3 + (seed * 7919 + i * 104729) % 30000 for i in range(n)]


# Locator records ------------------------------------------------------------
def test_locator_record_round_trip_and_garbage():
    entries = [(201, bytes(range(32))), (90, b"\xff" * 32)]
    record = fr.encode_locator(entries)
    assert fr.decode_locator(record) == entries
    assert fr.decode_locator(b"") == []
    assert fr.decode_locator(b"not a record") == []
    assert fr.decode_locator(record[:-1]) == []


def test_locator_merge_puts_the_new_frontier_first_and_keeps_at_most_four():
    old = [(100 + i, bytes([i]) * 32) for i in range(4)]
    merged = fr.merge_locator(old, 999, b"\x09" * 32)
    assert merged[0] == (999, b"\x09" * 32) and merged[1:] == old[:3]
    assert len(merged) == fr.LOCATOR_ENTRIES
    assert fr.merge_locator(merged, 999, b"\x09" * 32) == merged
    # Same length, other tokens: both kept.
    fork = fr.merge_locator(merged, 999, b"\x0a" * 32)
    assert fork[:2] == [(999, b"\x0a" * 32), (999, b"\x09" * 32)]


# Wire format ----------------------------------------------------------------
def test_probe_response_round_trip():
    regular = MooncakeLookupResult(80, (TailKeyBoundary(0, 80),))
    for probe in (
        FrontierProbe((), regular),
        FrontierProbe(((201, b"k" * 32), (90, b"j" * 32)), None),
        FrontierProbe(((201, b"k" * 32),), regular),
    ):
        assert pr.decode_frontier_probe(pr.encode_frontier_probe(probe)) == probe
    with pytest.raises(ValueError):
        pr.decode_frontier_probe(b"X")
    encoded = pr.encode_frontier_probe(
        FrontierProbe(((201, b"k" * 32), (90, b"j" * 32)), regular)
    )
    # A cut after the regular lookup or after an entry is a shorter probe;
    # any other is refused as malformed, never read past its end.
    head = 5 + len(pr.encode_lookup_response(regular))
    for end in range(1, len(encoded)):
        if end not in (head, head + 37):
            with pytest.raises(ValueError):
                pr.decode_frontier_probe(encoded[:end])


def test_probe_request_locators_are_checked():
    keys = [b"a" * 32, b"b" * 32]
    size = (32).to_bytes(2, "big")
    assert pr.decode_frontier_locators(size, b"".join(keys)) == keys
    for bad_size, blob in (
        ((0).to_bytes(2, "big"), b"".join(keys)),
        (size, b"".join(keys)[:-1]),
        (b"\x00", b"".join(keys)),
    ):
        with pytest.raises(ValueError):
            pr.decode_frontier_locators(bad_size, blob)


def test_verify_request_and_response_round_trip():
    key = bytes(range(32))
    request = pr.encode_frontier_verify(201, 1, True, key)
    assert pr.decode_frontier_verify(request) == (201, 1, True, key)
    hit = MooncakeLookupResult(201, frontier=key)
    regular = MooncakeLookupResult(80, (TailKeyBoundary(0, 80),))
    for result in (hit, regular, None):
        encoded = pr.encode_frontier_verified(result)
        assert pr.decode_frontier_verified(encoded, 201, key) == result


# Keys -----------------------------------------------------------------------
@pytest.mark.parametrize("length", [16, 17, U - 1 + 16, B - 1, B + 15, 3 * B + 7])
def test_a_later_request_computes_the_locator_from_its_own_tokens(length):
    writer = _Req(_tokens(length + 1))
    reader = _Req(_tokens(length + 1) + [7] * 200)
    h = frontier_locator_unit(length, U)
    # The locator's tokens lie inside the frontier.
    assert h * U + FRONTIER_LOCATOR_TOKENS <= length
    assert frontier_locator_key(sha256, writer, h, U) == frontier_locator_key(
        sha256, reader, h, U
    )
    assert frontier_key(sha256, writer, length, U) == frontier_key(
        sha256, reader, length, U
    )
    other = _Req(_tokens(length + 1, seed=1))
    assert frontier_locator_key(sha256, writer, h, U) != frontier_locator_key(
        sha256, other, h, U
    )
    assert frontier_locator_key(sha256, writer, h, U) != frontier_key(
        sha256, writer, h * U + FRONTIER_LOCATOR_TOKENS, U
    )


def test_workers_and_the_scheduler_find_the_same_ring_groups():
    """A worker gets its groups as UniformTypeKVCacheSpecs and the scheduler
    one layer's spec each; both must see the same rings and pool."""
    attn = FullAttentionSpec(
        block_size=B, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    state = MambaSpec(
        block_size=B,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    ring = CircularBufferSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=1,
        head_size_v=0,
        dtype=torch.bfloat16,
        dcp_sharded=False,
    )
    layers = [["attn0", "attn1"], ["m0"], ["m1"], ["ring0", "ring1"]]
    worker = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                names,
                UniformTypeKVCacheSpecs(
                    block_size=spec.block_size,
                    kv_cache_specs={name: spec for name in names},
                ),
            )
            for names, spec in zip(layers, [attn, state, state, ring])
        ],
    )
    scheduler = generate_scheduler_kv_cache_config([worker])
    assert fr.ring_group_ids(worker) == fr.ring_group_ids(scheduler) == (3,)
    assert fr.ring_unit(worker) == fr.ring_unit(scheduler) == 4
    # A group with no layer on this worker (pipeline parallel) holds no ring.
    worker.kv_cache_groups.append(
        KVCacheGroupSpec([], UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs={}))
    )
    assert fr.ring_group_ids(worker) == (3,)


def test_a_ring_with_draft_rows_keeps_its_pool_as_the_unit():
    """Under speculative decoding a ring holds more rows than a pool holds
    tokens (here 8 rows for pools of 4). A frontier keeps its ring unless its
    length is a multiple of the pool, as the scheduler's frontier index
    decides, so a reader does not ask for a ring the writer never stored."""
    ring = CircularBufferSpec(
        block_size=8,
        num_kv_heads=2,
        head_size=1,
        head_size_v=0,
        dtype=torch.bfloat16,
        dcp_sharded=False,
        compression_ratio=4,
    )
    names = ["ring0", "ring1"]
    worker = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                names,
                UniformTypeKVCacheSpecs(
                    block_size=ring.block_size,
                    kv_cache_specs={name: ring for name in names},
                ),
            )
        ],
    )
    scheduler = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(names, ring)],
    )
    assert fr.ring_unit(worker) == fr.ring_unit(scheduler) == ring.group_size == 4


# The option -----------------------------------------------------------------
def _vllm_config(
    extra: dict, frontier: bool = True, speculative=None, layout="per_position"
):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        cache_config=SimpleNamespace(
            mamba_frontier_checkpoint=frontier, mamba_spec_state_layout=layout
        ),
        speculative_config=speculative,
    )


def _config(extra: dict, **kwargs) -> fr.FrontierStoreConfig:
    return fr.FrontierStoreConfig.from_vllm_config(_vllm_config(extra, **kwargs))


def test_frontier_options_default_off_and_their_refusals():
    assert _config({}) == fr.FrontierStoreConfig()
    assert _config({}, frontier=False).write_through is False
    assert _config({"frontier_write_through": True}).write_through
    with pytest.raises(ValueError, match="needs mamba_frontier_checkpoint"):
        _config({"frontier_write_through": True}, frontier=False)
    # Under speculative decoding with either Mamba state layout: a state that
    # waits for its draft-column gather goes out with the step that runs it.
    for layout in ("per_position", "compact"):
        assert _config(
            {"frontier_write_through": True}, speculative=object(), layout=layout
        ).write_through
    with pytest.raises(ValueError, match="must be a bool"):
        _config({"frontier_write_through": "yes"})


# The store's keys and buffers -----------------------------------------------
def _db(group: int, tp_rank: int, namespace: str = "", slot: int = 64):
    meta = KeyMetadata("m", tp_rank, 0, 0, 0, group_id=group, store_namespace=namespace)
    db = ChunkedTokenDatabase(meta, B, U)
    db.set_kv_caches_base_addr([1 << 20])
    db.set_block_len([slot])
    return db


def _frontier_store(tp_rank: int, state_size: int | None = None) -> fr.FrontierStore:
    # Store group 0 is replicated (keys in tp_rank 0's namespace); groups 1
    # and 2 are Mamba states; KV-cache group 5 is an indexer ring.
    # ``state_size``: tokens per Mamba block, B (a page) by default.
    dbs = [_db(0, 0), _db(1, tp_rank), _db(2, tp_rank)]
    prefixes = (
        (PoolKey.build_prefix(dbs[0].metadata, tp_rank=0),),
        tuple(PoolKey.build_prefix(dbs[1].metadata, tp_rank=r) for r in range(2)),
        tuple(PoolKey.build_prefix(dbs[2].metadata, tp_rank=r) for r in range(2)),
    )
    ring = _db(5, tp_rank, "@ring")
    ring_prefixes = {
        5: tuple(PoolKey.build_prefix(ring.metadata, tp_rank=r) for r in range(2))
    }
    return fr.FrontierStore(
        dbs,
        prefixes,
        {5: ring},
        ring_prefixes,
        {1, 2},
        B,
        U,
        4,
        "m",
        writes_locators=tp_rank == 0,
        state_size=state_size,
    )


def test_object_puts_stripe_a_replicated_group_and_copy_states_and_rings():
    put = FrontierPut(
        key=b"\x01" * 32,
        length=B + 37,
        locator=b"\x02" * 32,
        blocks=((0, 7), (1, 8), (2, 9)),
        rings=((5, 10),),
    )
    keys = {
        rank: [p[0] for p in _frontier_store(rank).object_puts(put, rank, (2, 1, 1))]
        for rank in range(2)
    }
    # k = 1: chunk 1 of the replicated group is stored by rank 1 only.
    assert not any("@group:0@" in key for key in keys[0])
    assert sum("@group:0@" in key for key in keys[1]) == 1
    for rank in range(2):
        assert sum("@group:1@" in key for key in keys[rank]) == 1
        assert sum("@ring@" in key for key in keys[rank]) == 1
        assert all(key.endswith("@" + "01" * 32) for key in keys[rank])
    puts = _frontier_store(0).object_puts(put, 0, (2, 1, 1))
    assert puts[0][1] == [(1 << 20) + 8 * 64]


def test_verify_keys_cover_the_pages_after_the_local_hit_and_every_block():
    hashes = [bytes([i]) * 32 for i in range(12)]
    store = _frontier_store(0)
    keys = store.verify_keys(hashes, b"\x01" * 32, 2 * B + 37, 1)
    pages = [k for k in keys if "@group:0@" in k and not k.endswith("01" * 32)]
    # Page 1 (hash at its last unit), not page 0 the local hit covers.
    assert pages == [
        PoolKey.build_key_string(store.lookup_prefixes[0][0], hashes[7].hex())
    ]
    frontier = [k for k in keys if k.endswith("@" + "01" * 32)]
    # The replicated page once, two states on two ranks, the ring on two.
    assert len(frontier) == 1 + 2 + 2 + 2
    on_pool_boundary = store.verify_keys(hashes, b"\x01" * 32, 2 * B + 36, 1)
    assert not any("@ring@" in k for k in on_pool_boundary)


def test_load_plan_puts_each_object_into_the_requests_block():
    store = _frontier_store(1)
    load = FrontierLoad(b"\x01" * 32, 2 * B + 37, rings=((5, 30),))
    spec = LoadSpec(
        vllm_cached_tokens=B,
        kvpool_cached_tokens=2 * B + 37,
        can_load=True,
        frontier=load,
    )
    hashes = [bytes([i]) * 32 for i in range(12)]
    meta = ReqMeta(
        req_id="r",
        token_len_chunk=0,
        block_ids=([11, 12, 13], [0, 0, 21], [0, 0, 22]),
        block_hashes=hashes,
        load_spec=spec,
    )
    keys, addrs, sizes, block_ids = store.load_plan(meta)
    # Page 1, the frontier's page into block 2, both states, the ring.
    assert block_ids == [12, 13, 21, 22, 30]
    assert keys[0].endswith(hashes[7].hex())
    assert all(key.endswith("01" * 32) for key in keys[1:])
    assert addrs[0] == [(1 << 20) + 12 * 64] and sizes == [[64]] * 5


@pytest.mark.parametrize(
    ("length", "state_block"),
    # Inside a Mamba block, on a Mamba block boundary, in a page's first one.
    [(2 * B + 3 * U + 5, 11), (2 * B + 3 * U, 10), (2 * B + 5, 8)],
)
def test_load_plan_puts_each_state_into_the_block_holding_its_last_token(
    length: int, state_block: int
):
    """Under decode context parallelism a page holds several Mamba blocks (here
    four of U tokens): a state goes into the Mamba block holding the
    frontier's last token, where a local frontier hit puts it, and the page
    holding the frontier into page ``length // B``."""
    store = _frontier_store(1, state_size=U)
    load = FrontierLoad(b"\x01" * 32, length, rings=((5, 30),))
    spec = LoadSpec(
        vllm_cached_tokens=B,
        kvpool_cached_tokens=length,
        can_load=True,
        frontier=load,
    )
    # The requester's Mamba tables: nulls before the state block, the block a
    # load fills, then the draft-column blocks.
    drafts = [70, 71, 72]
    tables = [[0] * state_block + [s] + drafts for s in (100, 200)]
    meta = ReqMeta(
        req_id="r",
        token_len_chunk=0,
        block_ids=([11, 12, 13], *tables),
        block_hashes=[bytes([i]) * 32 for i in range(12)],
        load_spec=spec,
    )
    _, _, _, block_ids = store.load_plan(meta)
    assert block_ids == [12, 13, 100, 200, 30]


# The scheduler's lookup -----------------------------------------------------
class _StubClient:
    """LookupKeyClient over a dict of locator records and present keys."""

    def __init__(self, entries, regular: int = 0, complete: bool = True) -> None:
        self.entries = entries
        self.regular = MooncakeLookupResult(regular)
        self.complete = complete
        self.calls: list[str] = []

    def run(self, req_id, fn, non_block=False):
        return fn()

    def lookup_now(self, num_tokens, block_hashes):
        self.calls.append("lookup")
        return self.regular

    def frontier_probe(self, num_tokens, block_hashes, locators):
        self.calls.append("probe")
        self.locators = locators
        return FrontierProbe(
            tuple(self.entries), None if self.entries else self.regular
        )

    def frontier_verify(
        self, num_tokens, block_hashes, length, first_block, fallback, key
    ):
        self.calls.append("verify")
        if self.complete:
            return MooncakeLookupResult(length, frontier=key)
        return self.regular if fallback else None


def test_a_probe_looks_at_most_so_many_units_below_the_prompt_end(monkeypatch):
    monkeypatch.setattr(fr, "_PROBE_UNITS", 3)
    request = _Req(_tokens(4 * B))
    client = _StubClient([])
    fr.FrontierLookup(client, B, U, sha256).lookup(request, 0, False)
    last = fr.frontier_locator_unit(4 * B - 1, U)
    assert client.locators == [
        bytes(frontier_locator_key(sha256, request, h, U))
        for h in (last, last - 1, last - 2)
    ]


class _AsyncStubClient(_StubClient):
    """Runs each lookup on a thread of its own, as ``LookupKeyClient.run``,
    once ``gate`` is set."""

    def __init__(self, entries) -> None:
        super().__init__(entries)
        self.futures: dict = {}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.gate = threading.Event()
        self.gate.set()
        self.first_blocks: list[int] = []

    def run(self, req_id, fn, non_block=False):
        if req_id not in self.futures:
            self.futures[req_id] = self.executor.submit(self._gated, fn)
        if non_block and not self.futures[req_id].done():
            return None
        return self.futures.pop(req_id).result()

    def discard(self, req_id):
        future = self.futures.pop(req_id, None)
        if future is not None:
            future.cancel()

    def _gated(self, fn):
        self.gate.wait()
        return fn()

    def frontier_verify(
        self, num_tokens, block_hashes, length, first_block, fallback, key
    ):
        self.first_blocks.append(first_block)
        return super().frontier_verify(
            num_tokens, block_hashes, length, first_block, fallback, key
        )


def test_a_lookup_in_flight_for_another_local_hit_is_submitted_again():
    """The local hit shrank while an asynchronous lookup was in flight: its
    frontier check skipped pages the request no longer holds, so it is
    replaced."""
    request = _Req(_tokens(4 * B))
    key = bytes(frontier_key(sha256, request, 3 * B + 37, U))
    client = _AsyncStubClient([(3 * B + 37, key)])
    lookup = fr.FrontierLookup(client, B, U, sha256)
    client.gate.clear()
    assert lookup.lookup(request, 2 * B, True) is None
    assert lookup.lookup(request, B, True) is None
    client.gate.set()
    while (result := lookup.lookup(request, B, True)) is None:
        time.sleep(0.005)
    assert result.frontier == key
    # The verify of the replacing lookup starts at the new local hit.
    assert client.first_blocks[-1] == 1
    assert not lookup._submitted_at
    client.executor.shutdown()


def test_lookup_resumes_at_the_longest_stored_frontier_the_request_extends():
    tokens = _tokens(4 * B)
    request = _Req(tokens)
    entries = [
        (length, bytes(frontier_key(sha256, request, length, U)))
        for length in (B + 37, 2 * B + 37)
    ]
    # An entry another request wrote: same length, other tokens.
    entries.append((3 * B + 1, b"\x07" * 32))
    lookup = fr.FrontierLookup(_StubClient(entries), B, U, sha256)
    result = lookup.lookup(request, B, non_block=False)
    assert result == MooncakeLookupResult(2 * B + 37, frontier=entries[1][1])
    assert lookup.counts["hits"] == 1


def test_lookup_without_a_usable_frontier_gives_the_regular_lookup():
    request = _Req(_tokens(4 * B))
    client = _StubClient([], regular=2 * B)
    lookup = fr.FrontierLookup(client, B, U, sha256)
    assert lookup.lookup(request, B, non_block=False).hit_length == 2 * B
    assert client.calls == ["probe"]
    # Only an entry of another request: the regular lookup.
    client = _StubClient([(2 * B + 37, b"\x07" * 32)], regular=B)
    lookup = fr.FrontierLookup(client, B, U, sha256)
    assert lookup.lookup(request, 0, non_block=False) == MooncakeLookupResult(B)
    assert client.calls == ["probe", "lookup"]
    # A frontier the store no longer holds whole: the regular hit.
    key = bytes(frontier_key(sha256, request, 2 * B + 37, U))
    client = _StubClient([(2 * B + 37, key)], regular=B, complete=False)
    lookup = fr.FrontierLookup(client, B, U, sha256)
    assert lookup.lookup(request, 0, non_block=False) == MooncakeLookupResult(B)
    assert lookup.counts["incomplete"] == 1


def test_lookup_skips_frontiers_it_cannot_resume_at():
    request = _Req(_tokens(2 * B))
    within_local = bytes(frontier_key(sha256, request, B - 3, U))
    whole = bytes(frontier_key(sha256, request, 2 * B, U))
    client = _StubClient([(B - 3, within_local), (2 * B, whole)], regular=B)
    lookup = fr.FrontierLookup(client, B, U, sha256)
    # B - 3 is not past the local hit B; 2 * B leaves no token to compute.
    assert lookup.lookup(request, B, non_block=False) == MooncakeLookupResult(B)
    assert lookup.counts["no_match"] == 1
    # Too short for a locator: the regular lookup only.
    client = _StubClient([])
    lookup = fr.FrontierLookup(client, B, U, sha256)
    lookup.lookup(_Req(_tokens(FRONTIER_LOCATOR_TOKENS)), 0, non_block=False)
    assert client.calls == ["lookup"]


# The finish-time store job --------------------------------------------------
def _bare_scheduler(num_workers: int = 1) -> MooncakeStoreScheduler:
    scheduler = object.__new__(MooncakeStoreScheduler)
    scheduler.kv_role = "kv_both"
    scheduler._capacity_only = False
    scheduler._block_size = B
    scheduler._hash_block_size = U
    # KV-cache groups: attention 0, Mamba 1 and 2, ring 3.
    scheduler._store_group_ids = (0, 1, 2)
    scheduler._store_group_id_by_kv_cache_group_id = {0: 0, 1: 1, 2: 2}
    scheduler._ring_group_ids = (3,)
    scheduler._ring_unit = 4
    scheduler._frontiers = fr.FrontierLookup(_StubClient([]), B, U, sha256)
    scheduler._finished_frontier_metas = []
    scheduler._supersede = False
    scheduler._supersede_states = True
    scheduler._resumed_frontiers = {}
    scheduler._boundary_tokens = {}
    scheduler._store_units = False
    scheduler._load_local_pages = False
    scheduler._write_wait_s = 0.0
    scheduler._writing_frontiers = {}
    scheduler._frontier_jobs = {}
    scheduler._frontier_waiters = set()
    scheduler._supersede_lag = 0
    scheduler._supersede_queue = {}
    scheduler._keep_states = 0
    scheduler._keep_margin = 0
    scheduler._chains = {}
    scheduler._chain_heads = {}
    scheduler._unit_frontier = {}
    scheduler._unit_lookups = {}
    scheduler._unit_touches = []
    scheduler._request_trackers = {}
    scheduler.load_specs = {}
    scheduler._gpu_block_pool = BlockPool(
        num_gpu_blocks=64, enable_caching=True, hash_block_size=U
    )
    scheduler._num_workers = num_workers
    scheduler._next_store_job_id = 0
    scheduler._pinned_saves = {}
    scheduler._unreleased = {}
    scheduler._released_jobs = set()
    return scheduler


def _checkpoint(request: _Req, length: int) -> FrontierCheckpoint:
    k = length // B
    return FrontierCheckpoint(
        key=frontier_key(sha256, request, length, U),
        num_tokens=length,
        blocks={0: 10 + k, 1: 30, 2: 31, 3: 40},
        pages={0: tuple(range(10, 10 + k))},
    )


@pytest.mark.parametrize("num_workers", [1, 2])
def test_finished_frontier_becomes_a_pinned_store_job(num_workers):
    scheduler = _bare_scheduler(num_workers)
    request = _Req(_tokens(3 * B))
    length = 2 * B + 37
    # The request's normal saves covered its first block.
    scheduler._request_trackers["r"] = RequestTracker(
        req_id="r", token_len=B, allocated_block_ids=([],), num_saved_tokens=B
    )
    checkpoint = _checkpoint(request, length)
    pool = scheduler._gpu_block_pool
    owned = [10, 11, 12, 30, 31, 40]
    pool.touch([pool.blocks[i] for i in owned])

    scheduler.register_finished_frontier(request, checkpoint)
    # The request's own references go; the job keeps the blocks it reads.
    pool.free_blocks([pool.blocks[i] for i in owned])
    assert pool.blocks[10].ref_cnt == 0
    assert all(pool.blocks[i].ref_cnt == 1 for i in (11, 12, 30, 31, 40))

    (req_meta,) = scheduler._finished_frontier_metas
    assert req_meta.boundary_puts == [BoundaryPut(0, 11, 2 * B)]
    locator = frontier_locator_key(sha256, request, frontier_locator_unit(length, U), U)
    assert req_meta.frontier == FrontierPut(
        key=bytes(checkpoint.key),
        length=length,
        locator=bytes(locator),
        blocks=((0, 12), (1, 30), (2, 31)),
        rings=((3, 40),),
    )
    for _ in range(num_workers):
        assert pool.blocks[12].ref_cnt == 1
        scheduler.update_connector_output(
            SimpleNamespace(
                kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                    completed_saves={req_meta.store_job_id: 1}
                )
            )
        )
    assert all(pool.blocks[i].ref_cnt == 0 for i in owned)


def test_a_frontier_job_whose_state_waits_for_a_gather_goes_out_with_it():
    """With one Mamba state column per draft position a finished request's
    state may wait for the next step's draft-column gather: the job names its
    Mamba blocks, and the step that emits it must gather into them."""
    scheduler = _bare_scheduler()
    scheduler._store_coord = SimpleNamespace(mamba_group_ids={1, 2})
    request = _Req(_tokens(3 * B))
    checkpoint = dataclasses.replace(_checkpoint(request, 2 * B + 37), gathered=True)
    scheduler.register_finished_frontier(request, checkpoint)
    (req_meta,) = scheduler._finished_frontier_metas
    assert req_meta.frontier is not None
    assert req_meta.frontier.gathered_block_ids == (30, 31)
    without = SimpleNamespace(mamba_state_gathers=None)
    with pytest.raises(AssertionError, match="without its state gather"):
        scheduler._emit_finished_frontiers(
            MooncakeStoreConnectorMetadata(set(), set()), without
        )
    gather = MambaStateGather(
        column=2, group_ids=(1, 2), dst_block_ids=(30, 31), src_block_ids=(32, 33)
    )
    meta = MooncakeStoreConnectorMetadata(set(), set())
    scheduler._emit_finished_frontiers(
        meta, SimpleNamespace(mamba_state_gathers=[gather])
    )
    assert meta.requests == [req_meta] and not scheduler._finished_frontier_metas


def test_short_frontier_or_write_through_off_writes_nothing():
    scheduler = _bare_scheduler()
    request = _Req(_tokens(B))
    short = FRONTIER_LOCATOR_TOKENS - 1
    scheduler.register_finished_frontier(request, _checkpoint(request, short))
    assert scheduler._finished_frontier_metas == []
    scheduler._frontiers = None
    scheduler.register_finished_frontier(request, _checkpoint(request, B - 5))
    assert scheduler._finished_frontier_metas == []


def test_load_resuming_at_a_stored_frontier_names_the_rings_it_writes():
    scheduler = _bare_scheduler()
    scheduler._unfinished_requests = {}
    scheduler._unfinished_request_ids = set()
    scheduler.load_specs["r"] = LoadSpec(
        vllm_cached_tokens=B,
        kvpool_cached_tokens=2 * B + 37,
        can_load=False,
        frontier=FrontierLoad(b"\x01" * 32, 2 * B + 37),
    )
    request = SimpleNamespace(request_id="r")
    assert scheduler.frontier_ring_groups(request) == (3,)
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([1, 2, 3], [0, 0, 5], [0, 0, 6], [7])
    scheduler.update_state_after_alloc(request, blocks, B + 37)
    assert scheduler.load_specs["r"].frontier.rings == ((3, 7),)
    # On a pool boundary the frontier holds no ring.
    scheduler.load_specs["r"].frontier = FrontierLoad(b"\x01" * 32, 2 * B + 36)
    assert scheduler.frontier_ring_groups(request) == ()


# The worker's probe, verify, write and load ---------------------------------
class _DictStore:
    """MooncakeDistributedStore over a dict; records the keys each call asks
    about, as every such call grants a lease."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.asked: list[str] = []

    def batch_is_exist(self, keys):
        self.asked.extend(keys)
        return [1 if key in self.objects else 0 for key in keys]

    def get(self, key):
        self.asked.append(key)
        return self.objects[key]

    def upsert(self, key, value, config):
        self.objects[key] = bytes(value)
        return 0

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, config):
        for key in keys:
            self.objects[key] = b"x"
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        return [1 if key in self.objects else -704 for key in keys]


def _probe_worker(store: _DictStore, newest: bool) -> MooncakeStoreWorker:
    worker = object.__new__(MooncakeStoreWorker)
    worker.store = store
    worker.frontiers = _frontier_store(0)
    worker._newest_state_lookup = newest
    worker.lookup = MagicMock(return_value=MooncakeLookupResult(B))
    worker._record_kv_connector_operation = MagicMock()
    return worker


@pytest.mark.parametrize("newest", [False, True])
def test_probe_runs_the_regular_lookup_unless_lookup_newest_state(newest):
    store = _DictStore()
    worker = _probe_worker(store, newest)
    locator = b"\x05" * 32
    key = worker.frontiers.locator_key(locator)
    store.objects[key] = fr.encode_locator([(2 * B + 37, b"\x01" * 32)])
    probe = worker.frontier_probe(3 * B, [], [b"\x04" * 32, locator])
    assert probe.entries == ((2 * B + 37, b"\x01" * 32),)
    # With lookup_newest_state a probe that found a frontier leases nothing
    # else; the default also runs the regular lookup.
    assert (probe.regular is None) is newest
    assert worker.lookup.called is not newest
    # No record: the regular lookup always runs.
    assert worker.frontier_probe(3 * B, [], [b"\x04" * 32]).regular is not None


def test_verify_checks_every_object_on_every_rank():
    store = _DictStore()
    worker = _probe_worker(store, newest=True)
    hashes = [bytes([i]) * 32 for i in range(12)]
    keys = worker.frontiers.verify_keys(hashes, b"\x01" * 32, 2 * B + 37, 1)
    store.objects.update(dict.fromkeys(keys[:-1], b"x"))
    assert worker.frontier_verify(3 * B, hashes, 2 * B + 37, 1, True, b"\x01" * 32) == (
        MooncakeLookupResult(B)
    )
    assert (
        worker.frontier_verify(3 * B, hashes, 2 * B + 37, 1, False, b"\x01" * 32)
        is None
    )
    store.objects[keys[-1]] = b"x"
    assert worker.frontier_verify(
        3 * B, hashes, 2 * B + 37, 1, False, b"\x01" * 32
    ) == MooncakeLookupResult(2 * B + 37, frontier=b"\x01" * 32)


def _coord() -> MooncakeStoreCoordinator:
    attn = FullAttentionSpec(
        block_size=B, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    state = MambaSpec(
        block_size=B, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align"
    )
    groups = [KVCacheGroupSpec(["attn"], attn)]
    groups += [KVCacheGroupSpec([f"m{i}"], state) for i in range(2)]
    return MooncakeStoreCoordinator(groups, B, U, enable_partial_hash_hits=True)


def _sending_thread(store, frontiers, **kwargs) -> KVCacheStoreSendingThread:
    import threading

    return KVCacheStoreSendingThread(
        store=store,
        token_databases=frontiers.store_dbs,
        block_size=B,
        coord=_coord(),
        tp_rank=0,
        group_put_steps=(1, 1, 1),
        kv_role="kv_both",
        ready_event=threading.Event(),
        enable_kv_event=False,
        frontiers=frontiers,
        locator_config=SimpleNamespace(),
        **kwargs,
    )


def _frontier_job() -> tuple[_Req, FrontierPut, ReqMeta]:
    """A finished request's store job: one boundary put and a frontier at
    ``2 * B + 37``."""
    request = _Req(_tokens(3 * B))
    length = 2 * B + 37
    put = FrontierPut(
        key=bytes(frontier_key(sha256, request, length, U)),
        length=length,
        locator=b"\x02" * 32,
        blocks=((0, 12), (1, 30), (2, 31)),
        rings=((5, 40),),
    )
    req_meta = ReqMeta(
        req_id="r",
        token_len_chunk=0,
        block_ids=([], [], []),
        block_hashes=request.block_hashes,
        can_save=True,
        boundary_puts=[BoundaryPut(0, 11, 2 * B)],
        completed_token_len=length,
        frontier=put,
        store_job_id=0,
    )
    return request, put, req_meta


def test_sending_thread_puts_missing_objects_then_updates_the_locator():
    store = _DictStore()
    frontiers = _frontier_store(0)
    thread = _sending_thread(store, frontiers)
    request, put, req_meta = _frontier_job()
    key, length = put.key, put.length
    already = frontiers.store_dbs[1].key_for(BlockHash(key))
    store.objects[already] = b"old"
    thread.add_request(req_meta)
    thread._handle_request(thread.request_queue.get())

    puts = frontiers.object_puts(put, 0, (1, 1, 1))
    assert all(p[0] in store.objects for p in puts)
    assert store.objects[already] == b"old"
    page = frontiers.store_dbs[0].key_for(request.block_hashes[2 * B // U - 1])
    assert page in store.objects
    record = store.objects[frontiers.locator_key(b"\x02" * 32)]
    assert fr.decode_locator(record) == [(length, key)]
    assert thread.take_completed_saves() == {0: 1}


# frontier_soft_pin_rest ------------------------------------------------------
def test_soft_pin_rest_option():
    wt = {"frontier_write_through": True}
    assert _config({**wt, "frontier_soft_pin_rest": True}).soft_pin_rest
    with pytest.raises(ValueError, match="need frontier_write_through"):
        _config({"frontier_soft_pin_rest": True})
    with pytest.raises(ValueError, match="must be a bool"):
        _config({**wt, "frontier_soft_pin_rest": "true"})
    pin = {**wt, "frontier_soft_pin_rest": True}
    with pytest.raises(ValueError, match="exclusive"):
        _config({**pin, "unit_policy_url": "http://h:1"})
    remove = {**pin, "frontier_supersede_remove": True}
    with pytest.raises(ValueError, match="needs frontier_supersede_states false"):
        _config(remove)
    config = _config({**remove, "frontier_supersede_states": False})
    assert config.supersede_remove and not config.supersede_states


class _ConfigStore(_DictStore):
    """_DictStore that records the ReplicateConfig each object was put under."""

    def __init__(self) -> None:
        super().__init__()
        self.configs: dict[str, object] = {}

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, config):
        self.configs.update(dict.fromkeys(keys, config))
        return super().batch_put_from_multi_buffers(keys, addrs, sizes, config)


@pytest.mark.parametrize("own_config", [False, True])
def test_frontier_objects_go_out_under_their_own_config(own_config):
    store = _ConfigStore()
    frontiers = _frontier_store(0)
    pinned, unpinned = SimpleNamespace(), SimpleNamespace()
    thread = _sending_thread(
        store,
        frontiers,
        replicate_config=pinned,
        frontier_replicate_config=unpinned if own_config else None,
    )
    request, put, req_meta = _frontier_job()
    thread.add_request(req_meta)
    thread._handle_request(thread.request_queue.get())

    puts = frontiers.object_puts(put, 0, (1, 1, 1))
    assert puts
    frontier_config = unpinned if own_config else pinned
    assert all(store.configs[key] is frontier_config for key, *_ in puts)
    page = frontiers.store_dbs[0].key_for(request.block_hashes[2 * B // U - 1])
    assert store.configs[page] is pinned


@pytest.mark.parametrize("missing", [False, True])
def test_frontier_load_is_one_batch_and_fails_whole(missing):
    import queue as queue_module

    store = _DictStore()
    frontiers = _frontier_store(1)
    thread = KVCacheStoreRecvingThread(
        store=store,
        token_databases=frontiers.store_dbs,
        block_size=B,
        coord=_coord(),
        tp_rank=1,
        ready_event=MagicMock(),
        request_queue=queue_module.Queue(),
        frontiers=frontiers,
    )
    load = FrontierLoad(b"\x01" * 32, 2 * B + 37, rings=((5, 30),))
    spec = LoadSpec(
        vllm_cached_tokens=B,
        kvpool_cached_tokens=2 * B + 37,
        can_load=True,
        frontier=load,
    )
    hashes = [bytes([i]) * 32 for i in range(12)]
    req_meta = ReqMeta(
        req_id="r",
        token_len_chunk=0,
        block_ids=([11, 12, 13], [0, 0, 21], [0, 0, 22]),
        block_hashes=hashes,
        load_spec=spec,
    )
    keys, _, _, _ = frontiers.load_plan(req_meta)
    store.objects.update(dict.fromkeys(keys[: -1 if missing else None], b"x"))
    thread.request_queue.put(req_meta)
    thread._handle_request(thread.request_queue.get())
    assert thread.get_and_clear_finished_requests() == {"r"}
    failed = thread.get_and_clear_block_ids_with_load_errors()
    assert failed == ({30} if missing else set())


# frontier_supersede_remove and frontier_lookup_memo_s -----------------------
def test_supersede_and_memo_options():
    wt = {"frontier_write_through": True}
    assert _config(wt) == fr.FrontierStoreConfig(write_through=True)
    both = {**wt, "frontier_supersede_remove": True, "frontier_lookup_memo_s": 30}
    assert _config(both) == fr.FrontierStoreConfig(
        write_through=True, supersede_remove=True, lookup_memo_s=30.0
    )
    for extra, message in (
        ({"frontier_supersede_remove": True}, "need frontier_write_through"),
        ({"frontier_lookup_memo_s": 5}, "need frontier_write_through"),
        ({**wt, "frontier_lookup_memo_s": -1}, ">= 0"),
        ({**wt, "frontier_lookup_memo_s": "5"}, ">= 0"),
        ({**wt, "frontier_lookup_memo_s": float("inf")}, ">= 0"),
        ({**wt, "frontier_supersede_remove": "yes"}, "must be a bool"),
        ({**wt, "frontier_lookup_reuse_s": 2}, "needs frontier_lookup_memo_s"),
        (
            {**wt, "frontier_lookup_memo_s": 30, "frontier_lookup_reuse_s": -1},
            ">= 0",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            _config(extra)
    memo = {**wt, "frontier_lookup_memo_s": 30}
    assert _config(memo).lookup_reuse_s == fr.VERIFY_REUSE_S
    assert _config({**memo, "frontier_lookup_reuse_s": 2}).lookup_reuse_s == 2.0


@pytest.mark.parametrize("length", [B + 37, 2 * B + 36, 2 * B + 37])
@pytest.mark.parametrize("rank", [0, 1])
def test_object_keys_are_the_keys_object_puts_writes(rank, length):
    key = b"\x01" * 32
    rings = () if length % 4 == 0 else ((5, 10),)
    put = FrontierPut(
        key=key,
        length=length,
        locator=b"\x02" * 32,
        blocks=((0, 7), (1, 8), (2, 9)),
        rings=rings,
    )
    store = _frontier_store(rank)
    assert store.object_keys(key, length, rank, (2, 1, 1)) == [
        p[0] for p in store.object_puts(put, rank, (2, 1, 1))
    ]


def test_removal_codes_are_counted():
    store = _frontier_store(0)
    store.note_removals([0, 0, -704, -706, -1])
    assert dict(store.removals) == {"removed": 2, "absent": 1, "leased": 1, "error": 1}


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(fr.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        fr, "frontier_locator_key", lambda hash_fn, request, h, unit: bytes([h]) * 8
    )
    monkeypatch.setattr(
        fr,
        "frontier_key",
        lambda hash_fn, request, length, unit: length.to_bytes(8, "big"),
    )
    return now


MEMO_REQ = SimpleNamespace(
    request_id="r", num_tokens=3 * B, block_hashes=[b"\x00" * 32] * 12
)
MEMO_FRONTIER = (2 * B + 37, (2 * B + 37).to_bytes(8, "big"))


def _memo_lookup(client, memo_s):
    return fr.FrontierLookup(client, B, U, sha256, memo_s=memo_s)


def test_without_the_memo_every_lookup_probes(clock):
    client = _StubClient([])
    lookup = _memo_lookup(client, 0.0)
    lookup.lookup(MEMO_REQ, B, False)
    lookup.lookup(MEMO_REQ, B, False)
    assert client.calls == ["probe", "probe"]


def test_a_request_without_a_frontier_is_not_probed_again_within_the_memo(clock):
    client = _StubClient([])
    lookup = _memo_lookup(client, 30.0)
    lookup.lookup(MEMO_REQ, B, False)
    clock[0] += 10
    lookup.lookup(MEMO_REQ, B, False)
    assert client.calls == ["probe", "lookup"]
    assert lookup.counts["memo_no_frontier"] == 1
    clock[0] += 25
    lookup.lookup(MEMO_REQ, B, False)
    assert client.calls[-1] == "probe"


def test_a_verified_frontier_is_reused_briefly_then_verified_again(clock):
    client = _StubClient([MEMO_FRONTIER])
    lookup = _memo_lookup(client, 30.0)
    first = lookup.lookup(MEMO_REQ, B, False)
    assert first.frontier == MEMO_FRONTIER[1]
    assert client.calls == ["probe", "verify"]
    clock[0] += 3
    assert lookup.lookup(MEMO_REQ, B, False) is first
    assert client.calls == ["probe", "verify"]
    # Past the reuse time the verify alone renews the leases.
    clock[0] += 2
    again = lookup.lookup(MEMO_REQ, B, False)
    assert again.frontier == MEMO_FRONTIER[1]
    assert client.calls == ["probe", "verify", "verify"]
    assert (lookup.counts["memo_hit"], lookup.counts["memo_verify"]) == (1, 1)


def test_the_reuse_window_is_the_configured_one(clock):
    client = _StubClient([MEMO_FRONTIER])
    lookup = fr.FrontierLookup(client, B, U, sha256, memo_s=30.0, reuse_s=1.0)
    lookup.lookup(MEMO_REQ, B, False)
    clock[0] += 0.5
    lookup.lookup(MEMO_REQ, B, False)
    clock[0] += 1
    lookup.lookup(MEMO_REQ, B, False)
    assert client.calls == ["probe", "verify", "verify"]


def test_the_memo_can_be_pruned_while_another_thread_forgets():
    lookup = fr.FrontierLookup(None, B, U, sha256, memo_s=1e9)
    hit = MooncakeLookupResult(0)
    for i in range(5000):
        lookup._remember(f"r{i}", 0, None, hit)
    errors: list[BaseException] = []
    stop = threading.Event()

    def forget():
        i = 0
        while not stop.is_set():
            lookup.forget(f"r{i % 5000}")
            lookup._remember(f"r{i % 5000}", 0, None, hit)
            i += 1

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    forgetter = threading.Thread(target=forget)
    forgetter.start()
    try:
        # Every call past 4,096 entries walks the memo to prune it.
        for i in range(300):
            lookup._remember(f"x{i}", 0, None, hit)
    except RuntimeError as e:
        errors.append(e)
    finally:
        stop.set()
        forgetter.join()
        sys.setswitchinterval(interval)
    assert not errors


def test_a_frontier_that_no_longer_verifies_falls_back_and_is_forgotten(clock):
    client = _StubClient([MEMO_FRONTIER])
    lookup = _memo_lookup(client, 30.0)
    lookup.lookup(MEMO_REQ, B, False)
    client.complete = False
    clock[0] += 5
    assert lookup.lookup(MEMO_REQ, B, False).frontier is None
    clock[0] += 1
    lookup.lookup(MEMO_REQ, B, False)
    assert client.calls == ["probe", "verify", "verify", "lookup"]


def test_a_new_local_hit_or_a_forgotten_request_probes_again(clock):
    client = _StubClient([])
    lookup = _memo_lookup(client, 30.0)
    lookup.lookup(MEMO_REQ, B, False)
    lookup.lookup(MEMO_REQ, 2 * B, False)
    lookup.forget("r")
    lookup.lookup(MEMO_REQ, 2 * B, False)
    assert client.calls == ["probe", "probe", "probe"]


def _supersede_scheduler() -> MooncakeStoreScheduler:
    scheduler = _bare_scheduler()
    scheduler._supersede = True
    return scheduler


@pytest.mark.parametrize("states", [True, False])
def test_a_finish_supersedes_the_frontier_it_resumed_from_and_later_states(states):
    scheduler = _supersede_scheduler()
    scheduler._supersede_states = states
    request = _Req(_tokens(4 * B))
    pool = scheduler._gpu_block_pool
    # Resumed from a stored frontier at B + 37 that the lookup found.
    resumed_key = bytes(frontier_key(sha256, request, B + 37, U))
    scheduler._note_resumed_frontier(
        request, B, MooncakeLookupResult(B + 37, frontier=resumed_key)
    )
    # Its prefill stored the states at B + 48 and 2 * B; one earlier state
    # and one past the new frontier are not superseded.
    scheduler._boundary_tokens["r"] = {(1, B), (1, B + 48), (2, 2 * B), (1, 3 * B)}
    pool.touch([pool.blocks[i] for i in (10, 11, 12, 30, 31, 40)])
    scheduler.register_finished_frontier(request, _checkpoint(request, 2 * B + 37))
    (req_meta,) = scheduler._finished_frontier_metas
    assert req_meta.frontier.supersedes == ((resumed_key, B + 37),)
    # frontier_supersede_states false keeps the boundary states.
    assert req_meta.frontier.superseded_states == (
        ((1, B + 48), (2, 2 * B)) if states else ()
    )
    # A resume without a stored frontier supersedes nothing.
    scheduler._note_resumed_frontier(request, 2 * B, MooncakeLookupResult(2 * B))
    assert "r" not in scheduler._resumed_frontiers


class _RemovingStore(_DictStore):
    def __init__(self, leased: set[str]) -> None:
        super().__init__()
        self.leased = leased
        self.removed: list[str] = []

    def batch_remove(self, keys, force=False):
        codes = []
        for key in keys:
            if key not in self.objects:
                codes.append(-704)
            elif key in self.leased:
                codes.append(-706)
            else:
                del self.objects[key]
                self.removed.append(key)
                codes.append(0)
        return codes


def test_writing_a_frontier_removes_this_ranks_share_of_what_it_supersedes():
    import threading

    frontiers = _frontier_store(0)
    request = _Req(_tokens(4 * B))
    old_key = bytes(frontier_key(sha256, request, B + 37, U))
    old_keys = frontiers.object_keys(old_key, B + 37, 0, (1, 1, 1))
    state = frontiers.store_dbs[1].key_for(request.block_hashes[(B + 48) // U - 1])
    store = _RemovingStore(leased={old_keys[0]})
    store.objects.update(dict.fromkeys([*old_keys, state], b"x"))
    thread = KVCacheStoreSendingThread(
        store=store,
        token_databases=frontiers.store_dbs,
        block_size=B,
        coord=_coord(),
        tp_rank=0,
        group_put_steps=(1, 1, 1),
        kv_role="kv_both",
        ready_event=threading.Event(),
        enable_kv_event=False,
        frontiers=frontiers,
        locator_config=SimpleNamespace(),
    )
    put = FrontierPut(
        key=bytes(frontier_key(sha256, request, 2 * B + 37, U)),
        length=2 * B + 37,
        locator=b"\x02" * 32,
        blocks=((0, 12), (1, 30), (2, 31)),
        rings=((5, 40),),
        supersedes=((old_key, B + 37),),
        superseded_states=((1, B + 48),),
    )
    req_meta = ReqMeta(
        req_id="r",
        token_len_chunk=0,
        block_ids=([], [], []),
        block_hashes=request.block_hashes,
        can_save=True,
        completed_token_len=2 * B + 37,
        frontier=put,
        store_job_id=0,
    )
    thread.add_request(req_meta)
    thread._handle_request(thread.request_queue.get())
    # A leased object (a reader checked it) stays for the store to evict.
    assert old_keys[0] in store.objects
    assert sorted(store.removed) == sorted([*old_keys[1:], state])
    assert frontiers.removals == {"removed": len(old_keys), "leased": 1}


# frontier_load_local_pages ---------------------------------------------------
def test_load_local_pages_option():
    wt = {"frontier_write_through": True}
    assert _config({**wt, "frontier_load_local_pages": True}).load_local_pages
    with pytest.raises(ValueError, match="need frontier_write_through"):
        _config({"frontier_load_local_pages": True})


def test_a_frontier_load_reports_its_length_and_moves_its_start():
    scheduler = _bare_scheduler()
    request = SimpleNamespace(request_id="r")
    scheduler.load_specs["r"] = LoadSpec(
        vllm_cached_tokens=0,
        kvpool_cached_tokens=2 * B + 37,
        can_load=False,
        frontier=FrontierLoad(b"\x01" * 32, 2 * B + 37),
    )
    # Off: the scheduler never moves the load.
    assert scheduler.frontier_load_length(request) is None
    scheduler._load_local_pages = True
    assert scheduler.frontier_load_length(request) == 2 * B + 37
    scheduler.move_frontier_load_start(request, 2 * B)
    assert scheduler.load_specs["r"].vllm_cached_tokens == 2 * B
    # Not past the lookup's local hit, off the block grid, or past the
    # frontier's block start: refused.
    for num_tokens in (B, 2 * B + 3, 3 * B):
        with pytest.raises(AssertionError):
            scheduler.move_frontier_load_start(request, num_tokens)
    # A regular load is not a frontier load.
    scheduler.load_specs["r"] = LoadSpec(
        vllm_cached_tokens=0, kvpool_cached_tokens=2 * B, can_load=False
    )
    assert scheduler.frontier_load_length(request) is None


# frontier_write_wait_s -------------------------------------------------------
def test_write_wait_option():
    wt = {"frontier_write_through": True}
    assert _config({**wt, "frontier_write_wait_s": 3}).write_wait_s == 3.0
    with pytest.raises(ValueError, match="need frontier_write_through"):
        _config({"frontier_write_wait_s": 3})


def test_a_request_waits_while_its_own_frontier_is_being_written(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(fr.time, "monotonic", lambda: now[0])
    scheduler = _bare_scheduler()
    scheduler._write_wait_s = 3.0
    scheduler.enable_lookup = True
    scheduler.enable_partial_hash_hits = True
    scheduler.lookup_async = False
    writer = _Req(_tokens(3 * B), request_id="w")
    length = 2 * B + 37
    pool = scheduler._gpu_block_pool
    pool.touch([pool.blocks[i] for i in (10, 11, 12, 30, 31, 40)])
    scheduler.register_finished_frontier(writer, _checkpoint(writer, length))
    (req_meta,) = scheduler._finished_frontier_metas
    reader = _Req(_tokens(3 * B) + [5] * 50, request_id="r")
    counts = scheduler._frontiers.counts
    assert scheduler.defer_lookup(reader, B)
    assert scheduler.defer_lookup(reader, B)
    assert (counts["waited"], counts["wait_steps"]) == (1, 2)
    # A request that holds the frontier on the GPU, or another session's
    # request, does not wait.
    assert not scheduler.defer_lookup(reader, length)
    assert not scheduler.defer_lookup(_Req(_tokens(3 * B, seed=5), "o"), B)
    # Its next lookup is counted against the wait.
    scheduler.get_num_new_matched_tokens(reader, B)
    assert counts["waited_misses"] == 1
    # The wait ends once every rank completed the write.
    assert scheduler.defer_lookup(reader, B)
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
                completed_saves={req_meta.store_job_id: 1}
            )
        )
    )
    assert scheduler._writing_frontiers == {}
    assert not scheduler.defer_lookup(reader, B)
    # Or at its limit, after which the frontier is no longer tracked.
    pool.touch([pool.blocks[i] for i in (10, 11, 12, 30, 31, 40)])
    scheduler.register_finished_frontier(writer, _checkpoint(writer, length))
    assert scheduler.defer_lookup(reader, B)
    now[0] += 3.0
    assert not scheduler.defer_lookup(reader, B)
    assert counts["wait_timeouts"] == 1
    assert scheduler._writing_frontiers == {}


def test_a_late_frontier_neither_hides_a_fresh_one_nor_stays(monkeypatch):
    """A request extends two frontiers of its session in flight: the first
    finished past the wait, the second within it. The request waits for the
    second, and the first is no longer scanned."""
    now = [1000.0]
    monkeypatch.setattr(fr.time, "monotonic", lambda: now[0])
    scheduler = _bare_scheduler()
    scheduler._write_wait_s = 3.0
    writer = _Req(_tokens(3 * B), request_id="w")
    pool = scheduler._gpu_block_pool
    pool.touch([pool.blocks[i] for i in (10, 11, 12, 30, 31, 40)])
    scheduler.register_finished_frontier(writer, _checkpoint(writer, B + 20))
    now[0] += 5.0
    pool.touch([pool.blocks[i] for i in (10, 11, 12, 30, 31, 40)])
    scheduler.register_finished_frontier(writer, _checkpoint(writer, 2 * B + 37))
    reader = _Req(_tokens(3 * B) + [5] * 50, request_id="r")
    assert scheduler.defer_lookup(reader, B)
    assert [length for length, _, _ in scheduler._writing_frontiers.values()] == [
        2 * B + 37
    ]


def test_supersede_states_option():
    remove = {"frontier_write_through": True, "frontier_supersede_remove": True}
    assert _config(remove).supersede_states
    assert not _config({**remove, "frontier_supersede_states": False}).supersede_states
    for extra, message in (
        ({"frontier_write_through": True, "frontier_supersede_states": False}, "needs"),
        ({**remove, "frontier_supersede_states": "no"}, "must be a bool"),
    ):
        with pytest.raises(ValueError, match=message):
            _config(extra)


# frontier_supersede_lag ------------------------------------------------------
def test_supersede_lag_option():
    remove = {"frontier_write_through": True, "frontier_supersede_remove": True}
    assert _config({**remove, "frontier_supersede_lag": 2}).supersede_lag == 2
    for extra, message in (
        ({"frontier_write_through": True, "frontier_supersede_lag": 2}, "needs"),
        ({**remove, "frontier_supersede_lag": -1}, "integer >= 0"),
        ({**remove, "frontier_supersede_lag": 1.5}, "integer >= 0"),
        ({**remove, "frontier_supersede_lag": True}, "integer >= 0"),
    ):
        with pytest.raises(ValueError, match=message):
            _config(extra)


def test_a_finish_supersedes_what_the_finish_lag_turns_earlier_would_have():
    """A session's chain of turns, each resuming from the previous turn's
    frontier: with a lag of 2 each finish supersedes what the finish two turns
    earlier would have, so the frontiers and states of the turns between stay
    live for rewrites."""
    scheduler = _supersede_scheduler()
    scheduler._supersede_lag = 2
    tokens = _tokens(6 * B)
    pool = scheduler._gpu_block_pool
    lengths = [B + 9, 2 * B + 9, 3 * B + 9, 4 * B + 9, 5 * B + 9]
    emitted = []
    for turn, length in enumerate(lengths):
        request = _Req(tokens[: length + 1], request_id=f"t{turn}")
        if turn:
            previous = lengths[turn - 1]
            key = bytes(frontier_key(sha256, request, previous, U))
            scheduler._note_resumed_frontier(
                request, previous // B * B, MooncakeLookupResult(previous, frontier=key)
            )
            scheduler._boundary_tokens[request.request_id] = {(1, length // B * B)}
        pool.touch([pool.blocks[i] for i in range(10, 10 + length // B + 1)])
        pool.touch([pool.blocks[i] for i in (30, 31, 40)])
        scheduler.register_finished_frontier(request, _checkpoint(request, length))
        put = scheduler._finished_frontier_metas[-1].frontier
        emitted.append((put.supersedes, put.superseded_states))
    # Without the lag, turn t would supersede turn t - 1's frontier; the
    # first turn resumed from none.
    assert emitted[:3] == [((), ())] * 3
    first = bytes(frontier_key(sha256, _Req(tokens), lengths[0], U))
    assert emitted[3] == (((first, lengths[0]),), ((1, 2 * B),))
    second = bytes(frontier_key(sha256, _Req(tokens), lengths[1], U))
    assert emitted[4] == (((second, lengths[1]),), ((1, 3 * B),))
    # A request that resumed from no frontier starts a chain of its own.
    request = _Req(_tokens(3 * B, seed=4), request_id="new")
    pool.touch([pool.blocks[i] for i in (10, 11, 12, 30, 31, 40)])
    scheduler.register_finished_frontier(request, _checkpoint(request, 2 * B + 9))
    assert scheduler._finished_frontier_metas[-1].frontier.supersedes == ()


# frontier_keep_states and frontier_remove_retry_s --------------------------
def test_keep_states_and_remove_retry_options():
    wt = {"frontier_write_through": True}
    config = _config({**wt, "frontier_keep_states": 2, "frontier_remove_retry_s": 31})
    assert config.keep_states == 2 and config.remove_retry_s == 31.0
    remove = {**wt, "frontier_supersede_remove": True}
    assert _config({**remove, "frontier_remove_retry_s": 31}).remove_retry_s == 31.0
    for extra, message in (
        ({"frontier_keep_states": 2}, "needs frontier_write_through"),
        ({**wt, "frontier_keep_states": -1}, "integer >= 0"),
        ({**wt, "frontier_keep_states": 1.5}, "integer >= 0"),
        ({**wt, "frontier_keep_states": True}, "integer >= 0"),
        ({**remove, "frontier_keep_states": 2}, "are exclusive"),
        (
            {**wt, "frontier_soft_pin_rest": True, "frontier_keep_states": 2},
            "are exclusive",
        ),
        ({**wt, "frontier_remove_retry_s": 31}, "needs frontier_supersede_remove or"),
    ):
        with pytest.raises(ValueError, match=message):
            _config(extra)


def test_a_session_chain_keeps_its_newest_checkpoints_wherever_its_turns_resume():
    """A session's turns, each extending the previous turn's frontier with no
    stored frontier loaded (they resume from the GPU): each finish keeps the
    chain's two deepest checkpoints and removes the others, its own boundary
    states included, but leaves the states within the first prompt, which
    other sessions' first prompts may share, to the store's eviction."""
    scheduler = _bare_scheduler()
    scheduler._keep_states = 2
    tokens = _tokens(6 * B)
    pool = scheduler._gpu_block_pool
    # (prompt length, frontier length, boundary states the turn stored)
    turns = [
        (2 * B + 4, 2 * B + 9, {(1, B), (2, B), (1, 2 * B)}),
        (4 * B + 4, 4 * B + 9, {(1, 3 * B), (1, 4 * B)}),
        (4 * B + 40, 4 * B + 50, set()),
    ]
    emitted = []
    for turn, (prompt, length, states) in enumerate(turns):
        request = _Req(tokens[: length + 1], request_id=f"t{turn}")
        request.num_prompt_tokens = prompt
        scheduler._boundary_tokens[request.request_id] = states
        pool.touch([pool.blocks[i] for i in range(10, 10 + length // B + 1)])
        pool.touch([pool.blocks[i] for i in (30, 31, 40)])
        scheduler.register_finished_frontier(request, _checkpoint(request, length))
        put = scheduler._finished_frontier_metas[-1].frontier
        emitted.append((put.supersedes, put.superseded_states))
    keys = [bytes(frontier_key(sha256, _Req(tokens), n, U)) for _, n, _ in turns]
    # The first turn's states lie within its prompt.
    assert emitted[0] == ((), ())
    # The second keeps its own two deepest and removes the first frontier.
    assert emitted[1] == (((keys[0], turns[0][1]),), ((1, 3 * B),))
    # A turn that stored no state keeps the frontier before its own.
    assert emitted[2] == ((), ((1, 4 * B),))
    assert scheduler._chains[keys[2]][2] == [
        (turns[1][1], keys[1], ()),
        (turns[2][1], keys[2], ()),
    ]
    # A rewrite of the last turn extends no newest frontier: it starts a chain
    # of its own and leaves the session's alone.
    rewrite = _Req(tokens[: 4 * B + 20] + _tokens(B, seed=5), request_id="rw")
    rewrite.num_prompt_tokens = 5 * B + 4
    pool.touch([pool.blocks[i] for i in (*range(10, 16), 30, 31, 40)])
    scheduler.register_finished_frontier(rewrite, _checkpoint(rewrite, 5 * B + 9))
    assert scheduler._finished_frontier_metas[-1].frontier.supersedes == ()
    assert set(scheduler._chains) == {
        keys[2],
        bytes(frontier_key(sha256, rewrite, 5 * B + 9, U)),
    }


def test_keep_margin_option():
    keep = {"frontier_write_through": True, "frontier_keep_states": 2}
    config = _config({**keep, "frontier_keep_margin_tokens": 1024})
    assert config.keep_margin_tokens == 1024
    for extra, message in (
        ({**keep, "frontier_keep_states": 1, "frontier_keep_margin_tokens": 8}, ">= 2"),
        ({**keep, "frontier_keep_margin_tokens": -1}, "integer >= 0"),
        ({**keep, "frontier_keep_margin_tokens": 1.5}, "integer >= 0"),
    ):
        with pytest.raises(ValueError, match=message):
            _config(extra)


@pytest.mark.parametrize("margin", [0, B])
def test_a_margin_keeps_an_older_checkpoint_below_the_prompt_end(margin):
    """A short turn crosses a block boundary just under its prompt end. A
    client that rewrites the end of that prompt resumes below the boundary, so
    with a margin the chain keeps the previous frontier instead."""
    scheduler = _bare_scheduler()
    scheduler._keep_states = 2
    scheduler._keep_margin = margin
    tokens = _tokens(6 * B)
    pool = scheduler._gpu_block_pool
    # (prompt length, frontier length, boundary states the turn stored)
    turns = [(2 * B + 4, 2 * B + 9, set()), (4 * B + 30, 4 * B + 40, {(1, 4 * B)})]
    for turn, (prompt, length, states) in enumerate(turns):
        request = _Req(tokens[: length + 1], request_id=f"t{turn}")
        request.num_prompt_tokens = prompt
        scheduler._boundary_tokens[request.request_id] = states
        pool.touch([pool.blocks[i] for i in range(10, 10 + length // B + 1)])
        pool.touch([pool.blocks[i] for i in (30, 31, 40)])
        scheduler.register_finished_frontier(request, _checkpoint(request, length))
    keys = [bytes(frontier_key(sha256, _Req(tokens), n, U)) for _, n, _ in turns]
    put = scheduler._finished_frontier_metas[-1].frontier
    if margin:
        # The state 30 tokens under the prompt end goes; the first frontier,
        # further down, stays.
        assert (put.supersedes, put.superseded_states) == ((), ((1, 4 * B),))
        assert [c[0] for c in scheduler._chains[keys[1]][2]] == [2 * B + 9, 4 * B + 40]
    else:
        assert (put.supersedes, put.superseded_states) == (((keys[0], 2 * B + 9),), ())
        assert [c[0] for c in scheduler._chains[keys[1]][2]] == [4 * B, 4 * B + 40]


def test_a_removal_refused_under_a_lease_is_tried_again_once_due():
    frontiers = _frontier_store(0)
    frontiers.remove_retry_s = 0.001
    request = _Req(_tokens(4 * B))
    first, second = (
        frontiers.store_dbs[1].key_for(request.block_hashes[n // U - 1])
        for n in (B, 2 * B)
    )
    store = _RemovingStore(leased={first})
    store.objects.update(dict.fromkeys([first, second], b"x"))
    thread = KVCacheStoreSendingThread(
        store=store,
        token_databases=frontiers.store_dbs,
        block_size=B,
        coord=_coord(),
        tp_rank=0,
        group_put_steps=(1, 1, 1),
        kv_role="kv_both",
        ready_event=threading.Event(),
        enable_kv_event=False,
        frontiers=frontiers,
        locator_config=SimpleNamespace(),
    )

    def job(states: tuple[tuple[int, int], ...]) -> ReqMeta:
        put = FrontierPut(
            key=b"\x01" * 32,
            length=3 * B + 9,
            locator=b"\x02" * 32,
            blocks=(),
            rings=(),
            superseded_states=states,
        )
        return ReqMeta(
            req_id="r",
            token_len_chunk=0,
            block_ids=([], [], []),
            block_hashes=request.block_hashes,
            can_save=True,
            frontier=put,
        )

    thread._remove_superseded(job(((1, B),)))
    assert first in store.objects
    assert frontiers.removals == {"leased": 1, "deferred": 1}
    # The lease runs out; the next job's removal takes the refused key along.
    store.leased.clear()
    time.sleep(0.01)
    thread._remove_superseded(job(((1, 2 * B),)))
    assert store.removed == [second, first]
    assert frontiers.removals == {"leased": 1, "deferred": 1, "removed": 2}


# Debug log of frontier lookups ----------------------------------------------
def test_each_lookup_logs_its_path_at_debug_level(caplog_vllm):
    import logging

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
        scheduler as scheduler_module,
    )

    caplog_vllm.set_level(logging.DEBUG, logger=fr.logger.name)
    caplog_vllm.set_level(logging.DEBUG, logger=scheduler_module.logger.name)
    request = _Req(_tokens(4 * B))
    key = bytes(frontier_key(sha256, request, 2 * B + 37, U))
    scheduler = _bare_scheduler()
    scheduler._frontiers = fr.FrontierLookup(
        _StubClient([(2 * B + 37, key)]), B, U, sha256
    )
    scheduler.enable_lookup = True
    scheduler.enable_partial_hash_hits = True
    scheduler.lookup_async = False
    scheduler.load_async = True
    scheduler.get_num_new_matched_tokens(request, B)
    (line,) = [
        r.getMessage()
        for r in caplog_vllm.records
        if r.getMessage().startswith("Mooncake frontier lookup:")
    ]
    assert f"local={B} path=probe:hit" in line
    assert line.endswith(f"hit={2 * B + 37} frontier=1")
    # A frontier load whose start moves logs it too.
    scheduler._load_local_pages = True
    scheduler.move_frontier_load_start(request, 2 * B)
    assert any(
        r.getMessage().startswith("Mooncake frontier load: req=r starts at 128")
        for r in caplog_vllm.records
    )


def test_an_asynchronous_lookup_logs_the_time_it_took(caplog_vllm):
    import logging

    caplog_vllm.set_level(logging.DEBUG, logger=fr.logger.name)
    request = _Req(_tokens(4 * B))
    key = bytes(frontier_key(sha256, request, 2 * B + 37, U))
    client = _AsyncStubClient([(2 * B + 37, key)])
    lookup = fr.FrontierLookup(client, B, U, sha256)
    # The lookup itself takes 50 ms on the client's thread.
    real_probe = lookup._probe

    def slow_probe(request, local):
        time.sleep(0.05)
        return real_probe(request, local)

    lookup._probe = slow_probe
    while lookup.lookup(request, B, True) is None:
        time.sleep(0.005)
    path, elapsed_ms = lookup.paths.pop("r")
    assert path == "probe:hit" and elapsed_ms >= 50
    # A forgotten request leaves no path behind.
    lookup._note(request, "probe:hit")
    lookup.forget("r")
    assert not lookup._path_of
    client.executor.shutdown()


# frontier_locator_cache ------------------------------------------------------
def test_locator_cache_option():
    wt = {"frontier_write_through": True}
    assert _config({**wt, "frontier_locator_cache": True}).locator_cache
    with pytest.raises(ValueError, match="need frontier_write_through"):
        _config({"frontier_locator_cache": True})


class _CountingStore(_DictStore):
    def __init__(self) -> None:
        super().__init__()
        self.gets = 0
        self.upsert_code = 0

    def get(self, key):
        self.gets += 1
        return super().get(key)

    def upsert(self, key, value, config):
        if self.upsert_code:
            return self.upsert_code
        return super().upsert(key, value, config)


def _cached_frontier_store(writes_locators: bool = True) -> fr.FrontierStore:
    store = _frontier_store(0)
    return fr.FrontierStore(
        store.store_dbs,
        store.lookup_prefixes,
        store.ring_dbs,
        store.ring_prefixes,
        store.mamba_group_ids,
        B,
        U,
        4,
        "m",
        writes_locators=writes_locators,
        cache_locators=True,
    )


def test_the_locator_writer_reads_its_own_last_record():
    store = _CountingStore()
    frontiers = _cached_frontier_store()
    key = frontiers.locator_key(b"\x02" * 32)
    store.objects[key] = b"store copy"
    assert frontiers.read_record(store, key) == b"store copy"
    frontiers.remember_record(key, b"own copy")
    assert frontiers.read_record(store, key) == b"own copy"
    assert store.gets == 1
    assert frontiers.record_reads == {"store": 1, "cached": 1}
    # An unknown outcome (a failed upsert) forgets the record.
    frontiers.remember_record(key, None)
    assert frontiers.read_record(store, key) == b"store copy"
    # Only the rank that writes locators caches them.
    reader = _cached_frontier_store(writes_locators=False)
    reader.remember_record(key, b"own copy")
    assert reader.read_record(store, key) == b"store copy"


def test_the_locator_cache_keeps_the_newest_records(monkeypatch):
    monkeypatch.setattr(fr, "LOCATOR_CACHE_RECORDS", 2)
    frontiers = _cached_frontier_store()
    for i in range(3):
        frontiers.remember_record(f"k{i}", bytes([i]))
    assert list(frontiers._records) == ["k1", "k2"]


def _locator_thread(store, frontiers):
    import threading

    return KVCacheStoreSendingThread(
        store=store,
        token_databases=frontiers.store_dbs,
        block_size=B,
        coord=_coord(),
        tp_rank=0,
        group_put_steps=(1, 1, 1),
        kv_role="kv_both",
        ready_event=threading.Event(),
        enable_kv_event=False,
        frontiers=frontiers,
        locator_config=SimpleNamespace(),
    )


def _locator_put(length: int, locator: bytes) -> FrontierPut:
    return FrontierPut(
        key=bytes([length]) * 32, length=length, locator=locator, blocks=(), rings=()
    )


def test_merging_reads_the_store_and_probing_the_cached_record():
    store = _CountingStore()
    frontiers = _cached_frontier_store()
    thread = _locator_thread(store, frontiers)
    locator = b"\x02" * 32
    for length in (B + 20, B + 40):
        thread._add_to_locator(_locator_put(length, locator))
    key = frontiers.locator_key(locator)
    expected = [(B + 40, bytes([B + 40]) * 32), (B + 20, bytes([B + 20]) * 32)]
    assert fr.decode_locator(store.objects[key]) == expected
    # The second merge read the store's copy.
    assert store.gets == 1
    worker = _probe_worker(store, newest=True)
    worker.frontiers = frontiers
    assert list(worker.frontier_probe(3 * B, [], [locator]).entries) == expected
    # The probe read this rank's own record.
    assert store.gets == 1


def test_engines_sharing_a_store_keep_each_others_frontiers():
    store = _CountingStore()
    first, second = _cached_frontier_store(), _cached_frontier_store()
    a, b = _locator_thread(store, first), _locator_thread(store, second)
    locator = b"\x02" * 32
    a._add_to_locator(_locator_put(B + 20, locator))
    b._add_to_locator(_locator_put(B + 40, locator))
    a._add_to_locator(_locator_put(B + 60, locator))
    key = first.locator_key(locator)
    assert [length for length, _ in fr.decode_locator(store.objects[key])] == [
        B + 60,
        B + 40,
        B + 20,
    ]
