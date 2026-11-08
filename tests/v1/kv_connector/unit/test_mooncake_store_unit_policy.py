# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MooncakeStoreConnector with ``unit_policy_url``: recovery units registered
with a Mooncake master that evicts by unit."""

import json
import threading
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    frontier as fr,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    unit_policy as up,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BoundaryPut,
    ChunkedTokenDatabase,
    FrontierPut,
    KeyMetadata,
    MooncakeStoreConnectorMetadata,
    PoolKey,
    ReqMeta,
    RequestTracker,
    StoreUnit,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (  # noqa: E501
    MooncakeStoreScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (  # noqa: E501
    KVCacheStoreSendingThread,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.hybrid_checkpoint.cost import PrefillCostModel
from vllm.v1.core.hybrid_checkpoint.frontier import FrontierCheckpoint
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
)

B, U = 64, 16
URL = "http://127.0.0.1:9003"


def _vllm_config(extra: dict, coefficients=(1e-5, 0.0)):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        cache_config=SimpleNamespace(
            mamba_frontier_checkpoint=True, prefill_cost_coefficients=coefficients
        ),
        speculative_config=None,
    )


def test_unit_policy_url_and_its_refusals():
    wt = {"frontier_write_through": True}
    config = fr.FrontierStoreConfig.from_vllm_config(
        _vllm_config({**wt, "unit_policy_url": URL})
    )
    assert config.unit_policy_url == URL
    assert (
        fr.FrontierStoreConfig.from_vllm_config(_vllm_config(wt)).unit_policy_url
        is None
    )
    for extra, coefficients, message in (
        ({"unit_policy_url": URL}, (1e-5, 0.0), "needs frontier_write_through"),
        ({**wt, "unit_policy_url": URL}, None, "needs prefill_cost_coefficients"),
        ({**wt, "unit_policy_url": "127.0.0.1:9003"}, (1e-5, 0.0), "http://<host>"),
        ({**wt, "unit_policy_url": 9003}, (1e-5, 0.0), "http://<host>"),
    ):
        with pytest.raises(ValueError, match=message):
            fr.FrontierStoreConfig.from_vllm_config(_vllm_config(extra, coefficients))
    assert up.unit_policy_refusal(True, False) is None
    assert "group_ids" in up.unit_policy_refusal(False, False)
    assert "enable_group_semantics" in up.unit_policy_refusal(True, True)


# What a unit names ----------------------------------------------------------
def _db(group: int, tp_rank: int, namespace: str = "") -> ChunkedTokenDatabase:
    meta = KeyMetadata("m", tp_rank, 0, 0, 0, group_id=group, store_namespace=namespace)
    db = ChunkedTokenDatabase(meta, B, U)
    db.set_kv_caches_base_addr([1 << 20])
    db.set_block_len([64])
    return db


def _frontier_store() -> fr.FrontierStore:
    """A replicated attention group 0, Mamba groups 1 and 2 per rank, and an
    indexer ring in KV-cache group 5."""
    dbs = [_db(0, 0), _db(1, 0), _db(2, 0)]
    prefixes = (
        (PoolKey.build_prefix(dbs[0].metadata, tp_rank=0),),
        tuple(PoolKey.build_prefix(dbs[1].metadata, tp_rank=r) for r in range(2)),
        tuple(PoolKey.build_prefix(dbs[2].metadata, tp_rank=r) for r in range(2)),
    )
    ring = _db(5, 0, "@ring")
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
        writes_locators=True,
    )


def _units() -> up.StoreUnits:
    return up.StoreUnits(_frontier_store(), PrefillCostModel(1e-5, 0.0))


HASHES = [bytes([i]) * 32 for i in range(40)]


def test_an_aligned_state_owns_its_states_and_references_the_pages_before_it():
    update = _units().upsert(StoreUnit(HASHES[7], 2 * B, HASHES[3]), HASHES)
    key = HASHES[7].hex()
    assert update["op"] == "upsert" and update["id"] == f"m@unit@{key}"
    assert sorted(update["members"]) == sorted(
        f"m@tp_rank:{r}@pcp0@dcp0@pp_rank:0@group:{g}@{key}"
        for g in (1, 2)
        for r in range(2)
    )
    assert update["refs"] == [
        f"m@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@{HASHES[i].hex()}" for i in (3, 7)
    ]
    assert update["aligned"] is True
    assert update["value"] == pytest.approx(2 * B * 1e-5)
    assert update["parent"] == f"m@unit@{HASHES[3].hex()}"


def test_a_partial_tail_owns_its_page_and_a_frontier_its_rings_and_locator():
    units = _units()
    partial = units.upsert(StoreUnit(HASHES[9], 10 * U, None), HASHES)
    assert any("@group:0@" in key for key in partial["members"])
    assert partial["aligned"] is False and partial["parent"] == ""
    assert len(partial["refs"]) == 2
    key = b"\xaa" * 32
    frontier = units.upsert(
        StoreUnit(key, 10 * U + 7, HASHES[9], frontier=True),
        HASHES,
        locator=b"\xbb" * 32,
    )
    # The page, two Mamba groups on two ranks, the ring on two ranks.
    assert len(frontier["members"]) == 1 + 2 * 2 + 2
    assert all(member.endswith(key.hex()) for member in frontier["members"])
    assert frontier["refs"][-1] == "m@frontier_locator@" + "bb" * 32
    assert len(frontier["refs"]) == 3
    assert units.touch(key) == {"op": "touch", "id": "m@unit@" + "aa" * 32}
    assert units.dead(key) == {"op": "dead", "id": "m@unit@" + "aa" * 32}


# The client -----------------------------------------------------------------
class _Master:
    """The master's admin port: records each POST body; answers ``status``,
    or each POST with the next of ``replies`` (raw bodies) while any is left.
    A POST waits for ``gate``."""

    def __init__(self, status: int = 200, replies: Sequence[bytes] = ()) -> None:
        bodies: list[dict] = []
        self.bodies = bodies
        self.status = status
        self.replies = list(replies)
        self.gate = threading.Event()
        self.gate.set()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _reply(self, code, payload):
                data = payload if isinstance(payload, bytes) else json.dumps(payload)
                data = data if isinstance(data, bytes) else data.encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                self._reply(
                    200,
                    {"units": 0, "objects": 0, "checked": "check=1" in self.path},
                )

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                bodies.append(body)
                outer.gate.wait()
                if outer.replies:
                    self._reply(200, outer.replies.pop(0))
                    return
                self._reply(
                    outer.status, {"applied": len(body["updates"]), "errors": []}
                )

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


def test_the_client_sends_every_update_in_order_and_reads_the_self_check():
    master = _Master()
    try:
        client = up.UnitPolicyClient(master.url)
        updates = [{"op": "touch", "id": f"u{i}"} for i in range(600)]
        client.send(updates)
        client.close()
        sent = [u for body in master.bodies for u in body["updates"]]
        assert sent == updates
        assert all(len(body["updates"]) <= 256 for body in master.bodies)
        assert client.counts["touch"] == 600 and client.counts["failed_requests"] == 0
    finally:
        master.close()


def test_a_failing_master_is_counted_not_raised():
    master = _Master(status=500)
    try:
        client = up.UnitPolicyClient(master.url)
        client.send([{"op": "dead", "id": "u"}])
        client.close()
        assert client.counts["failed_requests"] == 1
        assert client.counts["failed_updates"] == 1
    finally:
        master.close()


def _wait_for(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def test_a_malformed_reply_is_counted_and_the_sender_goes_on():
    replies = [b"not json", b"[1, 2]", b'{"errors": "none"}']
    master = _Master(replies=replies)
    try:
        client = up.UnitPolicyClient(master.url)
        for i in range(4):
            client.send([{"op": "touch", "id": f"u{i}"}])
            _wait_for(lambda i=i: len(master.bodies) == i + 1)
        client.close()
        assert client.counts["failed_requests"] == 3
        assert client.counts["failed_updates"] == 3
        assert client.counts["touch"] == 4
    finally:
        master.close()


def test_updates_beyond_the_queue_bound_are_dropped_and_counted(monkeypatch):
    monkeypatch.setattr(up, "_QUEUE_MAX", 3)
    master = _Master()
    try:
        client = up.UnitPolicyClient(master.url)
        master.gate.clear()
        client.send([{"op": "touch", "id": "first"}])
        # The sender waits on a request the master has not answered.
        _wait_for(lambda: len(master.bodies) == 1)
        client.send([{"op": "touch", "id": f"u{i}"} for i in range(5)])
        assert client.counts["dropped_updates"] == 2
        master.gate.set()
        client.close()
        sent = [u["id"] for body in master.bodies for u in body["updates"]]
        assert sent == ["first", "u0", "u1", "u2"]
    finally:
        master.gate.set()
        master.close()


def test_an_address_without_the_unit_policy_is_refused():
    with pytest.raises(RuntimeError, match="does not run the recovery-unit"):
        up.UnitPolicyClient("http://127.0.0.1:1")
    with pytest.raises(ValueError, match="http://<host>:<port>"):
        up.UnitPolicyClient("127.0.0.1:9003")


# The units a scheduler names ------------------------------------------------
class _Req:
    def __init__(self, request_id: str = "r") -> None:
        self.request_id = request_id
        self.block_hashes = HASHES
        self.num_prompt_tokens = 4 * B
        self.num_computed_tokens = 4 * B
        self.all_token_ids = list(range(4 * B))
        self.mm_features: list = []
        self.lora_request = None
        self.cache_salt = None
        self.prompt_embeds = None


def _unit_scheduler() -> MooncakeStoreScheduler:
    scheduler = object.__new__(MooncakeStoreScheduler)
    scheduler.kv_role = "kv_both"
    scheduler._capacity_only = False
    scheduler._block_size = B
    scheduler._hash_block_size = U
    scheduler._store_group_ids = (0, 1, 2)
    scheduler._store_group_id_by_kv_cache_group_id = {0: 0, 1: 1, 2: 2}
    scheduler._boundary_state_group_ids = frozenset({1, 2})
    scheduler._frontiers = SimpleNamespace(
        hash_fn=sha256, key_at=lambda request, length: length.to_bytes(32, "big")
    )
    scheduler._finished_frontier_metas = []
    scheduler._supersede = False
    scheduler._resumed_frontiers = {}
    scheduler._boundary_tokens = {}
    scheduler._store_units = True
    scheduler._unit_frontier = {}
    scheduler._unit_lookups = {}
    scheduler._unit_touches = []
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
    scheduler._request_trackers = {}
    scheduler._unfinished_requests = {}
    scheduler._gpu_block_pool = BlockPool(
        num_gpu_blocks=64, enable_caching=True, hash_block_size=U
    )
    scheduler._num_workers = 1
    scheduler._next_store_job_id = 0
    scheduler._pinned_saves = {}
    scheduler._unreleased = {}
    scheduler._released_jobs = set()
    return scheduler


def test_the_resume_point_is_touched_and_parents_the_first_unit_written():
    scheduler = _unit_scheduler()
    request = _Req()
    # A store frontier hit at B + 37.
    scheduler._unit_lookups["r"] = (B, B + 37, b"\x0f" * 32)
    scheduler._note_resume_unit(request, B + 37)
    assert scheduler._unit_touches == [b"\x0f" * 32]
    unit = scheduler._next_unit("r", HASHES[7], 2 * B, frontier=False)
    assert unit == StoreUnit(HASHES[7], 2 * B, b"\x0f" * 32)
    # Each unit is the parent of the next one the request writes.
    assert (
        scheduler._next_unit("r", b"\x01" * 32, 2 * B + 9, frontier=True).parent
        == (HASHES[7])
    )
    # A local frontier hit (inside a hash unit) and a block-aligned hit.
    scheduler._note_resume_unit(request, B + 5)
    assert scheduler._unit_touches[-1] == (B + 5).to_bytes(32, "big")
    scheduler._note_resume_unit(request, 2 * B)
    assert scheduler._unit_touches[-1] == bytes(HASHES[2 * B // U - 1])
    # No hit: no touch, no parent.
    scheduler._note_resume_unit(request, 0)
    assert len(scheduler._unit_touches) == 3
    assert scheduler._next_unit("r", HASHES[3], B, frontier=False).parent is None


def test_boundary_states_and_frontiers_carry_their_units():
    scheduler = _unit_scheduler()
    request = _Req()
    scheduler._request_trackers["r"] = RequestTracker(
        req_id="r",
        token_len=2 * B,
        allocated_block_ids=([1, 2], [0, 3], [0, 4]),
        prefill_end_tokens=4 * B,
    )
    scheduler._unfinished_requests["r"] = (request, ([1, 2], [0, 3], [0, 4]))
    meta = MooncakeStoreConnectorMetadata(set(), set())
    scheduler._handle_boundary_state_offloads(
        {"r": [(1, 3, 2 * B), (2, 4, 2 * B), (1, 5, B + 48)]}, meta
    )
    (req_meta,) = meta.requests
    keys = [unit.key for unit in req_meta.store_units]
    assert keys == [bytes(HASHES[(B + 48) // U - 1]), bytes(HASHES[2 * B // U - 1])]
    assert req_meta.store_units[1].parent == keys[0]

    pool = scheduler._gpu_block_pool
    pool.touch([pool.blocks[i] for i in (1, 2, 3, 4, 6)])
    scheduler._frontiers = fr.FrontierLookup(SimpleNamespace(), B, U, sha256)
    checkpoint = FrontierCheckpoint(
        key=b"\x0a" * 32,
        num_tokens=2 * B + 9,
        blocks={0: 6, 1: 3, 2: 4},
        pages={0: (1, 2)},
    )
    scheduler.register_finished_frontier(request, checkpoint)
    (frontier_meta,) = scheduler._finished_frontier_metas
    assert frontier_meta.store_units == (
        StoreUnit(b"\x0a" * 32, 2 * B + 9, keys[1], frontier=True),
    )


# The worker's grouped puts and unit updates ---------------------------------
class _GroupStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.groups: dict[str, str] = {}

    def batch_is_exist(self, keys):
        return [1 if key in self.objects else 0 for key in keys]

    def get(self, key):
        return self.objects[key]

    def upsert(self, key, value, config):
        self.objects[key] = bytes(value)
        return 0

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, config):
        groups = config.group_ids or [""] * len(keys)
        for key, group in zip(keys, groups, strict=True):
            self.objects[key] = b"x"
            self.groups[key] = group
        return [0] * len(keys)


class _RecordingClient:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def send(self, updates):
        self.updates.extend(updates)


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


def _sending_thread(store, units, client):
    return KVCacheStoreSendingThread(
        store=store,
        token_databases=units.frontiers.store_dbs,
        block_size=B,
        coord=_coord(),
        tp_rank=0,
        group_put_steps=(1, 1, 1),
        kv_role="kv_both",
        ready_event=threading.Event(),
        enable_kv_event=False,
        replicate_config=SimpleNamespace(group_ids=None),
        frontiers=units.frontiers,
        locator_config=SimpleNamespace(),
        store_units=units,
        unit_client=client,
        unit_replicate_config=SimpleNamespace(group_ids=None),
    )


def test_a_frontier_job_groups_its_objects_and_registers_its_unit():
    store = _GroupStore()
    units = _units()
    client = _RecordingClient()
    thread = _sending_thread(store, units, client)
    key = b"\x0a" * 32
    put = FrontierPut(
        key=key,
        length=2 * B + 9,
        locator=b"\x0b" * 32,
        blocks=((0, 6), (1, 3), (2, 4)),
        rings=((5, 7),),
        supersedes=((b"\x0c" * 32, B + 9),),
        superseded_states=((1, B + 48),),
    )
    req_meta = ReqMeta(
        req_id="r",
        token_len_chunk=0,
        block_ids=([], [], []),
        block_hashes=HASHES,
        can_save=True,
        boundary_puts=[BoundaryPut(0, 2, 2 * B)],
        completed_token_len=2 * B + 9,
        frontier=put,
        store_job_id=0,
        store_units=(StoreUnit(key, 2 * B + 9, HASHES[5], frontier=True),),
    )
    thread.add_request(req_meta)
    thread._handle_request(thread.request_queue.get())
    unit_id = units.unit_id(key)
    frontier_objects = units.frontiers.object_keys(key, 2 * B + 9, 0, (1, 1, 1))
    assert all(store.groups[k] == unit_id for k in frontier_objects)
    # The page before the frontier is a path page: no group.
    page = units.frontiers.store_dbs[0].key_for(HASHES[2 * B // U - 1])
    assert store.groups[page] == ""
    upsert, *dead = client.updates
    assert upsert["op"] == "upsert" and upsert["id"] == unit_id
    assert upsert["refs"][-1] == units.frontiers.locator_key(b"\x0b" * 32)
    # With units, what frontier_supersede_remove would remove is marked dead.
    assert dead == [
        units.dead(b"\x0c" * 32),
        units.dead(bytes(HASHES[(B + 48) // U - 1])),
    ]


def test_a_boundary_state_job_groups_the_states_objects_only():
    store = _GroupStore()
    units = _units()
    client = _RecordingClient()
    thread = _sending_thread(store, units, client)
    state = HASHES[2 * B // U - 1]
    req_meta = ReqMeta(
        req_id="r",
        token_len_chunk=0,
        block_ids=([1, 2], [0, 3], [0, 4]),
        block_hashes=HASHES,
        can_save=True,
        boundary_puts=[BoundaryPut(1, 3, 2 * B), BoundaryPut(2, 4, 2 * B)],
        completed_token_len=2 * B,
        store_job_id=0,
        store_units=(StoreUnit(bytes(state), 2 * B, None),),
    )
    thread._maybe_offload_boundary_states(req_meta)
    assert set(store.groups.values()) == {units.unit_id(bytes(state))}
    assert [u["op"] for u in client.updates] == ["upsert"]
