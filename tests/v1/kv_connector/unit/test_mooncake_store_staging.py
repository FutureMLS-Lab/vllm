# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MooncakeStoreConnector with ``staging_buffer_mib``: a store job copies the
GPU bytes it puts into pinned staging, and its blocks are freed before the
puts."""

import ctypes
import threading
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BoundaryPut,
    ChunkedTokenDatabase,
    KeyMetadata,
    MooncakeStoreWorkerMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (  # noqa: E501
    MooncakeStoreScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (  # noqa: E501
    MOONCAKE_NOT_STAGED,
    STAGE_ALIGN,
    STAGE_WAIT_S,
    KVCacheStoreSendingThread,
    StagingArea,
    _staging_buffer_mib,
    _staging_wait_s,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import KVConnectorOutput


def test_staging_buffer_mib_option():
    assert _staging_buffer_mib({}) == 0
    assert _staging_buffer_mib({"staging_buffer_mib": 4096}) == 4096
    for bad in (True, -1, 1.5, "1"):
        with pytest.raises(ValueError, match="non-negative integer"):
            _staging_buffer_mib({"staging_buffer_mib": bad})


# The staging buffer ---------------------------------------------------------
class _Registry:
    def __init__(self) -> None:
        self.registered: list[tuple[int, int]] = []

    def register_buffer(self, addr, size):
        self.registered.append((addr, size))
        return 0


def _area(nbytes: int, region: torch.Tensor) -> StagingArea:
    return StagingArea(
        _Registry(), nbytes, [(region.data_ptr(), region.numel(), region)], None
    )


def test_a_staged_copy_holds_the_source_bytes_at_its_host_address():
    region = torch.arange(4096, dtype=torch.int64).to(torch.uint8)
    area = _area(4 * STAGE_ALIGN, region)
    base = region.data_ptr()
    _, translation = area.stage([(base + 100, 60), (base + 3000, 40), (base + 100, 60)])
    assert set(translation) == {base + 100, base + 3000}
    for src, size in ((100, 60), (3000, 40)):
        host = translation[base + src] - area.base
        assert torch.equal(area.buffer[host : host + size], region[src : src + size])
    assert area.counts["jobs"] == 1 and area.counts["bytes"] == 2 * STAGE_ALIGN


def test_the_ring_wraps_frees_in_job_order_and_times_out():
    region = torch.zeros(8192, dtype=torch.uint8)
    area = _area(4 * STAGE_ALIGN, region)
    base = region.data_ptr()
    first, _ = area.stage([(base, 2 * STAGE_ALIGN)])
    second, _ = area.stage([(base, STAGE_ALIGN)])
    assert (first, second) == (0, 2 * STAGE_ALIGN)
    # One slot is left at the end.
    assert area.stage([(base, 2 * STAGE_ALIGN)], timeout=0.05) is None
    assert area.counts["timeouts"] == 1
    area.free(first)
    third, _ = area.stage([(base, 2 * STAGE_ALIGN)])
    assert third == 0
    assert area.stage([(base, 5 * STAGE_ALIGN)]) is None
    assert area.counts["too_big"] == 1


def test_a_waiting_job_gets_room_once_the_oldest_is_freed():
    region = torch.zeros(8192, dtype=torch.uint8)
    area = _area(2 * STAGE_ALIGN, region)
    base = region.data_ptr()
    first, _ = area.stage([(base, 2 * STAGE_ALIGN)])
    got: list = []
    waiter = threading.Thread(
        target=lambda: got.append(area.stage([(base, STAGE_ALIGN)], timeout=10))
    )
    waiter.start()
    area.free(first)
    waiter.join(10)
    assert got and got[0] is not None and got[0][0] == 0


def test_a_source_outside_every_region_is_refused():
    region = torch.zeros(1024, dtype=torch.uint8)
    area = _area(4 * STAGE_ALIGN, region)
    with pytest.raises(ValueError):
        area.stage([(region.data_ptr() + 1000, 100)])
    # The failed job's room was freed.
    assert area.stage([(region.data_ptr(), STAGE_ALIGN)])[0] == 0


# The scheduler --------------------------------------------------------------
class _Pool:
    def __init__(self) -> None:
        self.blocks = list(range(16))
        self.freed: list[list[int]] = []

    def free_blocks(self, blocks):
        self.freed.append(list(blocks))


def _scheduler(num_workers: int = 2) -> MooncakeStoreScheduler:
    scheduler = object.__new__(MooncakeStoreScheduler)
    scheduler._gpu_block_pool = _Pool()
    scheduler._num_workers = num_workers
    scheduler._pinned_saves = {7: ([3, 4, 5], num_workers)}
    scheduler._unreleased = {}
    scheduler._released_jobs = set()
    scheduler._frontier_jobs = {7: b"frontier"}
    scheduler._writing_frontiers = {b"frontier": (100, None, 0.0)}
    return scheduler


def _report(scheduler, released=None, completed=None) -> None:
    meta = MooncakeStoreWorkerMetadata(
        completed_saves=completed or {}, released_saves=released or {}
    )
    scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))


def test_blocks_are_freed_once_every_rank_released_them():
    scheduler = _scheduler()
    _report(scheduler, released={7: 1})
    # One rank still reads them.
    assert scheduler._gpu_block_pool.freed == []
    _report(scheduler, released={7: 1})
    assert scheduler._gpu_block_pool.freed == [[5, 4, 3]]
    # The job is still in flight: a frontier it writes is not in the store.
    assert b"frontier" in scheduler._writing_frontiers
    assert 7 in scheduler._pinned_saves
    _report(scheduler, completed={7: 2})
    assert scheduler._gpu_block_pool.freed == [[5, 4, 3]]
    assert scheduler._writing_frontiers == {} and scheduler._pinned_saves == {}
    assert scheduler._released_jobs == set()


def test_a_job_completed_without_release_is_freed_on_completion():
    scheduler = _scheduler()
    _report(scheduler, released={7: 1}, completed={7: 1})
    assert scheduler._gpu_block_pool.freed == []
    _report(scheduler, completed={7: 1})
    assert scheduler._gpu_block_pool.freed == [[5, 4, 3]]
    assert scheduler._pinned_saves == {} and scheduler._unreleased == {}


def test_too_many_releases_are_refused():
    with pytest.raises(AssertionError):
        _report(_scheduler(), released={7: 3})


# The sending thread ---------------------------------------------------------
class _ByteStore:
    """Copies a put's bytes from the addresses it names."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def batch_is_exist(self, keys):
        return [1 if key in self.objects else 0 for key in keys]

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, config):
        for key, key_addrs, key_sizes in zip(keys, addrs, sizes, strict=True):
            self.objects[key] = b"".join(
                ctypes.string_at(addr, size)
                for addr, size in zip(key_addrs, key_sizes, strict=True)
            )
        return [0] * len(keys)


def _staged_thread(store, region: torch.Tensor, slot: int):
    attn = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    state = MambaSpec(
        block_size=4, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align"
    )
    coord = MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["attn"], attn), KVCacheGroupSpec(["m"], state)], 4, 4
    )
    dbs = []
    for group in range(2):
        db = ChunkedTokenDatabase(KeyMetadata("m", 0, 0, 0, 0, group_id=group), 4)
        db.set_kv_caches_base_addr([region.data_ptr()])
        db.set_block_len([slot])
        dbs.append(db)
    staging = _area(4 * STAGE_ALIGN, region)
    thread = KVCacheStoreSendingThread(
        store=store,
        token_databases=dbs,
        block_size=4,
        coord=coord,
        tp_rank=0,
        group_put_steps=(1, 1),
        kv_role="kv_both",
        ready_event=threading.Event(),
        enable_kv_event=False,
        replicate_config=SimpleNamespace(group_ids=None),
        staging=staging,
    )
    return thread, staging


def test_a_staged_job_is_released_once_copied_and_puts_the_copied_bytes():
    slot = 64
    region = torch.arange(8 * slot, dtype=torch.int64).to(torch.uint8)
    store = _ByteStore()
    thread, staging = _staged_thread(store, region, slot)
    hashes = [bytes([i]) * 4 for i in range(4)]
    req_meta = ReqMeta(
        req_id="r",
        token_len_chunk=0,
        block_ids=([1, 2], [0, 3]),
        block_hashes=hashes,
        can_save=True,
        boundary_puts=[BoundaryPut(1, 3, 8)],
        completed_token_len=8,
        store_job_id=5,
    )
    expected = bytes(region[3 * slot : 4 * slot].tolist())
    thread.add_request(req_meta)
    thread._stage_job(thread.request_queue.get())
    # The copy is done and the job's blocks are released before its puts.
    assert thread.take_released_saves() == {5: 1}
    # The block's GPU bytes change once released; the put carries the copy.
    region[3 * slot : 4 * slot] = 0
    completed: dict[int, int] = {}
    for _ in range(500):
        completed.update(thread.take_completed_saves())
        if completed:
            break
        threading.Event().wait(0.01)
    assert completed == {5: 1}
    (value,) = store.objects.values()
    assert value == expected
    assert staging.counts["jobs"] == 1


def test_a_key_not_copied_with_its_job_is_not_put():
    slot = 64
    region = torch.zeros(8 * slot, dtype=torch.uint8)
    store = _ByteStore()
    thread, _ = _staged_thread(store, region, slot)
    base = region.data_ptr()
    thread._translation = {base: base}
    codes = thread._put_buffers(
        ["a", "b"],
        [[base], [base + slot]],
        [[slot], [slot]],
        SimpleNamespace(group_ids=None),
    )
    assert codes == [0, MOONCAKE_NOT_STAGED]
    assert list(store.objects) == ["a"]


class _RacingStore(_ByteStore):
    """``evict``: keys the store drops right after the next existence check
    that finds them (an eviction between a job's copy and its puts).
    ``refuse``: keys whose puts fail."""

    def __init__(self) -> None:
        super().__init__()
        self.evict: set[str] = set()
        self.refuse: set[str] = set()

    def batch_is_exist(self, keys):
        found = super().batch_is_exist(keys)
        for key in self.evict & set(keys):
            self.objects.pop(key, None)
        self.evict -= set(keys)
        return found

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, config):
        codes = []
        for key, key_addrs, key_sizes in zip(keys, addrs, sizes, strict=True):
            if key in self.refuse:
                codes.append(-1)
                continue
            codes += super().batch_put_from_multi_buffers(
                [key], [key_addrs], [key_sizes], config
            )
        return codes


def _run_staged(thread, req_meta: ReqMeta) -> None:
    thread.add_request(req_meta)
    thread._stage_job(thread.request_queue.get())
    for _ in range(500):
        if req_meta.store_job_id in thread.take_completed_saves():
            return
        threading.Event().wait(0.01)
    raise AssertionError(f"store job {req_meta.store_job_id} did not complete")


def _page_keys(thread, hashes) -> list[str]:
    layout = thread.token_databases[0].store_layout
    return [
        layout.key_for(shard, block_hash)
        for block_hash in hashes
        for shard in layout.local_shard_ids
    ]


def test_a_key_evicted_between_the_copy_and_the_put_does_not_fail_the_job():
    slot = 64
    region = torch.arange(8 * slot, dtype=torch.int64).to(torch.uint8)
    store = _RacingStore()
    thread, staging = _staged_thread(store, region, slot)
    hashes = [bytes([i]) * 4 for i in range(4)]
    state_key = thread.token_databases[1].key_for(hashes[1])
    # The state is in the store when the job is copied, so it is not copied,
    # and it is evicted before the job's puts.
    store.objects[state_key] = b"stored before"
    store.evict = {state_key}
    _run_staged(
        thread,
        ReqMeta(
            req_id="r",
            token_len_chunk=8,
            block_ids=([1, 2], [0, 3]),
            block_hashes=hashes,
            can_save=True,
            boundary_puts=[BoundaryPut(1, 3, 8)],
            completed_token_len=8,
            store_job_id=5,
        ),
    )
    # The job's pages are put and recorded; the evicted state is not put again.
    assert set(store.objects) == set(_page_keys(thread, hashes[:2]))
    assert thread._saved_offset["r"] == 8
    assert staging.counts["evicted_after_copy"] == 1


def test_a_job_that_stops_early_has_its_pages_copied_again_by_the_next_job():
    slot = 64
    region = torch.arange(8 * slot, dtype=torch.int64).to(torch.uint8)
    store = _RacingStore()
    thread, _ = _staged_thread(store, region, slot)
    hashes = [bytes([i]) * 4 for i in range(4)]
    # The first job's state put fails, so the job stops before its pages.
    store.refuse = {thread.token_databases[1].key_for(hashes[0])}
    first = ReqMeta(
        req_id="r",
        token_len_chunk=4,
        block_ids=([1, 2], [3, 0]),
        block_hashes=hashes,
        can_save=True,
        boundary_puts=[BoundaryPut(1, 3, 4)],
        completed_token_len=4,
        store_job_id=5,
    )
    second = ReqMeta(
        req_id="r",
        token_len_chunk=8,
        block_ids=([1, 2], [3, 0]),
        block_hashes=hashes,
        can_save=True,
        completed_token_len=8,
        store_job_id=6,
    )
    _run_staged(thread, first)
    assert not store.objects
    _run_staged(thread, second)
    # The second job copied and put the first job's page as well.
    assert set(store.objects) == set(_page_keys(thread, hashes[:2]))
    assert thread._saved_offset["r"] == 8


def test_adjacent_segments_are_copied_at_once():
    region = torch.arange(4096, dtype=torch.int64).to(torch.uint8)
    area = _area(8 * STAGE_ALIGN, region)
    copies: list[int] = []
    copy = area._copy_segment

    def counted(dst, src, dst_addr, src_addr):
        copies.append(src.numel())
        copy(dst, src, dst_addr, src_addr)

    area._copy_segment = counted
    base = region.data_ptr()
    sources = [(base + i * STAGE_ALIGN, STAGE_ALIGN) for i in (3, 1, 2)]
    sources.append((base + 6 * STAGE_ALIGN, 100))
    _, translation = area.stage(sources)
    # Segments 1-3 are one run on both sides; the last one is apart.
    assert copies == [3 * STAGE_ALIGN, 100]
    for addr, size in sources:
        host = translation[addr] - area.base
        src = addr - base
        assert torch.equal(area.buffer[host : host + size], region[src : src + size])


def test_one_address_with_two_sizes_is_refused():
    region = torch.zeros(4096, dtype=torch.uint8)
    area = _area(8 * STAGE_ALIGN, region)
    base = region.data_ptr()
    with pytest.raises(ValueError, match="two sizes"):
        area.stage([(base, 64), (base, 128)])


def test_staging_wait_s_option():
    assert _staging_wait_s({}) == STAGE_WAIT_S
    assert _staging_wait_s({"staging_wait_s": 0}) == 0
    assert _staging_wait_s({"staging_wait_s": 0.5}) == 0.5
    for bad in (True, -1, "1"):
        with pytest.raises(ValueError, match="non-negative number"):
            _staging_wait_s({"staging_wait_s": bad})
    region = torch.zeros(1024, dtype=torch.uint8)
    area = StagingArea(
        _Registry(), STAGE_ALIGN, [(region.data_ptr(), 1024, region)], None, 0.0
    )
    base = region.data_ptr()
    area.stage([(base, STAGE_ALIGN)])
    # A full buffer does not wait with a zero wait.
    assert area.stage([(base, STAGE_ALIGN)]) is None
    assert area.counts["timeouts"] == 1
