# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba frontier checkpoints under decode context parallelism, where a
full-attention page holds ``DCP`` Mamba blocks: a frontier is the page and the
Mamba block that hold its position, and decode-boundary states sit on page
boundaries."""

import pytest

from vllm.utils.hashing import sha256
from vllm.v1.core.hybrid_checkpoint.keys import frontier_key
from vllm.v1.core.kv_cache_utils import init_none_hash, make_block_hash_with_group_id

from .utils import (
    add_request,
    admitted_at,
    frontier_recording_connector,
    make_hybrid_scheduler,
    prompt,
    run,
)

pytestmark = pytest.mark.cpu_test

STATE = 16
DCP = 4
PAGE = STATE * DCP
# The group ids in make_hybrid_scheduler's layout.
ATTENTION = 0
MAMBA = [1, 2]


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def make_scheduler(**kwargs):
    return make_hybrid_scheduler(
        block_size=STATE, dcp=DCP, ring=0, mamba_frontier_checkpoint=True, **kwargs
    )


def tables(scheduler, request_id: str) -> list[list]:
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    return [m.req_to_blocks[request_id] for m in managers]


def test_align_mode_caches_no_decode_state_inside_a_page():
    """Under the retention interval frontier checkpoints run with (0), align
    mode keeps the prompt's replay state and no state that decoding reaches."""
    scheduler = make_scheduler()
    assert scheduler.kv_cache_manager.coordinator.retention_interval == 0
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=11)
    run(scheduler, first)
    assert first.num_computed_tokens == 160
    pool = scheduler.kv_cache_manager.block_pool
    assert pool.get_cached_block(first.block_hashes[144 // STATE - 1], MAMBA)
    assert pool.get_cached_block(first.block_hashes[160 // STATE - 1], MAMBA) is None


def test_page_and_state_sizes():
    checkpoints = make_scheduler().mamba_checkpoints
    assert checkpoints is not None
    assert (checkpoints.page_size, checkpoints.state_size) == (PAGE, STATE)


def test_next_turn_resumes_at_the_previous_frontier():
    scheduler = make_scheduler()
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    frontier_tokens = first.num_computed_tokens
    assert frontier_tokens % STATE

    tokens = list(first.all_token_ids) + prompt(2, 40)
    second = add_request(scheduler, "t1", tokens, max_tokens=4)
    assert admitted_at(scheduler, second) == frontier_tokens
    # The page holding the frontier and the Mamba block holding its state:
    # nothing before the state in the Mamba groups.
    attention, *mamba = tables(scheduler, "t1")
    assert len(attention) >= frontier_tokens // PAGE + 1
    for table in mamba:
        state = frontier_tokens // STATE
        assert not table[state].is_null
        assert all(block.is_null for block in table[:state])


def test_next_turn_resumes_at_a_mamba_block_boundary_inside_a_page():
    scheduler = make_scheduler()
    # 150 prompt tokens and 11 outputs: the request ends on 160, a Mamba block
    # boundary inside a page.
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=11)
    run(scheduler, first)
    assert first.num_computed_tokens == 160
    frontiers = scheduler.mamba_checkpoints.frontiers
    assert [e.num_tokens for e in frontiers.entries.values()] == [160]

    tokens = list(first.all_token_ids) + prompt(2, 40)
    second = add_request(scheduler, "t1", tokens, max_tokens=2)
    assert admitted_at(scheduler, second) == 160
    # Past the prompt's replay state at 144.
    assert (frontiers.num_hits, frontiers.num_hit_tokens) == (1, 160 - 144)
    # The state is the Mamba block ending at 160: nothing before it.
    attention, *mamba = tables(scheduler, "t1")
    for table in mamba:
        assert not table[160 // STATE - 1].is_null
        assert all(block.is_null for block in table[: 160 // STATE - 1])


@pytest.mark.parametrize("max_tokens", [5, 21, 37, 53, 69, 85, 101, 117])
def test_every_mamba_block_boundary_of_a_page_at_dcp8(max_tokens: int):
    """A page of 8 Mamba blocks: an end on any of their boundaries inside the
    page leaves a frontier, and an end on the page boundary its
    decode-boundary state."""
    scheduler = make_hybrid_scheduler(
        block_size=STATE, dcp=8, ring=0, mamba_frontier_checkpoint=True
    )
    page = 8 * STATE
    first = add_request(scheduler, "t0", prompt(1, 300), max_tokens=max_tokens)
    run(scheduler, first)
    end = first.num_computed_tokens
    assert end == 299 + max_tokens and end % STATE == 0
    frontiers = scheduler.mamba_checkpoints.frontiers
    expected = [] if end % page == 0 else [end]
    assert [e.num_tokens for e in frontiers.entries.values()] == expected

    tokens = list(first.all_token_ids) + prompt(2, 40)
    second = add_request(scheduler, "t1", tokens, max_tokens=2)
    assert admitted_at(scheduler, second) == end


def test_a_page_boundary_end_keeps_its_decode_boundary_state():
    """An end on a page boundary leaves no frontier: the decode-boundary state
    there serves it, with every page before it whole."""
    scheduler = make_scheduler()
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=43)
    run(scheduler, first)
    assert first.num_computed_tokens == 3 * PAGE
    checkpoints = scheduler.mamba_checkpoints
    assert not checkpoints.frontiers.entries
    units = checkpoints.graph.units.values()
    assert [u.frontier for u in units if u.num_tokens == 3 * PAGE] == [False]

    tokens = list(first.all_token_ids) + prompt(2, 40)
    second = add_request(scheduler, "t1", tokens, max_tokens=2)
    assert admitted_at(scheduler, second) == 3 * PAGE


def test_session_leaf_moves_to_a_frontier_on_a_mamba_block_boundary():
    scheduler = make_scheduler()
    checkpoints = scheduler.mamba_checkpoints
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=10)
    run(scheduler, first)
    assert first.num_computed_tokens == 159
    tokens = list(first.all_token_ids) + prompt(2, 10)
    second = add_request(scheduler, "t1", tokens, max_tokens=7)
    assert admitted_at(scheduler, second) == 159
    run(scheduler, second)
    # 170 prompt tokens and 7 outputs end on 176, inside the page [128, 192):
    # the frontier there replaces the one at 159 as the session's leaf.
    assert second.num_computed_tokens == 176
    frontiers = checkpoints.frontiers
    [(key, entry)] = frontiers.entries.items()
    assert entry.num_tokens == 176
    assert key in checkpoints.graph.leaves

    tokens = list(second.all_token_ids) + prompt(3, 20)
    third = add_request(scheduler, "t2", tokens, max_tokens=2)
    assert admitted_at(scheduler, third) == 176


def test_a_block_boundary_frontier_is_dropped_whole():
    """Evicting a block-boundary frontier's Mamba block drops the frontier in
    every group; its page keeps the prompt's replay key."""
    scheduler = make_scheduler(
        prefix_cache_eviction="gds", prefill_cost_coefficients=(1e-5, 1e-6)
    )
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=11)
    run(scheduler, first)
    frontiers = scheduler.mamba_checkpoints.frontiers
    [entry] = frontiers.entries.values()
    assert entry.num_tokens == 160
    pool = scheduler.kv_cache_manager.block_pool
    page = entry.blocks[ATTENTION]
    page_key = make_block_hash_with_group_id(entry.key, ATTENTION)

    pool.evict_blocks({entry.blocks[MAMBA[0]].block_id})

    assert not frontiers.entries
    assert not pool.cached_block_hash_to_block.contain(page_key, page.block_id)
    assert page.block_hash == make_block_hash_with_group_id(
        first.block_hashes[144 // STATE - 1], ATTENTION
    )
    probe = add_request(scheduler, "p", prompt(1, 150) + prompt(4, 30), max_tokens=2)
    assert admitted_at(scheduler, probe) == 144


def test_a_block_boundary_frontier_is_handed_to_a_storing_connector():
    scheduler = make_scheduler(kv_transfer_config=frontier_recording_connector(True))
    first = add_request(scheduler, "t0", prompt(1, 150), max_tokens=11)
    run(scheduler, first)
    ((request_id, checkpoint, block_ids),) = scheduler.connector.frontiers
    assert (request_id, checkpoint.num_tokens) == ("t0", 160)
    assert checkpoint.key == frontier_key(sha256, first, 160, STATE)
    assert checkpoint.blocks[ATTENTION] == block_ids[ATTENTION][160 // PAGE]
    for group_id in MAMBA:
        assert checkpoint.blocks[group_id] == block_ids[group_id][160 // STATE - 1]
    assert checkpoint.pages == {ATTENTION: tuple(block_ids[ATTENTION][: 160 // PAGE])}


def test_prompt_tail_inside_a_page_dies_with_its_partial_page():
    """The prompt's replay state on a hash unit inside a page has a partial
    page of its own under its key: GreedyDual-Size counts it, and the unit
    can no longer be hit once that page is gone."""
    scheduler = make_scheduler(
        prefix_cache_eviction="gds", prefill_cost_coefficients=(1e-5, 1e-6)
    )
    first = add_request(scheduler, "t0", prompt(1, 100), max_tokens=3)
    run(scheduler, first)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None and checkpoints.eviction is not None
    tail = (100 - 1) // STATE * STATE
    assert tail % PAGE
    [unit] = [u for u in checkpoints.graph.units.values() if u.num_tokens == tail]
    assert unit.frontier
    pool = scheduler.kv_cache_manager.block_pool
    page = pool.cached_block_hash_to_block.get_one_block(
        make_block_hash_with_group_id(unit.key, ATTENTION)
    )
    assert page is not None
    assert not checkpoints.eviction.units[unit.key].dead

    pool.evict_blocks({page.block_id})

    assert checkpoints.eviction.units[unit.key].dead


def test_decode_boundary_state_sits_on_a_page_boundary():
    scheduler = make_scheduler()
    first = add_request(scheduler, "t0", prompt(1, 100), max_tokens=60)
    run(scheduler, first)
    assert first.num_computed_tokens > 2 * PAGE + STATE

    # Shares the first request's tokens past 2 * PAGE + STATE but not its
    # frontier: the last page boundary its output crossed is 2 * PAGE.
    tokens = list(first.all_token_ids[: 2 * PAGE + STATE + 4]) + prompt(3, 50)
    sibling = add_request(scheduler, "s", tokens, max_tokens=2)
    assert admitted_at(scheduler, sibling) == 2 * PAGE


def test_sessions_under_pool_pressure():
    scheduler = make_scheduler(num_blocks=120)
    tokens = {s: prompt(s, 90 + 37 * s) for s in range(6)}
    for turn in range(3):
        requests = [
            add_request(scheduler, f"c{s}t{turn}", tokens[s], max_tokens=5 + 13 * s)
            for s in range(6)
        ]
        run(scheduler, *requests)
        for s, request in enumerate(requests):
            tokens[s] = list(request.all_token_ids) + prompt(10 * s + turn, 21)
    checkpoints = scheduler.mamba_checkpoints
    assert checkpoints is not None
    assert checkpoints.frontiers.num_published > 0
    assert checkpoints.frontiers.num_hits > 0
