# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba frontier checkpoints under speculative decoding.

The real scheduler with three draft tokens per step from an n-gram drafter,
and a stand-in for the device (``spec_device``) that checks every state read
and every cached state against its token position. Sessions are multi-turn: a
turn's prompt is the previous turn's prompt and output plus new tokens.
Layouts: "per_position" keeps one state column per draft position,
"in_place" folds the accepted tokens into one column after sampling.
"""

import random
from collections import Counter
from types import SimpleNamespace

import pytest

from vllm.config import KVTransferConfig, SpeculativeConfig
from vllm.utils.hashing import sha256
from vllm.v1.core.hybrid_checkpoint.frontier import FrontierIndex
from vllm.v1.core.hybrid_checkpoint.speculative import (
    ExactState,
    check_speculative_config,
    commit_waits_for_next_verify,
)
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.request import RequestStatus

from .spec_device import Device, Engine, Fault
from .utils import (
    BoundaryStateRecordingKVConnector,
    FrontierStoringKVConnector,
    boundary_state_connector,
    frontier_storing_connector,
    make_hybrid_scheduler,
    mock_connector,
    prompt,
)

pytestmark = pytest.mark.cpu_test

BLOCK = 64
DRAFTS = 3
COMPRESSION_RATIO = 4
# The ring holds two compressed groups: a step stores its drafts' rows before
# acceptance is known.
RING = 8
LAYOUTS = ("per_position", "in_place")
# Decode context parallelism: a full-attention page of BLOCK tokens holds DCP
# Mamba blocks.
DCP = 4


@pytest.fixture(autouse=True)
def _init_hash():
    init_none_hash(sha256)


def speculative_config(truncate: bool = True) -> SpeculativeConfig:
    # A drafter without the EAGLE block drop, which asynchronous scheduling
    # supports.
    return SpeculativeConfig(
        method="ngram_gpu",
        num_speculative_tokens=DRAFTS,
        truncate_acceptance_at_stop=truncate,
    )


def default_accept(request_id: str, start: int) -> int:
    return (sum(map(ord, request_id)) * 7 + start * 13) % (DRAFTS + 1)


def make_engine(
    layout: str = "per_position",
    *,
    async_scheduling: bool = True,
    num_blocks: int = 400,
    truncate: bool = True,
    accept=default_accept,
    fault: Fault | None = None,
    dcp: int = 1,
    **scheduler_kwargs,
) -> Engine:
    scheduler = make_hybrid_scheduler(
        block_size=BLOCK // dcp,
        dcp=dcp,
        num_blocks=num_blocks,
        ring=RING,
        ring_compression_ratio=COMPRESSION_RATIO,
        speculative_config=speculative_config(),
        num_speculative_blocks=DRAFTS if layout == "per_position" else 0,
        async_scheduling=async_scheduling,
        mamba_frontier_checkpoint=True,
        **scheduler_kwargs,
    )
    device = Device(
        scheduler, num_drafts=DRAFTS, truncate=truncate, accept=accept, fault=fault
    )
    return Engine(scheduler, device, async_scheduling)


def session(
    engine: Engine,
    sid: int = 1,
    turns: int = 3,
    first: int = 900,
    out: tuple[int, ...] = (30, 41, 77),
    extra: tuple[int, ...] = (70, 33, 51),
    eos: dict[int, int] | None = None,
) -> list[int]:
    """Run a session; returns each turn's length. ``eos``: turn -> the output
    position at which the model samples EOS."""
    tokens = prompt(sid, first)
    lengths = []
    for turn in range(turns):
        request_id = f"s{sid}t{turn}"
        eos_at = None
        if eos is not None and turn in eos:
            eos_at = len(tokens) + eos[turn]
        engine.add(request_id, tokens, out[turn % len(out)], eos_at=eos_at)
        engine.run(request_id)
        all_tokens = list(engine.requests[request_id].all_token_ids)
        lengths.append(len(all_tokens))
        tokens = all_tokens + prompt(100 * sid + turn, extra[turn % len(extra)])
    return lengths


def assert_clean(engine: Engine) -> None:
    device = engine.device
    assert not device.violations, device.violations[:5]
    assert device.counts["reads"] > 0 and device.counts["cached_checks"] > 0


def checkpoints(engine: Engine):
    manager = engine.scheduler.mamba_checkpoints
    assert manager is not None and manager.speculative is not None
    return manager


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_session_resumes_at_its_frontiers(layout: str, async_scheduling: bool):
    engine = make_engine(layout, async_scheduling=async_scheduling)
    lengths = session(engine)
    assert_clean(engine)
    manager = checkpoints(engine)
    assert manager.frontiers.num_published == 3, manager.speculative.declined
    assert manager.frontiers.num_hits == 2
    for turn in (1, 2):
        # With a step in flight at the finish its first input token, the
        # request's last token, is in the state; otherwise it is not.
        (admitted,) = engine.admissions[f"s1t{turn}"]
        assert admitted in (lengths[turn - 1], lengths[turn - 1] - 1)
    if not async_scheduling:
        assert engine.admissions["s1t1"] == [lengths[0] - 1]
        if layout == "per_position":
            # The state is in an accepted draft column, folded into the first.
            assert engine.device.counts["gathers"] > 0
    if layout == "in_place":
        assert engine.device.counts["gathers"] == 0


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_eos_inside_accepted_drafts(layout: str, async_scheduling: bool):
    """Every draft is accepted, so EOS lands inside an accepted run; the
    truncated acceptance keeps the published state exact."""
    engine = make_engine(
        layout,
        async_scheduling=async_scheduling,
        accept=lambda request_id, start: DRAFTS,
    )
    session(engine, out=(500, 400, 400), eos={0: 22, 1: 9})
    assert_clean(engine)
    first = engine.requests["s1t0"]
    assert first.status == RequestStatus.FINISHED_STOPPED
    assert len(first.output_token_ids) == 23
    manager = checkpoints(engine)
    assert manager.frontiers.num_published == 3, manager.speculative.declined


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_a_deferred_commit_publishes_length_capped_frontiers_only(
    async_scheduling: bool,
):
    """Kimi K3's specialized decode under RecoverSSM commits a request's
    last steps at once only at its length limit: a request that samples EOS
    leaves its last accepted tokens pending and publishes no frontier; the
    length-capped turns publish theirs."""
    engine = make_engine(
        "in_place",
        async_scheduling=async_scheduling,
        accept=lambda request_id, start: DRAFTS,
    )
    checkpoints(engine).speculative.length_finishes_only = True
    session(engine, out=(500, 40, 40), eos={0: 22})
    assert_clean(engine)
    assert engine.requests["s1t0"].status == RequestStatus.FINISHED_STOPPED
    for turn in (1, 2):
        status = engine.requests[f"s1t{turn}"].status
        assert status == RequestStatus.FINISHED_LENGTH_CAPPED
    manager = checkpoints(engine)
    assert manager.speculative.declined["pending_commit"] == 1
    assert manager.frontiers.num_published == 2, manager.speculative.declined


@pytest.mark.parametrize(
    "use_kda_recoverssm,additional_config,waits",
    [
        (True, {"decode_backend": "latency_specialized"}, True),
        (True, {"decode_backend": "auto"}, False),
        (True, {}, False),
        (False, {"decode_backend": "latency_specialized"}, False),
    ],
)
def test_only_the_specialized_decode_under_recoverssm_defers_the_commit(
    use_kda_recoverssm: bool, additional_config: dict, waits: bool
):
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(use_kda_recoverssm=use_kda_recoverssm),
        additional_config=additional_config,
    )
    assert commit_waits_for_next_verify(vllm_config) is waits


@pytest.mark.parametrize("layout", LAYOUTS)
def test_without_truncation_a_stop_inside_accepted_drafts_publishes_nothing(
    layout: str,
):
    """A device that runs past the stop (startup refuses this configuration;
    here only the device misbehaves): the frontier is declined, never
    published wrong."""
    engine = make_engine(
        layout, truncate=False, accept=lambda request_id, start: DRAFTS
    )
    session(engine, out=(500, 400, 400), eos={0: 22, 1: 9})
    assert_clean(engine)
    assert checkpoints(engine).speculative.declined["stop_inside_accepted"] >= 1


@pytest.mark.parametrize("layout", LAYOUTS)
def test_decode_boundary_state_at_a_crossed_boundary(layout: str):
    """The output crosses several block boundaries; the state the last verify
    step left at the last of them is published as a decode-boundary state.
    Without a step in flight, align mode frees the block that holds it before
    it could cache it, so only the snapshot keeps it."""
    engine = make_engine(layout, async_scheduling=False)
    tokens = prompt(3, 3 * BLOCK - 40)
    engine.add("leaf", tokens, 300)
    engine.run("leaf")
    assert_clean(engine)
    num_tokens = len(engine.requests["leaf"].all_token_ids)
    boundary = (num_tokens - 1) // BLOCK * BLOCK
    units = checkpoints(engine).graph.units.values()
    assert any(u.num_tokens == boundary and not u.frontier for u in units), [
        u.num_tokens for u in units
    ]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_in_chunk_export(layout: str):
    """In 192-token steps the first request leaves attention pages up to 640
    and, under retention interval 0, no state before 300. A sibling sharing
    300 tokens starts from scratch and exports the state at its junction, 256,
    inside its second chunk; a third request sharing those tokens resumes
    there."""
    engine = make_engine(layout, max_num_batched_tokens=192, mamba_prefill_export=True)
    shared = prompt(1, 640)
    engine.add("a", shared, 20)
    engine.run("a")
    engine.add("b", shared[:300] + prompt(2, 400), 20)
    engine.run("b")
    engine.add("c", shared[:300] + prompt(3, 90), 20)
    engine.run("c")
    assert_clean(engine)
    assert engine.admissions["b"] == [0]
    assert engine.admissions["c"] == [256]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_concurrent_sessions_under_pool_pressure(layout: str):
    rng = random.Random(11)
    engine = make_engine(
        layout,
        num_blocks=180,
        accept=lambda request_id, start: hash((request_id, start)) % (DRAFTS + 1),
    )
    tokens = {s: prompt(s, rng.randrange(200, 1400)) for s in range(10)}
    for turn in range(3):
        request_ids = [f"c{s}t{turn}" for s in range(10)]
        for s, request_id in enumerate(request_ids):
            engine.add(request_id, tokens[s], rng.choice((17, 90, 200)))
        engine.run(*request_ids)
        for s, request_id in enumerate(request_ids):
            tokens[s] = list(engine.requests[request_id].all_token_ids) + prompt(
                1000 + 10 * s + turn, rng.randrange(10, 90)
            )
    assert_clean(engine)
    manager = checkpoints(engine)
    assert manager.frontiers.num_published > 0 and manager.frontiers.num_hits > 0


@pytest.mark.parametrize("fault", ["skip_gather", "no_conv_shift"])
def test_device_faults_are_detected(fault: Fault):
    """A gather skipped or applied without the window shift leaves a published
    state wrong, which the device's checks catch."""
    engine = make_engine("per_position", async_scheduling=False, fault=fault)
    session(engine)
    counts = engine.device.counts
    assert counts["gathers"] > 0
    assert counts["violation_cached"] + counts["violation_read"] > 0


@pytest.mark.parametrize("shift", [-1, 1])
def test_publishing_another_column_is_detected(monkeypatch, shift: int):
    original = ExactState.block_indices

    def wrong_column(self: ExactState, block_size: int, num_speculative_blocks: int):
        column = self.column + shift
        if not 0 <= column <= num_speculative_blocks:
            return original(self, block_size, num_speculative_blocks)
        moved = ExactState(self.num_tokens, self.step_start, self.step_len, column)
        return original(moved, block_size, num_speculative_blocks)

    monkeypatch.setattr(ExactState, "block_indices", wrong_column)
    engine = make_engine("per_position", async_scheduling=False)
    session(engine)
    counts = engine.device.counts
    assert counts["violation_cached"] + counts["violation_read"] > 0


def test_abort_with_a_step_in_flight_publishes_no_frontier():
    engine = make_engine("in_place")
    engine.add("aborted", prompt(5, 600), 2000)
    request = engine.requests["aborted"]
    while not (request.num_output_tokens > 50 and request.num_in_flight_tokens):
        engine.step()
    engine.scheduler.finish_requests("aborted", RequestStatus.FINISHED_ABORTED)
    engine.quiesce()
    assert_clean(engine)
    assert checkpoints(engine).frontiers.num_published == 0


def test_a_reset_with_a_gather_pending_drops_its_frontier():
    """A request finishes with its state in a draft column, and the prefix
    cache is reset before the step that would gather it into the first."""
    engine = make_engine(
        "per_position", async_scheduling=False, accept=lambda rid, start: DRAFTS
    )
    engine.add("r", prompt(1, 900), 41)
    while not engine.done("r"):
        engine.step()
    manager = checkpoints(engine)
    assert manager.frontiers.pending_gathers
    assert engine.scheduler.reset_prefix_cache(reset_running_requests=True)
    assert not manager.frontiers.pending_gathers and not manager.frontiers.entries
    pool = engine.scheduler.kv_cache_manager.block_pool
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
    # The engine goes on: the next request's states are all exact.
    engine.add("next", prompt(1, 900), 41)
    engine.run("next")
    assert_clean(engine)


def test_an_idle_reset_follows_a_tokenless_step_that_gathers_a_state():
    """With deferred block frees (a KV connector under asynchronous
    scheduling), blocks a step retains return to the pool once the step
    completes. A request finishes with its state in a draft column and nothing
    else runs, so a step without tokens runs the gather; the idle engine's
    prefix cache can then be reset. Synchronous scheduling leaves the state in
    a draft column at every such finish."""
    engine = make_engine(
        "per_position",
        async_scheduling=False,
        accept=lambda rid, start: DRAFTS,
        kv_transfer_config=boundary_state_connector(),
    )
    engine.scheduler.defer_block_free = True
    schedule = engine.scheduler.schedule
    tokenless_gathers = 0

    def counting_schedule(*args, **kwargs):
        nonlocal tokenless_gathers
        output = schedule(*args, **kwargs)
        if output.mamba_state_gathers and not output.total_num_scheduled_tokens:
            tokenless_gathers += 1
        return output

    engine.scheduler.schedule = counting_schedule  # type: ignore[method-assign]
    engine.add("r", prompt(1, 900), 41)
    engine.run("r")
    assert tokenless_gathers == 1
    scheduler = engine.scheduler
    assert scheduler.sched_step_seq == scheduler.processed_step_seq
    assert not scheduler.deferred_frees
    assert scheduler.reset_prefix_cache()
    assert_clean(engine)


def test_an_aborted_request_that_was_looked_up_leaves_no_lookup():
    engine = make_engine("in_place")
    session(engine, turns=1)
    request = engine.add("waiting", prompt(1, 900) + prompt(9, 40), 30)
    # The scheduler looked it up, then could not admit it.
    engine.scheduler.kv_cache_manager.get_computed_blocks(request)
    manager = checkpoints(engine)
    assert "waiting" in manager.frontiers._lookups
    engine.scheduler.finish_requests("waiting", RequestStatus.FINISHED_ABORTED)
    assert not manager.frontiers._lookups


def test_resume_inside_an_open_group_catches_up_to_the_group_not_the_ring():
    """Under speculative decoding the ring holds two groups; a request resumed
    inside an open group takes one token per step only until the group
    closes."""
    engine = make_engine("in_place", async_scheduling=False)
    lengths = session(engine, turns=2)
    manager = checkpoints(engine)
    assert manager.catchup_pool_size == COMPRESSION_RATIO
    (admitted,) = engine.admissions["s1t1"]
    assert admitted == lengths[0] - 1
    assert_clean(engine)


@pytest.mark.parametrize(
    ("kwargs", "needs"),
    [
        (
            dict(speculative_config=speculative_config(truncate=False)),
            "speculative_config.truncate_acceptance_at_stop",
        ),
        (
            dict(
                speculative_config=speculative_config(),
                kv_transfer_config=mock_connector(BLOCK),
            ),
            "no KV connector under speculative decoding",
        ),
        (
            dict(speculative_config=speculative_config(), resume_catchup="prefill"),
            "indexer rings that only the decode path reads",
        ),
        (
            dict(
                speculative_config=speculative_config(),
                kv_transfer_config=boundary_state_connector(),
                hash_block_size=BLOCK // 2,
            ),
            "a hash block size of at least the Mamba block size",
        ),
    ],
)
def test_unsupported_speculative_configurations_are_refused(kwargs, needs):
    with pytest.raises(ValueError, match=needs):
        make_hybrid_scheduler(
            block_size=BLOCK,
            ring=RING,
            num_speculative_blocks=DRAFTS,
            mamba_frontier_checkpoint=True,
            **kwargs,
        )


def storing_connector() -> KVTransferConfig:
    """A connector that stores finished requests' frontier checkpoints."""
    return KVTransferConfig(
        kv_connector="FrontierRecordingKVConnector",
        kv_role="kv_both",
        kv_load_failure_policy="recompute",
        kv_connector_extra_config={
            "matched_tokens": 0,
            "is_async": False,
            "stores_frontiers": True,
        },
    )


def connector(name: str, **extra_config) -> dict:
    return {
        "kv_connector": name,
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra_config,
    }


MOONCAKE = connector("MooncakeStoreConnector")
MOCK = connector("HybridMockKVConnector", matched_tokens=0, is_async=False)


@pytest.mark.parametrize(
    ("transfer", "refused"),
    [
        pytest.param(MOONCAKE, False, id="mooncake"),
        pytest.param(
            connector("MooncakeStoreConnector", save_decode_cache=True),
            True,
            id="mooncake-decode-cache",
        ),
        pytest.param(
            connector("MooncakeStoreConnector", frontier_write_through=True),
            False,
            id="mooncake-write-through",
        ),
        pytest.param(MOCK, True, id="mock"),
        pytest.param(
            connector("MultiConnector", connectors=[MOONCAKE, MOONCAKE]),
            False,
            id="multi",
        ),
        pytest.param(
            connector("MultiConnector", connectors=[MOONCAKE, MOCK]),
            True,
            id="multi-with-mock",
        ),
    ],
)
@pytest.mark.parametrize("layout", ["per_position", "compact"])
def test_a_kv_connector_with_per_position_columns_reads_boundary_states_only(
    transfer: dict, refused: bool, layout: str
):
    config = SimpleNamespace(
        speculative_config=speculative_config(),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        kv_transfer_config=KVTransferConfig(
            kv_load_failure_policy="recompute", **transfer
        ),
        cache_config=SimpleNamespace(mamba_spec_state_layout=layout),
    )
    refusal = check_speculative_config(config)  # type: ignore[arg-type]
    if refused and layout == "per_position":
        assert refusal == (
            "no KV connector under speculative decoding unless it reads only "
            "the boundary states handed to it (supports_per_position_spec_states)"
            " or mamba_spec_state_layout is 'compact'"
        )
    else:
        assert refusal is None


def watch_boundary_states(engine: Engine, mutate: bool = False) -> Counter[str]:
    """Check, after each step's device work, every Mamba block the scheduler
    hands the connector that step: it holds the exact state at its boundary, a
    prompt position, and no state gather of the step touches it. ``mutate``
    checks the request's running state block in its place instead."""
    recorder = engine.scheduler.connector
    assert isinstance(recorder, BoundaryStateRecordingKVConnector)
    device = engine.device
    counts: Counter[str] = Counter()
    step = engine.step

    def checked_step() -> bool:
        busy = step()
        offloads, gathered = recorder.steps[-1]
        for request_id, offers in offloads.items():
            for group_id, block_id, boundary in offers:
                if mutate:
                    state = device.requests.get(request_id)
                    if state is None:
                        continue
                    block_id = state.table[group_id][state.state_idx]
                counts["offers"] += 1
                held = device.blocks.get(block_id)
                if (
                    held is None
                    or (held.rec, held.conv) != (boundary, boundary)
                    or boundary > engine.requests[request_id].num_prompt_tokens
                    or block_id in gathered
                ):
                    counts["violations"] += 1
        return busy

    engine.step = checked_step  # type: ignore[method-assign]
    return counts


@pytest.mark.parametrize("dcp", [1, DCP])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_a_boundary_state_connector_reads_exact_prefill_states(
    dcp: int, async_scheduling: bool
):
    engine = make_engine(
        "per_position",
        async_scheduling=async_scheduling,
        dcp=dcp,
        kv_transfer_config=boundary_state_connector(),
    )
    counts = watch_boundary_states(engine)
    session(engine, sid=1)
    session(engine, sid=2, eos={1: 3})
    # Short outputs: frontiers on and between Mamba block boundaries.
    session(engine, sid=3, first=100, out=(5, 21, 37), extra=(20,))
    assert_clean(engine)
    assert counts["offers"] > 0 and counts["violations"] == 0, counts


def test_the_boundary_state_check_catches_a_running_state_block():
    engine = make_engine(
        "per_position", dcp=DCP, kv_transfer_config=boundary_state_connector()
    )
    counts = watch_boundary_states(engine, mutate=True)
    session(engine)
    assert counts["violations"] > 0, counts


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_compact_layout_hands_each_frontier_to_a_storing_connector(
    async_scheduling: bool,
):
    """Under the compact layout the step that holds a finished request's state
    leaves it final, so a connector that stores frontier checkpoints gets each
    one at its exact length, and the checkpoint's Mamba blocks hold the state
    after that many tokens when the connector takes them."""
    engine = make_engine(
        "in_place",
        async_scheduling=async_scheduling,
        kv_transfer_config=storing_connector(),
        compact_layout=True,
    )
    connector = engine.scheduler.connector
    states: dict[str, list[tuple[int | None, int | None]]] = {}
    register = connector.register_finished_frontier

    def recording(request, frontier):
        states[request.request_id] = [
            (engine.device.blocks[block_id].rec, engine.device.blocks[block_id].conv)
            for group_id, block_id in frontier.blocks.items()
            if group_id in checkpoints(engine).mamba_groups
        ]
        register(request, frontier)

    connector.register_finished_frontier = recording
    lengths = session(engine)
    assert_clean(engine)
    assert [request_id for request_id, *_ in connector.frontiers] == [
        "s1t0",
        "s1t1",
        "s1t2",
    ]
    for turn, (request_id, frontier, _) in enumerate(connector.frontiers):
        # A step in flight at the finish has the request's last token in the
        # state.
        assert frontier.num_tokens in (lengths[turn], lengths[turn] - 1)
        assert states[request_id] == [(frontier.num_tokens, frontier.num_tokens)] * 2


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_session_resumes_at_its_frontiers_under_dcp(
    layout: str, async_scheduling: bool
):
    engine = make_engine(layout, async_scheduling=async_scheduling, dcp=DCP)
    session(engine)
    assert_clean(engine)
    manager = checkpoints(engine)
    assert (manager.page_size, manager.state_size) == (BLOCK, BLOCK // DCP)
    # Turn 1 ends on 1040, a Mamba block boundary inside a page.
    assert manager.frontiers.num_published == 3, manager.speculative.declined
    assert manager.frontiers.num_hits == 2


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("last_step", ["lands", "crosses"])
def test_block_boundary_frontier_inside_a_page_under_dcp(layout: str, last_step: str):
    """Turn 0 ends on 112, a Mamba block boundary inside a page. Its last
    verify step lands there with every draft accepted (a step from 108), or
    crosses into it (one accepted from 108, then a step from 110 cut at the
    stop). The frontier is the block ending at 112, where align mode left the
    state after the step: no draft column is gathered into it again."""

    def accept(request_id: str, start: int) -> int:
        return 1 if last_step == "crosses" and start == 108 else DRAFTS

    engine = make_engine(layout, async_scheduling=False, dcp=DCP, accept=accept)
    lengths = session(engine, turns=2, first=100, out=(13, 1), extra=(20,))
    assert_clean(engine)
    assert lengths[0] == 113
    assert engine.admissions["s1t1"] == [112]
    assert engine.device.counts["gathers"] == 0


@pytest.mark.parametrize("layout", LAYOUTS)
def test_block_boundary_frontiers_with_a_step_in_flight(layout: str):
    """Under asynchronous scheduling a turn that ends with a step in flight
    has its last token in the state: ends on a Mamba block boundary inside a
    page are resumed there too."""
    engine = make_engine(layout, async_scheduling=True, dcp=DCP)
    on_boundary = 0
    for sid, out in enumerate(range(8, 81), start=1):
        lengths = session(
            engine, sid=sid, turns=2, first=100, out=(out, 1), extra=(20,)
        )
        if lengths[0] % BLOCK == 0:
            # An end on a page boundary is the decode-boundary state's, which
            # a step in flight at the finish does not snapshot.
            continue
        (admitted,) = engine.admissions[f"s{sid}t1"]
        assert admitted in (lengths[0], lengths[0] - 1)
        if admitted == lengths[0] and admitted % (BLOCK // DCP) == 0:
            on_boundary += admitted % BLOCK != 0
    assert_clean(engine)
    assert on_boundary > 0


@pytest.mark.parametrize("layout", LAYOUTS)
def test_decode_boundary_states_sit_on_page_boundaries_under_dcp(layout: str):
    """The output crosses Mamba block boundaries inside pages and page
    boundaries; only the last page boundary keeps a decode-boundary state, as
    the attention pages before it are whole."""
    engine = make_engine(layout, async_scheduling=False, dcp=DCP)
    tokens = prompt(3, 3 * BLOCK - 40)
    engine.add("leaf", tokens, 300)
    engine.run("leaf")
    assert_clean(engine)
    num_tokens = len(engine.requests["leaf"].all_token_ids)
    boundary = (num_tokens - 1) // BLOCK * BLOCK
    decode = [
        u.num_tokens
        for u in checkpoints(engine).graph.units.values()
        if not u.frontier and u.num_tokens > len(tokens)
    ]
    assert boundary in decode, decode
    assert all(n % BLOCK == 0 for n in decode), decode


@pytest.mark.parametrize("layout", LAYOUTS)
def test_concurrent_sessions_under_pool_pressure_under_dcp(layout: str):
    rng = random.Random(13)
    engine = make_engine(
        layout,
        num_blocks=180,
        accept=lambda request_id, start: hash((request_id, start)) % (DRAFTS + 1),
        dcp=DCP,
    )
    tokens = {s: prompt(s, rng.randrange(200, 1400)) for s in range(10)}
    for turn in range(3):
        request_ids = [f"d{s}t{turn}" for s in range(10)]
        for s, request_id in enumerate(request_ids):
            engine.add(request_id, tokens[s], rng.choice((17, 90, 200)))
        engine.run(*request_ids)
        for s, request_id in enumerate(request_ids):
            tokens[s] = list(engine.requests[request_id].all_token_ids) + prompt(
                2000 + 10 * s + turn, rng.randrange(10, 90)
            )
    assert_clean(engine)
    manager = checkpoints(engine)
    assert manager.frontiers.num_published > 0 and manager.frontiers.num_hits > 0


def watch_frontier_jobs(engine: Engine) -> Counter[str]:
    """Check, after each step's device work, the frontier jobs a storing
    connector sent with the step: a state that waits for its draft-column
    gather is gathered by that step, and every job's Mamba blocks hold the
    state after exactly the frontier's tokens."""
    connector = engine.scheduler.connector
    assert isinstance(connector, FrontierStoringKVConnector)
    device = engine.device
    mamba = set(device.mamba_groups)
    counts: Counter[str] = Counter()
    step = engine.step

    def checked_step() -> bool:
        busy = step()
        jobs, gathered = connector.steps[-1]
        for frontier in jobs:
            counts["jobs"] += 1
            counts["gathered"] += frontier.gathered
            blocks = [b for g, b in frontier.blocks.items() if g in mamba]
            if frontier.gathered and not gathered.issuperset(blocks):
                counts["violations"] += 1
            for block_id in blocks:
                held = device.blocks.get(block_id)
                length = frontier.num_tokens
                if held is None or (held.rec, held.conv) != (length, length):
                    counts["violations"] += 1
        return busy

    engine.step = checked_step  # type: ignore[method-assign]
    return counts


@pytest.mark.parametrize("dcp", [1, DCP])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_a_storing_connector_reads_each_frontier_after_its_gather(
    dcp: int, async_scheduling: bool
):
    """Under one state column per draft position a finished request's state
    may wait for the next step's draft-column gather; the job that stores its
    frontier goes out with that step, which gathers first."""
    engine = make_engine(
        "per_position",
        async_scheduling=async_scheduling,
        dcp=dcp,
        kv_transfer_config=frontier_storing_connector(),
    )
    counts = watch_frontier_jobs(engine)
    session(engine, sid=1)
    session(engine, sid=2, eos={1: 3})
    session(engine, sid=3, first=100, out=(5, 21, 37), extra=(20,))
    assert_clean(engine)
    assert counts["jobs"] > 0 and counts["violations"] == 0, counts
    if not async_scheduling:
        # Without a step in flight a finish leaves its state in a draft column.
        assert counts["gathered"] > 0, counts


def test_the_frontier_job_check_catches_a_skipped_gather():
    engine = make_engine(
        "per_position",
        async_scheduling=False,
        fault="skip_gather",
        kv_transfer_config=frontier_storing_connector(),
    )
    counts = watch_frontier_jobs(engine)
    session(engine)
    assert counts["gathered"] > 0 and counts["violations"] > 0, counts


def test_a_gathered_frontier_is_stored_only_when_published(monkeypatch):
    """``publish`` queues a state's gather only when it publishes the
    frontier (a duplicate key does not), so a checkpoint whose state waits for
    a gather is described only then."""
    monkeypatch.setattr(
        FrontierIndex, "publish", lambda self, request, exact=None: None
    )
    engine = make_engine(
        "per_position",
        async_scheduling=False,
        accept=lambda rid, start: DRAFTS,
        kv_transfer_config=frontier_storing_connector(),
    )
    counts = watch_frontier_jobs(engine)
    session(engine)
    assert counts["gathered"] == 0 and counts["violations"] == 0, counts


def test_a_reset_waits_for_a_queued_frontier_job_and_its_gather():
    """A request finishes with its state in a draft column; the prefix cache
    reset before the step that gathers it fails and drops nothing, since the
    queued job reads the gathered state. Once that step and the job have run,
    the reset succeeds and no block stays held."""
    engine = make_engine(
        "per_position",
        async_scheduling=False,
        accept=lambda rid, start: DRAFTS,
        kv_transfer_config=frontier_storing_connector(),
    )
    # Deferred frees, as a KV connector under asynchronous scheduling has.
    engine.scheduler.defer_block_free = True
    counts = watch_frontier_jobs(engine)
    engine.add("r", prompt(1, 900), 41)
    while not engine.done("r"):
        engine.step()
    manager = checkpoints(engine)
    assert manager.frontiers.pending_gathers
    scheduler = engine.scheduler
    assert not scheduler.reset_prefix_cache()
    assert manager.frontiers.pending_gathers
    engine.step()
    assert (counts["jobs"], counts["gathered"], counts["violations"]) == (1, 1, 0)
    # The job's pins go with the next step.
    engine.step()
    assert scheduler.sched_step_seq == scheduler.processed_step_seq
    assert not scheduler.deferred_frees
    assert scheduler.reset_prefix_cache()
    pool = scheduler.kv_cache_manager.block_pool
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1
