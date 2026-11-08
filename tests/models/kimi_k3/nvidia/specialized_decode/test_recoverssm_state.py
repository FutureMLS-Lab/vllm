# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The runner side of the deferred RecoverSSM (``DeferredRecoverSSMState``) and how
the specialized decode installs it on ``KimiK3ModelState`` and the KDA metadata
builders: CPU only, the hybrid state's base class mocked."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadataBuilder
from vllm.models.kimi_k3.nvidia.model_state import KimiK3ModelState
from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm.state import (
    DeferredRecoverSSMState,
    _build_deferred,
    install_model_state,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


def _kda_builder(use_recoverssm: bool = True) -> KimiK3KDAMetadataBuilder:
    builder = KimiK3KDAMetadataBuilder.__new__(KimiK3KDAMetadataBuilder)
    builder.use_recoverssm = use_recoverssm
    builder.recoverssm_context = None
    builder.device = torch.device("cpu")
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=6)
    )
    return builder


class _Group:
    def __init__(self, builder) -> None:
        self.builder = builder

    def get_metadata_builder(self, ubatch_id: int = 0):
        assert ubatch_id == 0
        return self.builder


def _groups():
    deferring = [_kda_builder(), _kda_builder()]
    others = [SimpleNamespace(), _kda_builder(use_recoverssm=False)]
    groups = [
        [_Group(deferring[0]), _Group(others[0])],
        [_Group(deferring[1]), _Group(others[1])],
    ]
    return groups, deferring, others


def test_add_request_zeroes_the_pending_count() -> None:
    state = DeferredRecoverSSMState(4, torch.device("cpu"))
    assert isinstance(state, RecoverSSMState)
    assert state.pending.dtype == torch.int32 and state.pending.tolist() == [0] * 4
    state.pending.copy_(torch.tensor([3, 2, 4, 1], dtype=torch.int32))
    state.add_request(2)
    assert state.pending.tolist() == [3, 2, 0, 1]


def test_bind_installs_only_on_recoverssm_kda_builders() -> None:
    state = DeferredRecoverSSMState(4, torch.device("cpu"))
    groups, deferring, others = _groups()
    idx_mapping = torch.tensor([2, 0], dtype=torch.int32)
    state.bind(groups, idx_mapping)
    for builder in deferring:
        assert builder.recoverssm_pending is state.pending
        assert builder.recoverssm_request_slots is idx_mapping
        assert builder.recoverssm_record_rows.tolist() == [-1] * 6
        # Per-instance overrides; the class is unchanged.
        assert "build" in vars(builder)
        assert "_get_recoverssm_context" in vars(builder)
        assert type(builder) is KimiK3KDAMetadataBuilder
    for other in others:
        assert not hasattr(other, "recoverssm_pending")
        assert not hasattr(other, "recoverssm_request_slots")
        assert "build" not in vars(other)

    # Capture / dummy batches own no request: no slots, the same pending buffer,
    # installed once.
    builds = [builder.build for builder in deferring]
    state.bind(groups, None)
    for builder, build in zip(deferring, builds):
        assert builder.recoverssm_pending is state.pending
        assert builder.recoverssm_request_slots is None
        assert builder.build is build


def _fake_base_init(use_recoverssm: bool, frontiers: bool):
    def init(self, vllm_config, model, encoder_cache, device):
        self.max_num_reqs = 8
        self.device = device
        self.recoverssm = RecoverSSMState() if use_recoverssm else None
        self.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(mamba_frontier_checkpoint=frontiers),
            model_config=SimpleNamespace(max_model_len=1000),
        )

    return init


def _make_state(
    use_recoverssm: bool, model=None, frontiers: bool = False
) -> KimiK3ModelState:
    model = model or SimpleNamespace(specialized_decode_config=None)
    with patch.object(
        MambaHybridModelState, "__init__", _fake_base_init(use_recoverssm, frontiers)
    ):
        return KimiK3ModelState(None, model, None, torch.device("cpu"))


def test_model_state_hands_itself_to_the_specialized_decode() -> None:
    bound = []
    decode = SimpleNamespace(bind_model_state=bound.append)
    model = SimpleNamespace(specialized_decode_config=object())
    model.build_specialized_decode = lambda: setattr(
        model, "specialized_decode", decode
    )
    state = _make_state(use_recoverssm=True, model=model)
    assert bound == [state]
    # Without the specialized decode the stock state is left as it is.
    assert type(_make_state(use_recoverssm=True).recoverssm) is RecoverSSMState


def test_install_model_state_only_with_recoverssm() -> None:
    state = _make_state(use_recoverssm=True)
    install_model_state(state)
    assert type(state.recoverssm) is DeferredRecoverSSMState
    assert state.recoverssm.pending.shape == (8,)
    plain = _make_state(use_recoverssm=False)
    install_model_state(plain)
    assert plain.recoverssm is None
    assert "add_request" not in vars(plain) and "prepare_attn" not in vars(plain)


def test_installed_add_request_resets_the_slot_after_the_base() -> None:
    calls = []

    def base_add_request(self, req_index, new_req_data):
        calls.append((req_index, self.recoverssm.pending[req_index].item()))

    with patch.object(MambaHybridModelState, "add_request", base_add_request):
        state = _make_state(use_recoverssm=True)
        install_model_state(state)
        state.recoverssm.pending.fill_(3)
        state.add_request(5, SimpleNamespace())
    assert calls == [(5, 3)]
    assert state.recoverssm.pending.tolist() == [3] * 5 + [0] + [3] * 2


def test_installed_prepare_attn_binds_before_the_metadata_build() -> None:
    groups, deferring, _ = _groups()
    idx_mapping = torch.tensor([1, 4, 0], dtype=torch.int32)
    input_batch = SimpleNamespace(
        idx_mapping=idx_mapping, idx_mapping_np=np.array([1, 4, 0])
    )
    seen = []

    def base_prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
        model_specific_attn_metadata=None,
    ):
        seen.append(
            [
                (
                    getattr(b, "recoverssm_pending", None) is self.recoverssm.pending,
                    b.recoverssm_request_slots,
                )
                for b in deferring
            ]
        )
        return {"for_capture": for_capture}

    with patch.object(MambaHybridModelState, "prepare_attn", base_prepare_attn):
        state = _make_state(use_recoverssm=True)
        install_model_state(state)
        out = state.prepare_attn(input_batch, None, (), None, groups, None)
        assert out == {"for_capture": False}
        assert seen[-1] == [(True, idx_mapping), (True, idx_mapping)]

        out = state.prepare_attn(
            input_batch, None, (), None, groups, None, for_capture=True
        )
        assert out == {"for_capture": True}
        assert seen[-1] == [(True, None), (True, None)]

    # Without the install (stock) nothing is bound.
    plain = _make_state(use_recoverssm=True)
    groups, deferring, _ = _groups()
    with patch.object(MambaHybridModelState, "prepare_attn", lambda self, *a, **k: {}):
        plain.prepare_attn(input_batch, None, (), None, groups, None)
    assert all(getattr(b, "recoverssm_pending", None) is None for b in deferring)


def _new_request(prompt_len: int, max_tokens: int | None):
    params = None if max_tokens is None else SimpleNamespace(max_tokens=max_tokens)
    return SimpleNamespace(prompt_len=prompt_len, sampling_params=params)


def test_add_request_keeps_the_length_limit_with_frontiers() -> None:
    # The device's stop condition: min(prompt_len + max_tokens, max_model_len).
    state = DeferredRecoverSSMState(4, torch.device("cpu"), max_model_len=1000)
    state.add_request(0, _new_request(300, 200))
    state.add_request(1, _new_request(900, 200))
    state.add_request(2, _new_request(300, None))
    assert state.max_len is not None
    assert state.max_len.tolist() == [500, 1000, 1000, 1000]
    # Without frontier checkpoints no limit is kept.
    plain = DeferredRecoverSSMState(4, torch.device("cpu"))
    plain.add_request(0, _new_request(300, 200))
    assert plain.max_len is None


def test_bind_hands_the_batch_rows_length_limits() -> None:
    state = DeferredRecoverSSMState(4, torch.device("cpu"), max_model_len=1000)
    for slot, max_tokens in enumerate((10, 20, 30, 40)):
        state.add_request(slot, _new_request(100, max_tokens))
    groups, deferring, _ = _groups()
    idx_mapping = torch.tensor([2, 0], dtype=torch.int32)
    state.bind(groups, idx_mapping, np.array([2, 0]))
    for builder in deferring:
        assert builder.recoverssm_length_limits.tolist() == [130, 110]
    # Capture and dummy batches: no limits.
    state.bind(groups, None)
    assert all(b.recoverssm_length_limits is None for b in deferring)


def test_installed_state_keeps_limits_only_with_frontiers() -> None:
    for frontiers in (False, True):
        state = _make_state(use_recoverssm=True, frontiers=frontiers)
        install_model_state(state)
        deferred = state.recoverssm
        assert isinstance(deferred, DeferredRecoverSSMState)
        assert (deferred.max_len is not None) is frontiers


BLOCK = 1024
NUM_SPEC = 5


def _deferred_builder(limits: list[int] | None) -> KimiK3KDAMetadataBuilder:
    """A KDA builder bound by a deferred RecoverSSM state: batch rows 0-3 are the
    request slots 3, 1, 0, 2, whose length limits are ``limits`` (None: no
    frontier checkpoints)."""
    slots = [3, 1, 0, 2]
    state = DeferredRecoverSSMState(
        6, torch.device("cpu"), None if limits is None else 5000
    )
    for slot, limit in zip(slots, limits or ()):
        state.add_request(slot, _new_request(100, limit - 100))
    builder = _kda_builder()
    builder.num_spec = NUM_SPEC
    spec: Any = MambaSpec(
        block_size=BLOCK,
        shapes=((4,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    builder.kv_cache_spec = spec
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=6),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
    )
    state.bind(
        [[_Group(builder)]], torch.tensor(slots, dtype=torch.int32), np.array(slots)
    )
    return builder


@pytest.mark.parametrize(
    "limits,eager",
    [
        # Row 1 may cross a block boundary within this step or the next.
        (None, [0, 1, 0, 0]),
        # Row 2 may reach its length limit (710) by then; row 3 (800) not.
        ([5000, 5000, 710, 800], [0, 1, 1, 0]),
        # A request whose step reaches its limit exactly (701 = 700 + 1).
        ([5000, 5000, 5000, 701], [0, 1, 0, 1]),
    ],
)
def test_build_deferred_commits_rows_that_may_reach_their_length_limit(
    limits: list[int] | None, eager: list[int]
) -> None:
    q = NUM_SPEC + 1
    upper = torch.tensor([500, 1020, 700, 700], dtype=torch.int32)
    m = SimpleNamespace(
        num_reqs=4,
        seq_lens_cpu_upper_bound=upper,
        query_start_loc_cpu=torch.arange(0, 5 * q, q, dtype=torch.int32),
        is_prefilling=torch.zeros(4, dtype=torch.bool),
    )
    builder = _deferred_builder(limits)
    # The eager mask's host-to-device copy (pinned memory needs a GPU).
    with patch(
        "vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm.state."
        "async_tensor_h2d",
        lambda data, device: data.to(device),
    ):
        deferred = _build_deferred(
            builder,
            m,
            SimpleNamespace(),
            4,
            None,
            torch.full((4,), NUM_SPEC, dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
        )
    assert deferred.record_rows[:4].tolist() == [3, 1, 0, 2]
    assert deferred.eager is not None and deferred.eager.tolist() == eager
