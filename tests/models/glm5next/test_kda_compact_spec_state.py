# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA's compact speculative state against its per-position
verify.

Random q, k, v, gate and beta logits for a batch of verify steps go through
both paths with GLM-5.3-Flash's bounded gate:

- per position: a float32 transcription of GLM's recurrent KDA step gives the
  state after each of the step's tokens (one column per position) and the
  outputs;
- compact: upstream's ``kda_recoverssm_verify`` reads the state, writes
  nothing to it and records per-position corrections and keys in the layer's
  records (a row per verify request, outside the KV cache); the commit
  context, built over GLM's layer view, then folds the accepted tokens into
  the state (and, in align mode, writes the state at a block boundary the
  accepted tokens reached into the block before it).

The outputs agree, and the committed state for ``a`` accepted tokens agrees
with per-position column ``a - 1``, within float32 rounding (the fold sums in
another order). The convolution window ends at the accepted position.
"""

import os
import random
from types import SimpleNamespace

import pytest
import torch

from vllm.models.glm5next.common import kda as glm_kda
from vllm.models.kimi_k3.nvidia.ops import recoverssm
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs CUDA or the Triton interpreter",
)

DEVICE = torch.device("cuda" if current_platform.is_cuda() else "cpu")
HEADS, DIM, QUERY_LEN, CONV = 2, 32, 4, 4
LOWER_BOUND = -5.0


def layer(conv, state, records, a_log, dt_bias):
    """The attributes of a GLM KDA layer that ``_recoverssm_view`` reads;
    ``A_log`` four-dimensional, as the model keeps it."""
    return SimpleNamespace(
        kv_cache=[conv, state],
        recoverssm_records=records,
        A_log=a_log,
        dt_bias=dt_bias,
        local_num_heads=HEADS,
        head_dim=DIM,
        kda_lower_bound=LOWER_BOUND,
    )


def per_position(state, q, k, v, raw_g, raw_beta, a_log, dt_bias):
    """GLM's recurrent KDA, one token at a time: l2-normalized q and k, q
    scaled by ``DIM ** -0.5``, the bounded gate
    ``LOWER_BOUND * sigmoid(exp(A_log) * (g + dt_bias))`` and beta through a
    sigmoid. Returns the outputs [T, H, D] and the state after every token
    [T, H, D, D]."""
    h = state.clone().float()
    outs, states = [], []
    a = torch.exp(a_log.view(-1).float())
    bias = dt_bias.view(HEADS, DIM).float()
    for t in range(q.shape[1]):
        qt, kt, vt = (x[0, t].float() for x in (q, k, v))
        qt = qt / torch.sqrt((qt * qt).sum(-1, keepdim=True) + 1e-6) * DIM**-0.5
        kt = kt / torch.sqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
        gate = LOWER_BOUND / (
            1.0 + torch.exp(-(a[:, None] * (raw_g[0, t].float() + bias)))
        )
        h = h * torch.exp(gate)[:, None, :]
        vt = vt - (h * kt[:, None, :]).sum(-1)
        vt = vt * torch.sigmoid(raw_beta[0, t].float())[:, None]
        h = h + vt[:, :, None] * kt[:, None, :]
        outs.append((h * qt[:, None, :]).sum(-1))
        states.append(h.clone())
    return torch.stack(outs), torch.stack(states)


def inputs(seed: int, num_steps: int):
    generator = torch.Generator().manual_seed(seed)
    num_tokens = num_steps * QUERY_LEN

    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    q, k, v, raw_g = (
        randn(1, num_tokens, HEADS, DIM).to(torch.bfloat16) for _ in range(4)
    )
    raw_beta = randn(1, num_tokens, HEADS).to(torch.bfloat16)
    a_log = (randn(1, 1, HEADS, 1) * 0.3).float()
    dt_bias = (randn(HEADS * DIM) * 0.1).float()
    return tuple(x.to(DEVICE) for x in (q, k, v, raw_g, raw_beta, a_log, dt_bias))


def verify(q, k, v, raw_g, raw_beta, a_log, dt_bias, state, cu, state_indices):
    records = recoverssm.KDARecoverSSMRecords.allocate(
        state_indices.shape[0], HEADS, QUERY_LEN, DIM, torch.bfloat16, DEVICE
    )
    out = recoverssm.kda_recoverssm_verify(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=a_log.view(-1),
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        checkpoint_state=state,
        records=records,
        query_start_loc=cu,
        state_indices=state_indices,
        spec_query_len=QUERY_LEN,
    )
    return out, records


@pytest.mark.parametrize("seed", range(3))
def test_verify_and_commit_match_the_per_position_columns(seed: int):
    rng = random.Random(seed)
    num_steps = 3
    q, k, v, raw_g, raw_beta, a_log, dt_bias = inputs(seed, num_steps)
    cu = torch.arange(
        0, (num_steps + 1) * QUERY_LEN, QUERY_LEN, dtype=torch.int32, device=DEVICE
    )
    num_blocks = 1 + num_steps * QUERY_LEN + 2
    initial = 0.2 * torch.randn(
        num_blocks, HEADS, DIM, DIM, generator=torch.Generator().manual_seed(seed)
    ).to(DEVICE)

    # Per position: step r starts from block 1 + r * QUERY_LEN, one column
    # per token.
    expected_out = torch.zeros(num_steps * QUERY_LEN, HEADS, DIM, device=DEVICE)
    columns = initial.clone()
    for r in range(num_steps):
        tokens = slice(r * QUERY_LEN, (r + 1) * QUERY_LEN)
        out, states = per_position(
            initial[1 + r * QUERY_LEN],
            q[:, tokens],
            k[:, tokens],
            v[:, tokens],
            raw_g[:, tokens],
            raw_beta[:, tokens],
            a_log,
            dt_bias,
        )
        expected_out[tokens] = out
        columns[1 + r * QUERY_LEN : 1 + (r + 1) * QUERY_LEN] = states

    # Compact: step r's state is block 1 + r * QUERY_LEN, its records row r.
    state = initial.clone()
    state_indices = torch.arange(
        1, 1 + num_steps * QUERY_LEN, QUERY_LEN, dtype=torch.int32, device=DEVICE
    )
    out, records = verify(
        q, k, v, raw_g, raw_beta, a_log, dt_bias, state, cu, state_indices
    )
    assert torch.equal(state, initial), "verify must not write the state"
    torch.testing.assert_close(out[0].float(), expected_out, atol=2e-2, rtol=2e-2)

    accepted = torch.tensor(
        [rng.randrange(1, QUERY_LEN + 1) for _ in range(num_steps)],
        dtype=torch.int32,
        device=DEVICE,
    )
    width = CONV - 1 + QUERY_LEN - 1
    # Token-major window; row i of each step's window holds the value i + 1.
    conv = torch.zeros(num_blocks, width, 3 * HEADS * DIM, dtype=torch.bfloat16)
    conv[:, :, :] = torch.arange(1, width + 1, dtype=torch.bfloat16)[:, None]
    conv = conv.to(DEVICE)
    context = recoverssm.KDARecoverSSMCommitContext.create(
        [glm_kda._recoverssm_view(layer(conv, state, records, a_log, dt_bias))],
        spec_query_len=QUERY_LEN,
        max_num_reqs=8,
    )
    context.commit(accepted, state_indices, cu)
    for r in range(num_steps):
        a = int(accepted[r])
        block = 1 + r * QUERY_LEN
        torch.testing.assert_close(
            state[block], columns[block + a - 1], atol=1e-4, rtol=1e-3
        )
        # The window now starts at the accepted position.
        assert float(conv[block, 0, 0]) == a


def test_align_commit_writes_the_boundary_state():
    """A step that crosses a block boundary: the final state goes to the
    final position's block and the state at the boundary to the block before
    it, each equal to the matching per-position column."""
    q, k, v, raw_g, raw_beta, a_log, dt_bias = inputs(7, 1)
    cu = torch.tensor([0, QUERY_LEN], dtype=torch.int32, device=DEVICE)
    num_blocks = 8
    initial = 0.2 * torch.randn(
        num_blocks, HEADS, DIM, DIM, generator=torch.Generator().manual_seed(9)
    ).to(DEVICE)
    _, columns = per_position(initial[2], q, k, v, raw_g, raw_beta, a_log, dt_bias)

    state = initial.clone()
    # The pre-copy moved the state into the new block.
    state[6] = initial[2]
    state_indices = torch.tensor([6], dtype=torch.int32, device=DEVICE)
    _, records = verify(
        q, k, v, raw_g, raw_beta, a_log, dt_bias, state, cu, state_indices
    )
    conv = torch.zeros(
        num_blocks,
        CONV - 1 + QUERY_LEN - 1,
        3 * HEADS * DIM,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    context = recoverssm.KDARecoverSSMCommitContext.create(
        [glm_kda._recoverssm_view(layer(conv, state, records, a_log, dt_bias))],
        spec_query_len=QUERY_LEN,
        max_num_reqs=4,
    )
    # Block size 8: 6 tokens computed, all 4 accepted, so the state after 10
    # tokens goes to block 1 (id 6) and the state at 8 to block 0 (id 7).
    block_table = torch.tensor([[7, 6, 0, 0]], dtype=torch.int32, device=DEVICE)
    context.commit(
        torch.tensor([4], dtype=torch.int32, device=DEVICE),
        state_indices,
        cu,
        block_table=block_table,
        num_computed_tokens=torch.tensor([6], dtype=torch.int32, device=DEVICE),
        mamba_block_size=8,
    )
    torch.testing.assert_close(state[6], columns[3], atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(state[7], columns[1], atol=1e-4, rtol=1e-3)
