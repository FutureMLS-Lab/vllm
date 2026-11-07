# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An SSD intermediate state equals an independent run over the same prefix.

This is the central claim of internal prefill checkpoints: the state the SSD
scan already materializes at the checkpoint's chunk boundary is exactly the
state a fresh prefill over `[0, checkpoint)` would end with, so it can be
cached and later resumed from. The store test covers the other half: moving
that state, and the conv window, into the paged cache. The last test checks
both bit for bit at Nemotron 3 Super's shapes, with the offsets in-chunk
export (``mamba_prefill_export``) asks for.
"""

import math

import pytest
import torch

from tests.kernels.mamba.test_mamba_ssm_ssd import generate_random_inputs
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.ops.ssd_checkpoint import (
    store_prefill_checkpoint,
)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

DEVICE = current_platform.device_type
compute_chunk_metadata = BaseMambaAttentionMetadataBuilder._compute_chunk_metadata


def _run(inputs, seqlen, chunk_size, checkpoint_offset):
    """Run one prefill of `seqlen` tokens, optionally exporting a checkpoint.

    Returns (states, checkpoint_chunk_index). `states` holds every chunk's
    state when a checkpoint was requested, else the per-sequence final state.
    """
    A, dt, X, B, C = inputs
    offsets = [checkpoint_offset] if checkpoint_offset else None
    cu, seq_idx, last, ckpt = compute_chunk_metadata(
        chunk_size,
        1,
        torch.tensor([0]),
        torch.tensor([0, seqlen]),
        checkpoint_offsets_p=offsets,
    )

    x = X[:seqlen]
    out = torch.empty_like(x)
    states = mamba_chunk_scan_combined_varlen(
        x,
        dt[:seqlen],
        A,
        B[:seqlen],
        C[:seqlen],
        chunk_size,
        cu_seqlens=torch.tensor([0, seqlen], dtype=torch.int32, device=DEVICE),
        cu_chunk_seqlens=torch.tensor(cu, dtype=torch.int32, device=DEVICE),
        last_chunk_indices=torch.tensor(last, dtype=torch.int32, device=DEVICE),
        seq_idx=torch.tensor(seq_idx, dtype=torch.int32, device=DEVICE),
        out=out,
        D=None,
        return_intermediate_states=checkpoint_offset > 0,
        state_dtype=torch.float32,
    )
    return states, ckpt[0]


@pytest.mark.parametrize("checkpoint_offset", [32, 48, 96])
def test_checkpoint_state_matches_an_independent_prefix_run(checkpoint_offset):
    """states[checkpoint_chunk_idx] == the final state of a prefix-only run.

    48 is deliberately not a multiple of chunk_size: align-mode block sizes
    carry no chunk_size factor, so off-grid checkpoints are the common case
    and are what the chunk split exists to support.
    """
    seqlen, chunk_size, n_heads, d_head = 128, 32, 4, 32
    inputs = generate_random_inputs(1, seqlen, n_heads, d_head, torch.float32)
    inputs = tuple(t.squeeze(0) if t.dim() > 1 else t for t in inputs)

    states, ckpt_idx = _run(inputs, seqlen, chunk_size, checkpoint_offset)
    assert ckpt_idx >= 0, "the checkpoint did not land on a chunk boundary"
    checkpoint_state = states[ckpt_idx]

    prefix_final, _ = _run(inputs, checkpoint_offset, chunk_size, 0)

    torch.testing.assert_close(
        checkpoint_state.to(torch.float32),
        prefix_final[0].to(torch.float32),
        atol=8e-3,
        rtol=5e-3,
    )


def test_checkpoint_does_not_perturb_the_final_state():
    """Exporting a checkpoint must not change what the prefill produces.

    The chunk split is an extra boundary, not a different computation, so the
    sequence's final state has to match a run that took no checkpoint.
    """
    seqlen, chunk_size, n_heads, d_head = 128, 32, 4, 32
    inputs = generate_random_inputs(1, seqlen, n_heads, d_head, torch.float32)
    inputs = tuple(t.squeeze(0) if t.dim() > 1 else t for t in inputs)

    with_ckpt, ckpt_idx = _run(inputs, seqlen, chunk_size, 48)
    without_ckpt, _ = _run(inputs, seqlen, chunk_size, 0)

    torch.testing.assert_close(
        with_ckpt[-1].to(torch.float32),
        without_ckpt[0].to(torch.float32),
        atol=8e-3,
        rtol=5e-3,
    )


@pytest.mark.parametrize("spec_slots", [0, 3])
@pytest.mark.parametrize("dim_first", [True, False])
def test_store_selects_the_checkpoint_from_all_chunk_states(dim_first, spec_slots):
    """The store reads the checkpoint row out of every chunk state.

    Row 0 checkpoints 32 tokens in, its state being chunk 3 rather than the
    program's own row, so the store only passes if ``chunk_idx`` is used.
    Row 1 declines and must write nothing. The conv input is a column slice of
    a wider projection, the SD conv layout is a transposed view, and SSM rows
    are padded pages, matching the strides the mixer passes. Spec decoding
    widens the conv state, and the store must still fill only the first
    ``state_len`` slots with the newest inputs.
    """
    dim, state_len, heads, head_dim, d_state = 16, 4, 2, 4, 8
    projection = torch.randn(64, dim + 8, device=DEVICE)
    conv_input = projection[:, :dim]
    num_slots = state_len + spec_slots
    if dim_first:
        conv_state = torch.zeros(3, dim, num_slots, device=DEVICE)
    else:
        conv_state = torch.zeros(3, num_slots, dim, device=DEVICE).transpose(-1, -2)
    varlen_states = torch.randn(5, heads, head_dim, d_state, device=DEVICE)
    # Blocks are pages holding more than the SSM state, so rows are padded.
    row = heads * head_dim * d_state
    ssm_storage = torch.zeros(3, row + 8, device=DEVICE)
    ssm_state = ssm_storage[:, :row].view(3, heads, head_dim, d_state)
    checkpoint = MambaPrefillCheckpointMetadata(
        checkpoint_offsets=torch.tensor([32, 0], dtype=torch.int32, device=DEVICE),
        state_indices=torch.tensor(
            [2, NULL_BLOCK_ID], dtype=torch.int32, device=DEVICE
        ),
        offsets=[32, 0],
    )

    store_prefill_checkpoint(
        checkpoint,
        torch.tensor([3, 0], device=DEVICE),
        conv_input.T,
        conv_state,
        varlen_states,
        ssm_state,
        torch.tensor([0, 40, 64], dtype=torch.int32, device=DEVICE),
        state_len=state_len,
    )

    exact = dict(rtol=0, atol=0)
    torch.testing.assert_close(
        conv_state[2, :, :state_len], conv_input[32 - state_len : 32].T, **exact
    )
    assert not conv_state[2, :, state_len:].any(), "wrote into the spec slots"
    torch.testing.assert_close(ssm_state[2], varlen_states[3], **exact)
    assert not conv_state[:2].any() and not ssm_storage[:2].any()
    assert not ssm_storage[:, row:].any(), "wrote into the page padding"


# Nemotron 3 Super's Mamba2 layer at tensor parallel 2, per GPU: 64 heads of
# 64 channels, 4 groups with a 128-wide state, chunks of 128, and a
# convolution over x, B and C (5,120 channels) that keeps its last 3 inputs.
# The model keeps its SSM state in float32.
NEMOTRON_HEADS, NEMOTRON_HEAD_DIM = 64, 64
NEMOTRON_GROUPS, NEMOTRON_STATE = 4, 128
NEMOTRON_CHUNK, NEMOTRON_CONV_DIM, NEMOTRON_CONV_STATE_LEN = 128, 5120, 3


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(
        torch.int16 if tensor.element_size() == 2 else torch.int32
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Checked on CUDA.")
@torch.inference_mode()
def test_prefill_export_is_a_cut_run_bit_for_bit_at_nemotron_shapes():
    """The mixer's prefill step on Nemotron 3 Super's per-GPU shapes, with
    the checkpoints in-chunk export asks for.

    Each row resumes from ``computed`` tokens and exports the state ``offset``
    tokens into its query: at a block boundary after a block-aligned start, at
    the next block boundary after a start inside a block (and inside a chunk),
    at a 2,080-token hash unit off the 128-token chunk grid, and one token past
    a chunk end; one row exports nothing.

    Each exported SSM state is, bit for bit, the final state of its row cut at
    the offset, from the same computed tokens and initial state, as a chunk
    that ends there leaves it; the state 128 tokens later differs. Each conv
    window is the 3 inputs before the offset. A row whose offset is on the
    chunk grid, or that has none, keeps the output and final state of the step
    without export, bit for bit; an offset off the grid adds a chunk, which
    may change the row's rounding. The null block, a row's unused checkpoint
    block and a guard block stay untouched.
    """
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(0)
    # (computed tokens, query tokens, export offset into the query)
    rows = [
        (8320, 9000, 8320),
        (2080, 7000, 6240),
        (0, 5000, 2080),
        (4160, 3000, 0),
        (6240, 4000, 2080),
        (0, 300, 129),
    ]
    starts = [0]
    for _, length, _ in rows:
        starts.append(starts[-1] + length)
    num_tokens = starts[-1]

    def randn(*shape, scale=1.0, dtype=torch.bfloat16):
        return (scale * torch.randn(*shape, generator=gen)).to(device, dtype)

    x = randn(num_tokens, NEMOTRON_HEADS, NEMOTRON_HEAD_DIM)
    B = randn(num_tokens, NEMOTRON_GROUPS, NEMOTRON_STATE)
    C = randn(num_tokens, NEMOTRON_GROUPS, NEMOTRON_STATE)
    dt = randn(num_tokens, NEMOTRON_HEADS, scale=0.5)
    conv_input = randn(num_tokens, NEMOTRON_CONV_DIM)
    # Mamba2's initialization: A = -U(1, 16), dt_bias the inverse softplus of
    # a log-uniform dt in [1e-3, 1e-1], D near 1.
    A = -torch.empty(NEMOTRON_HEADS).uniform_(1, 16, generator=gen).to(device)
    dt0 = torch.exp(
        torch.empty(NEMOTRON_HEADS).uniform_(
            math.log(1e-3), math.log(1e-1), generator=gen
        )
    )
    dt_bias = (dt0 + torch.log(-torch.expm1(-dt0))).to(device)
    D = randn(NEMOTRON_HEADS, scale=0.1, dtype=torch.float32) + 1.0
    state_shape = (NEMOTRON_HEADS, NEMOTRON_HEAD_DIM, NEMOTRON_STATE)
    initial_state = randn(len(rows), *state_shape, scale=0.1, dtype=torch.float32)

    # Block 0 is the null block, then a state and a checkpoint block per row
    # and a guard block. Every block starts as NaN, so any write shows.
    state_blocks = [1 + row for row in range(len(rows))]
    checkpoint_blocks = [1 + len(rows) + row for row in range(len(rows))]
    guard = 1 + 2 * len(rows)

    def ints(values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device=device)

    def step(pieces, export):
        """The mixer's prefill SSM step over ``pieces``: (row, tokens, offset)
        each, from the row's first token. Returns the output and the caches."""
        ssm = torch.full((guard + 1, *state_shape), float("nan"), device=device)
        conv = torch.full(
            (guard + 1, NEMOTRON_CONV_STATE_LEN, NEMOTRON_CONV_DIM),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
        for row, (computed, _, _) in enumerate(rows):
            if computed:
                ssm[state_blocks[row]] = initial_state[row]
        index = torch.cat(
            [torch.arange(starts[row], starts[row] + n) for row, n, _ in pieces]
        ).to(device)
        cu = [0]
        for _, n, _ in pieces:
            cu.append(cu[-1] + n)
        computed = [rows[row][0] for row, _, _ in pieces]
        offsets = [offset for _, _, offset in pieces] if export else None
        cu_chunk, seq_idx, last_chunk, ckpt_idx = compute_chunk_metadata(
            NEMOTRON_CHUNK,
            len(pieces),
            torch.tensor(computed),
            torch.tensor(cu),
            offsets,
        )
        read = ints([state_blocks[row] for row, _, _ in pieces])
        initial_states = None
        if any(computed):
            has_initial = ints([c > 0 for c in computed], torch.bool)
            initial_states = torch.where(has_initial[:, None, None, None], ssm[read], 0)
        out = torch.empty(
            len(index),
            NEMOTRON_HEADS,
            NEMOTRON_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        states = mamba_chunk_scan_combined_varlen(
            x[index],
            dt[index],
            A,
            B[index],
            C[index],
            chunk_size=NEMOTRON_CHUNK,
            cu_seqlens=ints(cu),
            cu_chunk_seqlens=ints(cu_chunk),
            last_chunk_indices=ints(last_chunk),
            seq_idx=ints(seq_idx),
            out=out,
            D=D,
            z=None,
            dt_bias=dt_bias,
            initial_states=initial_states,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
            return_intermediate_states=export,
            state_dtype=ssm.dtype,
            final_states_out=None if export else ssm,
            final_state_indices=None if export else read,
        )
        if export:
            final_states = states[ints(last_chunk)]
            checkpoint = MambaPrefillCheckpointMetadata(
                checkpoint_offsets=ints(offsets),
                state_indices=ints(
                    [
                        checkpoint_blocks[row] if offset else NULL_BLOCK_ID
                        for row, _, offset in pieces
                    ]
                ),
                offsets=offsets,
            )
            store_prefill_checkpoint(
                checkpoint,
                ints([max(i, 0) for i in ckpt_idx], torch.int64),
                conv_input[index].T,
                conv.transpose(-1, -2),
                states,
                ssm,
                ints(cu),
                state_len=NEMOTRON_CONV_STATE_LEN,
            )
            ssm[read] = final_states
        return out, ssm, conv

    out, ssm, conv = step([(row, n, c) for row, (_, n, c) in enumerate(rows)], True)
    out_off, ssm_off, _ = step(
        [(row, n, 0) for row, (_, n, _) in enumerate(rows)], False
    )

    for row, (computed, length, offset) in enumerate(rows):
        tokens = slice(starts[row], starts[row] + length)
        if not offset or (computed + offset) % NEMOTRON_CHUNK == 0:
            assert torch.equal(_bits(out[tokens]), _bits(out_off[tokens])), row
            block = state_blocks[row]
            assert torch.equal(_bits(ssm[block]), _bits(ssm_off[block])), row
        if not offset:
            assert ssm[checkpoint_blocks[row]].isnan().all(), row
            assert conv[checkpoint_blocks[row]].isnan().all(), row
            continue
        exported = ssm[checkpoint_blocks[row]]
        _, cut, _ = step([(row, offset, 0)], False)
        assert torch.equal(_bits(exported), _bits(cut[state_blocks[row]])), row
        _, later, _ = step([(row, offset + NEMOTRON_CHUNK, 0)], False)
        assert not torch.equal(_bits(exported), _bits(later[state_blocks[row]])), row
        end = starts[row] + offset
        window = conv_input[end - NEMOTRON_CONV_STATE_LEN : end]
        assert torch.equal(_bits(conv[checkpoint_blocks[row]]), _bits(window)), row
    for block in (NULL_BLOCK_ID, guard):
        assert ssm[block].isnan().all() and conv[block].isnan().all(), block
