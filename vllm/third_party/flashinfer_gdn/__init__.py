# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
"""FlashInfer's Blackwell (SM100) chunked Gated DeltaNet prefill, exporting the
recurrent state at one position per sequence inside the call.

The three modules are copies of FlashInfer 0.7.0.post1's
``flashinfer/gdn_kernels/blackwell/`` files (Apache-2.0 for ``gdn_prefill.py``,
see ``LICENSE``, and NVIDIA's BSD-3-Clause, in each kernel file's header). The
changes:

- FlashInfer writes states at one stride shared by the whole batch. Here each
  sequence has one offset (FlashKDA's checkpoint interface): the state after
  ``min(c, L)`` tokens when ``c > 0``, ``c % 16 == 0`` and ``c <= 16 * ceil(L / 16)``,
  written to the sequence's row of ``checkpoint_state``.
- The chunk holding the position is the bound of the chunk loops, so every other
  chunk runs FlashInfer's chunk body unchanged and every operand of the call's own
  matrix products keeps its bits: the output and the final state equal FlashInfer's
  call without checkpoints, and the exported state equals FlashInfer's final state
  of the sequence cut at the position, on the same (non context-parallel) route.
- ``v_split=2`` runs each (sequence, value head) as two tiles that own 64 of the
  state's 128 rows along V (the rows evolve independently and every matrix
  product keeps its reduction tiling), so calls with few sequences use twice as
  many SMs; the tile scheduler runs every other round of tiles backwards, and
  split tiles issue the state chain's matrix products first, keep deeper
  shared-memory rings, run the two halves of a (sequence, head) as one 2-CTA
  cluster and give compute group 0 more registers than compute group 1 (248
  against 232).
- The forget gate arrives in log space and the gate warp applies the accurate
  ``expf`` that ``torch.exp`` uses; state tiles are loaded and stored in
  coalesced rows; the kernel is launched with programmatic dependent launch and
  waits for the preceding grid before touching global memory.
- ``gdn_prefill.py`` takes the per-sequence offsets, an optional row of
  ``checkpoint_state`` per sequence and ``v_split``, and compiles under its own
  CuTe DSL module name, so it never shares FlashInfer's kernel cache.
"""

from vllm.third_party.flashinfer_gdn.gdn_prefill import (
    chunk_gated_delta_rule_export_sm100,
)

__all__ = ["chunk_gated_delta_rule_export_sm100"]
