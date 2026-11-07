# Hybrid State Checkpoints

Hybrid models interleave full-attention layers with recurrent layers whose state vLLM manages as Mamba state (Mamba, gated delta rule, Kimi delta attention and similar). With prefix caching in Mamba `align` mode, vLLM caches a recurrent state only on the block grid (see [Hybrid KV Cache Manager](hybrid_kv_cache_manager.md#case-3-mamba-models)). This document describes three opt-in extensions, implemented in `vllm/v1/core/hybrid_checkpoint/`, that keep more of those states resumable and choose which ones to keep.

| Option | Default | Effect |
| --- | --- | --- |
| `mamba_frontier_checkpoint` | `False` | Caches a finished request's state at its exact end, which may lie inside a block, and at the last block boundary its output crossed. |
| `mamba_prefill_export` | `False` | Lets a non-final prefill chunk run past one position where a state must be cached, and has the backend write the state there. |
| `prefix_cache_eviction` with `prefill_cost_coefficients` | `"lru"`, `None` | With `"gds"`, evicts cached states, each with the attention pages only it needs, by GreedyDual-Size on the prefill time they save. |
| `prefix_cache_leaf_last`, `prefix_cache_leaf_idle_seconds` | `False`, `None` | With `"gds"`, evicts every state that has a deeper cached state on its path before any session's latest state, optionally demoting a latest state unused for that long. |

How to enable them is described in [Automatic Prefix Caching](../features/automatic_prefix_caching.md#resuming-past-the-block-grid).

## Motivation

In `align` mode a request's recurrent state is cached at the block boundaries its prefill reaches. Multi-turn and agentic traffic, where each turn's prompt extends the previous turn's prompt and output, pays three costs for this:

1. **Resume granularity.** The next turn resumes at the last block boundary the previous prompt reached. It recomputes the rest of that prompt, up to a block, plus every token the previous turn generated. Blocks of hybrid models are large (one page holds a whole recurrent state), often hundreds to thousands of tokens.
2. **Chunk boundaries.** A non-final prefill chunk must end at every position where a state has to be cached: a block boundary after a mid-block start, the last cacheable block boundary, or a shared-prefix junction. Each early stop costs a forward pass with a short chunk.
3. **Eviction granularity.** The free queue evicts one block at a time, least recently freed first. A recurrent state is usable only together with the attention pages on its path, so evicting one of them wastes the rest, and a long session's state goes as readily as a short one's.

## Design principles

- All logic lives in `vllm/v1/core/hybrid_checkpoint/`. The scheduler, the KV cache manager, the block pool and `MambaManager` call it from narrow hooks that do nothing while the options are off.
- Every option defaults off, and every unsupported combination is refused at startup with a `ValueError` that names the missing requirement.
- Model-specific behaviour is declared on KV cache specs, not by model names (see [Indexer rings](#indexer-rings)).
- Upstream mechanisms are reused: `hash_block_tokens` for keys, the partial-hit copy-on-write for resuming, the backends' prefill checkpoints for in-chunk states, the free queue for the eviction order and the KV connector interface for other cache tiers.

## Recovery units

A recovery unit (`units.RecoveryUnit`) is one recurrent state that a request can resume from, cached in every Mamba group:

- a block-boundary state that `align` mode caches itself;
- a decode-boundary state: the running state at the last block boundary a request's output crossed, published when the request finishes;
- a frontier: the state at a finished request's exact processed length, inside a block.

A unit is usable only while all of its blocks remain cached. The block pool reports every key a block loses (`BlockPool.key_removal_listener`), and when any block of a frontier is evicted, the frontier's other blocks lose the key too, so a lookup never finds part of a unit.

`units.CheckpointGraph` relates the units. A unit's parent is the nearest unit before it on its prefix path: the one its request resumed from, or the one the request cached before it. When a unit is removed, its children move up to its parent. A unit is a branch point when two cached continuations diverge there or when a request other than its session's next turn has resumed there.

### Frontier keys

A frontier is cached under a key that is not the hash of its block's token range. `keys.frontier_key` chains the request's block hash at the last hash boundary with the remaining tokens, their extra keys (multimodal inputs, LoRA, cache salt) and the length, through `hash_block_tokens` with a marker that separates frontier keys from block hashes. Equal keys mean equal prefixes of equal length. `BlockPool.cache_block_under_key` caches a block under such a key.

## Frontier checkpoints

When a request is freed after processing `L` tokens, with `L` inside block `k` (block size `B`), `FrontierIndex.publish` caches the following under the frontier key of `L`, all of them or none:

- the attention page holding `[kB, L)` in every full-attention group;
- the running state block in every Mamba group;
- the ring block of an indexer whose compression group is open at `L` (see [Indexer rings](#indexer-rings)).

A later request whose tokens extend those `L` tokens first gets its block-aligned hit as before; `FrontierIndex.extend` then lengthens it to the longest frontier that the request's tokens extend. The request resumes at `L` through the partial-hit copy-on-write, which gives it private copies of the partial page and the state.

With a `prefix_match_unit` smaller than the block size (partial hits on a finer hash unit, as deployments that cache states in an external store use), the page holding `[kB, L)` may already be cached under the request's own key at the last hash-unit boundary inside it. The frontier key joins that key (the block pool keeps several keys per block); removing the frontier removes only its own key (`BlockPool.uncache_key`), and the page stays cached under the request's key. Mamba state and ring blocks must still be unkeyed.

### Decode-boundary states

A request's output can cross block boundaries, where `align` mode caches no state. With frontier checkpoints, when a running request's output crosses a block boundary, `HybridCheckpointManager.snapshot_decode_boundary` copies its running state into a scratch block in every Mamba group. The state at the last such boundary is published when the request finishes. A scratch block never preempts a request: it comes from the blocks the scheduled requests left free, and a running request that cannot allocate frees scratch blocks first.

### Session leaves

When a request finishes, the newest state it published becomes its session's leaf, and the leaf it resumed from is released unless that leaf is a branch point or a running request still holds its state. A frontier that differs from the request only in its last token, such as a stop token the client replaced, is released too. An aborted request publishes no frontier, because its last state may come from a step or a load that never completed.

### Indexer rings

Some models keep an auxiliary ring beside their attention layers, for example a compressor whose open group of tokens is summarised only once the group is full. A request restored inside an open group needs a copy of the ring, and the ring's spec declares how the request continues (`CircularBufferSpec.resume_catchup`):

- `"decode"`: only the decode path reads the ring, so the scheduler feeds the restored request one token per step until the group closes;
- `"prefill"`: the prefill path reads the ring too, so the restored request prefills as usual;
- `None`: the ring cannot be restored inside a group, and frontier checkpoints are refused for the model.

The ring group's manager takes at most one hit per request, the frontier's ring block, and copies it into the request's own ring when that is allocated.

A ring can be wider than one group: under speculative decoding a verify step writes its drafts' rows before it knows which are accepted. Such a ring declares its group size (`CircularBufferSpec.compression_ratio`; GLM-5.3-Flash's indexer: `index_kpool`). A frontier then keeps the ring only when `L` is inside an open group, and a restored request catches up to the end of the group, not of the ring.

## In-chunk state export

With `mamba_prefill_export`, the scheduler plans each non-final prefill chunk with `export.plan_export` so that the chunk contains at most one position where a state must be cached. If the backend's prefill checkpoint can write the first such position (`is_mamba_prefill_checkpoint_valid`), the chunk runs past it and the state there is exported; the chunk then ends at the second such position, if there is one. Otherwise the chunk ends at the first position, as without export.

The scheduler chooses one position per request per step, and every consumer takes it from there:

1. `MambaManager` reserves and re-keys the checkpoint block from the scheduler's per-request dictionary.
2. `SchedulerOutput.prefill_checkpoint_positions` carries it to the Model Runner V2 input batch.
3. `CommonAttentionMetadata.prefill_checkpoint_positions` carries it to `compute_mamba_prefill_checkpoints`, which writes the state there instead of at the default checkpoint position.

A later cap on the chunk that moves the position out of it drops the export.

A non-final chunk that the block-aligned split cannot schedule at all, because the step's remaining budget is smaller than a block, may instead end on a 64-token boundary when that leaves at least 1024 tokens (`export.plan_filler`).

A position where a state must be cached falls strictly inside a non-final chunk only when the chunk spans more than one block or such positions lie off the block grid: with a prefix-match unit finer than the block (`prefix_match_unit`) or with shared-prefix checkpoints (`enable_mamba_shared_prefix_checkpoint`). Otherwise, for example with a long-prefill threshold of one block and fewer than two blocks of batched tokens, the option only adds fill chunks.

### Backends

The exported state comes from the backend's own prefill call: no extra prefill runs and the chunk is not cut. A Mamba layer can export when it declares a prefill checkpoint alignment (`MambaBase.get_prefill_checkpoint_alignment`), and `mamba_prefill_export` is refused at startup unless every Mamba layer of the model does.

| Layer (model) | Prefill backend | Alignment | Compute capability 10.0 | Other GPUs |
| --- | --- | --- | --- | --- |
| Mamba2 (Nemotron 3 Super) | Triton SSD | 1 | `_compute_chunk_metadata` ends an SSD chunk at the offset, the scan returns every chunk's state and `store_prefill_checkpoint` writes it with the convolution window | the same |
| Kimi delta attention (GLM-5.3-Flash) | FlashKDA | 16 | `vllm._flashkda_export_C` ([Export kernels](#export-kernels)); without that build, as on other GPUs | `vllm._flashkda_C`'s own checkpoint path (SM90, SM10x, SM12x), with an extra pass in calls that export |
| Gated delta rule (Qwen3.8-Flash-Next) | FlashInfer | the prefix-cache grid | the vendored per-sequence kernel ([Export kernels](#export-kernels)) for 128-wide keys and values, else as on other GPUs | FlashInfer's stride checkpoints (SM90, and SM10x and SM12x with 128-wide keys and CUDA 13) |
| PLE short convolution (Qwen3.8-Flash-Next) | short convolution | the prefix-cache grid | `MambaPrefillCheckpointExporter` copies the convolution window | the same |

The Kimi delta attention and gated delta rule layers above declare no alignment on any other prefill backend (Triton, CuTe DSL, AITER, SYCL), so the option is refused where those run, for example on GPUs without the support above and on ROCm, XPU and CPU.

Two of these paths change rounding:

- An SSD offset off the chunk grid (multiples of the model's chunk size in absolute positions, 128 tokens for Nemotron 3 Super) adds a chunk to its sequence. The exported state is still the state a chunk ending there leaves, bit for bit, but the sequence's later outputs and final state may change by rounding. An offset on the grid changes nothing; block boundaries are on it when the block size is a multiple of the chunk size.
- FlashInfer's stride checkpoints follow its route choice: a call with few sequences takes the context-parallel route, whose states differ from those of the sequence cut at the offset by rounding (about 1e-3 on B200).

### Export kernels

On compute capability 10.0, two backends write the exported state inside their prefill kernel:

- **Gated delta rule, FlashInfer backend.** FlashInfer's chunked prefill writes checkpoints at one stride for every sequence of the call (`checkpoint_every_n_tokens`, the gcd of the batch's offsets) into a workspace. With `mamba_prefill_export`, `GDNAttentionMetadataBuilder` instead numbers one workspace row per sequence that exports (`plan_gdn_export_rows`), and the layer calls a vendored copy of that kernel (`vllm/third_party/flashinfer_gdn`) that writes each sequence's state at its own offset into its row; `MambaPrefillCheckpointExporter` then copies the rows into the cache blocks as before. These calls always take FlashInfer's non-context-parallel route; when few sequences leave SMs idle, the builder has each (sequence, value head) run as two tiles of half the state's rows (`plan_gdn_export_split`).
- **Kimi delta attention, FlashKDA backend.** Calls with checkpoint offsets use `vllm._flashkda_export_C`: FlashKDA built with the kernel files of `csrc/flashkda_export/`, which store the checkpoint and the final state from registers.

Each export kernel keeps four invariants:

1. A call's output and final state are bit-identical to the backend's call without export on the same route, and each exported state is bit-identical to the final state of its sequence cut at the offset.
2. A sequence whose offset is 0 exports nothing, and no other row is written.
3. Calls without checkpoint offsets run the backend's original kernel.
4. Offsets keep the alignment the layer declares (64 tokens for the gated delta rule, 16 for FlashKDA), and the scheduler places them on the prefix-cache grid (`get_mamba_prefill_checkpoint_grid`).

Both export kernels write the states they export while computing the chunk, without another pass over the data. A long single sequence still runs on FlashInfer's non-context-parallel route when it exports, which can be slower than the context-parallel route its call without checkpoints would take.

The vendored gated-delta kernel runs the two tiles of a split (sequence, value head) as one 2-CTA cluster, so a split call keeps its grid even on any SM count. Its value ring's release under that launch rests on the CuTe DSL version vLLM pins (4.8.0): after a CuTe DSL upgrade, `tests/kernels/mamba/test_gdn_prefill_flashinfer.py`, split cases included, must pass on SM100 again.

## GreedyDual-Size eviction

With `prefix_cache_eviction="gds"`, evictions follow GreedyDual-Size over recovery units (`eviction.GreedyDualSizeEviction`). Each unit is a state together with the attention pages on its path that no other unit needs. For each unit:

- `gain` is the prefill time it saves: the seconds to prefill from its fallback, the deepest unit before it on its path (or position 0), to its own position. A unit that lost a page on its path can no longer be hit and has gain 0.
- `size` is the number of blocks that evicting it frees: its state blocks and its exclusive pages.
- Its score is `L + gain / size`, where `L` is the score of the last evicted unit at the time the unit was last cached, resumed at or released by a request.

The unit with the lowest score is evicted whole: its state blocks, then its exclusive pages, deepest first. A frontier's page that is also cached under the request's own key at a hash unit finer than the block only loses the frontier's key: it is neither freed nor counted in the frontier's size. A cached free block that no unit needs scores `L`.

The free queue stays the block pool's only record of the order. The policy implements `CachedBlockEvictionPolicy`: the block pool tells it about cached blocks as they are freed (`on_blocks_freed`), and before an allocation takes cached blocks, the policy moves the blocks it chooses to the front of the free queue (`prepare_eviction`). Without a policy the pool keeps its LRU order.

With `prefix_cache_leaf_last`, a unit that has a deeper live unit on its path (one holding its key as a page, or a child in the graph) is inner unless it is a branch point, and every inner unit goes before any leaf: the score becomes `(tier, score)` with dead 0, inner 1 and leaf 2, GreedyDual-Size orders each tier, and `L` rises only when a leaf goes. A session's latest state, where its next turn resumes, thus outlasts the earlier states on its path, which serve only rewrites and forks. With `prefix_cache_leaf_idle_seconds`, a leaf that is not a branch point and was not used for that long ranks as inner.

Units that a waiting request's prefix hit reads (among the first 256 waiting requests whose lookups walk at most 262,144 hash units in all, refreshed at most every 0.5 s of wall-clock time) and units a running request holds are not evicted while pinned; when no other unit is left, the pool's order applies, pinned blocks last, those of the waiting queue's back first, so that its head, the next request admitted, keeps its hit longest. A new unit whose score is below every other unit's is released as soon as its blocks are free while the pool has no uncached free block. Under this policy, leaf replacement no longer releases block-boundary states; the eviction order decides when they go.

### Prefill cost model

`cost.PrefillCostModel` models the time to prefill positions `[a, b)` as

```text
d * (b - a) + e * (b^2 - a^2) / 200000
```

seconds: `d` seconds per token, plus `e` seconds per token per 100,000 tokens of context for the attention work that grows with position. `prefill_cost_coefficients=(d, e)` sets the two coefficients. Because every score is a past score plus `gain / size`, scaling both coefficients by one constant leaves the eviction order unchanged; only `e / d` matters. Fit them by least squares to the model's prefill times of `n` tokens after `c` tokens of context, `d * n + e * n * (2 * c + n) / 200000`, at a few context lengths.

With a hash unit finer than the block, a frontier's fallback is its graph parent or the deepest block-boundary unit before it, not a state at a hash-unit boundary inside its own block, so its gain counts from there.

## KV connectors

Frontier checkpoints run with a KV connector whose `kv_role` is `"kv_both"` and whose `kv_load_failure_policy` is `"recompute"`. Under `"fail"`, a request whose load failed would finish with a computed length over blocks the load did not write, where no frontier may be cut.

The scheduler shows a connector only the block-aligned part of a local hit that ends inside a block, and lets a longer external hit replace the rest. A frontier hit is such a hit: when the connector's hit is longer, the frontier's partial page, state and ring block are dropped from the request's hit, and the request resumes at the state the load ends with. This applies to the native CPU offloading connector (`--kv-offloading-size`) like any other connector.

A connector can also keep frontier checkpoints in a store other engines share. Three opt-in hooks of `KVConnectorBase_V1` serve this, all inert by default:

- `stores_frontier_checkpoints` and `register_finished_frontier`: before a finished request's blocks return to the pool, the scheduler hands such a connector the request's frontier checkpoint (`frontier.FrontierCheckpoint`: the key, the length, the block of every group holding the state and the attention pages before it), whether or not it is cached locally;
- `frontier_load_length` and `move_frontier_load_start`: when a load resumes a request at a stored frontier past attention pages the GPU still caches, the scheduler takes those pages as local hits and the load starts after them;
- `defer_lookup`: the scheduler tries a request again at a later step when the connector knows that a write which would extend its hit is still in flight.

`MooncakeStoreConnector` implements them with `frontier_write_through` and related options; see [Mamba Frontier Checkpoints in the Mooncake Store](mooncake_store_frontier_checkpoints.md).

## Speculative decoding

Under speculative decoding a finished request's computed-token count does not say where its exact state is: a verify step writes one state column per draft position, and under asynchronous scheduling one more step of the request may be in flight when it finishes. `speculative.SpeculativeFrontiers` keeps each request's last completed step (its device start, its query length, whether it verified drafts, and the tokens the device sampled and the scheduler kept), which the scheduler records in `update_from_output`. When the request finishes, it resolves where the exact state is:

- with a step in flight, the state after the request's last token, in that step's first column;
- otherwise, the state after the kept tokens, in the column of the last accepted draft or, for a layout without per-position columns, in the block the accepted tokens end in.

A state in a later draft column is folded into the step's first column on the GPU before anything reads it: the scheduler sends a `MambaStateGather` in the next step's output (`SchedulerOutput.mamba_state_gathers`), Model Runner V2 applies it with the `align`-mode pre-copy kernel before any copy of that step, and both blocks stay referenced until the step completes. When the scheduler kept fewer tokens than the device sampled, for example at a stop string, the request publishes no frontier; `SpeculativeFrontiers.declined` counts the reasons.

A decode-boundary state comes from the block in which a verify step leaves the state at a block boundary its accepted tokens crossed, snapshotted when the step's output is processed.

`speculative_config.truncate_acceptance_at_stop` makes the rejection sampler stop accepting at the request's stop: the first sampled token that is its EOS token (unless `ignore_eos`) or one of its stop token ids, or that reaches its length limit. The output is unchanged, since the scheduler drops the tokens after a stop anyway, but the device's state then ends at the request's last token. Frontier checkpoints require it.

### Compact state layout

`mamba_spec_state_layout` chooses how Mamba layers keep their state through a verify step. With `"per_position"`, the default, the step writes the state after every draft position, so a running request holds one more state block per draft token in every Mamba group. With `"compact"`, the step writes no state: it records each position's update in buffers outside the KV cache, and the accepted ones are folded into the state after sampling, so a running request holds one state block per group. Frontier checkpoints treat such a step as an in-place commit. A model opts in through the `SupportsCompactSpecState` interface; GLM-5.3-Flash's KDA layers implement it with upstream's RecoverSSM verify and commit, the mechanism Kimi-K3 uses under `--use-replayssm`.

## Metrics

`vllm:prefix_cache_hits` includes the tokens a frontier serves. `vllm:prefix_cache_frontier_hits`, registered only when `mamba_frontier_checkpoint` is on, counts the part of those hits that lies past the block-aligned hit, which is what frontier checkpoints save. Re-admissions of preempted requests are not counted.

## Requirements

`mamba_frontier_checkpoint` requires prefix caching with `mamba_cache_mode="align"`, Model Runner V2, `prefix_cache_retention_interval=0`, no `long_prefill_token_threshold` below the block size, and a KV connector, if any, with `kv_role="kv_both"` and `kv_load_failure_policy="recompute"`. It is refused with ReplaySSM, KV cache events, and decode or prefill context parallelism. With speculative decoding it also requires `truncate_acceptance_at_stop`, no EAGLE block drop (`disable_eagle_block_drop` for drafters that need it, such as MTP and EAGLE), pipeline parallel size 1, no KV connector unless `mamba_spec_state_layout="compact"`, and indexer rings that only the decode path reads.

`mamba_prefill_export` requires `mamba_cache_mode="align"`, Model Runner V2 and a backend that writes prefill checkpoints in every Mamba group. It is refused with the EAGLE block drop under speculative decoding (an exported position follows neither the drop nor the lookahead block hashes that replace it), ubatching, and context parallelism.

The export kernels take effect only on compute capability 10.0, with the FlashInfer prefill backend for the gated delta rule and the FlashKDA backend for Kimi delta attention; FlashKDA's export build `vllm._flashkda_export_C` is built for 10.0f and 10.0a. Elsewhere those backends export through their upstream checkpoint paths, and Mamba2 exports through the SSD on every GPU (see [Backends](#backends)).

`mamba_spec_state_layout="compact"` requires speculative decoding, a model whose Mamba layers all support it (`MambaBase.supports_compact_spec_state`), CUDA, the Triton Mamba backend, pipeline parallel size 1 and, with prefix caching, Model Runner V2. It excludes `--use-replayssm` and every KV connector but `MooncakeStoreConnector`.

`prefix_cache_eviction="gds"` requires `mamba_frontier_checkpoint` and `prefill_cost_coefficients` with `d > 0` and `e >= 0`. `prefix_cache_leaf_last` requires `prefix_cache_eviction="gds"`, and `prefix_cache_leaf_idle_seconds` (positive) requires `prefix_cache_leaf_last`.

## Testing

The tests in `tests/v1/core/hybrid_checkpoint/` run on CPU; all but the graph tests drive the real scheduler and KV cache manager:

- `test_checkpoint_graph.py`: a removed unit's children move to its parent, branch points, and frontier keys that separate tokens, lengths and extra keys.
- `test_frontier_checkpoints.py`: resuming at a frontier, catch-up inside an indexer group, decode-boundary states, evicting one block of a frontier, aborted requests, leaf replacement, a prefix-match unit smaller than the block, KV connectors (the frontier a connector that stores frontiers is handed, a load moved past cached pages, a deferred lookup) and refused configurations.
- `test_prefill_export.py`: the planning rules, a sibling's chunk that exports the state at its shared-prefix junction under each backend's checkpoint alignment, fill chunks and refused configurations.
- `test_gds_eviction.py`: GreedyDual-Size against LRU under memory pressure, a frontier pinned by a waiting request, an exhausted pool giving up the pins of the waiting queue's back first, the pool's consistency after every step of a seeded multi-session run, leaf-last tiers and idle demotion, and refused configurations.
- `test_speculative_frontiers.py`: frontier checkpoints under speculative decoding with an n-gram drafter, in both state layouts, under synchronous and asynchronous scheduling, with stops inside accepted drafts, an in-chunk export, pool pressure, aborts and refused configurations. A device stand-in (`spec_device.py`) checks every state read and every cached state against its token position, and must catch injected faults.

The export kernels' invariants are checked on SM100 GPUs by `tests/kernels/mamba/test_gdn_prefill_flashinfer.py::test_prefill_export_is_flashinfer_bit_for_bit` (with and without the value split, with an untouched row and a state 64 tokens later as controls) and `tests/models/glm5next/test_kda_recurrent.py::test_flashkda_export_build_is_flashkda_bit_for_bit`; `tests/v1/attention/test_gdn_metadata_builder.py` checks the export rows and the value split on CPU.

The paths other GPUs take are checked against cut sequences, bit for bit, at the served models' per-GPU shapes at tensor parallel 2, with a state one chunk later as a control: `tests/models/glm5next/test_kda_recurrent.py::test_flashkda_checkpoint_is_a_cut_run_bit_for_bit` (FlashKDA's own checkpoint path and the export build) and, on CUDA GPUs, `tests/kernels/mamba/test_mamba2_checkpoint_state.py::test_prefill_export_is_a_cut_run_bit_for_bit_at_nemotron_shapes` (the SSD path, where rows whose offset is on the chunk grid also keep the output and final state of the step without export). FlashInfer's stride checkpoints are checked within a tolerance, on both routes, by `tests/kernels/mamba/test_gdn_prefill_flashinfer.py::test_flashinfer_checkpoints_export_the_state_at_each_offset`. `tests/v1/attention/test_mamba2_prefill_checkpoints.py` checks on CPU that the Mamba2 metadata builder ends a chunk at the scheduler's position.

Elsewhere, `tests/v1/worker/test_gpu_stop_truncation.py` checks the truncated acceptance against the scheduler's `check_stop`, `tests/kernels/mamba/test_precopy_mamba_align.py` checks the gather (CUDA), and `tests/models/glm5next/test_kda_compact_spec_state.py` checks GLM-5.3-Flash's compact verify and commit against its per-position verify.
