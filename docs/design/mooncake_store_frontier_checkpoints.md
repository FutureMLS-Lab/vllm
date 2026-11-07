# Mamba Frontier Checkpoints in the Mooncake Store

With `--mamba-frontier-checkpoint`, a hybrid model's finished request leaves its
recurrent state at its exact end resumable on the GPU that ran it: the
attention page holding the tokens after the last full block, the running state
of every Mamba group and, when the end is inside an indexer compression pool,
the indexer ring, all cached under one *frontier key*. A later request that
extends those tokens resumes there instead of at the last block boundary.

`MooncakeStoreConnector` stores whole blocks keyed by their prefix hashes and
the Mamba state at hash-unit boundaries. Its option `frontier_write_through`
also writes each finished request's frontier checkpoint to the store, so that
a request on any engine sharing the store resumes at the frontier, for example
the next turn of a session whose previous turn finished on another instance.

## What a finish writes

Before a finished request's blocks return to the pool, the scheduler hands its
frontier checkpoint to the connector
(`KVConnectorBase_V1.register_finished_frontier`, for connectors whose
`stores_frontier_checkpoints` is true). The checkpoint is the frontier key,
its length `L` inside block `k = L // block_size`, the block of every group
that holds the state, and the request's attention pages `[0, k)`. The
connector writes, as one store job that references those blocks until every
rank has written them:

1. the full attention pages `[s, k)` the request's normal saves did not cover
   (usually the pages its output tokens filled), under their block hashes,
   exactly as normal saves key them, so they also serve ordinary prefix hits;
2. every frontier block under the frontier key, in its group's key namespace.
   An indexer ring is not a prefix-cacheable group, so rings get a namespace of
   their own (`...@ring...`), one copy per tensor-parallel rank;
3. on one rank, the entry `(L, frontier key)` at the head of the locator
   record under the frontier's locator key (below). A record keeps at most
   four entries, newest first.

Each rank checks which objects the store already holds and puts only the
others. Nothing is written for an aborted request, for a frontier on a block
boundary (the block-boundary state already covers it), for one shorter than
16 tokens, or when a block is missing.

## How another engine finds a frontier

A frontier key chains the request's block hash at the frontier's last hash unit
with the tokens after it and `L`, so a reader can only check a key whose `L` it
knows. The *locator key* chains the request's block hash at the start of hash
unit `h = (L - 16) // hash_block_size` with the next 16 tokens and their extra
keys, without `L`, so it lies inside the frontier and every request that
extends the frontier computes it from its own tokens
(`vllm/v1/core/hybrid_checkpoint/keys.py`).

For a request with a block-aligned local hit `s`, the scheduler-side lookup
(`FrontierLookup` in
`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/frontier.py`):

1. computes the locator keys of every hash unit a frontier in `(s, n - 1]` could
   be anchored in (`n` the request's tokens), at most the 4,096 deepest, and
   asks the worker which records exist; the worker returns their entries;
2. keeps the longest entry whose key equals the request's own frontier key at
   that length, so a stale or foreign entry never matches;
3. asks the worker to check that every object a load needs exists on every
   rank: the pages `[s // block_size, k)` and every frontier block. Only then is
   the hit `L`; any missing object gives the regular lookup's hit instead.

That is two round trips on the lookup socket when a locator exists and one
otherwise. A frontier that is not longer than what the GPU already holds is
not used.

## Loading

The load is one batch per rank: the pages into the request's blocks
`[s // block_size, k)`, the page holding `[k * block_size, L)` into block `k`,
each Mamba state into the running block, and each ring into the request's ring
block. The rings are reported as loaded (`get_loaded_kv_cache_group_ids`) so
the worker does not zero them. Any failed key fails the request's whole load,
which `kv_load_failure_policy: recompute` recomputes. The request then resumes
at `L` as after a local frontier hit.

The local hit a lookup starts from ends at the deepest Mamba state every group
caches, which can be well before the frontier while the GPU still caches the
attention pages up to it. With `frontier_load_local_pages`, the scheduler asks
the connector whether a pending load resumes at a stored frontier
(`KVConnectorBase_V1.frontier_load_length`); if so it takes the pages it caches
after the local hit under the request's own block hashes as local hits, up to
the frontier's block or the first page missing, and moves the load's start
there (`move_frontier_load_start`), so the load reads only the remaining pages
and the frontier's blocks.

## Leases and eviction

Mooncake evicts, above its high watermark, the objects whose lease expired,
oldest first, and every existence check grants a lease, so the store's eviction
order follows what lookups ask about. With `lookup_newest_state` a lookup that
finds a frontier asks for nothing else: only the newest usable state and the
pages it needs are renewed, and earlier states on the session's path age out
first. Without it the regular lookup runs as well, as without frontiers.

## Superseded frontiers

A session that resumes from its previous turn's frontier and finishes again
leaves the store holding the older frontier and the Mamba states its prefill
stored at hash-unit boundaries past it, none of which another prefix shares.
With `frontier_supersede_remove`, writing the new frontier also removes them:
the scheduler remembers the frontier each request resumes from (a stored one it
loads or a local one) and the boundary states it stores, and the finish's store
job names those past the resumed frontier. Each rank removes its own share
without force; Mooncake refuses an object whose lease is still running, so a
reader that just verified it keeps it until its load, and a refused object is
left to the store's eviction. Pages are never removed.

Superseding at once also drops the block-boundary state under the session's own
latest frontier, which a client that rewrites its last output resumes at. With
`frontier_supersede_lag` set to `n`, a finish supersedes instead what the finish
`n` turns earlier on its session's chain would have: the scheduler keeps, by the
key of each chain's newest frontier, the lists of its last finishes, and a
finish that resumed from a frontier continues that frontier's chain.

## Writes in flight

A frontier reaches the store only once its write job has completed on every
rank, while the request's next turn can arrive first and find its GPU copy
evicted. With `frontier_write_wait_s`, the scheduler asks the connector, before
each lookup and with the request's whole local hit, whether to defer it
(`KVConnectorBase_V1.defer_lookup`): the connector does while the request
extends past that hit a frontier of its own session whose write has not
completed, finished less than the wait ago, and the scheduler tries the request
again at its next step.

## Repeated lookups

A request waiting for space is looked up at every step. With
`frontier_lookup_memo_s`, within that many seconds at the same local hit, a
request whose probe found no usable frontier runs only the regular lookup, and
one whose frontier verified reuses that hit for up to `frontier_lookup_reuse_s`
(default 4 seconds, under half of Mooncake's default 10-second lease), then
repeats only the verification, which renews the leases. The reuse window must
stay below the master's `default_kv_lease_ttl`: past the lease a reused hit's
objects may be evicted, and its load then fails into a recompute. A new local
hit or a finished request starts afresh.

## GPU memory held by writes

A finish's store job references the frontier's blocks and the pages it writes
until every rank has put them, as normal saves hold theirs. With
`staging_buffer_mib`, each job first copies the bytes it will put into pinned
host memory registered with the store; once the copy completes the rank reports
the job's blocks released and the scheduler frees them, while the puts run from
the copies on a second thread in job order. A frontier stays in flight for
`frontier_write_wait_s` until its job completes. A job that does not fit the
buffer within `staging_wait_s` keeps its blocks until it completes. A job
copies only what the store does not hold when it is copied; a key evicted
between that check and the job's puts counts as stored, since its bytes are no
longer on the GPU, and a job that ends without recording its pages (a failed
or skipped put) makes the request's next job copy them again.

## Recovery units

Mooncake's default eviction knows objects, not the resume points they make up:
it can evict a frontier's state while keeping its page, or a page that a
hundred states need. With `unit_policy_url`, the connector describes *recovery
units* to a master that evicts by unit (`--eviction_policy=unit_gds`). A unit
is a resume point that a store job wrote, a Mamba state at a hash-unit boundary,
a prompt's partial tail or a frontier checkpoint, with:

- its *members*, the objects only it owns (its states and, for a partial tail
  or a frontier, its page and rings), which every rank puts under the unit's id
  as their Mooncake group, so the master evicts them together;
- its *refs*, the full pages before it (and a frontier's locator record), which
  units share and the master reference-counts;
- its *value*, the seconds of prefill from position 0 under
  `prefill_cost_coefficients`, and its *parent*, the deepest unit before it on
  its request's path.

The rank that writes locators sends the updates from one thread, in order and in
batches, to the master's admin port (`POST /api/v1/units` with
`{"updates": [...]}`; `GET /api/v1/units/stats` answers whether the policy
runs): `upsert` when a job wrote a unit, `touch` when an admitted request
resumes at one, `dead` for what `frontier_supersede_remove` would otherwise
remove. A failed request, a malformed reply and an update dropped because
65,536 updates already wait are counted and logged, and the thread goes on:
objects of a unit the master never learns about are evicted as orphans after
its grace period, and a lost touch or dead mark only changes the eviction
order.

## Observability

Store operations are recorded with the connector's other metrics under the
operations `frontier_put`, `frontier_exists`, `frontier_locator`,
`frontier_probe`, `frontier_verify`, `frontier_get` and `frontier_remove`. Once a
minute the scheduler logs the frontier lookup's counts (probes, hits, hit tokens,
foreign or incomplete frontiers, memo uses, waits) and the locator-writing rank
its removal outcomes and unit updates. With debug logging, each lookup logs the
path it took (`probe:hit`, `probe:no_entries`, `probe:no_match`,
`probe:incomplete`, `memo:hit`, `memo:no_frontier`, `memo:verify`, `short`),
its time, the local hit it started from and its hit, and the scheduler logs each
deferred lookup and each frontier load whose start moved.

## Limits

- Needs `--mamba-frontier-checkpoint`, and so `kv_role: kv_both` and
  `kv_load_failure_policy: recompute`. Under speculative decoding only with
  `--mamba-spec-state-layout compact`: with one state column per draft position
  a finished request's state may still wait for its draft-column gather when the
  store job reads it. Not supported with KV regions per group (HiSparse).
- A request shorter than one hash unit (or one block without partial hash hits)
  skips the store lookup, so it does not find a frontier inside its first unit.
- Two engines updating one locator record at once can drop an entry; a dropped
  frontier is then not found, never misread.
- With `frontier_locator_cache` the locator-writing rank's probes read its own
  last record under a key; a record another engine changed meanwhile is read
  stale, so that engine's frontier is not found here until this rank writes the
  record again (merges always read the store's copy, so no entry is dropped).
- A finish writes the frontier's page, its states and its rings: one block per
  group, several times the size of one page for models with large states.
