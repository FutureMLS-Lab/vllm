# MooncakeStoreConnector Usage Guide

MooncakeStoreConnector is a KV cache connector that uses [MooncakeDistributedStore](https://github.com/kvcache-ai/Mooncake) as a shared KV cache pool. Unlike `MooncakeConnector` which does direct point-to-point KV transfer between prefiller and decoder, MooncakeStoreConnector enables KV cache offloading to an external distributed store, supporting:

- **CPU/disk offloading**: Extend effective KV cache capacity by offloading to CPU memory or disk via Mooncake's transfer engine.
- **Prefix caching across instances**: Hash-based deduplication allows multiple vLLM instances to share cached KV blocks through the store.
- **Single-node and multi-node deployment**: Works both as a standalone KV cache extension and in disaggregated prefill-decode setups.

## Prerequisites

### Install Mooncake

Install mooncake through pip:

```bash
uv pip install mooncake-transfer-engine
```

Refer to the [Mooncake official repository](https://github.com/kvcache-ai/Mooncake) for more installation instructions and building from source.

### Start the Mooncake Master Server

The Mooncake master manages metadata and coordinates the distributed store. Start it before launching vLLM:

```bash
mooncake_master --port 50051
```

Default ports:

- RPC: 50051

Multiple vLLM instances can share the same master server.

### Configure Mooncake

Create a JSON configuration file (e.g., `mooncake_config.json`):

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": "80GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false
}
```

- `mode`: Topology selection. `"embedded"` (default, PR-40900 baseline) has each
  vLLM rank contribute `global_segment_size` to the pool in-process.
  `"standalone-store"` makes ranks pure requesters — an external
  `mooncake_client` process owns the CPU pool and (optionally) the SSD tier.
- `protocol`: Use `"rdma"` for best performance. `"tcp"` works as a fallback.
- `global_segment_size`: CPU memory contributed to the distributed pool (per
  GPU). Must be `> 0` in `embedded` mode and `0` in `standalone-store` mode.
- `local_buffer_size`: Private buffer for this node's own operations (per GPU).
- `enable_offload`: When `true`, vLLM allocates a DirectIO staging buffer so
  large prefills do not exceed the owner's SSD-write budget. Set this together
  with the matching `--enable_offload=true` flag on `mooncake_master` and on
  the external `mooncake_client` (if any).
- `tenant_id`: Optional Mooncake tenant namespace. Producers and consumers
  that should share store data must use the same tenant id. Default:
  `"default"`.

Set the config path via environment variable:

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```

## Usage

### Single-Node KV Cache Offloading

Use MooncakeStoreConnector to offload KV cache to CPU memory, extending the effective cache size:

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
```

### Disaggregated Prefill-Decode (XpYd)

In disaggregated prefill-decode mode, use `MultiConnector` to combine `MooncakeConnector` (point-to-point KV transfer) with `MooncakeStoreConnector` (shared KV cache pool). This enables both direct P2P transfer between prefiller and decoder, and cross-instance prefix cache sharing via the distributed store.
**Prefiller Node:**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
VLLM_MOONCAKE_BOOTSTRAP_PORT=50052 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8100 \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_producer"
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both"
                }
            ]
        }
    }'
```

**Decoder Node:**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
VLLM_MOONCAKE_BOOTSTRAP_PORT=50053 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8200 \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_consumer"
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_consumer"
                }
            ]
        }
    }'
```

To also offload newly completed decode KV blocks, add the following extra
configuration to the decoder's `MooncakeStoreConnector` entry.

When decode processing starts, the consumer checks the block-aligned prompt
prefix and fills any blocks missing from the Store. Subsequent saves append
newly completed decode blocks. This keeps a complete, reusable prefix in the
Store. It also covers prompt KV delivered directly by `MooncakeConnector`.

```json
{
    "kv_connector_extra_config": {
        "save_decode_cache": true
    }
}
```

#### Sharing one Store across multiple Prefill TP sizes

Heterogeneous-TP sharing normally uses a fixed `store_tp_size`. When several
prefillers use different TP sizes, opt in to a common Store TP derived from
their least common multiple:

```json
{
    "kv_connector_extra_config": {
        "enable_store_tp_lcm": true,
        "prefill_tp_sizes": [4, 2]
    }
}
```

Every prefiller and decoder that shares these entries must use the same list.
The example selects Store TP 4: a TP4 endpoint maps each rank to one Store
shard, while TP2 endpoints map each rank to two Store shards. Runtime TP sizes
remain unchanged. A decoder configured with `"save_decode_cache": true` uses
the same Store TP for decode KV from every prefiller.

The list may contain positive integer TP sizes. Sharing requires a Store TP that
is at least the local TP and divisible by it, an LBHNC or LBNHC local KV cache,
and the existing topology and KV-head constraints. The Store namespace includes
the attention backend's selected layout. Different layouts use separate Store
entries. Malformed lists and unsupported endpoints use an isolated rank-local
key layout. When
`enable_store_tp_lcm` is absent or false, `prefill_tp_sizes` has no effect and
the existing `store_tp_size` behavior is unchanged.

**Proxy:**

A disaggregation proxy routes requests between prefiller and decoder nodes.
When `MooncakeConnector` is also used for direct P2P transfer, refer to its
[usage guide](mooncake_connector_usage.md) for proxy setup details.

### Disk Offloading

Disk offloading is most commonly run in `standalone-store` mode: an external
`mooncake_client` process owns the CPU pool and the SSD tier, and each vLLM
rank is a pure requester. This avoids per-rank duplication of the SSD pool
and keeps DirectIO budget tracking on a single process.

Three things need to be aligned for end-to-end disk offloading:

1. **`mooncake_master`** is started with `--enable_offload=true`.
2. **`mooncake_client`** (the owner) is started with `--enable_offload=true`
   plus an SSD path via `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`.
3. **vLLM-side** sets `"enable_offload": true` in the JSON config file (this is
   read by the connector and is **not** an environment variable).

Example `mooncake_config.json` for the vLLM side:

```json
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": 0,
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "mlx5_0",
  "enable_offload": true
}
```

Steer this rank to the local owner segment with:

```bash
export MOONCAKE_PREFERRED_SEGMENT=127.0.0.1:50053
```

The owner's SSD directory, on-disk eviction policy, and the DirectIO staging
buffer size are controlled on the `mooncake_client` side via the standard
Mooncake environment variables (`MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`,
`MOONCAKE_BUCKET_EVICTION_POLICY`, `MOONCAKE_USE_URING`,
`MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`,
`MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES`, etc.). Those are independent of
the vLLM JSON config.

### Tenant Isolation

Set `tenant_id` in the Mooncake JSON config when different vLLM deployments should use separate Mooncake tenant namespaces:

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": "80GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false,
  "tenant_id": "tenant-a"
}
```

Strict isolation requires a Mooncake master started with `--enable_multi_tenants=true` and a tenant quota policy that registers each tenant. Non-default `tenant_id` also requires a Mooncake version whose `MooncakeDistributedStore.setup()` accepts the `tenant_id` parameter. In `standalone-store` mode, start the external `mooncake_client` with the matching tenant id because that process owns the real store client.

## Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `MOONCAKE_CONFIG_PATH` | Path to Mooncake JSON config file | (required) |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | Bootstrap port for MooncakeConnector P2P transfer (disagg mode only) | 8998 |
| `MOONCAKE_PREFERRED_SEGMENT` | Pin this rank's replicas to a specific owner segment (`host:port`); used in `standalone-store` mode | — |
| `MOONCAKE_REQUESTER_LOCAL_HOSTNAME` | Override the hostname the vLLM rank registers with Mooncake as a requester. Defaults to the rank's resolved IP. | — |
| `VLLM_MOONCAKE_STORE_TIER_LOG` | When `1`, logs a per-batch tier summary (memory vs disk hits) for observability | disabled |
| `VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO` | Fraction of the owner's DirectIO staging buffer that the requester will fill in a single `batch_get_into_multi_buffers` call. Lower → more conservative pre-split, more round trips. | 0.9 |

## KV Transfer Config

### KV Role Options

- **kv_producer**: For instances that store KV caches to the pool.
- **kv_consumer**: For instances that load KV caches from the pool.
- **kv_both**: The instance both stores and loads KV caches. Use this for single-node CPU offloading or prefiller instances.

### kv_connector_extra_config

- `load_async` (bool): Enable asynchronous loading for better compute-I/O overlap. Default: `true`.
- `lookup_async` (bool): Run the external prefix-cache lookup on a background thread so it never blocks the scheduler step. The request is held until the in-flight lookup completes, then resumed on a later step. Default: `false`.
- `lookup_rpc_port` (int): Custom port for the ZMQ lookup RPC socket. Default: `0`.
- `lookup_timeout_s` (float): Seconds the scheduler's lookup waits for worker rank 0's reply. A lookup without a reply in time, or one the worker failed to answer, is a miss and is logged; the store reset waits without a limit. Default: `30`.
- `lookup_newest_state` (bool): For hybrid models with partial hash hits (`--prefix-match-unit` smaller than the block size), probe the Mamba state keys from the deepest hash unit down and stop at the first unit that gives a hit, instead of checking the state at every unit on the prompt's path. The hit is the same. Every existence check grants the object a Mooncake lease, and the master evicts objects in lease order, so the default keeps every earlier turn's state on a session's path as fresh as its newest one; with this option only the state a request can resume from is renewed. Default: `false`.
- `frontier_write_through` (bool): With `--mamba-frontier-checkpoint`, also write each finished request's Mamba frontier checkpoint (its state at its exact end inside a block: the attention page, every Mamba state and, inside an indexer pool, the indexer ring) to the store, with the full pages its normal saves did not cover and a small locator record that names the frontier's length, so that a later request on any engine sharing the store resumes at that length instead of the last stored block boundary. The lookup finds a frontier through its locator and uses it only when every object a load needs is in the store; with `lookup_newest_state` a lookup that finds one leases nothing else. Needs `kv_role` `kv_both` and `kv_load_failure_policy` `recompute` (as frontier checkpoints do with any connector); under speculative decoding only with `--mamba-spec-state-layout compact`. Design: [Mamba frontier checkpoints in the Mooncake store](../design/mooncake_store_frontier_checkpoints.md). Default: `false`.
- `frontier_supersede_remove` (bool): With `frontier_write_through`, once a request that resumed from a stored frontier checkpoint (one it loaded; not one the GPU still held) has written its own, each rank removes from the store its share of the frontier it resumed from and of the Mamba boundary states it stored past that frontier; no other prefix shares them. The removal is not forced: Mooncake refuses to remove an object whose lease has not expired, so a reader that just checked it keeps it until its load. Pages are never removed. Default: `false`.
- `frontier_supersede_lag` (int): With `frontier_supersede_remove`, a finish supersedes not what its own request resumed from but what the finish this many turns earlier on its session's chain of frontiers would have, so the last turns' frontiers and the block-boundary states under them stay in the store for a client that rewrites one of its recent outputs. A chain that does not continue supersedes nothing more. Default: `0`.
- `frontier_lookup_memo_s` (float): With `frontier_write_through`, a request waiting for space is looked up again at every step. Within this many seconds at the same local hit, a request whose probe found no usable frontier is not probed again (the regular lookup alone runs), and one whose frontier verified reuses that hit for up to `frontier_lookup_reuse_s`, then repeats only the verification, which renews the leases. Default: `0` (off).
- `frontier_lookup_reuse_s` (float): With `frontier_lookup_memo_s`, seconds a verified frontier hit is reused before its verification repeats. Keep it below the master's `default_kv_lease_ttl` (Mooncake's default is 10 s): past the lease the hit's objects may be evicted, and its load then fails into a recompute. Default: `4`.
- `unit_policy_url` (str): With `frontier_write_through`, the admin address (`http://<host>:<port>`) of a Mooncake master that evicts by recovery unit (`--eviction_policy=unit_gds`). Each state the connector writes (a Mamba state at a hash-unit boundary, a prompt's partial tail) and each frontier checkpoint is a unit: its own objects go under the unit's id as their Mooncake group, and the rank that writes locators registers it with the master, with the full pages before it (shared by units and reference-counted by the master), its value (the prefill seconds from position 0 under `--prefill-cost-coefficients`, which it needs) and the deepest unit before it on its request's path. A request touches the unit it resumes at; with `frontier_supersede_remove` the units it supersedes are marked dead instead of removed. Needs a Mooncake client with `ReplicateConfig.group_ids`; excludes `enable_group_semantics`. Default: unset.
- `staging_buffer_mib` (int): MiB of pinned host memory per worker rank, registered with the store, into which each store job copies the GPU bytes it is about to put (after the step's fence, on a copy stream of its own) before any put. Once the copy completes the rank reports the job's GPU blocks released, so the scheduler frees them while the puts from host memory and the rest of the job run on a second thread, in job order; the job still counts as in flight until it completes. A key the store held when the job was copied (so it was not copied) and evicted before the job's puts counts as stored, as an eviction right after a put would leave it. A job that does not fit the buffer within `staging_wait_s` puts from GPU memory as before and keeps its blocks until it completes, and so does a job larger than the whole buffer; both are counted in the rank's periodic staging log line, with a warning the first time. Size the buffer to hold at least the largest job this rank writes (a finish of the longest prompt: the rank's share of its new pages and states) plus the jobs that arrive while the store is still putting earlier ones. Default: `0` (off).
- `staging_wait_s` (float): With `staging_buffer_mib`, seconds a store job waits for room in the staging buffer, which the put thread frees as it finishes earlier jobs. The copying thread serves jobs in order, so later jobs wait behind it. `0` never waits. Default: `5`.
- `frontier_load_local_pages` (bool): With `frontier_write_through`, a request that resumes at a stored frontier checkpoint reads from the store only what the GPU does not hold. The local hit a lookup starts from ends at the deepest Mamba state every group caches, often well before the frontier although the GPU still caches the attention pages up to it; the request then also takes, as local hits, the pages after that hit that the GPU caches under its own block hashes, up to the frontier's block or the first one missing, and the load starts at the first page it does not take. A failed load recomputes the request, as before. Default: `false`.
- `frontier_write_wait_s` (float): With `frontier_write_through`, a finished request's frontier checkpoint reaches the store only when its write job has completed on every rank, and its GPU copy may be gone before that. A waiting request that extends such a frontier of its own session past its local hit is not looked up yet (the scheduler tries it again at its next step, as for a lookup in flight) until the job completes, at most this many seconds after the finish; without it the request's first lookup can miss the frontier, which `frontier_lookup_memo_s` then keeps, and it resumes at a block boundary. Default: `0` (off).
- `frontier_locator_cache` (bool): With `frontier_write_through`, the worker rank that writes the frontier locator records keeps the record it last wrote under each locator key and, while the store still holds that key, reads it in place of the store's copy in a lookup's probe (a small `get` queues behind the store's large transfers). Merging the next frontier into a record always reads the store's copy, so engines sharing a store keep each other's entries. A record another engine changed is read stale by this rank's probes until this rank writes it again: a frontier that engine added is not found here meanwhile, never misread. Default: `false`.
- `cache_prefix` (str): Namespace prepended to every store key. Lets separate deployments share one Mooncake master without polluting each other — instances configured with different prefixes never see each other's cached blocks, even for identical prompts. All instances that should share a prefix cache must use the same value. Default: `""` (no prefix; keys are byte-identical to the unprefixed format).
- `save_decode_cache` (bool): Enable offloading decode tokens' KV cache. A `kv_consumer` does not save during prefill; when decode starts, it fills any missing block-aligned prompt prefix before appending completed decode blocks. Default: `false`.
- `store_tp_size` (int): Common Store TP for endpoints with different local TP sizes. It supports LBHNC and LBNHC local KV caches, with `store_tp_size >= local_tp_size` and `store_tp_size % local_tp_size == 0`. The current topology is one full-attention cache group, PCP/DCP disabled, and cross-layer blocks disabled. For GQA and MHA, the total KV-head count must be divisible by `store_tp_size`. Store shards contain fixed global KV-head ranges in the local layout. Shared endpoints use the same KV cache layout, pipeline-parallel size, and Store TP. The Store namespace includes the layout and PP size. Unsupported configurations use a topology-specific rank-local namespace.

LBHNC/HND is strongly recommended for TP-sharded Store when supported. LBNHC/NHD
creates many transfer segments and may significantly reduce PUT/GET performance.

For example, with prefill TP 4, decode TP 2, and eight KV heads, set
`store_tp_size` to 4 on both instances. Each decode rank reads and writes two
of the four Store shards.

MQA with one total KV head uses a replicated-head layout. For the supported
prefill TP 4 to decode TP 2 case, every rank uses the
same rank-0 key namespace. The four prefill replicas stripe block PUTs so each
object is stored once, while both decode ranks GET every block into their local
KV replica. `store_tp_size` does not appear in MQA keys, so identical MQA
objects written at different store TP sizes share the same pool entry when PP
sizes match.

Tensor-parallel collectives and low-precision arithmetic are not bitwise
invariant across TP sizes, so heterogeneous-TP reuse does not guarantee the
same greedy output as recomputing the prefix at the decode TP size.

## Notes

### Reproducible Block Hashes Across Processes

The `MooncakeStoreConnector` relies on consistent block hashes across all vLLM processes sharing the distributed store. Block hashes chain from `NONE_HASH`, which is derived from a fixed default seed, so identical prompts produce identical block hashes across processes by default — enabling cross-process prefix cache hits without extra configuration.

The exception is the non-cryptographic `xxhash`/`xxhash_cbor` values of `--prefix-caching-hash-algo`, which seed `NONE_HASH` randomly per process; sharing a store with those requires `PYTHONHASHSEED`.

To use a custom shared seed, set the same `PYTHONHASHSEED` on every instance that shares the store (DP ranks, separate prefiller/decoder nodes, and any other vLLM process pointed at the same Mooncake store):

```bash
PYTHONHASHSEED=<shared-value> vllm serve ...
```

### GPU Memory Ordering of Store Transfers

The store's transfer engine reads and writes GPU memory without waiting for work queued on vLLM's CUDA streams: RDMA accesses the memory directly, and Mooncake's `MC_GPU_COPY=pinned` mode copies on a stream of its own. The connector orders these accesses itself:

- A store job's puts, and with `staging_buffer_mib` its copy into the staging buffer, start only after the CUDA event recorded after the step's forward pass has completed, so they read the KV cache that step wrote.
- A load writes only into blocks allocated to the loading request, which no scheduled step reads or writes: the request runs only once its load has completed, and with asynchronous scheduling or pipeline parallelism the scheduler returns a finished request's blocks to the pool only after the steps that may still write them have completed.
