# vLLM with FlashStates for Kimi-K3.1-Pro on GB300

This is vLLM `266e88c05`, the commit inside the `vllm/vllm-openai:kimi-k31-cu134` image, with FlashStates, a serving
system for agentic workloads on hybrid models (attention plus linear-attention or state-space layers). FlashStates saves
recurrent checkpoints at any token position and keeps the complete hybrid states (a recurrent checkpoint and the
attention KV it depends on) that save the most prefill per unit of cache. On Kimi-K3.1-Pro it runs with decode context
parallelism, DSpark speculative decoding and the specialized decode path, and keeps the states in a Mooncake store.

**Results, the benchmark setups and installation: [FutureMLS-Lab/FlashStates](https://github.com/FutureMLS-Lab/FlashStates).**
The GB300 image is `ghcr.io/futuremls-lab/flashstates-gb300:kimi-k31-cu134-266e88c-fs88bad5a-mcf74d68c-gb300`, also
tagged `v0.1.0-kimi-k31-gb300` (on Docker Hub `weili0234/flashstates:kimi-k31-cu134-266e88c-fs88bad5a-mcf74d68c-gb300`
and `weili0234/flashstates-vllm:v0.1.0-kimi-k31-gb300`): the `vllm/vllm-openai:kimi-k31-cu134` image with the vLLM of this
branch's commit 88bad5a and the Mooncake build of the `flashstate-k31pro-gb300` branch of
[FutureMLS-Lab/Mooncake](https://github.com/FutureMLS-Lab/Mooncake/tree/flashstate-k31pro-gb300) (commit f74d68c). The design
is described in
[`docs/design/hybrid_state_checkpoints.md`](../docs/design/hybrid_state_checkpoints.md) and
[`docs/design/mooncake_store_frontier_checkpoints.md`](../docs/design/mooncake_store_frontier_checkpoints.md).

## Options

FlashStates' changes are options that are off by default:

| Option | Effect |
| --- | --- |
| `--mamba-frontier-checkpoint` | keep a finished request's recurrent state at its exact end, so the next turn resumes there instead of at the last block boundary of its prompt |
| `--mamba-prefill-export` | let a prefill chunk run past a position where a state must be cached and have the backend write the state there |
| `--prefix-cache-eviction gds`, `--prefill-cost-coefficients D E` | evict whole resumable states by GreedyDual-Size over the prefill seconds a state saves per block it frees |
| `--prefix-cache-leaf-last`, `--prefix-cache-leaf-idle-seconds S` | evict every inner state before any leaf; a leaf unused for `S` seconds ranks as inner |
| `--mamba-spec-state-layout compact` | under MTP speculative decoding, keep one state block per running request and fold accepted updates in after sampling |
| `truncate_acceptance_at_stop` (speculative config) | accept no draft past the request's stop, so a finished request's state ends at its last token |
| Mooncake store connector keys (`frontier_write_through`, `lookup_newest_state`, `frontier_keep_states`, `unit_policy_url`, ...) | write each finished request's state through to a Mooncake store, look up the newest state first, keep each session's newest recurrent checkpoints, and register recovery units with a unit-evicting master; see [`docs/features/mooncake_store_connector_usage.md`](../docs/features/mooncake_store_connector_usage.md) |

With every option off, this branch behaves as vLLM `266e88c05` except for two fixes that are always on: a Mamba block
whose state column a chunk did not write is not cached, and GLM-5.3-Flash's MTP layer is built unquantized when the
checkpoint stores it in BF16.

## Kimi-K3.1-Pro on GB300 with a Mooncake store

One engine runs on two GB300 trays of one NVL72 domain (TP8, DCP8) with a 640 GiB store (80 GB per GPU). The Mooncake
master, on the head node:

```bash
mooncake_master -rpc_port=50051 -rpc_thread_num=4 -default_kv_lease_ttl=30000 \
    -eviction_high_watermark_ratio=0.95 -eviction_ratio=0.1 \
    --eviction_policy=unit_gds --unit_leaf_last=true --unit_idle_demote_s=300 --unit_orphan_grace_s=120 \
    --unit_anchor_tier=true --unit_dead_headroom_ratio=0.05
```

The client configuration, the same file on both nodes (`MOONCAKE_CONFIG_PATH`):

```json
{"mode": "embedded", "metadata_server": "P2PHANDSHAKE", "master_server_address": "<head address>:50051",
 "global_segment_size": "80GB", "local_buffer_size": "4GB", "protocol": "rdma",
 "device_name": "<the compute RDMA NICs>", "enable_offload": false}
```

On each node (node rank 0 on the head; rank 1 adds `--headless`):

```bash
MC_GPU_COPY=pinned PYTHONHASHSEED=0 MOONCAKE_REQUESTER_LOCAL_HOSTNAME=<this node's address> \
vllm serve moonshotai/Kimi-K3.1-Pro --trust-remote-code \
    --tensor-parallel-size 8 --nnodes 2 --node-rank <rank> --master-addr <head address> \
    --decode-context-parallel-size 8 --load-format fastsafetensors \
    --speculative-config '{"model": "Inferact/Kimi-K3.1-Pro-DSpark-preview", "num_speculative_tokens": 5, "method": "dspark", "draft_sample_method": "probabilistic", "truncate_acceptance_at_stop": true}' \
    --additional-config '{"decode_backend": "latency_specialized"}' \
    --enable-prefix-caching --gpu-memory-utilization 0.95 --max-num-seqs 256 --max-num-batched-tokens 16384 \
    --mamba-frontier-checkpoint --prefix-cache-eviction gds --prefill-cost-coefficients 3.407e-05 7.329e-06 \
    --prefix-cache-leaf-last --prefix-cache-leaf-idle-seconds 60 --prefix-cache-retention-interval 0 \
    --kv-transfer-config '{"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both", "kv_load_failure_policy": "recompute", "kv_connector_extra_config": {"frontier_write_through": true, "lookup_newest_state": true, "frontier_lookup_memo_s": 30, "frontier_load_local_pages": true, "frontier_write_wait_s": 3, "frontier_locator_cache": true, "lookup_async": true, "unit_policy_url": "http://127.0.0.1:9003", "frontier_keep_states": 2, "frontier_remove_retry_s": 31}}'
```

`unit_policy_url` is the master's admin port (`--metrics_port`, 9003 by default) on the head node, where the scheduler
runs. The prefill cost coefficients
were fitted from timed prefills on this setup; only their ratio changes the eviction order. GB300 trays have no GPU
peer-memory module, so the image sets `WITH_NVIDIA_PEERMEM=0` and the store registers GPU memory through DMA-BUF. For
chat use, add the base command's `--enable-auto-tool-choice --tool-call-parser kimi_k3 --reasoning-parser kimi_k3`.
