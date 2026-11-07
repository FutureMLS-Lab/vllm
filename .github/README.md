# vLLM with FlashStates

This is vLLM v0.33.0 with FlashStates, a serving system for agentic workloads on hybrid models (attention plus
linear-attention or state-space layers). FlashStates saves recurrent checkpoints at any token position and keeps the
complete hybrid states (a recurrent checkpoint and the attention KV it depends on) that save the most prefill per unit
of cache.

**Results, the B200 image, the benchmark setups and installation: [FutureMLS-Lab/FlashStates](https://github.com/FutureMLS-Lab/FlashStates).**
The design is described in [`docs/design/hybrid_state_checkpoints.md`](../docs/design/hybrid_state_checkpoints.md).

## Options

FlashStates' changes are options that are off by default:

| Option | Effect |
| --- | --- |
| `--mamba-frontier-checkpoint` | keep a finished request's recurrent state at its exact end, so the next turn resumes there instead of at the last block boundary of its prompt |
| `--mamba-prefill-export` | let a prefill chunk run past a position where a state must be cached and have the backend write the state there |
| `--prefix-cache-eviction gds`, `--prefill-cost-coefficients D E` | evict whole resumable states by GreedyDual-Size over the prefill seconds a state saves per block it frees |
| `--prefix-cache-leaf-last`, `--prefix-cache-leaf-idle-seconds S` | evict every inner state before any leaf; a leaf unused for `S` seconds ranks as inner |
| `--mamba-spec-state-layout compact` | under speculative decoding, keep one state block per running request and fold accepted updates in after sampling |
| `VLLM_DETOKENIZE_PRIME_SUFFIX=1` | prime the incremental detokenizer with the prompt's last tokens |
| Mooncake store connector keys (`lookup_newest_state`, `frontier_write_through`, ...) | write each finished request's state through to a Mooncake store and look up the newest state first; see `docs/features/mooncake_store_connector_usage.md` |

With every option off, this branch behaves as vLLM v0.33.0 except for three fixes that are always on: a Mamba block
whose state column a chunk did not write is not cached; GLM-5.3-Flash's MTP layer is built unquantized when the
checkpoint stores it in BF16; and the upstream fix #55112 (copy every overlaid KV view in
`copy_kv_cache_blocks_inplace`).
