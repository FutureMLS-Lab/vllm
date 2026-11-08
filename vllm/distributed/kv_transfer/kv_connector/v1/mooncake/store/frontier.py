# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba frontier checkpoints in the Mooncake store (``frontier_write_through``).

A frontier checkpoint (``vllm/v1/core/hybrid_checkpoint/frontier.py``) is a
finished request's state after its first ``L`` tokens, ``L`` inside a cache
block ``k = L // B``: the page holding ``[kB, L)`` in every full-attention
group, the running state in every Mamba group and, when ``L`` is inside an
indexer compression pool, the indexer ring. Locally a later request extending
those tokens resumes at ``L``; this module lets another engine sharing the
store do the same.

Write. A finish's frontier becomes one store job (``FrontierPut``): the full
pages ``[s, k)`` its normal saves did not cover, under their block hashes as
normal saves key them (``ReqMeta.boundary_puts``), then every frontier block
under the frontier key in its group's namespace (a ring, not a
prefix-cacheable group, in a ring namespace of its own), then, on one rank,
``(L, frontier key)`` in the locator record under the frontier's locator key,
newest first. The locator key chains the request's hash at a hash-unit
boundary with the next ``FRONTIER_LOCATOR_TOKENS`` tokens, so another engine
computes it from its own tokens (``keys.frontier_locator_key``).

Lookup (``FrontierLookup``, scheduler side). For each hash unit a frontier
longer than the local hit could be anchored in, the scheduler computes the
locator key; the worker returns the entries of the records that exist; the
scheduler keeps the longest entry whose key equals its own frontier key at
that length; the worker then checks that every object the load needs exists
on every rank: the pages after the local hit and every frontier block. Each
check grants a Mooncake lease, which keeps the objects until the load. Any
missing object gives no frontier hit. The load writes the pages, the page
holding ``[kB, L)``, each state into the running block and each ring into the
request's ring.

Lease rule. Mooncake evicts the objects whose lease expired, oldest first.
With ``lookup_newest_state`` a lookup asks for the frontier first and, when it
hits, for nothing else, so only the newest usable state and its pages are
renewed. Without it the regular lookup runs too, as it does without
frontiers.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    FrontierPut,
    KeyMetadata,
    MooncakeLookupResult,
    PoolKey,
    ReqMeta,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.hybrid_checkpoint.keys import (
    FRONTIER_LOCATOR_TOKENS,
    frontier_key,
    frontier_locator_key,
    frontier_locator_unit,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    iter_layer_specs,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
        LookupKeyClient,
    )
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Entries a locator record keeps, newest first.
LOCATOR_ENTRIES = 4
# frontier_locator_cache: records the locator-writing rank keeps, the oldest
# written dropped first.
LOCATOR_CACHE_RECORDS = 65536
LOCATOR_MAGIC = b"MFL1"
# Store namespaces of locator records and of indexer rings.
LOCATOR_NAMESPACE = "frontier_locator"
RING_NAMESPACE = "ring"
# Seconds between the scheduler's count log lines.
_LOG_EVERY = 60.0
# Default of frontier_lookup_reuse_s: under half of Mooncake's default 10 s
# lease, so a reused hit's objects stay leased.
VERIFY_REUSE_S = 4.0
# frontier_lookup_memo_s: entries the memo keeps before pruning old ones.
_MEMO_ENTRIES = 4096
# Hash units a probe looks for locators in at most, the deepest first: a
# frontier further below the prompt's end is not found, and the request resumes
# where its regular lookup does.
_PROBE_UNITS = 4096
# Mooncake ErrorCode values (mooncake-store/include/types.h).
_OBJECT_NOT_FOUND = -704
_OBJECT_HAS_LEASE = -706
# frontier_remove_retry_s: times a removal refused under a lease is tried
# again, and the refused keys waiting at most (the oldest dropped first).
_REMOVE_RETRIES = 2
_REMOVE_RETRY_KEYS = 1 << 16

# One object to put: its key, buffer addresses and sizes, and key metadata.
Put = tuple[str, list[int], list[int], KeyMetadata]


@dataclass(frozen=True)
class FrontierStoreConfig:
    """The frontier-checkpoint options of ``kv_connector_extra_config``, all
    off by default.

    Attributes:
        write_through: ``frontier_write_through``: finished requests' Mamba
            frontier checkpoints go to the store, and lookups find them.
        supersede_remove: ``frontier_supersede_remove``: once a request that
            resumed from a frontier has written its own, the frontier it
            resumed from and the boundary states it stored past that
            frontier are removed from the store.
        lookup_memo_s: ``frontier_lookup_memo_s``: seconds a request waiting
            at the same local hit reuses its last frontier lookup.
        lookup_reuse_s: ``frontier_lookup_reuse_s``: within ``lookup_memo_s``,
            seconds a verified frontier hit is reused before its verification
            repeats, which renews the objects' leases; keep it below the
            master's ``default_kv_lease_ttl``.
        unit_policy_url: ``unit_policy_url``: the admin address
            (``http://<host>:<port>``) of a Mooncake master running the
            recovery-unit eviction policy; store jobs register the units they
            write with it.
        load_local_pages: ``frontier_load_local_pages``: a load resuming at a
            stored frontier reads only the pages the GPU does not cache.
        write_wait_s: ``frontier_write_wait_s``: at most this many seconds
            after a finish, a request extending that finish's frontier past
            its local hit is not looked up until the frontier is written.
        supersede_lag: ``frontier_supersede_lag``: with ``supersede_remove``,
            a finish supersedes what the finish this many turns earlier on
            its session's chain of frontiers would have.
        locator_cache: ``frontier_locator_cache``: the rank that writes the
            locator records reads its own last record under a key instead of
            the store's copy.
        soft_pin_rest: ``frontier_soft_pin_rest``: every object the connector
            puts except the frontiers' is soft-pinned, so the master's
            eviction takes frontier checkpoints first and pages and
            block-boundary states only to make up a round's minimum.
        supersede_states: ``frontier_supersede_states``: with
            ``supersede_remove``, a finish also removes the boundary states
            it stored past the frontier it resumed from; false removes only
            that frontier.
        keep_states: ``frontier_keep_states``: a session's chain of turns
            keeps at most this many recurrent checkpoints in the store, the
            deepest. A request continues the chain when its tokens extend the
            chain's newest frontier, wherever it resumed: from the store,
            from the GPU or by recompute. Once its own frontier is written,
            the chain's other frontiers and boundary states, its own included,
            are removed. The boundary states within the prompt of the chain's
            first request, which other sessions' first prompts may share, are
            left to the store's eviction. 0 removes none.
        keep_margin_tokens: ``frontier_keep_margin_tokens``: with
            ``keep_states``, every kept checkpoint but the newest lies at
            least this many tokens below the finishing request's prompt end,
            the deepest such first. A client that rewrites the end of its last
            prompt resumes below the rewrite, so a boundary state just under
            the prompt end serves it worse than an older checkpoint. 0 keeps
            the deepest checkpoints whatever their place.
        remove_retry_s: ``frontier_remove_retry_s``: seconds after which a
            removal the store refused under a lease is tried again, up to
            ``_REMOVE_RETRIES`` times; 0 leaves refused objects to the
            store's eviction.

    """

    write_through: bool = False
    supersede_remove: bool = False
    lookup_memo_s: float = 0.0
    lookup_reuse_s: float = VERIFY_REUSE_S
    unit_policy_url: str | None = None
    load_local_pages: bool = False
    write_wait_s: float = 0.0
    supersede_lag: int = 0
    locator_cache: bool = False
    soft_pin_rest: bool = False
    supersede_states: bool = True
    keep_states: int = 0
    remove_retry_s: float = 0.0
    keep_margin_tokens: int = 0

    @staticmethod
    def from_vllm_config(vllm_config: VllmConfig) -> FrontierStoreConfig:
        """The options of ``vllm_config``'s KV connector.

        Raises:
            ValueError: for an option of the wrong type, or ``write_through``
                without ``mamba_frontier_checkpoint`` or with a KDA state commit
                deferred to the next verify (RecoverSSM).

        """
        assert vllm_config.kv_transfer_config is not None
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        write_through = _flag(extra_config, "frontier_write_through")
        supersede_remove = _flag(extra_config, "frontier_supersede_remove")
        lookup_memo_s = _seconds(extra_config, "frontier_lookup_memo_s")
        load_local_pages = _flag(extra_config, "frontier_load_local_pages")
        write_wait_s = _seconds(extra_config, "frontier_write_wait_s")
        locator_cache = _flag(extra_config, "frontier_locator_cache")
        soft_pin_rest = _flag(extra_config, "frontier_soft_pin_rest")
        if not write_through and (
            supersede_remove
            or lookup_memo_s
            or load_local_pages
            or write_wait_s
            or locator_cache
            or soft_pin_rest
        ):
            raise ValueError(
                "frontier_supersede_remove, frontier_lookup_memo_s, "
                "frontier_load_local_pages, frontier_write_wait_s, "
                "frontier_locator_cache and frontier_soft_pin_rest need "
                "frontier_write_through"
            )
        supersede_states = extra_config.get("frontier_supersede_states", True)
        if not isinstance(supersede_states, bool):
            raise ValueError("frontier_supersede_states must be a bool")
        if "frontier_supersede_states" in extra_config and not supersede_remove:
            raise ValueError(
                "frontier_supersede_states needs frontier_supersede_remove"
            )
        if soft_pin_rest and supersede_remove and supersede_states:
            # Removing the boundary states under a frontier the store evicts
            # first would leave the session nothing to resume at.
            raise ValueError(
                "frontier_soft_pin_rest with frontier_supersede_remove needs "
                "frontier_supersede_states false"
            )
        lookup_reuse_s = _seconds(
            extra_config, "frontier_lookup_reuse_s", VERIFY_REUSE_S
        )
        if "frontier_lookup_reuse_s" in extra_config and not lookup_memo_s:
            raise ValueError("frontier_lookup_reuse_s needs frontier_lookup_memo_s")
        supersede_lag = extra_config.get("frontier_supersede_lag", 0)
        if isinstance(supersede_lag, bool) or not isinstance(supersede_lag, int):
            raise ValueError("frontier_supersede_lag must be an integer >= 0")
        if supersede_lag < 0:
            raise ValueError("frontier_supersede_lag must be an integer >= 0")
        if supersede_lag and not supersede_remove:
            raise ValueError("frontier_supersede_lag needs frontier_supersede_remove")
        keep_states = extra_config.get("frontier_keep_states", 0)
        if isinstance(keep_states, bool) or not isinstance(keep_states, int):
            raise ValueError("frontier_keep_states must be an integer >= 0")
        if keep_states < 0:
            raise ValueError("frontier_keep_states must be an integer >= 0")
        if keep_states and not write_through:
            raise ValueError("frontier_keep_states needs frontier_write_through")
        if keep_states and supersede_remove:
            # Both remove a session's older checkpoints; keep_states also
            # follows sessions that resume from the GPU.
            raise ValueError(
                "frontier_keep_states and frontier_supersede_remove are exclusive"
            )
        if keep_states and soft_pin_rest:
            # The store evicts frontiers first under soft pins, which could
            # leave a session none of its kept checkpoints.
            raise ValueError(
                "frontier_keep_states and frontier_soft_pin_rest are exclusive"
            )
        keep_margin_tokens = extra_config.get("frontier_keep_margin_tokens", 0)
        if isinstance(keep_margin_tokens, bool) or not isinstance(
            keep_margin_tokens, int
        ):
            raise ValueError("frontier_keep_margin_tokens must be an integer >= 0")
        if keep_margin_tokens < 0:
            raise ValueError("frontier_keep_margin_tokens must be an integer >= 0")
        if keep_margin_tokens and keep_states < 2:
            # Only checkpoints other than the newest have a margin.
            raise ValueError(
                "frontier_keep_margin_tokens needs frontier_keep_states >= 2"
            )
        remove_retry_s = _seconds(extra_config, "frontier_remove_retry_s")
        if remove_retry_s and not (supersede_remove or keep_states):
            raise ValueError(
                "frontier_remove_retry_s needs frontier_supersede_remove or "
                "frontier_keep_states"
            )
        unit_policy_url = extra_config.get("unit_policy_url")
        if unit_policy_url is not None:
            parts = (
                urlsplit(unit_policy_url) if isinstance(unit_policy_url, str) else None
            )
            if parts is None or parts.scheme != "http" or parts.port is None:
                raise ValueError(
                    "unit_policy_url must be http://<host>:<port>, got "
                    f"{unit_policy_url!r}"
                )
            if not write_through:
                raise ValueError("unit_policy_url needs frontier_write_through")
            if soft_pin_rest:
                # The unit policy orders eviction itself.
                raise ValueError(
                    "frontier_soft_pin_rest and unit_policy_url are exclusive"
                )
            if vllm_config.cache_config.prefill_cost_coefficients is None:
                # A unit's value is the prefill it saves.
                raise ValueError("unit_policy_url needs prefill_cost_coefficients")
        if write_through and not vllm_config.cache_config.mamba_frontier_checkpoint:
            raise ValueError("frontier_write_through needs mamba_frontier_checkpoint")
        return FrontierStoreConfig(
            write_through=write_through,
            supersede_remove=supersede_remove,
            lookup_memo_s=lookup_memo_s,
            lookup_reuse_s=lookup_reuse_s,
            unit_policy_url=unit_policy_url,
            load_local_pages=load_local_pages,
            write_wait_s=write_wait_s,
            supersede_lag=supersede_lag,
            locator_cache=locator_cache,
            soft_pin_rest=soft_pin_rest,
            supersede_states=supersede_states,
            keep_states=keep_states,
            remove_retry_s=remove_retry_s,
            keep_margin_tokens=keep_margin_tokens,
        )


def _flag(extra_config: dict[str, Any], name: str) -> bool:
    value = extra_config.get(name, False)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _seconds(extra_config: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = extra_config.get(name, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 <= value < math.inf
    ):
        raise ValueError(f"{name} must be a finite number >= 0")
    return float(value)


def ring_group_ids(kv_cache_config: KVCacheConfig) -> tuple[int, ...]:
    """The indexer-ring groups (not prefix-cacheable) a frontier holds. A
    worker's groups carry ``UniformTypeKVCacheSpecs`` where the scheduler's
    carry one layer's spec, so a group is judged by its layers' specs."""
    out = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        specs = list(iter_layer_specs(group.kv_cache_spec))
        if specs and all(isinstance(spec, CircularBufferSpec) for spec in specs):
            out.append(group_id)
    return tuple(out)


def ring_unit(kv_cache_config: KVCacheConfig) -> int:
    """Tokens per indexer pool (``CircularBufferSpec.group_size``, as the
    scheduler's frontier index counts them): a frontier holds its rings unless
    its length is a multiple. Under speculative decoding a ring has more rows
    than a pool holds tokens."""
    sizes = []
    for group_id in ring_group_ids(kv_cache_config):
        for spec in iter_layer_specs(
            kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
        ):
            assert isinstance(spec, CircularBufferSpec)
            sizes.append(spec.group_size)
    return math.lcm(1, *sizes)


def encode_locator(entries: Sequence[tuple[int, bytes]]) -> bytes:
    """A locator record: ``(frontier length, frontier key)`` entries, newest
    first."""
    out = bytearray(LOCATOR_MAGIC)
    out.append(len(entries))
    for length, key in entries:
        out.extend(length.to_bytes(4, "big"))
        out.append(len(key))
        out.extend(key)
    return bytes(out)


def decode_locator(data: bytes) -> list[tuple[int, bytes]]:
    """The entries of a locator record; none for anything else, since a
    reader checks every entry against its own frontier key anyway."""
    if data[:4] != LOCATOR_MAGIC or len(data) < 5:
        return []
    entries: list[tuple[int, bytes]] = []
    offset = 5
    for _ in range(data[4]):
        if offset + 5 > len(data):
            return []
        length = int.from_bytes(data[offset : offset + 4], "big")
        key_len = data[offset + 4]
        key = bytes(data[offset + 5 : offset + 5 + key_len])
        if len(key) != key_len:
            return []
        entries.append((length, key))
        offset += 5 + key_len
    return entries


def merge_locator(
    entries: Sequence[tuple[int, bytes]], length: int, key: bytes
) -> list[tuple[int, bytes]]:
    """``(length, key)`` first, then the other entries, at most
    ``LOCATOR_ENTRIES``."""
    others = [entry for entry in entries if entry != (length, key)]
    return [(length, key), *others][:LOCATOR_ENTRIES]


class FrontierStore:
    """Worker side: keys and buffers of one rank's frontier objects.

    ``store_dbs`` and ``lookup_prefixes`` are the connector's per store group
    (pages and Mamba states); ``ring_dbs`` and ``ring_prefixes`` key the
    indexer rings by KV-cache group id, each rank its own copy. One rank
    (``writes_locators``) writes the locator records and, with
    ``cache_locators`` (``frontier_locator_cache``), keeps the record it last
    wrote under each key. ``block_size`` is a page's tokens and
    ``state_size`` a Mamba block's (a page's by default; under decode context
    parallelism a page holds several Mamba blocks).
    """

    def __init__(
        self,
        store_dbs: Sequence[ChunkedTokenDatabase],
        lookup_prefixes: Sequence[tuple[str, ...]],
        ring_dbs: dict[int, ChunkedTokenDatabase],
        ring_prefixes: dict[int, tuple[str, ...]],
        mamba_group_ids: set[int],
        block_size: int,
        hash_block_size: int,
        unit: int,
        namespace: str,
        *,
        writes_locators: bool,
        cache_locators: bool = False,
        state_size: int | None = None,
        remove_retry_s: float = 0.0,
    ) -> None:
        self.store_dbs = store_dbs
        self.lookup_prefixes = lookup_prefixes
        self.ring_dbs = ring_dbs
        self.ring_prefixes = ring_prefixes
        self.mamba_group_ids = mamba_group_ids
        self.block_size = block_size
        self.state_size = state_size or block_size
        self.units_per_block = block_size // hash_block_size
        self.unit = unit
        self.namespace = namespace
        self.writes_locators = writes_locators
        # frontier_locator_cache: locator key -> the record this rank last
        # upserted there. The rank's sending thread writes it and its lookup
        # server thread reads it.
        self.cache_locators = cache_locators and writes_locators
        self._records: dict[str, bytes] = {}
        self.record_reads: Counter[str] = Counter()
        self._reads_logged_at = time.monotonic()
        # frontier_supersede_remove: batch_remove outcomes.
        self.removals: Counter[str] = Counter()
        self._removals_logged_at = time.monotonic()
        # frontier_remove_retry_s: (time due, key, retries left after it) of
        # the removals refused under a lease, in time order.
        self.remove_retry_s = remove_retry_s
        self._retries: deque[tuple[float, str, int]] = deque()

    def note_removals(self, codes: Sequence[int]) -> None:
        """Count batch_remove codes and log the counts once a minute."""
        for code in codes:
            if code == 0:
                self.removals["removed"] += 1
            elif code == _OBJECT_NOT_FOUND:
                self.removals["absent"] += 1
            elif code == _OBJECT_HAS_LEASE:
                self.removals["leased"] += 1
            else:
                self.removals["error"] += 1
        now = time.monotonic()
        if now - self._removals_logged_at >= _LOG_EVERY:
            self._removals_logged_at = now
            logger.info(
                "Mooncake frontier removals: %s", dict(sorted(self.removals.items()))
            )

    def due_retries(self) -> list[tuple[str, int]]:
        """``frontier_remove_retry_s``: the refused removals due again, each
        with the retries left after it."""
        now = time.monotonic()
        due: list[tuple[str, int]] = []
        while self._retries and self._retries[0][0] <= now:
            _, key, left = self._retries.popleft()
            due.append((key, left))
        return due

    def defer_leased(
        self, keys: Sequence[str], codes: Sequence[int], retries_left: Sequence[int]
    ) -> None:
        """``frontier_remove_retry_s``: the removals refused under a lease
        that have retries left are due again once the lease can have run
        out."""
        if not self.remove_retry_s:
            return
        due = time.monotonic() + self.remove_retry_s
        for key, code, left in zip(keys, codes, retries_left):
            if code == _OBJECT_HAS_LEASE and left > 0:
                self._retries.append((due, key, left - 1))
                self.removals["deferred"] += 1
        while len(self._retries) > _REMOVE_RETRY_KEYS:
            self._retries.popleft()

    def locator_key(self, locator: bytes) -> str:
        return f"{self.namespace}@{LOCATOR_NAMESPACE}@{locator.hex()}"

    def read_record(self, store: Any, key: str, cached_ok: bool = True) -> bytes:
        """The locator record under ``key``, which the store holds: with
        ``cache_locators`` and ``cached_ok`` the one this rank last wrote
        there, else the store's copy."""
        cached = self._records.get(key) if self.cache_locators and cached_ok else None
        record = cached if cached is not None else bytes(store.get(key))
        if self.cache_locators:
            self.record_reads["cached" if cached is not None else "store"] += 1
            now = time.monotonic()
            if now - self._reads_logged_at >= _LOG_EVERY:
                self._reads_logged_at = now
                logger.info(
                    "Mooncake frontier locator reads: %s",
                    dict(sorted(self.record_reads.items())),
                )
        return record

    def remember_record(self, key: str, record: bytes | None) -> None:
        """``cache_locators``: ``record`` is what the store now holds under
        ``key``; None when that is unknown (a failed upsert)."""
        if not self.cache_locators:
            return
        self._records.pop(key, None)
        if record is None:
            return
        self._records[key] = record
        if len(self._records) > LOCATOR_CACHE_RECORDS:
            self._records.pop(next(iter(self._records)))

    def object_puts(
        self, put: FrontierPut, tp_rank: int, put_steps: Sequence[int]
    ) -> list[Put]:
        """This rank's frontier objects: a replicated group's block from the
        rank that stores chunk ``k`` (as normal saves stripe chunks), a Mamba
        state and a ring from every rank."""
        k = put.length // self.block_size
        key = BlockHash(put.key)
        puts: list[Put] = []
        for group_id, block_id in put.blocks:
            step = put_steps[group_id]
            if k % step != (tp_rank + group_id) % step:
                continue
            db = self.store_dbs[group_id]
            addrs, sizes = db.prepare_value_for_block(block_id)
            puts.append((db.key_for(key), addrs, sizes, db.metadata))
        for group_id, block_id in put.rings:
            db = self.ring_dbs[group_id]
            addrs, sizes = db.prepare_value_for_block(block_id)
            puts.append((db.key_for(key), addrs, sizes, db.metadata))
        return puts

    def object_keys(
        self, key: bytes, length: int, tp_rank: int, put_steps: Sequence[int]
    ) -> list[str]:
        """This rank's keys of the frontier ``(key, length)``, as
        ``object_puts`` writes them; every ring inside a pool."""
        k = length // self.block_size
        chunk = BlockHash(key)
        keys = [
            db.key_for(chunk)
            for group_id, db in enumerate(self.store_dbs)
            if k % put_steps[group_id] == (tp_rank + group_id) % put_steps[group_id]
        ]
        if length % self.unit:
            keys.extend(db.key_for(chunk) for db in self.ring_dbs.values())
        return keys

    def page_keys(
        self, block_hashes: Sequence[BlockHash], num_blocks: int
    ) -> list[str]:
        """Every namespace's key of the full pages ``[0, num_blocks)``, block
        by block (``unit_policy_url``: a unit's path pages)."""
        per = self.units_per_block
        groups = [
            prefixes
            for group_id, prefixes in enumerate(self.lookup_prefixes)
            if group_id not in self.mamba_group_ids
        ]
        keys: list[str] = []
        for i in range(num_blocks):
            page_hex = block_hashes[(i + 1) * per - 1].hex()
            for prefixes in groups:
                keys.extend(PoolKey.build_key_string(p, page_hex) for p in prefixes)
        return keys

    def verify_keys(
        self,
        block_hashes: Sequence[BlockHash],
        key: bytes,
        length: int,
        first_block: int,
    ) -> list[str]:
        """Every object a load resuming at the frontier needs, in every
        namespace it is read from: the pages ``[first_block, k)``, the
        frontier block of every store group and, inside a pool, every ring."""
        k = length // self.block_size
        per = self.units_per_block
        frontier_hex = key.hex()
        keys: list[str] = []
        for group_id, prefixes in enumerate(self.lookup_prefixes):
            if group_id not in self.mamba_group_ids:
                for i in range(first_block, k):
                    page_hex = block_hashes[(i + 1) * per - 1].hex()
                    keys.extend(PoolKey.build_key_string(p, page_hex) for p in prefixes)
            keys.extend(PoolKey.build_key_string(p, frontier_hex) for p in prefixes)
        if length % self.unit:
            for prefixes in self.ring_prefixes.values():
                keys.extend(PoolKey.build_key_string(p, frontier_hex) for p in prefixes)
        return keys

    def load_plan(
        self, req_meta: ReqMeta
    ) -> tuple[list[str], list[list[int]], list[list[int]], list[int]]:
        """Keys, buffers and block ids of a load resuming at the frontier:
        pages ``[s, k)`` after the local hit ``s``, the page holding
        ``[kB, L)`` into block ``k``, each state into the Mamba block holding
        token ``L - 1``, where a local frontier hit puts it, and each ring
        into the request's ring."""
        spec = req_meta.load_spec
        assert spec is not None and spec.frontier is not None
        frontier = spec.frontier
        k = frontier.length // self.block_size
        per = self.units_per_block
        key = BlockHash(frontier.key)
        keys: list[str] = []
        addrs: list[list[int]] = []
        sizes: list[list[int]] = []
        block_ids: list[int] = []

        def add(db: ChunkedTokenDatabase, chunk_hash: BlockHash, block_id: int) -> None:
            assert block_id != NULL_BLOCK_ID, (req_meta.req_id, frontier.length)
            addr, size = db.prepare_value_for_block(block_id)
            keys.append(db.key_for(chunk_hash))
            addrs.append(addr)
            sizes.append(size)
            block_ids.append(block_id)

        state_block = cdiv(frontier.length, self.state_size) - 1
        for group_id, db in enumerate(self.store_dbs):
            table = req_meta.block_ids[group_id]
            if group_id in self.mamba_group_ids:
                add(db, key, table[state_block])
                continue
            for i in range(spec.vllm_cached_tokens // self.block_size, k):
                add(db, req_meta.block_hashes[(i + 1) * per - 1], table[i])
            add(db, key, table[k])
        for group_id, block_id in frontier.rings:
            add(self.ring_dbs[group_id], key, block_id)
        return keys, addrs, sizes, block_ids


class FrontierLookup:
    """Scheduler side: the longest stored frontier a request extends past its
    local hit, or the regular lookup. Whether the regular lookup also runs
    when a frontier is found is the worker's lease rule
    (``lookup_newest_state``)."""

    def __init__(
        self,
        client: LookupKeyClient,
        block_size: int,
        hash_block_size: int,
        hash_fn: Callable[[Any], bytes],
        memo_s: float = 0.0,
        reuse_s: float = VERIFY_REUSE_S,
    ) -> None:
        self.client = client
        self.block_size = block_size
        self.hash_block_size = hash_block_size
        self.hash_fn = hash_fn
        self.counts: Counter[str] = Counter()
        self._logged_at = time.monotonic()
        # frontier_lookup_memo_s: request id -> (local hit, probe time,
        # (length, key) of the verified frontier or None, its hit, verify
        # time). The client's thread fills it while the scheduler's forgets.
        self.memo_s = memo_s
        self.reuse_s = reuse_s
        self._memo_lock = threading.Lock()
        # With debug logging: request id -> the path its last lookup took and
        # the lookup's milliseconds.
        self.record_paths = logger.isEnabledFor(logging.DEBUG)
        self.paths: dict[str, tuple[str, float]] = {}
        self._path_of: dict[str, str] = {}
        # Request -> the local hit its lookup in flight was submitted at.
        self._submitted_at: dict[str, int] = {}
        self._memo: dict[
            str,
            tuple[int, float, tuple[int, bytes] | None, MooncakeLookupResult, float],
        ] = {}

    def key_at(self, request: Request, length: int) -> bytes:
        """The frontier key of ``request``'s first ``length`` tokens."""
        return bytes(frontier_key(self.hash_fn, request, length, self.hash_block_size))

    def forget(self, req_id: str) -> None:
        with self._memo_lock:
            self._memo.pop(req_id, None)
        self._path_of.pop(req_id, None)
        self._submitted_at.pop(req_id, None)

    def _remember(
        self,
        req_id: str,
        local: int,
        frontier: tuple[int, bytes] | None,
        hit: MooncakeLookupResult,
        probed: float | None = None,
    ) -> None:
        now = time.monotonic()
        with self._memo_lock:
            if len(self._memo) > _MEMO_ENTRIES:
                for old in [
                    r for r, m in self._memo.items() if now - m[1] > self.memo_s
                ]:
                    del self._memo[old]
            self._memo[req_id] = (
                local,
                now if probed is None else probed,
                frontier,
                hit,
                now,
            )

    def lookup(
        self, request: Request, num_computed_tokens: int, non_block: bool
    ) -> MooncakeLookupResult | None:
        """As ``LookupKeyClient.lookup``: None while an asynchronous lookup
        is in flight."""

        def run() -> MooncakeLookupResult:
            # On the client's thread, which an asynchronous lookup runs on.
            start = time.perf_counter()
            result = self._lookup(request, num_computed_tokens)
            if self.record_paths:
                path = self._path_of.pop(request.request_id, "none")
                elapsed_ms = (time.perf_counter() - start) * 1e3
                self.paths[request.request_id] = (path, elapsed_ms)
            return result

        req_id = request.request_id
        submitted = self._submitted_at.get(req_id)
        if submitted is None or submitted > num_computed_tokens:
            if submitted is not None:
                # The local hit shrank: the frontier check of the lookup in
                # flight skipped the pages the request no longer holds.
                self.client.discard(req_id)
            self._submitted_at[req_id] = num_computed_tokens
        result = self.client.run(req_id, run, non_block)
        if result is not None:
            del self._submitted_at[req_id]
        now = time.monotonic()
        if now - self._logged_at >= _LOG_EVERY:
            self._logged_at = now
            logger.info(
                "Mooncake frontier checkpoints: %s", dict(sorted(self.counts.items()))
            )
        return result

    def _note(self, request: Request, path: str) -> None:
        if self.record_paths:
            self._path_of[request.request_id] = path

    def _lookup(self, request: Request, local: int) -> MooncakeLookupResult:
        """Runs on the client's thread, which owns its socket. A request
        waiting for space is looked up at every step: within ``memo_s`` at
        the same local hit, one whose probe found no usable frontier runs the
        regular lookup alone, and one whose frontier verified reuses that hit
        for up to ``reuse_s``, then repeats only the verify, which
        renews the leases."""
        if self.memo_s > 0:
            with self._memo_lock:
                memo = self._memo.get(request.request_id)
            now = time.monotonic()
            if memo is not None and memo[0] == local and now - memo[1] < self.memo_s:
                _, probed, frontier, hit, verified_at = memo
                if frontier is None:
                    self.counts["memo_no_frontier"] += 1
                    self._note(request, "memo:no_frontier")
                    return self.client.lookup_now(
                        request.num_tokens, request.block_hashes
                    )
                if now - verified_at < self.reuse_s:
                    self.counts["memo_hit"] += 1
                    self._note(request, "memo:hit")
                    return hit
                self.counts["memo_verify"] += 1
                self._note(request, "memo:verify")
                length, key = frontier
                verified = self.client.frontier_verify(
                    request.num_tokens,
                    request.block_hashes,
                    length,
                    local // self.block_size,
                    fallback=True,
                    key=key,
                )
                assert verified is not None
                ok = verified.frontier is not None
                self._remember(
                    request.request_id,
                    local,
                    frontier if ok else None,
                    verified,
                    probed,
                )
                return verified
            result = self._probe(request, local)
            found = (
                None
                if result.frontier is None
                else (result.hit_length, bytes(result.frontier))
            )
            self._remember(request.request_id, local, found, result)
            return result
        return self._probe(request, local)

    def _probe(self, request: Request, local: int) -> MooncakeLookupResult:
        unit = self.hash_block_size
        num_tokens = request.num_tokens
        hashes = request.block_hashes
        # A frontier leaves at least one token to compute.
        max_length = num_tokens - 1
        if max_length < FRONTIER_LOCATOR_TOKENS:
            self._note(request, "short")
            return self.client.lookup_now(num_tokens, hashes)
        last = frontier_locator_unit(max_length, unit)
        first = max(0, frontier_locator_unit(local + 1, unit), last + 1 - _PROBE_UNITS)
        locators = [
            bytes(frontier_locator_key(self.hash_fn, request, h, unit))
            for h in range(last, first - 1, -1)
        ]
        if not locators:
            self._note(request, "short")
            return self.client.lookup_now(num_tokens, hashes)
        self.counts["probes"] += 1
        probe = self.client.frontier_probe(num_tokens, hashes, locators)
        regular = probe.regular
        if not probe.entries:
            assert regular is not None
            self._note(request, "probe:no_entries")
            return regular
        self.counts["entries"] += len(probe.entries)
        best: tuple[int, bytes] | None = None
        for length, key in sorted(set(probe.entries), reverse=True):
            if regular is not None and length <= regular.hit_length:
                break
            if not local < length <= max_length or length % self.block_size == 0:
                continue
            if bytes(frontier_key(self.hash_fn, request, length, unit)) == key:
                best = (length, key)
                break
        if best is None:
            self.counts["no_match"] += 1
            self._note(request, "probe:no_match")
            if regular is not None:
                return regular
            return self.client.lookup_now(num_tokens, hashes)
        length, key = best
        verified = self.client.frontier_verify(
            num_tokens,
            hashes,
            length,
            local // self.block_size,
            fallback=regular is None,
            key=key,
        )
        if verified is None or verified.frontier is None:
            self.counts["incomplete"] += 1
            self._note(request, "probe:incomplete")
            if verified is not None:
                return verified
            # Without a regular lookup the verify falls back to one.
            assert regular is not None
            return regular
        self.counts["hits"] += 1
        self.counts["hit_tokens"] += length - local
        self._note(request, "probe:hit")
        return verified
