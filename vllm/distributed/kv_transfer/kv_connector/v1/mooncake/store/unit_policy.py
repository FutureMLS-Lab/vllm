# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recovery units in the Mooncake store (``unit_policy_url``).

A Mooncake master that runs the recovery-unit eviction policy
(``--eviction_policy=unit_gds``) evicts the store's objects by unit. A unit is
a resume point (a Mamba state at a hash-unit boundary, a prompt's partial
tail, or a frontier checkpoint) with the objects it owns (its states, its
partial or frontier page, its indexer rings) and the full pages it needs, which
units share and the master reference-counts. The master ranks units by
GreedyDual-Size over the prefill each saves, the bytes that go with it and the
clock of its last use, dead units first.

This module is the client side. The rank that writes frontier locators
registers every unit a store job wrote (``StoreUnits.upsert``, with every
rank's keys), touches the units requests resume at, and marks superseded units
dead, through the master's admin port (``UnitPolicyClient``). Each rank puts a
unit's own objects under the unit's id as their Mooncake group, so the master
evicts them together.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import queue
import threading
import time
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    PoolKey,
    StoreUnit,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.frontier import (  # noqa: E501
        FrontierStore,
    )
    from vllm.v1.core.hybrid_checkpoint.cost import PrefillCostModel

logger = init_logger(__name__)

# The namespace of unit ids in the store.
UNIT_NAMESPACE = "unit"
_UNITS_PATH = "/api/v1/units"
_STATS_PATH = "/api/v1/units/stats"
# Updates one request to the master carries at most.
_BATCH = 256
# Updates the client holds for the master at most; more are dropped and
# counted.
_QUEUE_MAX = 65536
# Seconds between the client's count log lines.
_LOG_EVERY = 60.0
_TIMEOUT_S = 5.0


def unit_policy_refusal(
    supports_group_ids: bool, enable_group_semantics: bool
) -> str | None:
    """Why ``unit_policy_url`` cannot be used with this connector's Mooncake
    client, or None."""
    if not supports_group_ids:
        return "unit_policy_url needs a Mooncake package with ReplicateConfig.group_ids"
    if enable_group_semantics:
        return (
            "unit_policy_url excludes enable_group_semantics: it groups each "
            "unit's objects itself"
        )
    return None


class StoreUnits:
    """The master's unit updates for the units a job wrote, with every rank's
    keys, as ``FrontierStore`` names them."""

    def __init__(self, frontiers: FrontierStore, cost: PrefillCostModel) -> None:
        self.frontiers = frontiers
        self.cost = cost

    def unit_id(self, key: bytes) -> str:
        """The unit's id, also its objects' Mooncake group id."""
        return f"{self.frontiers.namespace}@{UNIT_NAMESPACE}@{key.hex()}"

    def upsert(
        self,
        unit: StoreUnit,
        block_hashes: Sequence[BlockHash],
        locator: bytes | None = None,
    ) -> dict[str, Any]:
        """Members: a state's Mamba states (and, inside a block, the partial
        page keyed at it), or a frontier's page, states and rings. Refs: the
        full pages before it and a frontier's locator record."""
        frontiers = self.frontiers
        key_hex = unit.key.hex()
        inside_block = unit.position % frontiers.block_size != 0
        members: list[str] = []
        for group_id, prefixes in enumerate(frontiers.lookup_prefixes):
            if unit.frontier or inside_block or group_id in frontiers.mamba_group_ids:
                members.extend(
                    PoolKey.build_key_string(prefix, key_hex) for prefix in prefixes
                )
        if unit.frontier and unit.position % frontiers.unit:
            for prefixes in frontiers.ring_prefixes.values():
                members.extend(
                    PoolKey.build_key_string(prefix, key_hex) for prefix in prefixes
                )
        refs = frontiers.page_keys(block_hashes, unit.position // frontiers.block_size)
        if unit.frontier and locator is not None:
            refs.append(frontiers.locator_key(locator))
        return {
            "op": "upsert",
            "id": self.unit_id(unit.key),
            "value": self.cost.seconds(0, unit.position),
            "aligned": not inside_block,
            "parent": "" if unit.parent is None else self.unit_id(unit.parent),
            "members": members,
            "refs": refs,
        }

    def touch(self, key: bytes) -> dict[str, Any]:
        return {"op": "touch", "id": self.unit_id(key)}

    def dead(self, key: bytes) -> dict[str, Any]:
        return {"op": "dead", "id": self.unit_id(key)}


class UnitPolicyClient:
    """Sends unit updates to the master's admin port from one thread, in
    order and in batches. A failed request or a malformed reply is counted
    and logged, not raised, and so is an update dropped because ``_QUEUE_MAX``
    are waiting: objects of a unit the master never learns about are evicted
    as orphans after its grace period, and a lost touch or dead mark only
    changes the eviction order."""

    def __init__(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme != "http" or not parts.hostname or parts.port is None:
            raise ValueError(
                f"unit_policy_url must be http://<host>:<port>, got {url!r}"
            )
        self.host = parts.hostname
        self.port = parts.port
        self.counts: Counter[str] = Counter()
        self._counts_lock = threading.Lock()
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(_QUEUE_MAX)
        self._conn: http.client.HTTPConnection | None = None
        self._logged_at = time.monotonic()
        stats = self._request("GET", _STATS_PATH, None)
        if stats is None:
            raise RuntimeError(
                f"the Mooncake master at {url} does not run the recovery-unit "
                "eviction policy (--eviction_policy=unit_gds)"
            )
        logger.info(
            "Mooncake unit policy at %s: %d units, %d objects",
            url,
            stats.get("units", 0),
            stats.get("objects", 0),
        )
        self._thread = threading.Thread(
            target=self._run, name="MooncakeUnitPolicy", daemon=True
        )
        self._thread.start()

    def send(self, updates: Sequence[dict[str, Any]]) -> None:
        dropped = 0
        for update in updates:
            try:
                self._queue.put_nowait(update)
            except queue.Full:
                dropped += 1
        if dropped:
            self._count("dropped_updates", dropped)
            logger.warning_once(
                "Mooncake unit updates are dropped: the master's admin port "
                "takes them slower than they come."
            )

    def close(self) -> None:
        with contextlib.suppress(queue.Full):
            self._queue.put(None, timeout=_TIMEOUT_S)
        self._thread.join(timeout=_TIMEOUT_S)
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _count(self, name: str, n: int = 1) -> None:
        with self._counts_lock:
            self.counts[name] += n

    def _run(self) -> None:
        while True:
            update = self._queue.get()
            if update is None:
                return
            batch = [update]
            closing = False
            while len(batch) < _BATCH:
                try:
                    update = self._queue.get_nowait()
                except queue.Empty:
                    break
                if update is None:
                    closing = True
                    break
                batch.append(update)
            try:
                self._post(batch)
            except Exception:
                logger.exception("Mooncake unit updates failed")
                self._count("failed_requests")
                self._count("failed_updates", len(batch))
            if closing:
                return

    def _post(self, batch: list[dict[str, Any]]) -> None:
        reply = self._request("POST", _UNITS_PATH, {"updates": batch})
        errors = None if reply is None else reply.get("errors", [])
        with self._counts_lock:
            for update in batch:
                self.counts[update["op"]] += 1
            if not isinstance(errors, list):
                self.counts["failed_requests"] += 1
                self.counts["failed_updates"] += len(batch)
            elif errors:
                self.counts["rejected"] += len(errors)
            counts = dict(sorted(self.counts.items()))
        if isinstance(errors, list) and errors:
            logger.warning("Mooncake unit updates rejected: %s", errors[:3])
        elif reply is not None and not isinstance(errors, list):
            logger.error("Mooncake unit policy reply without an error list: %r", reply)
        now = time.monotonic()
        if now - self._logged_at >= _LOG_EVERY:
            self._logged_at = now
            logger.info("Mooncake unit updates: %s", counts)

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        body = None if payload is None else json.dumps(payload).encode()
        headers = {} if body is None else {"Content-Type": "application/json"}
        for attempt in range(2):
            if self._conn is None:
                self._conn = http.client.HTTPConnection(
                    self.host, self.port, timeout=_TIMEOUT_S
                )
            try:
                self._conn.request(method, path, body=body, headers=headers)
                response = self._conn.getresponse()
                data = response.read()
            except (OSError, http.client.HTTPException) as e:
                # A dropped keep-alive connection: reconnect once.
                self._conn.close()
                self._conn = None
                if attempt:
                    logger.error("Mooncake unit policy request failed: %s", e)
                continue
            if response.status != 200:
                logger.error(
                    "Mooncake unit policy %s %s: HTTP %d %s",
                    method,
                    path,
                    response.status,
                    data[:200],
                )
                return None
            try:
                reply = json.loads(data)
            except ValueError:
                reply = None
            if not isinstance(reply, dict):
                logger.error(
                    "Mooncake unit policy %s %s: not a JSON object: %r",
                    method,
                    path,
                    data[:200],
                )
                return None
            return reply
        return None
