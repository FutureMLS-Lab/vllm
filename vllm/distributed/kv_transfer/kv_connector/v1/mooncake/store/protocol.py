# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire-format constants for the LookupKey ZMQ admin channel.

This is the single source of truth shared by ``LookupKeyClient`` and
``LookupKeyServer`` on the scheduler<->worker rank-0 admin channel.

Wire format (REQ/REP over IPC):

    Request: [msg_type: bytes] [payload_frames...]

      msg_type == LOOKUP_MSG:
          frame 1: num_tokens (u32 big-endian, 4 bytes); the worker derives
                   the aligned lookup length
          frame 2: hash_len (u16 big-endian, 2 bytes) — byte length of each
                   fixed-size block hash (0 when there are no hashes)
          frame 3: raw block hashes concatenated back-to-back (each hash_len
                   bytes); the server splits on hash_len
        Response: hit_length (u32 big-endian, first 4 bytes), followed by zero
                  or more 8-byte tail-key boundaries. Each entry is group_id
                  (u32), then boundary_tokens (u32).

      msg_type == RESET_MSG:
          (no payload frames)
        Response: [RESP_OK] or [RESP_ERR]

      msg_type == FRONTIER_PROBE_MSG (``frontier_write_through``):
          frames 1-3: as LOOKUP_MSG
          frame 4: locator key length (u16)
          frame 5: the locator keys concatenated
        Response: b"R" and a LOOKUP_MSG response when no locator exists, else
                  b"C", then a LOOKUP_MSG response's length (u32, 0 for
                  none) and that response, then the locator entries, each
                  frontier length (u32), key length (u8) and key.

      msg_type == FRONTIER_VERIFY_MSG:
          frames 1-3: as LOOKUP_MSG
          frame 4: frontier length (u32), first block to load (u32), whether
                   to fall back to the regular lookup (u8), then the frontier
                   key
        Response: b"T" when every object of the frontier's load exists, else
                  b"R" and a LOOKUP_MSG response (with the fallback) or b"M".

The first frame of every request is a named bytes tag (not a numeric
sentinel that aliases the data field) so the protocol stays
self-describing and extensible: adding new admin commands requires
only a new tag and a new dispatch branch.

Mirrors the named-tag convention used by the NIXL connector (see
``vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py``).
"""

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    FrontierProbe,
    MooncakeLookupResult,
    TailKeyBoundary,
)

# Request message-type tags. Frame 0 of every request.
LOOKUP_MSG: bytes = b"lookup"
RESET_MSG: bytes = b"reset"
FRONTIER_PROBE_MSG: bytes = b"frontier_probe"
FRONTIER_VERIFY_MSG: bytes = b"frontier_verify"

# Single-byte response status codes for admin commands.
RESP_OK: bytes = b"\x01"
RESP_ERR: bytes = b"\x00"

# group_id (u32), boundary_tokens (u32).
TAIL_KEY_BOUNDARY_ENTRY_SIZE: int = 8


def encode_lookup_response(result: MooncakeLookupResult) -> bytes:
    hit_length = result.hit_length.to_bytes(4, "big")
    if not result.tail_key_boundaries:
        return hit_length

    payload = bytearray(hit_length)
    for boundary in result.tail_key_boundaries:
        payload.extend(boundary.group_id.to_bytes(4, "big"))
        payload.extend(boundary.num_tokens.to_bytes(4, "big"))
    return bytes(payload)


def decode_lookup_response(payload: bytes) -> MooncakeLookupResult:
    if len(payload) < 4 or (len(payload) - 4) % TAIL_KEY_BOUNDARY_ENTRY_SIZE:
        raise ValueError("Invalid Mooncake lookup response")

    hit_length = int.from_bytes(payload[:4], "big")
    if len(payload) == 4:
        return MooncakeLookupResult(hit_length)

    boundaries = []
    for offset in range(4, len(payload), TAIL_KEY_BOUNDARY_ENTRY_SIZE):
        boundaries.append(
            TailKeyBoundary(
                group_id=int.from_bytes(payload[offset : offset + 4], "big"),
                num_tokens=int.from_bytes(payload[offset + 4 : offset + 8], "big"),
            )
        )
    return MooncakeLookupResult(hit_length, tuple(boundaries))


def encode_frontier_probe(probe: FrontierProbe) -> bytes:
    if not probe.entries:
        assert probe.regular is not None
        return b"R" + encode_lookup_response(probe.regular)
    regular = b"" if probe.regular is None else encode_lookup_response(probe.regular)
    payload = bytearray(b"C")
    payload.extend(len(regular).to_bytes(4, "big"))
    payload.extend(regular)
    for length, key in probe.entries:
        payload.extend(length.to_bytes(4, "big"))
        payload.extend(len(key).to_bytes(1, "big"))
        payload.extend(key)
    return bytes(payload)


def decode_frontier_locators(size_frame: bytes, blob: bytes) -> list[bytes]:
    """FRONTIER_PROBE_MSG's frames 4 and 5: the locator keys."""
    size = int.from_bytes(size_frame, "big")
    if len(size_frame) != 2 or not size or len(blob) % size:
        raise ValueError("Invalid Mooncake frontier probe request")
    return [blob[i : i + size] for i in range(0, len(blob), size)]


def decode_frontier_probe(payload: bytes) -> FrontierProbe:
    if payload[:1] == b"R":
        return FrontierProbe((), decode_lookup_response(payload[1:]))
    if payload[:1] != b"C" or len(payload) < 5:
        raise ValueError("Invalid Mooncake frontier probe response")
    size = int.from_bytes(payload[1:5], "big")
    if 5 + size > len(payload):
        raise ValueError("Invalid Mooncake frontier probe response")
    regular = decode_lookup_response(payload[5 : 5 + size]) if size else None
    entries = []
    offset = 5 + size
    while offset < len(payload):
        if offset + 5 > len(payload):
            raise ValueError("Invalid Mooncake frontier probe response")
        length = int.from_bytes(payload[offset : offset + 4], "big")
        key_len = payload[offset + 4]
        key = bytes(payload[offset + 5 : offset + 5 + key_len])
        if len(key) != key_len:
            raise ValueError("Invalid Mooncake frontier probe response")
        entries.append((length, key))
        offset += 5 + key_len
    return FrontierProbe(tuple(entries), regular)


def encode_frontier_verify(
    length: int, first_block: int, fallback: bool, key: bytes
) -> bytes:
    return (
        length.to_bytes(4, "big")
        + first_block.to_bytes(4, "big")
        + bytes([fallback])
        + key
    )


def decode_frontier_verify(payload: bytes) -> tuple[int, int, bool, bytes]:
    if len(payload) < 9:
        raise ValueError("Invalid Mooncake frontier verify request")
    return (
        int.from_bytes(payload[:4], "big"),
        int.from_bytes(payload[4:8], "big"),
        bool(payload[8]),
        bytes(payload[9:]),
    )


def encode_frontier_verified(result: MooncakeLookupResult | None) -> bytes:
    """``result``: the frontier's hit (its ``frontier`` set), the regular
    lookup that replaced it, or None for a miss without fallback."""
    if result is None:
        return b"M"
    if result.frontier is not None:
        return b"T"
    return b"R" + encode_lookup_response(result)


def decode_frontier_verified(
    payload: bytes, length: int, key: bytes
) -> MooncakeLookupResult | None:
    if payload == b"T":
        return MooncakeLookupResult(length, frontier=key)
    if payload == b"M":
        return None
    if payload[:1] != b"R":
        raise ValueError("Invalid Mooncake frontier verify response")
    return decode_lookup_response(payload[1:])
