"""Binary framing for the versioned ARDY bridge protocol."""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping


MAGIC = 0x59445241  # bytes b"ARDY" when packed little-endian
PROTOCOL_VERSION = 1
HEADER_STRUCT = struct.Struct("<IHHII")
HEADER_SIZE = HEADER_STRUCT.size
MAX_JSON_BYTES = 1024 * 1024
MAX_BLOB_BYTES = 64 * 1024 * 1024


class MessageType(IntEnum):
    HELLO = 1
    CHUNK = 2
    PLAYHEAD = 3
    INVALIDATE = 4
    STATUS = 5
    ERROR = 6

    SET_PROMPT = 100
    TRANSPORT = 101
    SET_PARAMS = 102
    WAYPOINT = 103
    REQUEST_RANGE = 104
    BYE = 105


class ProtocolError(ValueError):
    """Raised when a frame violates the wire protocol."""


class ConnectionClosed(EOFError):
    """Raised when a peer closes while a frame is being received."""


@dataclass(frozen=True, slots=True)
class Frame:
    msg_type: int
    header: Mapping[str, Any]
    blob: bytes = b""
    version: int = PROTOCOL_VERSION

    def to_bytes(self) -> bytes:
        return encode_frame(
            self.msg_type,
            self.header,
            self.blob,
            version=self.version,
        )


def _validate_lengths(json_len: int, blob_len: int) -> None:
    if json_len > MAX_JSON_BYTES:
        raise ProtocolError(f"JSON header is too large: {json_len} bytes")
    if blob_len > MAX_BLOB_BYTES:
        raise ProtocolError(f"Binary blob is too large: {blob_len} bytes")


def _encode_json(header: Mapping[str, Any] | None) -> bytes:
    if header is None:
        header = {}
    if not isinstance(header, Mapping):
        raise TypeError("frame header must be a mapping")
    try:
        text = json.dumps(
            dict(header),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"frame header is not valid JSON: {exc}") from exc
    return text.encode("utf-8")


def _decode_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON header: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("JSON header must decode to an object")
    return value


def encode_frame(
    msg_type: int | MessageType,
    header: Mapping[str, Any] | None = None,
    blob: bytes | bytearray | memoryview = b"",
    *,
    version: int = PROTOCOL_VERSION,
) -> bytes:
    """Encode one complete length-prefixed frame."""

    msg_type_value = int(msg_type)
    if not 0 <= msg_type_value <= 0xFFFF:
        raise ProtocolError(f"message type is outside uint16: {msg_type_value}")
    if not 0 <= version <= 0xFFFF:
        raise ProtocolError(f"protocol version is outside uint16: {version}")

    json_payload = _encode_json(header)
    blob_payload = bytes(blob)
    _validate_lengths(len(json_payload), len(blob_payload))
    prefix = HEADER_STRUCT.pack(
        MAGIC,
        version,
        msg_type_value,
        len(json_payload),
        len(blob_payload),
    )
    return prefix + json_payload + blob_payload


def _decode_prefix(prefix: bytes) -> tuple[int, int, int, int]:
    if len(prefix) != HEADER_SIZE:
        raise ProtocolError(f"frame prefix must be {HEADER_SIZE} bytes")
    magic, version, msg_type, json_len, blob_len = HEADER_STRUCT.unpack(prefix)
    if magic != MAGIC:
        raise ProtocolError(f"invalid magic 0x{magic:08X}")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol version {version}; expected {PROTOCOL_VERSION}"
        )
    _validate_lengths(json_len, blob_len)
    return version, msg_type, json_len, blob_len


def decode_frame(payload: bytes | bytearray | memoryview) -> Frame:
    """Decode exactly one frame and reject truncated or trailing bytes."""

    data = bytes(payload)
    if len(data) < HEADER_SIZE:
        raise ProtocolError("truncated frame prefix")
    version, msg_type, json_len, blob_len = _decode_prefix(data[:HEADER_SIZE])
    expected = HEADER_SIZE + json_len + blob_len
    if len(data) != expected:
        raise ProtocolError(f"frame size mismatch: expected {expected}, got {len(data)}")
    json_end = HEADER_SIZE + json_len
    return Frame(
        msg_type=msg_type,
        header=_decode_json(data[HEADER_SIZE:json_end]),
        blob=data[json_end:],
        version=version,
    )


def _recv_exact(peer: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = peer.recv(remaining)
        if not chunk:
            received = size - remaining
            raise ConnectionClosed(f"peer closed after {received} of {size} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(peer: socket.socket) -> Frame:
    """Receive one frame from a stream socket, tolerating TCP fragmentation."""

    prefix = _recv_exact(peer, HEADER_SIZE)
    version, msg_type, json_len, blob_len = _decode_prefix(prefix)
    json_payload = _recv_exact(peer, json_len)
    blob = _recv_exact(peer, blob_len) if blob_len else b""
    return Frame(
        msg_type=msg_type,
        header=_decode_json(json_payload),
        blob=blob,
        version=version,
    )


def send_frame(
    peer: socket.socket,
    msg_type: int | MessageType,
    header: Mapping[str, Any] | None = None,
    blob: bytes | bytearray | memoryview = b"",
    *,
    version: int = PROTOCOL_VERSION,
) -> None:
    peer.sendall(encode_frame(msg_type, header, blob, version=version))
