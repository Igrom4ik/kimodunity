"""ARDY-to-Unity bridge protocol and local TCP transport."""

from .adapters import chunk_from_ardy_output, hello_from_model
from .messages import ChunkMessage, HelloMessage
from .protocol import (
    MAGIC,
    PROTOCOL_VERSION,
    ConnectionClosed,
    Frame,
    MessageType,
    ProtocolError,
    decode_frame,
    encode_frame,
    recv_frame,
    send_frame,
)
from .server import BridgeServer, ClientConnection

__all__ = [
    "MAGIC",
    "PROTOCOL_VERSION",
    "BridgeServer",
    "ChunkMessage",
    "ClientConnection",
    "ConnectionClosed",
    "Frame",
    "HelloMessage",
    "MessageType",
    "ProtocolError",
    "decode_frame",
    "encode_frame",
    "recv_frame",
    "send_frame",
    "chunk_from_ardy_output",
    "hello_from_model",
]
