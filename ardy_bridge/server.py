"""Single-client loopback TCP server for the ARDY bridge."""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

from .protocol import (
    ConnectionClosed,
    Frame,
    MessageType,
    ProtocolError,
    recv_frame,
    send_frame as send_wire_frame,
)


MessageCallback = Callable[["ClientConnection", Frame], None]
ConnectionCallback = Callable[["ClientConnection"], None]


@dataclass(slots=True)
class ClientConnection:
    socket: socket.socket
    address: tuple[str, int]
    _send_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def send(
        self,
        msg_type: int | MessageType,
        header: Mapping[str, Any] | None = None,
        blob: bytes | bytearray | memoryview = b"",
    ) -> None:
        with self._send_lock:
            send_wire_frame(self.socket, msg_type, header, blob)

    def send_frame(self, frame: Frame) -> None:
        with self._send_lock:
            self.socket.sendall(frame.to_bytes())

    def close(self) -> None:
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.socket.close()
        except OSError:
            pass


class BridgeServer:
    """Background TCP server accepting one active Unity client at a time."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8801,
        *,
        on_connect: ConnectionCallback | None = None,
        on_message: MessageCallback | None = None,
        on_disconnect: ConnectionCallback | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.on_connect = on_connect
        self.on_message = on_message
        self.on_disconnect = on_disconnect
        self._listener: socket.socket | None = None
        self._client: ClientConnection | None = None
        self._accept_thread: threading.Thread | None = None
        self._client_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._client_ready = threading.Event()

    @property
    def address(self) -> tuple[str, int]:
        listener = self._listener
        if listener is None:
            return self.host, self.port
        host, port = listener.getsockname()[:2]
        return str(host), int(port)

    @property
    def has_client(self) -> bool:
        with self._lock:
            return self._client is not None

    def wait_for_client(self, timeout: float | None = None) -> bool:
        return self._client_ready.wait(timeout)

    def start(self) -> tuple[str, int]:
        if self._accept_thread is not None and self._accept_thread.is_alive():
            return self.address
        self._stopping.clear()
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Windows allows multiple listeners on the same address when
            # SO_REUSEADDR is enabled. That can route Unity to a stale ARDY
            # process while the UI reports the newly started generator PID.
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            listener.listen(4)
            listener.settimeout(0.2)
        except BaseException:
            listener.close()
            raise
        self._listener = listener
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="ArdyBridgeAccept",
            daemon=True,
        )
        self._accept_thread.start()
        return self.address

    def stop(self) -> None:
        self._stopping.set()
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        with self._lock:
            client = self._client
        if client is not None:
            client.close()
        current = threading.current_thread()
        for thread in (self._client_thread, self._accept_thread):
            if thread is not None and thread is not current:
                thread.join(timeout=2.0)
        self._client_thread = None
        self._accept_thread = None
        self._client_ready.clear()

    def send(
        self,
        msg_type: int | MessageType,
        header: Mapping[str, Any] | None = None,
        blob: bytes | bytearray | memoryview = b"",
    ) -> bool:
        with self._lock:
            client = self._client
        if client is None:
            return False
        client.send(msg_type, header, blob)
        return True

    def send_frame(self, frame: Frame) -> bool:
        with self._lock:
            client = self._client
        if client is None:
            return False
        client.send_frame(frame)
        return True

    def _accept_loop(self) -> None:
        while not self._stopping.is_set():
            listener = self._listener
            if listener is None:
                return
            try:
                peer, address = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection = ClientConnection(peer, (str(address[0]), int(address[1])))
            with self._lock:
                occupied = self._client is not None
                if not occupied:
                    self._client = connection
                    self._client_ready.set()
            if occupied:
                try:
                    connection.send(
                        MessageType.ERROR,
                        {
                            "code": "client_already_connected",
                            "message": "ARDY bridge accepts one active client at a time",
                        },
                    )
                finally:
                    connection.close()
                continue
            self._client_thread = threading.Thread(
                target=self._client_loop,
                args=(connection,),
                name="ArdyBridgeClient",
                daemon=True,
            )
            self._client_thread.start()

    def _client_loop(self, client: ClientConnection) -> None:
        try:
            if self.on_connect is not None:
                self.on_connect(client)
            while not self._stopping.is_set():
                frame = recv_frame(client.socket)
                if self.on_message is not None:
                    self.on_message(client, frame)
                if frame.msg_type == MessageType.BYE:
                    break
        except ConnectionClosed:
            pass
        except ProtocolError as exc:
            try:
                client.send(MessageType.ERROR, {"code": "protocol_error", "message": str(exc)})
            except OSError:
                pass
        finally:
            client.close()
            with self._lock:
                if self._client is client:
                    self._client = None
                    self._client_ready.clear()
            if self.on_disconnect is not None:
                self.on_disconnect(client)

    def __enter__(self) -> "BridgeServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
