"""Connect to a running ARDY bridge and print received frame summaries."""

from __future__ import annotations

import argparse
import socket

from ardy_bridge.protocol import MessageType, recv_frame, send_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8801)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    with socket.create_connection((args.host, args.port), timeout=args.timeout) as peer:
        peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for _ in range(args.count):
            frame = recv_frame(peer)
            try:
                name = MessageType(frame.msg_type).name.lower()
            except ValueError:
                name = "unknown"
            print(
                f"type={frame.msg_type} ({name}) header={dict(frame.header)} "
                f"blobBytes={len(frame.blob)}"
            )
        send_frame(peer, MessageType.BYE, {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
