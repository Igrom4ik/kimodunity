from __future__ import annotations

import socket
import threading
import time
import unittest

import numpy as np

from ardy_bridge.messages import ChunkMessage, HelloMessage
from ardy_bridge.protocol import MessageType, encode_frame, recv_frame, send_frame
from ardy_bridge.server import BridgeServer


class BridgeServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.received = []
        self.message_ready = threading.Event()

        def on_message(client, frame) -> None:
            self.received.append(frame)
            self.message_ready.set()

        self.server = BridgeServer(port=0, on_message=on_message)
        self.address = self.server.start()

    def tearDown(self) -> None:
        self.server.stop()

    def connect(self) -> socket.socket:
        peer = socket.create_connection(self.address, timeout=2.0)
        peer.settimeout(2.0)
        return peer

    def test_bidirectional_frame_flow_and_bye(self) -> None:
        with self.connect() as peer:
            self.assertTrue(self.server.wait_for_client(2.0))
            self.assertTrue(self.server.send(MessageType.STATUS, {"state": "ready"}))
            status = recv_frame(peer)
            self.assertEqual(MessageType.STATUS, status.msg_type)
            self.assertEqual("ready", status.header["state"])

            send_frame(peer, MessageType.TRANSPORT, {"action": "play"})
            self.assertTrue(self.message_ready.wait(2.0))
            self.assertEqual(MessageType.TRANSPORT, self.received[0].msg_type)
            self.assertEqual("play", self.received[0].header["action"])
            send_frame(peer, MessageType.BYE, {})

        deadline = time.monotonic() + 2.0
        while self.server.has_client and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(self.server.has_client)

    def test_second_client_is_rejected(self) -> None:
        with self.connect() as first:
            self.assertTrue(self.server.wait_for_client(2.0))
            with self.connect() as second:
                error = recv_frame(second)
                self.assertEqual(MessageType.ERROR, error.msg_type)
                self.assertEqual("client_already_connected", error.header["code"])
            send_frame(first, MessageType.BYE, {})

    def test_second_listener_cannot_reuse_same_port(self) -> None:
        duplicate = BridgeServer(host=self.address[0], port=self.address[1])
        try:
            with self.assertRaises(OSError):
                duplicate.start()
        finally:
            duplicate.stop()

    def test_fragmented_tcp_frame_is_reassembled(self) -> None:
        with self.connect() as peer:
            self.assertTrue(self.server.wait_for_client(2.0))
            encoded = encode_frame(MessageType.SET_PROMPT, {"text": "walk", "fromFrame": 4})
            for value in encoded:
                peer.sendall(bytes((value,)))
            self.assertTrue(self.message_ready.wait(2.0))
            self.assertEqual(MessageType.SET_PROMPT, self.received[0].msg_type)
            self.assertEqual("walk", self.received[0].header["text"])
            send_frame(peer, MessageType.BYE, {})

    def test_typed_hello_and_chunk_stream(self) -> None:
        hello = HelloMessage.create(
            model="core40",
            skeleton="test",
            fps=20,
            joint_names=("Hips", "LeftFoot", "LeftToe", "RightFoot", "RightToe"),
            parents=(-1, 0, 1, 0, 3),
            root_idx=0,
            foot_contact_joints=(1, 2, 3, 4),
            gen_horizon=40,
            num_frames_per_token=4,
            neutral_joints=np.zeros((5, 3), dtype=np.float32),
        )
        quaternions = np.zeros((2, 5, 4), dtype=np.float32)
        quaternions[..., 3] = 1.0
        chunk = ChunkMessage.create(
            start_frame=0,
            revision=0,
            root_positions=np.zeros((2, 3), dtype=np.float32),
            local_quaternions=quaternions,
            contacts=np.zeros((2, 4), dtype=np.float32),
        )

        with self.connect() as peer:
            self.assertTrue(self.server.wait_for_client(2.0))
            self.assertTrue(self.server.send_frame(hello.to_frame()))
            self.assertTrue(self.server.send_frame(chunk.to_frame()))
            received_hello = HelloMessage.from_frame(recv_frame(peer))
            received_chunk = ChunkMessage.from_frame(recv_frame(peer), joint_count=5)
            self.assertEqual("core40", received_hello.model)
            self.assertEqual(2, received_chunk.count)
            send_frame(peer, MessageType.BYE, {})


if __name__ == "__main__":
    unittest.main()
