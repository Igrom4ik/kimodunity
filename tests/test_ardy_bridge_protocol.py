from __future__ import annotations

import struct
import unittest
from pathlib import Path

import numpy as np

from ardy_bridge.adapters import chunk_from_ardy_output, hello_from_model
from ardy_bridge.messages import ChunkMessage, HelloMessage, encode_invalidate
from ardy_bridge.protocol import (
    HEADER_SIZE,
    MAGIC,
    MAX_BLOB_BYTES,
    PROTOCOL_VERSION,
    MessageType,
    ProtocolError,
    decode_frame,
    encode_frame,
)
from tools.bridge_smoke_server import build_messages


class ProtocolFrameTests(unittest.TestCase):
    def test_header_is_exact_little_endian_layout(self) -> None:
        encoded = encode_frame(MessageType.STATUS, {"state": "ready"}, b"abc")
        magic, version, msg_type, json_len, blob_len = struct.unpack("<IHHII", encoded[:HEADER_SIZE])
        self.assertEqual(MAGIC, magic)
        self.assertEqual(PROTOCOL_VERSION, version)
        self.assertEqual(MessageType.STATUS, msg_type)
        self.assertEqual(b"ARDY", encoded[:4])
        self.assertEqual(3, blob_len)
        self.assertEqual(len(encoded), HEADER_SIZE + json_len + blob_len)

    def test_unicode_json_and_blob_round_trip(self) -> None:
        encoded = encode_frame(MessageType.ERROR, {"message": "ошибка"}, b"\x00\x01")
        frame = decode_frame(encoded)
        self.assertEqual(MessageType.ERROR, frame.msg_type)
        self.assertEqual("ошибка", frame.header["message"])
        self.assertEqual(b"\x00\x01", frame.blob)

    def test_decode_rejects_trailing_bytes_and_wrong_version(self) -> None:
        encoded = encode_frame(MessageType.STATUS, {})
        with self.assertRaises(ProtocolError):
            decode_frame(encoded + b"extra")
        wrong_version = bytearray(encoded)
        wrong_version[4:6] = struct.pack("<H", 2)
        with self.assertRaises(ProtocolError):
            decode_frame(wrong_version)

    def test_decode_rejects_oversized_blob_before_allocation(self) -> None:
        prefix = struct.pack(
            "<IHHII",
            MAGIC,
            PROTOCOL_VERSION,
            MessageType.CHUNK,
            2,
            MAX_BLOB_BYTES + 1,
        )
        with self.assertRaises(ProtocolError):
            decode_frame(prefix + b"{}")


class TypedMessageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.joint_names = ("Hips", "LeftFoot", "RightFoot")
        self.parents = (-1, 0, 0)
        self.neutral = np.array(
            [[0.0, 0.0, 0.0], [0.1, -0.9, 0.0], [-0.1, -0.9, 0.0]],
            dtype=np.float32,
        )

    def test_hello_round_trip(self) -> None:
        hello = HelloMessage.create(
            model="ARDY-Core-RP-20FPS-Horizon40",
            skeleton="cskel27",
            fps=20,
            joint_names=self.joint_names,
            parents=self.parents,
            root_idx=0,
            foot_contact_joints=(1, 2),
            gen_horizon=40,
            num_frames_per_token=4,
            neutral_joints=self.neutral,
        )
        restored = HelloMessage.from_frame(decode_frame(hello.to_bytes()))
        self.assertEqual(hello.model, restored.model)
        self.assertEqual(hello.joint_names, restored.joint_names)
        self.assertEqual(hello.parents, restored.parents)
        np.testing.assert_array_equal(hello.neutral_joints, restored.neutral_joints)

    def test_core40_chunk_round_trip_and_wire_size(self) -> None:
        frame_count, joint_count = 40, 27
        roots = np.arange(frame_count * 3, dtype=np.float32).reshape(frame_count, 3) / 100.0
        quaternions = np.zeros((frame_count, joint_count, 4), dtype=np.float32)
        quaternions[..., 3] = 1.0
        contacts = np.zeros((frame_count, 4), dtype=np.float32)
        contacts[::2, (0, 2)] = 1.0
        chunk = ChunkMessage.create(
            start_frame=80,
            revision=3,
            root_positions=roots,
            local_quaternions=quaternions,
            contacts=contacts,
        )
        frame = decode_frame(chunk.to_bytes())
        self.assertEqual(40 * (3 + 27 * 4 + 4) * 4, len(frame.blob))
        restored = ChunkMessage.from_frame(frame, joint_count=joint_count)
        self.assertEqual(80, restored.start_frame)
        self.assertEqual(3, restored.revision)
        np.testing.assert_array_equal(roots, restored.root_positions)
        np.testing.assert_array_equal(quaternions, restored.local_quaternions)
        np.testing.assert_array_equal(contacts, restored.contacts)

    def test_chunk_without_contacts_round_trip(self) -> None:
        roots = np.zeros((2, 3), dtype=np.float32)
        quaternions = np.zeros((2, 3, 4), dtype=np.float32)
        quaternions[..., 3] = 1.0
        chunk = ChunkMessage.create(
            start_frame=0,
            revision=0,
            root_positions=roots,
            local_quaternions=quaternions,
        )
        restored = ChunkMessage.from_frame(decode_frame(chunk.to_bytes()), joint_count=3)
        self.assertIsNone(restored.contacts)

    def test_invalidate_uses_absolute_frame(self) -> None:
        frame = decode_frame(encode_invalidate(123))
        self.assertEqual(MessageType.INVALIDATE, frame.msg_type)
        self.assertEqual({"fromFrame": 123}, frame.header)

    def test_chunk_rejects_non_boolean_has_contacts(self) -> None:
        frame = decode_frame(
            encode_frame(
                MessageType.CHUNK,
                {"startFrame": 0, "count": 1, "revision": 0, "hasContacts": "false"},
                np.zeros(15, dtype=np.float32).tobytes(),
            )
        )
        with self.assertRaises(ProtocolError):
            ChunkMessage.from_frame(frame, joint_count=3)


class AdapterTests(unittest.TestCase):
    def test_golden_smoke_messages_have_core40_layout(self) -> None:
        golden = Path(__file__).resolve().parents[1] / "tools" / "golden_transform.json"
        hello, chunk = build_messages(golden, 40)
        self.assertEqual(27, len(hello.joint_names))
        self.assertEqual((25, 26, 21, 22), hello.foot_contact_joints)
        self.assertEqual(40, chunk.count)
        self.assertEqual(18400, len(chunk.to_frame().blob))

    def test_model_metadata_becomes_hello(self) -> None:
        class Skeleton:
            bone_order_names = ["Hips", "LeftFoot", "LeftToe", "RightFoot", "RightToe"]
            joint_parents = np.array([-1, 0, 1, 0, 3])
            root_idx = 0
            left_foot_joint_indices = [1, 2]
            right_foot_joint_indices = [3, 4]
            neutral_joints = np.zeros((5, 3), dtype=np.float32)

        class MotionRep:
            fps = 20

        class Model:
            skeleton = Skeleton()
            motion_rep = MotionRep()
            gen_horizon_len = 40
            num_frames_per_token = 4

        hello = hello_from_model(Model(), model_name="core40", skeleton_name="cskel27")
        self.assertEqual((1, 2, 3, 4), hello.foot_contact_joints)
        self.assertEqual(40, hello.gen_horizon)
        self.assertEqual("cskel27", hello.skeleton)

    def test_real_skeleton_name_metadata_becomes_contact_indices(self) -> None:
        class Skeleton:
            bone_order_names = ["Hips", "LeftFoot", "LeftToe", "RightFoot", "RightToe"]
            joint_parents = np.array([-1, 0, 1, 0, 3])
            root_idx = 0
            left_foot_joint_names = ["LeftFoot", "LeftToe"]
            right_foot_joint_names = ["RightFoot", "RightToe"]
            neutral_joints = np.zeros((5, 3), dtype=np.float32)

        class MotionRep:
            fps = 20

        class Model:
            skeleton = Skeleton()
            motion_rep = MotionRep()
            gen_horizon_len = 40
            num_frames_per_token = 4

        hello = hello_from_model(Model(), model_name="core40")
        self.assertEqual((1, 2, 3, 4), hello.foot_contact_joints)

    def test_ardy_output_matrices_become_xyzw_chunk(self) -> None:
        rotations = np.broadcast_to(np.eye(3), (1, 3, 2, 3, 3)).copy()
        rotations[0, 2, 0] = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        roots = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
        contacts = np.array([[[0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0]]], dtype=bool)
        chunk = chunk_from_ardy_output(
            {
                "local_rot_mats": rotations,
                "root_positions": roots,
                "foot_contacts": contacts,
            },
            start_frame=20,
            revision=2,
            history_frames=1,
        )
        self.assertEqual(2, chunk.count)
        np.testing.assert_array_equal(roots[0, 1:], chunk.root_positions)
        np.testing.assert_allclose(chunk.local_quaternions[0], [[0, 0, 0, 1]] * 2)
        np.testing.assert_allclose(
            chunk.local_quaternions[1, 0],
            [0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)],
            atol=1e-6,
        )
        np.testing.assert_array_equal(contacts[0, 1:].astype(np.float32), chunk.contacts)

    def test_ardy_output_can_be_limited_to_twenty_frames(self) -> None:
        rotations = np.broadcast_to(np.eye(3), (1, 40, 2, 3, 3)).copy()
        roots = np.arange(120, dtype=np.float32).reshape(1, 40, 3)
        chunk = chunk_from_ardy_output(
            {"local_rot_mats": rotations, "root_positions": roots},
            start_frame=0,
            revision=0,
            output_frames=20,
        )
        self.assertEqual(20, chunk.count)
        np.testing.assert_array_equal(roots[0, :20], chunk.root_positions)


if __name__ == "__main__":
    unittest.main()
