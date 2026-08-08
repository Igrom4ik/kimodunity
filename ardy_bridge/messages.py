"""Typed hello/chunk payloads for the ARDY bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import Frame, MessageType, PROTOCOL_VERSION, ProtocolError, encode_frame


FLOAT32_LE = np.dtype("<f4")


def _as_float32_le(values: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=FLOAT32_LE)
    if array.shape != shape:
        raise ProtocolError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ProtocolError(f"{label} contains NaN or infinity")
    return np.ascontiguousarray(array, dtype=FLOAT32_LE)


def encode_json_message(msg_type: int | MessageType, header: Mapping[str, Any]) -> bytes:
    return encode_frame(msg_type, header)


def encode_invalidate(from_frame: int) -> bytes:
    if from_frame < 0:
        raise ProtocolError("fromFrame must be non-negative")
    return encode_frame(MessageType.INVALIDATE, {"fromFrame": int(from_frame)})


@dataclass(frozen=True, slots=True)
class HelloMessage:
    model: str
    skeleton: str
    fps: int
    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    root_idx: int
    foot_contact_joints: tuple[int, ...]
    gen_horizon: int
    num_frames_per_token: int
    neutral_joints: np.ndarray
    protocol: int = PROTOCOL_VERSION

    @classmethod
    def create(
        cls,
        *,
        model: str,
        skeleton: str,
        fps: int,
        joint_names: Sequence[str],
        parents: Sequence[int],
        root_idx: int,
        foot_contact_joints: Sequence[int],
        gen_horizon: int,
        num_frames_per_token: int,
        neutral_joints: Any,
    ) -> "HelloMessage":
        names = tuple(str(name) for name in joint_names)
        parent_values = tuple(int(parent) for parent in parents)
        contacts = tuple(int(index) for index in foot_contact_joints)
        joints = _as_float32_le(neutral_joints, (len(names), 3), "neutral_joints")
        message = cls(
            model=str(model),
            skeleton=str(skeleton),
            fps=int(fps),
            joint_names=names,
            parents=parent_values,
            root_idx=int(root_idx),
            foot_contact_joints=contacts,
            gen_horizon=int(gen_horizon),
            num_frames_per_token=int(num_frames_per_token),
            neutral_joints=joints,
        )
        message.validate()
        return message

    def validate(self) -> None:
        joint_count = len(self.joint_names)
        if joint_count == 0:
            raise ProtocolError("hello must contain at least one joint")
        if len(self.parents) != joint_count:
            raise ProtocolError("jointNames and parents must have equal length")
        if self.neutral_joints.shape != (joint_count, 3):
            raise ProtocolError("neutral_joints shape does not match jointNames")
        if not 0 <= self.root_idx < joint_count:
            raise ProtocolError("rootIdx is outside the joint array")
        if self.parents[self.root_idx] not in (-1, self.root_idx):
            raise ProtocolError("root parent must be -1 or rootIdx")
        if any(parent >= joint_count or parent < -1 for parent in self.parents):
            raise ProtocolError("parents contains an invalid joint index")
        if any(index < 0 or index >= joint_count for index in self.foot_contact_joints):
            raise ProtocolError("footContactJoints contains an invalid joint index")
        if self.fps <= 0 or self.gen_horizon <= 0 or self.num_frames_per_token <= 0:
            raise ProtocolError("fps, genHorizon and numFramesPerToken must be positive")

    def to_bytes(self) -> bytes:
        return self.to_frame().to_bytes()

    def to_frame(self) -> Frame:
        self.validate()
        header = {
            "protocol": self.protocol,
            "model": self.model,
            "skeleton": self.skeleton,
            "fps": self.fps,
            "jointNames": list(self.joint_names),
            "parents": list(self.parents),
            "rootIdx": self.root_idx,
            "footContactJoints": list(self.foot_contact_joints),
            "genHorizon": self.gen_horizon,
            "numFramesPerToken": self.num_frames_per_token,
        }
        return Frame(
            msg_type=MessageType.HELLO,
            header=header,
            blob=self.neutral_joints.tobytes(order="C"),
        )

    @classmethod
    def from_frame(cls, frame: Frame) -> "HelloMessage":
        if frame.msg_type != MessageType.HELLO:
            raise ProtocolError(f"expected hello frame, got message type {frame.msg_type}")
        header = frame.header
        try:
            names = tuple(str(value) for value in header["jointNames"])
            expected_bytes = len(names) * 3 * FLOAT32_LE.itemsize
            if len(frame.blob) != expected_bytes:
                raise ProtocolError(
                    f"hello blob must be {expected_bytes} bytes, got {len(frame.blob)}"
                )
            joints = np.frombuffer(frame.blob, dtype=FLOAT32_LE).reshape(len(names), 3).copy()
            message = cls(
                protocol=int(header["protocol"]),
                model=str(header["model"]),
                skeleton=str(header["skeleton"]),
                fps=int(header["fps"]),
                joint_names=names,
                parents=tuple(int(value) for value in header["parents"]),
                root_idx=int(header["rootIdx"]),
                foot_contact_joints=tuple(int(value) for value in header["footContactJoints"]),
                gen_horizon=int(header["genHorizon"]),
                num_frames_per_token=int(header["numFramesPerToken"]),
                neutral_joints=joints,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"invalid hello header: {exc}") from exc
        if message.protocol != PROTOCOL_VERSION:
            raise ProtocolError(f"hello protocol must be {PROTOCOL_VERSION}")
        message.validate()
        return message


@dataclass(frozen=True, slots=True)
class ChunkMessage:
    start_frame: int
    revision: int
    root_positions: np.ndarray
    local_quaternions: np.ndarray
    contacts: np.ndarray | None = None

    @classmethod
    def create(
        cls,
        *,
        start_frame: int,
        revision: int,
        root_positions: Any,
        local_quaternions: Any,
        contacts: Any | None = None,
    ) -> "ChunkMessage":
        roots_source = np.asarray(root_positions)
        rotations_source = np.asarray(local_quaternions)
        if roots_source.ndim != 2:
            raise ProtocolError("root_positions must have shape [frames, 3]")
        if rotations_source.ndim != 3:
            raise ProtocolError("local_quaternions must have shape [frames, joints, 4]")
        frame_count = roots_source.shape[0]
        joint_count = rotations_source.shape[1]
        roots = _as_float32_le(roots_source, (frame_count, 3), "root_positions")
        rotations = _as_float32_le(
            rotations_source,
            (frame_count, joint_count, 4),
            "local_quaternions",
        )
        contact_array = None
        if contacts is not None:
            contact_array = _as_float32_le(contacts, (frame_count, 4), "contacts")
        message = cls(
            start_frame=int(start_frame),
            revision=int(revision),
            root_positions=roots,
            local_quaternions=rotations,
            contacts=contact_array,
        )
        message.validate()
        return message

    @property
    def count(self) -> int:
        return int(self.root_positions.shape[0])

    @property
    def joint_count(self) -> int:
        return int(self.local_quaternions.shape[1])

    def validate(self) -> None:
        if self.start_frame < 0 or self.revision < 0:
            raise ProtocolError("startFrame and revision must be non-negative")
        if self.count <= 0 or self.joint_count <= 0:
            raise ProtocolError("chunk must contain frames and joints")
        if self.root_positions.shape != (self.count, 3):
            raise ProtocolError("invalid root_positions shape")
        if self.local_quaternions.shape != (self.count, self.joint_count, 4):
            raise ProtocolError("invalid local_quaternions shape")
        if self.contacts is not None and self.contacts.shape != (self.count, 4):
            raise ProtocolError("invalid contacts shape")

    def to_bytes(self) -> bytes:
        return self.to_frame().to_bytes()

    def to_frame(self) -> Frame:
        self.validate()
        fields = [
            self.root_positions.reshape(self.count, 3),
            self.local_quaternions.reshape(self.count, self.joint_count * 4),
        ]
        if self.contacts is not None:
            fields.append(self.contacts.reshape(self.count, 4))
        blob = np.ascontiguousarray(np.concatenate(fields, axis=1), dtype=FLOAT32_LE)
        header = {
            "startFrame": self.start_frame,
            "count": self.count,
            "revision": self.revision,
            "hasContacts": self.contacts is not None,
        }
        return Frame(
            msg_type=MessageType.CHUNK,
            header=header,
            blob=blob.tobytes(order="C"),
        )

    @classmethod
    def from_frame(cls, frame: Frame, *, joint_count: int) -> "ChunkMessage":
        if frame.msg_type != MessageType.CHUNK:
            raise ProtocolError(f"expected chunk frame, got message type {frame.msg_type}")
        try:
            start_frame = int(frame.header["startFrame"])
            count = int(frame.header["count"])
            revision = int(frame.header["revision"])
            has_contacts_value = frame.header["hasContacts"]
            if not isinstance(has_contacts_value, bool):
                raise ProtocolError("hasContacts must be a JSON boolean")
            has_contacts = has_contacts_value
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"invalid chunk header: {exc}") from exc
        if count <= 0 or joint_count <= 0:
            raise ProtocolError("chunk count and joint_count must be positive")
        floats_per_frame = 3 + joint_count * 4 + (4 if has_contacts else 0)
        expected_bytes = count * floats_per_frame * FLOAT32_LE.itemsize
        if len(frame.blob) != expected_bytes:
            raise ProtocolError(
                f"chunk blob must be {expected_bytes} bytes, got {len(frame.blob)}"
            )
        values = np.frombuffer(frame.blob, dtype=FLOAT32_LE).reshape(count, floats_per_frame)
        rotation_end = 3 + joint_count * 4
        return cls.create(
            start_frame=start_frame,
            revision=revision,
            root_positions=values[:, :3],
            local_quaternions=values[:, 3:rotation_end].reshape(count, joint_count, 4),
            contacts=values[:, rotation_end:] if has_contacts else None,
        )
