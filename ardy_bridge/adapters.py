"""Adapters from ARDY model objects/output tensors to bridge messages."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .messages import ChunkMessage, HelloMessage
from .protocol import ProtocolError


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def hello_from_model(model: Any, *, model_name: str, skeleton_name: str | None = None) -> HelloMessage:
    """Build the complete rig description sent when Unity connects."""

    skeleton = model.skeleton
    joint_names = tuple(str(value) for value in skeleton.bone_order_names)
    left_contacts = _foot_contact_indices(skeleton, joint_names, "left")
    right_contacts = _foot_contact_indices(skeleton, joint_names, "right")
    contact_joints = left_contacts + right_contacts
    if len(contact_joints) != 4:
        raise ProtocolError(
            f"bridge v1 expects four foot contact joints, got {len(contact_joints)}"
        )
    return HelloMessage.create(
        model=model_name,
        skeleton=skeleton_name or type(skeleton).__name__,
        fps=int(model.motion_rep.fps),
        joint_names=joint_names,
        parents=_to_numpy(skeleton.joint_parents).tolist(),
        root_idx=int(skeleton.root_idx),
        foot_contact_joints=contact_joints,
        gen_horizon=int(model.gen_horizon_len),
        num_frames_per_token=int(model.num_frames_per_token),
        neutral_joints=_to_numpy(skeleton.neutral_joints),
    )


def _foot_contact_indices(
    skeleton: Any,
    joint_names: tuple[str, ...],
    side: str,
) -> tuple[int, ...]:
    for attribute in (f"{side}_foot_joint_idx", f"{side}_foot_joint_indices"):
        values = getattr(skeleton, attribute, None)
        if values is not None:
            return tuple(int(value) for value in values)

    names = getattr(skeleton, f"{side}_foot_joint_names", None)
    if names is None:
        raise ProtocolError(f"skeleton has no {side} foot contact metadata")
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    try:
        return tuple(name_to_index[str(name)] for name in names)
    except KeyError as exc:
        raise ProtocolError(
            f"{side} foot contact joint {exc.args[0]!r} is not in bone_order_names"
        ) from exc


def chunk_from_ardy_output(
    output: Mapping[str, Any],
    *,
    start_frame: int,
    revision: int,
    sample_index: int = 0,
    history_frames: int = 0,
    output_frames: int | None = None,
) -> ChunkMessage:
    """Convert ``motion_rep.inverse`` output into one v1 wire chunk.

    ``history_frames`` skips the history prefix returned by an autoregressive
    step so only newly generated frames are transmitted. Matrices are converted
    to scipy quaternions in explicit ``(x, y, z, w)`` order. Coordinate-system
    conversion intentionally remains on the Unity side.
    """

    try:
        rotations = _to_numpy(output["local_rot_mats"])
        roots = _to_numpy(output["root_positions"])
    except KeyError as exc:
        raise ProtocolError(f"ARDY output is missing {exc.args[0]!r}") from exc
    if rotations.ndim != 5 or rotations.shape[-2:] != (3, 3):
        raise ProtocolError(
            f"local_rot_mats must have shape [B,T,J,3,3], got {rotations.shape}"
        )
    if roots.ndim != 3 or roots.shape[-1] != 3:
        raise ProtocolError(f"root_positions must have shape [B,T,3], got {roots.shape}")
    if rotations.shape[:2] != roots.shape[:2]:
        raise ProtocolError("local_rot_mats and root_positions batch/frame dimensions differ")
    batch_size, frame_count, joint_count = rotations.shape[:3]
    if not 0 <= sample_index < batch_size:
        raise ProtocolError(f"sample_index {sample_index} is outside batch size {batch_size}")
    if not 0 <= history_frames < frame_count:
        raise ProtocolError(
            f"history_frames must be in [0, {frame_count - 1}], got {history_frames}"
        )

    remaining_frames = frame_count - history_frames
    if output_frames is None:
        output_frames = remaining_frames
    if not 1 <= output_frames <= remaining_frames:
        raise ProtocolError(
            f"output_frames must be in [1, {remaining_frames}], got {output_frames}"
        )
    output_end = history_frames + output_frames

    selected_rotations = rotations[sample_index, history_frames:output_end]
    quaternions = Rotation.from_matrix(
        selected_rotations.reshape(-1, 3, 3)
    ).as_quat().reshape(output_frames, joint_count, 4)
    selected_roots = roots[sample_index, history_frames:output_end]

    contacts = None
    if "foot_contacts" in output and output["foot_contacts"] is not None:
        contact_values = _to_numpy(output["foot_contacts"])
        if contact_values.shape != (batch_size, frame_count, 4):
            raise ProtocolError(
                f"foot_contacts must have shape {(batch_size, frame_count, 4)}, "
                f"got {contact_values.shape}"
            )
        contacts = contact_values[sample_index, history_frames:output_end]

    return ChunkMessage.create(
        start_frame=start_frame,
        revision=revision,
        root_positions=selected_roots,
        local_quaternions=quaternions,
        contacts=contacts,
    )
