#!/usr/bin/env python3
"""Generate deterministic cskel27 golden data for the Unity bridge tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ardy.skeleton.definitions import CoreSkeleton27


def vec3(value: torch.Tensor) -> dict[str, float]:
    return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}


def quat4(value: torch.Tensor) -> dict[str, float]:
    return {
        "x": float(value[0]),
        "y": float(value[1]),
        "z": float(value[2]),
        "w": float(value[3]),
    }


def axis_angle_xyzw(axis: tuple[float, float, float], degrees: float) -> torch.Tensor:
    axis_tensor = torch.tensor(axis, dtype=torch.float64)
    axis_tensor = axis_tensor / torch.linalg.vector_norm(axis_tensor)
    half_angle = math.radians(degrees) * 0.5
    xyz = axis_tensor * math.sin(half_angle)
    return torch.cat((xyz, torch.tensor([math.cos(half_angle)], dtype=torch.float64)))


def quaternion_xyzw_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    quaternions = quaternions / torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
    x, y, z, w = quaternions.unbind(dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    return torch.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - zw),
            2.0 * (xz + yw),
            2.0 * (xy + zw),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - xw),
            2.0 * (xz - yw),
            2.0 * (yz + xw),
            1.0 - 2.0 * (xx + yy),
        ),
        dim=-1,
    ).reshape(quaternions.shape[:-1] + (3, 3))


def build_diagnostic_frame(skeleton: CoreSkeleton27) -> dict[str, object]:
    local_quaternions = torch.zeros((skeleton.nbjoints, 4), dtype=torch.float64)
    local_quaternions[:, 3] = 1.0

    diagnostic_rotations = {
        "Hips": ((0.0, 1.0, 0.0), 20.0),
        "Spine2": ((1.0, 0.0, 0.0), 8.0),
        "RightArm": ((0.0, 0.0, 1.0), -22.0),
        "RightForeArm": ((0.0, 1.0, 0.0), 28.0),
        "LeftArm": ((0.0, 0.0, 1.0), 17.0),
        "LeftForeArm": ((0.0, 1.0, 0.0), -31.0),
        "RightUpLeg": ((1.0, 0.0, 0.0), -12.0),
        "RightLeg": ((1.0, 0.0, 0.0), 24.0),
        "RightFoot": ((1.0, 0.0, 0.0), -9.0),
        "LeftUpLeg": ((1.0, 0.0, 0.0), 14.0),
        "LeftLeg": ((1.0, 0.0, 0.0), -19.0),
        "LeftFoot": ((1.0, 0.0, 0.0), 6.0),
    }

    for joint_name, (axis, degrees) in diagnostic_rotations.items():
        local_quaternions[skeleton.bone_index[joint_name]] = axis_angle_xyzw(axis, degrees)

    local_rotations = quaternion_xyzw_to_matrix(local_quaternions)
    root_position = torch.tensor([0.35, 0.96, 1.20], dtype=torch.float64)
    _, posed_joints, _ = skeleton.fk(local_rotations.unsqueeze(0), root_position.unsqueeze(0))

    return {
        "name": "diagnostic_pose",
        "root_position_ardy": vec3(root_position),
        "local_rotations_xyzw_ardy": [quat4(value) for value in local_quaternions],
        "posed_joints_ardy": [vec3(value) for value in posed_joints[0]],
    }


def build_test_quaternions() -> list[dict[str, object]]:
    cases = [
        ("identity", (0.0, 1.0, 0.0), 0.0),
        ("x_plus_90", (1.0, 0.0, 0.0), 90.0),
        ("x_minus_90", (1.0, 0.0, 0.0), -90.0),
        ("y_plus_90", (0.0, 1.0, 0.0), 90.0),
        ("y_minus_90", (0.0, 1.0, 0.0), -90.0),
        ("z_plus_90", (0.0, 0.0, 1.0), 90.0),
        ("z_minus_90", (0.0, 0.0, 1.0), -90.0),
    ]

    result = []
    for name, axis, degrees in cases:
        ardy = axis_angle_xyzw(axis, degrees)
        unity = torch.tensor([ardy[0], -ardy[1], -ardy[2], ardy[3]], dtype=torch.float64)
        result.append(
            {
                "name": name,
                "ardy_xyzw": quat4(ardy),
                "unity_xyzw": quat4(unity),
            }
        )
    return result


def generate() -> dict[str, object]:
    skeleton = CoreSkeleton27()
    neutral_joints = skeleton.neutral_joints.to(dtype=torch.float64)

    local_offsets = neutral_joints.clone()
    for joint_index, parent_index in enumerate(skeleton.joint_parents.tolist()):
        if parent_index >= 0:
            local_offsets[joint_index] -= neutral_joints[parent_index]

    return {
        "schema_version": 1,
        "generator": "tools/gen_golden.py",
        "coordinate_system": {
            "handedness": "right",
            "up_axis": "+Y",
            "forward_axis": "+Z",
            "unit": "meter",
            "quaternion_order": "xyzw",
        },
        "skeleton": {
            "name": skeleton.name,
            "root_index": skeleton.root_idx,
            "joint_names": skeleton.bone_order_names,
            "parents": [int(value) for value in skeleton.joint_parents.tolist()],
            "neutral_joints_ardy": [vec3(value) for value in neutral_joints],
            "local_offsets_ardy": [vec3(value) for value in local_offsets],
        },
        "test_quaternions": build_test_quaternions(),
        "frames": [build_diagnostic_frame(skeleton)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("golden_transform.json"),
        help="Destination JSON path.",
    )
    args = parser.parse_args()

    payload = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(payload['skeleton']['joint_names'])} joints).")


if __name__ == "__main__":
    main()
