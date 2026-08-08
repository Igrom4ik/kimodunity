"""Serve golden cskel27 hello/chunk frames without loading the ARDY model."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import numpy as np

from ardy_bridge.messages import ChunkMessage, HelloMessage
from ardy_bridge.server import BridgeServer, ClientConnection


def _xyz(values: list[dict[str, float]]) -> np.ndarray:
    return np.asarray([[value["x"], value["y"], value["z"]] for value in values], dtype=np.float32)


def _xyzw(values: list[dict[str, float]]) -> np.ndarray:
    return np.asarray(
        [[value["x"], value["y"], value["z"], value["w"]] for value in values],
        dtype=np.float32,
    )


def build_messages(golden_path: Path, frame_count: int) -> tuple[HelloMessage, ChunkMessage]:
    data = json.loads(golden_path.read_text(encoding="utf-8"))
    skeleton = data["skeleton"]
    frame = data["frames"][0]
    joint_names = skeleton["joint_names"]
    foot_contact_joints = tuple(
        joint_names.index(name)
        for name in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")
    )
    hello = HelloMessage.create(
        model="ARDY-Core-RP-20FPS-Horizon40-golden-smoke",
        skeleton=skeleton["name"],
        fps=20,
        joint_names=joint_names,
        parents=skeleton["parents"],
        root_idx=skeleton["root_index"],
        foot_contact_joints=foot_contact_joints,
        gen_horizon=40,
        num_frames_per_token=4,
        neutral_joints=_xyz(skeleton["neutral_joints_ardy"]),
    )
    root = np.asarray(
        [[frame["root_position_ardy"][axis] for axis in ("x", "y", "z")]],
        dtype=np.float32,
    )
    rotations = _xyzw(frame["local_rotations_xyzw_ardy"])[None, ...]
    chunk = ChunkMessage.create(
        start_frame=0,
        revision=0,
        root_positions=np.repeat(root, frame_count, axis=0),
        local_quaternions=np.repeat(rotations, frame_count, axis=0),
        contacts=np.zeros((frame_count, 4), dtype=np.float32),
    )
    return hello, chunk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8801)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path(__file__).resolve().parent / "golden_transform.json",
    )
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be positive")

    hello, chunk = build_messages(args.golden, args.frames)
    disconnected = threading.Event()

    def on_connect(client: ClientConnection) -> None:
        client.send_frame(hello.to_frame())
        client.send_frame(chunk.to_frame())

    def on_disconnect(client: ClientConnection) -> None:
        disconnected.set()

    with BridgeServer(
        args.host,
        args.port,
        on_connect=on_connect,
        on_disconnect=on_disconnect,
    ) as server:
        host, port = server.address
        print(f"ARDY bridge smoke server listening on {host}:{port}", flush=True)
        if not disconnected.wait(args.timeout):
            raise TimeoutError("no smoke client completed before timeout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
