# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish the interactive browser session to the Unity runtime bridge."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ardy_bridge.adapters import chunk_from_ardy_output, hello_from_model
from ardy_bridge.protocol import MessageType
from ardy_bridge.server import BridgeServer, ClientConnection


class UnityBridgeMixin:
    def init_unity_bridge(self, port: int) -> None:
        self._unity_bridge_port = int(port)
        self._unity_browser_client_id: int | None = None
        self._unity_hello = None
        self._unity_revision = 0
        self.unity_bridge = BridgeServer(
            "127.0.0.1",
            self._unity_bridge_port,
            on_connect=self._on_unity_connect,
            on_message=self._on_unity_message,
        )
        host, bound_port = self.unity_bridge.start()
        print(f"[Unity Bridge] Browser runtime listening on {host}:{bound_port}", flush=True)

    def _on_unity_connect(self, client: ClientConnection) -> None:
        print(f"[Unity Bridge] Unity connected from {client.address}", flush=True)
        if self._unity_hello is not None:
            client.send_frame(self._unity_hello.to_frame())
        self._publish_unity_motion()
        self._publish_unity_playhead()

    def _on_unity_message(self, client: ClientConnection, frame: Any) -> None:
        if frame.msg_type != MessageType.TRANSPORT:
            return
        action = frame.header.get("action")
        client_id = self._unity_browser_client_id
        if client_id is None or not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        if action == "play":
            session.playing = True
        elif action == "pause":
            session.playing = False
        elif action == "stop":
            session.playing = False
            self.set_frame(client_id, 0)
        self._publish_unity_playhead()

    def on_client_connect(self, client) -> None:
        self._unity_browser_client_id = client.client_id
        super().on_client_connect(client)
        self._publish_unity_motion()
        self._publish_unity_playhead()

    def on_client_disconnect(self, client) -> None:
        super().on_client_disconnect(client)
        if self._unity_browser_client_id == client.client_id:
            self._unity_browser_client_id = None
            self._unity_hello = None

    def load_model(self, client_id: int, model_name: str, progress=None):
        model = super().load_model(client_id, model_name, progress=progress)
        if model is not None:
            _, _, skeleton_name = self.get_skeleton_info(model.motion_rep.skeleton)
            self._unity_hello = hello_from_model(
                model,
                model_name=model_name,
                skeleton_name=skeleton_name,
            )
            if self.unity_bridge.has_client:
                self.unity_bridge.send_frame(self._unity_hello.to_frame())
        return model

    def _generate_step(self, client_id: int):
        result = super()._generate_step(client_id)
        if client_id == self._unity_browser_client_id:
            self._publish_unity_motion()
            self._publish_unity_playhead()
        return result

    def set_frame(self, client_id: int, frame_idx: int, trigger_by_gui_timeline: bool = False):
        result = super().set_frame(client_id, frame_idx, trigger_by_gui_timeline)
        if client_id == self._unity_browser_client_id:
            self._publish_unity_playhead()
        return result

    def _publish_unity_motion(self) -> None:
        if not self.unity_bridge.has_client:
            return
        client_id = self._unity_browser_client_id
        if client_id is None or not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        if session.motion_tensor is None or session.motion_rep is None:
            return

        with session.motion_tensor_lock:
            motion = session.motion_tensor.detach().clone()
        unnormalized = session.motion_rep.unnormalize(motion)
        decoded = session.motion_rep.inverse(unnormalized, is_normalized=False)
        frame_count = int(motion.shape[1])
        self._unity_revision += 1
        chunk = chunk_from_ardy_output(
            decoded,
            start_frame=0,
            revision=self._unity_revision,
            output_frames=frame_count,
        )
        self.unity_bridge.send(MessageType.INVALIDATE, {"fromFrame": 0})
        self.unity_bridge.send_frame(chunk.to_frame())
        print(
            f"[Unity Bridge] Published frames 0-{frame_count - 1} "
            f"(revision {self._unity_revision})",
            flush=True,
        )

    def _publish_unity_playhead(self) -> None:
        if not self.unity_bridge.has_client:
            return
        client_id = self._unity_browser_client_id
        if client_id is None or not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        frame = max(0, int(session.frame_idx))
        range_end = max(frame, int(session.max_frame_idx))
        self.unity_bridge.send(
            MessageType.PLAYHEAD,
            {
                "frame": frame,
                "playing": bool(session.playing),
                "rangeEnd": range_end,
            },
        )
