"""Run the cached ARDY core40 model as a paced Unity bridge server.

The runner loads the already cached NF4 LLM2Vec encoder directly, with no
Viser/Gradio web UI. It defaults to one 20-frame English-prompt preview.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Keep one bridge process alive across prompt changes. The memory manager may
# temporarily offload the motion model while the 5.4GB text encoder is active,
# then restore it without dropping Unity's TCP connection.
os.environ.setdefault("ARDY_OFFLOAD", "1")

from ardy_bridge.adapters import chunk_from_ardy_output, hello_from_model
from ardy_bridge.protocol import MessageType
from ardy_bridge.server import BridgeServer, ClientConnection


DEFAULT_MODEL = "ARDY-Core-RP-20FPS-Horizon40"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8801)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--prompt", default="A person walks.")
    parser.add_argument(
        "--output-frames",
        type=int,
        default=0,
        help="Frames sent per generation step; 0 sends the complete horizon.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Generate but do not transmit this many warm-up frames.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Stop after this many chunks; 0 continues until disconnected.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--total-frames",
        type=int,
        default=0,
        help="Generate exactly this many frames, then remain ready; 0 streams continuously.",
    )
    parser.add_argument("--connect-timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch

    from ardy.model.load_model import load_model, load_text_encoder

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the core40 Unity bridge")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.prompt.isascii():
        raise ValueError("--prompt must contain English/ASCII text only")
    if args.max_chunks < 0:
        raise ValueError("--max-chunks must be non-negative")
    if args.total_frames < 0:
        raise ValueError("--total-frames must be non-negative")

    from ardy.model.memory_manager import manager as memory_manager

    text_encoder_holder: list[Any] = []

    def encode_prompt(prompt: str) -> tuple[Any, list[int]]:
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = Path(".cache") / "text_embeddings" / f"{cache_key}.npy"
        if cache_path.exists():
            import numpy as np

            cached = np.load(cache_path)
            if cached.ndim != 2:
                raise ValueError(f"invalid cached embedding shape {cached.shape}")
            print(f"Loaded cached embedding for {prompt!r}", flush=True)
            return (
                torch.from_numpy(cached).unsqueeze(0).to(dtype=torch.float32),
                [int(cached.shape[0])],
            )

        print("Loading cached NF4 text encoder without web UI...", flush=True)
        encoder_started = time.perf_counter()
        if not text_encoder_holder:
            text_encoder_holder.append(load_text_encoder(mode="local", device="cuda"))
            memory_manager.register_encoder(text_encoder_holder[0])
        text_encoder = text_encoder_holder[0]
        encoded_features, encoded_lengths = text_encoder([prompt])
        text_features_cpu = encoded_features.detach().to(device="cpu", dtype=torch.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.save(cache_path, text_features_cpu[0, : encoded_lengths[0]].numpy())
        if hasattr(text_encoder, "unload"):
            text_encoder.unload()
        del encoded_features
        torch.cuda.empty_cache()
        print(
            f"Encoded and cached English prompt {prompt!r} in "
            f"{time.perf_counter() - encoder_started:.3f}s",
            flush=True,
        )
        return text_features_cpu, encoded_lengths

    text_features_cpu, encoded_lengths = encode_prompt(args.prompt)

    print(f"Loading cached {args.model} on CUDA...", flush=True)
    load_started = time.perf_counter()
    model = load_model(args.model, device="cuda", text_encoder=False)
    model.eval()
    memory_manager.register_model(args.model, model)
    torch.cuda.synchronize()
    print(f"Model loaded in {time.perf_counter() - load_started:.3f}s", flush=True)

    horizon = int(model.gen_horizon_len)
    fps = int(model.motion_rep.fps)
    token_frames = int(model.num_frames_per_token)
    if args.diffusion_steps < 1 or args.diffusion_steps > int(model.diffusion.num_base_steps):
        raise ValueError("--diffusion-steps is outside the model range")
    if args.history_frames < token_frames or args.history_frames % token_frames:
        raise ValueError(
            f"--history-frames must be a positive multiple of {token_frames}"
        )
    if args.skip_frames < 0 or args.skip_frames >= horizon:
        raise ValueError(f"--skip-frames must be in [0, {horizon - 1}]")
    if args.output_frames < 0 or args.output_frames > horizon:
        raise ValueError(f"--output-frames must be in [0, {horizon}]")
    if args.output_frames and args.skip_frames + args.output_frames > horizon:
        raise ValueError(
            f"--skip-frames + --output-frames must be at most {horizon}"
        )

    hello = hello_from_model(model, model_name=args.model, skeleton_name="cskel27")

    disconnected = threading.Event()
    generation_failed: list[BaseException] = []
    preview_frames: list[Any] = []
    generation_thread: threading.Thread | None = None
    generation_stop: threading.Event | None = None
    revision = -1
    server: BridgeServer

    def generate(
        job_stop: threading.Event,
        prompt: str,
        text_features: Any,
        text_mask: Any,
        total_frames: int,
        seed: int,
        job_revision: int,
    ) -> None:
        history = None
        start_frame = 0
        chunks_sent = 0
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                while not job_stop.is_set() and not disconnected.is_set():
                    history_count = 0 if history is None else int(history.shape[1])
                    warmup_count = args.skip_frames if chunks_sent == 0 else 0
                    output_count = args.output_frames or (horizon - warmup_count)
                    if total_frames:
                        output_count = min(output_count, total_frames - start_frame)
                        if output_count <= 0:
                            return
                    step_started = time.perf_counter()
                    samples = model.autoregressive_step(
                        num_frames=history_count + horizon,
                        num_denoising_steps=args.diffusion_steps,
                        motion_mask=None,
                        observed_motion=None,
                        text_feat=text_features,
                        text_pad_mask=text_mask,
                        init_history_sequence=history,
                    )
                    torch.cuda.synchronize()
                    unnormalized = model.motion_rep.unnormalize(samples)
                    decoded: dict[str, Any] = model.motion_rep.inverse(
                        unnormalized,
                        is_normalized=False,
                    )
                    chunk = chunk_from_ardy_output(
                        decoded,
                        start_frame=start_frame,
                        revision=job_revision,
                        history_frames=history_count + warmup_count,
                        output_frames=output_count,
                    )
                    if job_stop.is_set() or disconnected.is_set():
                        return
                    preview_frame = chunk.to_frame()
                    preview_frames[:] = [preview_frame]
                    if not server.send_frame(preview_frame):
                        return

                    elapsed = time.perf_counter() - step_started
                    print(
                        f"Sent frames {start_frame}-{start_frame + chunk.count - 1} "
                        f"in {elapsed * 1000.0:.1f}ms",
                        flush=True,
                    )
                    server.send(
                        MessageType.STATUS,
                        {
                            "state": "generating",
                            "prompt": prompt,
                            "lastStepMs": elapsed * 1000.0,
                            "vramMb": 0.0,
                            "message": f"Generated {start_frame + chunk.count} of {total_frames or 'continuous'} frames",
                        },
                    )
                    start_frame += chunk.count
                    chunks_sent += 1
                    history = samples[:, -args.history_frames :].contiguous()
                    if total_frames and start_frame >= total_frames:
                        server.send(
                            MessageType.STATUS,
                            {
                                "state": "ready",
                                "prompt": prompt,
                                "lastStepMs": elapsed * 1000.0,
                                "vramMb": 0.0,
                                "message": f"Finite sequence ready: {start_frame} frames",
                            },
                        )
                        print(
                            f"Finite sequence complete: {start_frame} frames for {prompt!r}",
                            flush=True,
                        )
                        return
                    if args.max_chunks and chunks_sent >= args.max_chunks:
                        print(
                            f"One-shot preview complete: {start_frame} frames for {prompt!r}",
                            flush=True,
                        )
                        return
                    job_stop.wait(max(0.0, horizon / fps - elapsed))
        except BaseException as exception:
            generation_failed.append(exception)
            try:
                server.send(
                    MessageType.ERROR,
                    {"code": "generation_failed", "message": str(exception)},
                )
            except OSError:
                pass
            job_stop.set()

    def begin_generation(
        client: ClientConnection,
        prompt: str,
        seed: int,
        total_frames: int,
        prepared: tuple[Any, list[int]] | None = None,
    ) -> None:
        nonlocal generation_thread, generation_stop, revision
        prompt = prompt.strip()
        if not prompt or not prompt.isascii():
            client.send(
                MessageType.ERROR,
                {"code": "invalid_prompt", "message": "Prompt must use English/ASCII text"},
            )
            return
        if total_frames < 0:
            client.send(
                MessageType.ERROR,
                {"code": "invalid_frame_count", "message": "totalFrames must be non-negative"},
            )
            return

        previous_thread = generation_thread
        previous_stop = generation_stop
        if previous_thread is not None and previous_thread.is_alive():
            client.send(
                MessageType.STATUS,
                {
                    "state": "switching_prompt",
                    "prompt": prompt,
                    "lastStepMs": 0.0,
                    "vramMb": 0.0,
                    "message": "Stopping the current generation without disconnecting Unity",
                },
            )
            if previous_stop is not None:
                previous_stop.set()
            previous_thread.join()

        if disconnected.is_set():
            return

        if revision >= 0:
            client.send(MessageType.INVALIDATE, {"fromFrame": 0})
        revision += 1
        preview_frames.clear()
        client.send(
            MessageType.STATUS,
            {
                "state": "encoding_prompt",
                "prompt": prompt,
                "lastStepMs": 0.0,
                "vramMb": 0.0,
                "message": "Preparing text conditioning on the existing bridge connection",
            },
        )
        features_cpu, lengths = prepared or encode_prompt(prompt)
        memory_manager.touch_and_move(args.model, "cuda")
        model.eval()
        text_features = features_cpu.to(device="cuda", dtype=torch.bfloat16)
        text_mask = torch.arange(text_features.shape[1], device="cuda").expand(
            text_features.shape[0], text_features.shape[1]
        ) < torch.tensor(lengths, device="cuda")[:, None]
        generation_stop = threading.Event()
        client.send(
            MessageType.STATUS,
            {
                "state": "generating",
                "prompt": prompt,
                "lastStepMs": 0.0,
                "vramMb": 0.0,
                "message": f"Generating revision {revision}, {total_frames or 'continuous'} frames",
            },
        )
        generation_thread = threading.Thread(
            target=generate,
            args=(
                generation_stop,
                prompt,
                text_features,
                text_mask,
                total_frames,
                seed,
                revision,
            ),
            name="ArdyCore40Generation",
            daemon=True,
        )
        generation_thread.start()

    def on_connect(client: ClientConnection) -> None:
        print(f"Unity client connected from {client.address[0]}:{client.address[1]}", flush=True)
        client.send(
            MessageType.STATUS,
            {
                "state": "handshake",
                "prompt": args.prompt,
                "lastStepMs": 0.0,
                "vramMb": 0.0,
                "message": "Unity connected; sending skeleton and model metadata",
            },
        )
        client.send_frame(hello.to_frame())
        print(
            f"Sent hello: {args.model}, {fps} FPS, {len(hello.joint_names)} joints",
            flush=True,
        )
        begin_generation(
            client,
            args.prompt,
            args.seed,
            args.total_frames,
            prepared=(text_features_cpu, encoded_lengths),
        )

    def on_message(client: ClientConnection, frame: Any) -> None:
        if frame.msg_type == MessageType.SET_PROMPT:
            try:
                prompt = str(frame.header.get("text", ""))
                seed = int(frame.header.get("seed", 0))
                total_frames = int(frame.header.get("totalFrames", horizon))
            except (TypeError, ValueError) as exception:
                client.send(
                    MessageType.ERROR,
                    {"code": "invalid_generation_request", "message": str(exception)},
                )
                return
            begin_generation(client, prompt, seed, total_frames)
            return
        if frame.msg_type != MessageType.TRANSPORT:
            return
        if frame.header.get("action") != "replay":
            return
        if not preview_frames:
            client.send(
                MessageType.ERROR,
                {"code": "preview_not_ready", "message": "No generated preview is cached yet"},
            )
            return
        client.send(MessageType.INVALIDATE, {"fromFrame": 0})
        client.send_frame(preview_frames[0])
        print("Replayed the most recently generated chunk", flush=True)

    def on_disconnect(client: ClientConnection) -> None:
        if generation_stop is not None:
            generation_stop.set()
        disconnected.set()

    server = BridgeServer(
        args.host,
        args.port,
        on_connect=on_connect,
        on_message=on_message,
        on_disconnect=on_disconnect,
    )
    try:
        host, port = server.start()
        print(f"ARDY core40 bridge listening on {host}:{port}", flush=True)
        if not server.wait_for_client(args.connect_timeout):
            raise TimeoutError("Unity did not connect before --connect-timeout")
        while not disconnected.wait(0.2):
            if generation_failed:
                raise RuntimeError("core40 generation failed") from generation_failed[0]
    finally:
        if generation_stop is not None:
            generation_stop.set()
        server.stop()
        if generation_thread is not None:
            generation_thread.join(timeout=5.0)

    if generation_failed:
        raise RuntimeError("core40 generation failed") from generation_failed[0]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
