"""Reproducible CUDA benchmark for ARDY Core Horizon40 autoregressive steps.

The benchmark deliberately bypasses the text encoder and supplies a synthetic
embedding with the same shape expected by the denoiser. Text content does not
change the tensor shapes or the amount of generation work. This isolates model
load and autoregressive generation latency from prompt-encoding latency.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL = "ARDY-Core-RP-20FPS-Horizon40"
DEFAULT_STEPS = (10, 8, 6, 4, 2, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--diffusion-steps",
        default=",".join(map(str, DEFAULT_STEPS)),
        help="Comma-separated denoising-step counts (default: 10,8,6,4,2,1).",
    )
    parser.add_argument("--warm-steps", type=int, default=10)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument(
        "--text-tokens",
        type=int,
        default=None,
        help="Synthetic text-token count (default: checkpoint llm_shape token count).",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmarks",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads. By default only the local cache is used.",
    )
    return parser.parse_args()


def parse_step_counts(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("--diffusion-steps must contain positive integers")
    return values


def query_gpu() -> tuple[str, str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(command, text=True, timeout=10).strip().splitlines()[0]
        name, driver, memory_mib = [part.strip() for part in line.split(",")]
        return name, driver, memory_mib
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return "unknown", "unknown", "unknown"


def timed_cuda_call(torch, operation):
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - started) * 1000.0


def fmt(value: float) -> str:
    return f"{value:.2f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    *,
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    load_ms: float,
    python_version: str,
    torch_version: str,
    cuda_version: str,
    gpu_name: str,
    driver_version: str,
    gpu_memory_mib: str,
    fps: int,
    horizon: int,
    base_steps: int,
) -> None:
    full_quality = next((row for row in rows if row["diffusion_steps"] == base_steps), None)
    if full_quality is None:
        decision = "Полный режим модели не измерен; решение отложено."
    else:
        warm_mean_ms = float(full_quality["warm_mean_ms"])
        chunk_budget_ms = horizon / fps * 1000.0
        if warm_mean_ms <= chunk_budget_ms:
            decision = (
                f"Полный режим ({base_steps} steps) укладывается в длительность core40-chunk "
                f"{chunk_budget_ms / 1000.0:.2f} с. Для текущего этапа подходит буферизованная "
                "генерация core40; core8 пока не требуется."
            )
        else:
            decision = (
                f"Полный режим ({base_steps} steps) не укладывается в длительность core40-chunk "
                f"{chunk_budget_ms / 1000.0:.2f} с. Нужен сниженный diffusion preset или отдельная "
                "проверка core8/ускорения."
            )

    lines = [
        "# ARDY core40 benchmark",
        "",
        f"Дата UTC: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`  ",
        f"Модель: `{args.model}`  ",
        f"GPU: `{gpu_name}` (`{gpu_memory_mib} MiB`, driver `{driver_version}`)  ",
        f"Python: `{python_version}`  ",
        f"PyTorch: `{torch_version}`, CUDA runtime: `{cuda_version}`  ",
        f"Загрузка модели без text encoder: `{load_ms / 1000.0:.3f} с`  ",
        f"Horizon: `{horizon}` кадров при `{fps} FPS` = `{horizon / fps:.2f} с` движения  ",
        f"История прогретого шага: `{args.history_frames}` кадра; повторов: `{args.warm_steps}`  ",
        f"Синтетический text embedding: `[1, {args.text_tokens}, 4096]`.",
        "",
        "## Результаты",
        "",
        "| Diffusion steps | Первый шаг без истории, мс | Прогретый mean, мс | median, мс | min–max, мс | x realtime | Peak PyTorch alloc, MiB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['diffusion_steps']} | {row['initial_no_history_ms']} | "
            f"{row['warm_mean_ms']} | {row['warm_median_ms']} | "
            f"{row['warm_min_ms']}–{row['warm_max_ms']} | {row['realtime_factor']} | "
            f"{row['peak_vram_mib']} |"
        )

    lines.extend(
        [
            "",
            "`x realtime` = длительность выдаваемого 2-секундного chunk / среднее время прогретого шага; больше 1 означает, что генерация быстрее воспроизведения.",
            "",
            "## Решение",
            "",
            decision,
            "",
            "Это замер PyTorch/CUDA без TensorRT и без `torch.compile`. Text encoder исключён намеренно: в live-пайплайне embedding кэшируется и не входит в каждый autoregressive step.",
            "",
            "## Воспроизведение",
            "",
            "Запускать из корня ARDY через wrapper, который выбирает исправный venv или локальный Python 3.12 fallback:",
            "",
            "```powershell",
            ".\\tools\\run_core40_benchmark.ps1",
            "```",
            "",
            "Скрипт по умолчанию работает offline и не загружает другие checkpoints, включая core8.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    step_counts = parse_step_counts(args.diffusion_steps)
    if args.warm_steps < 1:
        raise ValueError("--warm-steps must be >= 1")
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"

    import torch

    from ardy.model.load_model import load_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda"
    cuda_device = torch.device("cuda:0")

    started = time.perf_counter()
    model = load_model(args.model, device=device, text_encoder=False)
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - started) * 1000.0

    base_steps = int(model.diffusion.num_base_steps)
    if any(value > base_steps for value in step_counts):
        raise ValueError(f"diffusion steps cannot exceed model maximum {base_steps}")
    if args.history_frames < model.num_frames_per_token or args.history_frames % model.num_frames_per_token:
        raise ValueError(
            f"--history-frames must be a positive multiple of {model.num_frames_per_token}"
        )

    fps = int(model.motion_rep.fps)
    horizon = int(model.gen_horizon_len)
    llm_tokens, llm_dim = [int(value) for value in model.denoiser.model.llm_shape]
    text_tokens = args.text_tokens if args.text_tokens is not None else llm_tokens
    if text_tokens < 1:
        raise ValueError("--text-tokens must be >= 1")
    args.text_tokens = text_tokens
    text_feat = torch.randn(1, text_tokens, llm_dim, device=device)
    text_pad_mask = torch.ones(1, text_tokens, dtype=torch.bool, device=device)
    chunk_duration_ms = horizon / fps * 1000.0

    rows: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for diffusion_steps in step_counts:
            torch.cuda.reset_peak_memory_stats(cuda_device)

            def initial_step():
                return model.autoregressive_step(
                    num_frames=horizon,
                    num_denoising_steps=diffusion_steps,
                    motion_mask=None,
                    observed_motion=None,
                    text_feat=text_feat,
                    text_pad_mask=text_pad_mask,
                )

            output, initial_ms = timed_cuda_call(torch, initial_step)
            warm_times: list[float] = []
            history = output[:, -args.history_frames :].contiguous()

            for _ in range(args.warm_steps):
                def warm_step():
                    return model.autoregressive_step(
                        num_frames=args.history_frames + horizon,
                        num_denoising_steps=diffusion_steps,
                        motion_mask=None,
                        observed_motion=None,
                        text_feat=text_feat,
                        text_pad_mask=text_pad_mask,
                        init_history_sequence=history,
                    )

                output, elapsed_ms = timed_cuda_call(torch, warm_step)
                warm_times.append(elapsed_ms)
                history = output[:, -args.history_frames :].contiguous()

            warm_mean_ms = statistics.fmean(warm_times)
            rows.append(
                {
                    "model": args.model,
                    "diffusion_steps": diffusion_steps,
                    "initial_no_history_ms": fmt(initial_ms),
                    "warm_mean_ms": fmt(warm_mean_ms),
                    "warm_median_ms": fmt(statistics.median(warm_times)),
                    "warm_min_ms": fmt(min(warm_times)),
                    "warm_max_ms": fmt(max(warm_times)),
                    "realtime_factor": fmt(chunk_duration_ms / warm_mean_ms),
                    "peak_vram_mib": fmt(torch.cuda.max_memory_allocated(cuda_device) / (1024**2)),
                    "warm_repetitions": args.warm_steps,
                    "history_frames": args.history_frames,
                    "horizon_frames": horizon,
                    "fps": fps,
                    "model_load_ms": fmt(load_ms),
                }
            )
            print(
                f"steps={diffusion_steps}: initial={initial_ms:.2f} ms, "
                f"warm_mean={warm_mean_ms:.2f} ms, "
                f"peak={rows[-1]['peak_vram_mib']} MiB",
                flush=True,
            )

    gpu_name, driver_version, gpu_memory_mib = query_gpu()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "core40_benchmark.csv"
    markdown_path = args.output_dir / "core40_benchmark.md"
    write_csv(csv_path, rows)
    write_markdown(
        markdown_path,
        args=args,
        rows=rows,
        load_ms=load_ms,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_version=str(torch.version.cuda),
        gpu_name=gpu_name,
        driver_version=driver_version,
        gpu_memory_mib=gpu_memory_mib,
        fps=fps,
        horizon=horizon,
        base_steps=base_steps,
    )
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
