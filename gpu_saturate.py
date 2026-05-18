#!/usr/bin/env python3
"""Occupy all visible CUDA GPUs with memory allocations and matmul load."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
from typing import Iterable


def parse_gpus(value: str, count: int) -> list[int]:
    if value.lower() == "all":
        return list(range(count))
    gpus = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        gpu = int(item)
        if gpu < 0 or gpu >= count:
            raise ValueError(f"GPU index {gpu} is outside visible range 0..{count - 1}")
        gpus.append(gpu)
    if not gpus:
        raise ValueError("No GPUs selected")
    return sorted(set(gpus))


def dtype_from_name(torch, name: str):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def allocate_memory(torch, device: str, fraction: float, dtype) -> list:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    target_bytes = int(total_bytes * fraction)
    bytes_to_allocate = max(0, min(target_bytes, int(free_bytes * 0.75)))
    element_size = torch.tensor([], dtype=dtype, device=device).element_size()
    chunk_bytes = 256 * 1024 * 1024
    chunks = []

    while bytes_to_allocate > 0:
        this_chunk = min(chunk_bytes, bytes_to_allocate)
        elements = max(1, this_chunk // element_size)
        try:
            tensor = torch.empty(elements, dtype=dtype, device=device)
            tensor.normal_()
            chunks.append(tensor)
            bytes_to_allocate -= elements * element_size
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            if chunk_bytes <= 16 * 1024 * 1024:
                break
            chunk_bytes //= 2

    return chunks


def gpu_worker(
    gpu: int,
    memory_fraction: float,
    matrix_size: int,
    dtype_name: str,
    seconds: int,
    report_every: int,
) -> None:
    import torch

    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    dtype = dtype_from_name(torch, dtype_name)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    held_tensors = allocate_memory(torch, device, memory_fraction, dtype)

    n = matrix_size
    while n >= 512:
        try:
            a = torch.randn((n, n), device=device, dtype=dtype)
            b = torch.randn((n, n), device=device, dtype=dtype)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            n //= 2
    else:
        raise RuntimeError(f"GPU {gpu}: not enough free memory for compute tensors")

    print(
        f"[gpu {gpu}] holding {sum(t.numel() * t.element_size() for t in held_tensors) / 2**30:.2f} GiB; "
        f"matmul {n}x{n} {dtype_name}",
        flush=True,
    )

    start = time.monotonic()
    last_report = start
    iters = 0
    with torch.no_grad():
        while seconds <= 0 or time.monotonic() - start < seconds:
            c = a @ b
            a, b = b, c
            iters += 1
            now = time.monotonic()
            if report_every > 0 and now - last_report >= report_every:
                torch.cuda.synchronize(device)
                elapsed = now - start
                print(f"[gpu {gpu}] {iters} matmuls in {elapsed:.1f}s", flush=True)
                last_report = now


def terminate(processes: Iterable[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Use all visible CUDA GPUs for a controllable stress/occupancy test."
    )
    parser.add_argument("--gpus", default="all", help="GPU ids such as '0,1' or 'all'.")
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--seconds", type=int, default=0, help="0 means run until Ctrl-C.")
    parser.add_argument("--report-every", type=int, default=30)
    args = parser.parse_args()

    if not 0.0 < args.memory_fraction < 1.0:
        raise SystemExit("--memory-fraction must be between 0 and 1")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this environment.")

    gpu_count = torch.cuda.device_count()
    gpus = parse_gpus(args.gpus, gpu_count)
    print(f"Visible CUDA GPUs: {gpu_count}; selected: {gpus}", flush=True)

    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(
            target=gpu_worker,
            args=(
                gpu,
                args.memory_fraction,
                args.matrix_size,
                args.dtype,
                args.seconds,
                args.report_every,
            ),
        )
        for gpu in gpus
    ]

    def handle_stop(signum, _frame):
        print(f"Received signal {signum}; stopping workers...", flush=True)
        terminate(processes)
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    return max((process.exitcode or 0) for process in processes)


if __name__ == "__main__":
    raise SystemExit(main())
