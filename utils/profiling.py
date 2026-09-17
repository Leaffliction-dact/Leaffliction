from pathlib import Path

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)


def get_activities(device: torch.device) -> list[ProfilerActivity]:
    """Return CPU + optional CUDA activities for the given device."""
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def make_training_profiler(
    device: torch.device,
    trace_dir: Path,
    *,
    wait: int = 1,
    warmup: int = 1,
    active: int = 5,
    repeat: int = 1,
) -> profile:
    trace_dir.mkdir(parents=True, exist_ok=True)

    return profile(
        activities=get_activities(device),
        schedule=schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        ),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    )


def make_inference_profiler(device: torch.device) -> profile:
    return profile(
        activities=get_activities(device),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    )


def print_profiler_summary(
    prof: profile,
    *,
    row_limit: int = 20,
) -> None:
    sep = "=" * 80
    averages = prof.key_averages(group_by_input_shape=True)

    print(f"\n{sep}")
    print("PROFILER  ·  top operators sorted by CPU time")
    print(sep)
    print(averages.table(sort_by="cpu_time_total", row_limit=row_limit))

    if ProfilerActivity.CUDA in prof.activities:
        print(f"\n{sep}")
        print("PROFILER  ·  top operators sorted by CUDA time")
        print(sep)
        print(averages.table(sort_by="cuda_time_total", row_limit=row_limit))

    try:
        print(f"\n{sep}")
        print("PROFILER  ·  top operators sorted by CPU memory usage")
        print(sep)
        print(
            averages.table(
                sort_by="self_cpu_memory_usage",
                row_limit=row_limit,
            )
        )
    except Exception:
        pass  # older torch builds may not support this sort key
