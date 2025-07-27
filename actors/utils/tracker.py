import functools
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager

import torch

from .logger import get_logging_level
from .wandb import is_wandb_active


class StepProfiler:
    """
    Accumulates timing and memory metrics across multiple operations within a single training step.
    Only logs to WandB once per step at the end, avoiding overwrites and memory stat resets.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all accumulated metrics for a new step."""
        self.metrics = defaultdict(dict)
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self.step_start_mem = None
        self.step_peak_mem = 0
        self.has_measurements = False

    def start_step(self):
        """Call at the beginning of each training step to reset memory tracking."""
        if self.device is not None:
            torch.cuda.reset_peak_memory_stats(self.device)
            self.step_start_mem = torch.cuda.memory_allocated(self.device)
            self.step_peak_mem = self.step_start_mem
        self.has_measurements = False

    @contextmanager
    def track(
        self,
        operation_name: str,
        no_memory_measurement: bool = False,
        actor_name: str = None,
    ):
        """
        Track timing and memory for a specific operation within the current step.
        Memory tracking shows actual GPU memory usage like nvidia-smi.
        """
        start_time = time.perf_counter()
        operation_start_mem = None

        if not no_memory_measurement and self.device is not None:
            # Clear cache before measuring to get accurate baseline
            import gc

            gc.collect()
            torch.cuda.empty_cache()

            operation_start_mem = torch.cuda.memory_allocated(self.device)
            # Reset peak stats just for this operation
            torch.cuda.reset_peak_memory_stats(self.device)

        yield

        elapsed = time.perf_counter() - start_time

        # Create unique key for this operation (with actor name if provided)
        if actor_name:
            metric_key = f"{operation_name}_{actor_name}"
        else:
            metric_key = operation_name

        self.metrics[metric_key]["time_s"] = elapsed

        if (
            not no_memory_measurement
            and self.device is not None
            and operation_start_mem is not None
        ):
            current_mem = torch.cuda.memory_allocated(self.device)
            peak_mem = torch.cuda.max_memory_allocated(self.device)

            mem_change_mb = (current_mem - operation_start_mem) / 1e6
            peak_usage_mb = peak_mem / 1e6

            self.metrics[metric_key]["mem_change_mb"] = mem_change_mb
            self.metrics[metric_key]["peak_memory_mb"] = peak_usage_mb

        self.has_measurements = True

        # Log timing information in verbose mode
        if get_logging_level() == "verbose":
            from .logger import logger

            mem_info = ""
            if not no_memory_measurement and self.device is not None:
                current_mem = torch.cuda.memory_allocated(self.device)
                mem_info = f" | GPU: {current_mem / 1e6:.1f}MB"
                if (
                    metric_key in self.metrics
                    and "mem_change_mb" in self.metrics[metric_key]
                ):
                    mem_change = self.metrics[metric_key]["mem_change_mb"]
                    mem_info += f" | Change: {mem_change:+.1f}MB"
            logger.verbose(f"⏱️  {metric_key}: {elapsed:.3f}s{mem_info}")

    def log_step_metrics(self, step: int, accel=None, use_wandb: bool = True):
        """
        Log all accumulated metrics for this step to WandB.
        Should be called once at the end of each training step.
        """
        if not self.has_measurements:
            return

        if use_wandb and (accel is None or accel.is_main_process) and is_wandb_active():
            import wandb

            from .logger import get_logging_level

            wandb_log = {}

            # Always log timing metrics
            for operation, metrics in self.metrics.items():
                if "time_s" in metrics:
                    wandb_log[f"Profile/{operation}/time_s"] = metrics["time_s"]

                # Only log memory metrics in verbose mode
                if get_logging_level() == "verbose":
                    for metric_name, value in metrics.items():
                        if metric_name != "time_s":  # Don't double-log timing
                            wandb_log[f"Profile/{operation}/{metric_name}"] = value

            if wandb_log:
                wandb.log(wandb_log, step=step)


# Global profiler instance that accumulates metrics per step
_step_profiler = StepProfiler()


@contextmanager
def gpu_tracker(
    name: str,
    step: int,
    accel,
    log_to_wandb: bool,
    no_memory_measurement: bool = False,
    extra: dict | None = None,
    actor_name: str = None,
):
    with _step_profiler.track(name, no_memory_measurement, actor_name):
        yield


def gpu_profiler(
    name: str | None = None, use_wandb: bool = True, no_memory_measurement: bool = False
):
    """
    Legacy decorator for backward compatibility.
    Now uses the new StepProfiler under the hood.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            tag = name or func.__name__

            with _step_profiler.track(tag, no_memory_measurement):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


def start_step_profiling():
    """Call at the beginning of each training step."""
    _step_profiler.start_step()


def log_step_profiling(step: int, accel=None, use_wandb: bool = True):
    """Call at the end of each training step to log all metrics."""
    _step_profiler.log_step_metrics(step, accel, use_wandb)


def reset_step_profiling():
    """Reset the profiler for a new step."""
    _step_profiler.reset()
