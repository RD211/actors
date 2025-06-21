import functools
import time
from contextlib import contextmanager
from typing import Callable, Optional

import torch
from .wandb import is_wandb_active
from .logger import get_logging_level

@contextmanager
def gpu_tracker(
    name: str,
    step: int,
    accel,
    log_to_wandb: bool,
    no_memory_measurement: bool = False,
    extra: Optional[dict] = None,
):
    device = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device)
    start_mem = torch.cuda.memory_allocated(device)
    start_time = time.perf_counter()

    yield  # <<< run the wrapped code >>>

    elapsed = time.perf_counter() - start_time
    peak_mem = torch.cuda.max_memory_allocated(device)

    # Log timing information in verbose mode
    if get_logging_level() == "verbose":
        from .logger import logger
        mem_info = f" | Peak mem: {peak_mem/1e6:.1f}MB" if not no_memory_measurement else ""
        logger.verbose(f"⏱️  {name}: {elapsed:.3f}s{mem_info}")

    # ----------------- WandB logging guard ---------------- #
    if (
        log_to_wandb
        and (accel is None or accel.is_main_process)
        and is_wandb_active()
    ):
        import wandb  # safe: we already know wandb exists

        wandb_log = {
            f"{name}/time_s": elapsed,
            "step": step,
        }

        if not no_memory_measurement:
            wandb_log.update(
                {
                    f"{name}/mem_peak_mb": peak_mem / 1e6,
                    f"{name}/mem_start_mb": start_mem / 1e6,
                    f"{name}/mem_diff_mb": (peak_mem - start_mem) / 1e6,
                }
            )

        if extra:
            wandb_log.update(extra)

        wandb.log(wandb_log, step=step)


def gpu_profiler(name: str | None = None, use_wandb: bool = True, no_memory_measurement: bool = False):

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            tag = name or func.__name__

            # user should set these attributes once in Trainer/Module
            step = getattr(self, "_step", 0)
            # Get accelerator if available, otherwise None
            accel = getattr(self, "accel", None)

            with gpu_tracker(
                tag,
                step,
                accel,
                log_to_wandb=use_wandb,
                no_memory_measurement=no_memory_measurement,
            ):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator