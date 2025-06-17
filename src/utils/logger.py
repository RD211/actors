from __future__ import annotations

import datetime
import logging
import os
from typing import Dict

import colorama
import psutil
import pynvml
import torch

# ────────────────────────────── palette ────────────────────────────────
colorama.init(autoreset=True)
pynvml.nvmlInit()


class Palette:
    SUCCESS = colorama.Fore.GREEN + colorama.Style.BRIGHT
    WARNING = colorama.Fore.YELLOW + colorama.Style.BRIGHT
    ERROR = colorama.Fore.RED + colorama.Style.BRIGHT
    INFO = colorama.Fore.CYAN
    MUTED = colorama.Fore.BLUE
    RESET = colorama.Style.RESET_ALL


def colorize(text: str, style: str) -> str:
    """Wrap *text* in ANSI colour codes defined by *style*."""
    return f"{style}{text}{Palette.RESET}"


# ─────────────────────── GPU + RAM formatter ───────────────────────────
_LEVEL_COLOURS: Dict[int, str] = {
    logging.DEBUG: colorama.Fore.CYAN,
    logging.INFO: colorama.Fore.GREEN,
    logging.WARNING: colorama.Fore.YELLOW,
    logging.ERROR: colorama.Fore.RED,
    logging.CRITICAL: colorama.Fore.MAGENTA,
}


class GPUFormatter(logging.Formatter):
    def __init__(self, *, show_rank: bool = False, show_date: bool = False) -> None:
        super().__init__()
        self.show_rank = show_rank
        self.show_date = show_date

    # ----------------------------------------- helpers
    @staticmethod
    def _gpu_summary() -> str:
        if not torch.cuda.is_available():
            return "GPU% N/A"
        percentages = []
        for idx in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            percentages.append(f"{idx}:{int(mem.used / mem.total * 100)}%")
        return "GPU% [" + " ".join(percentages) + "]"

    @staticmethod
    def _ram_summary() -> str:
        vm = psutil.virtual_memory()
        used = vm.used / 1024**3
        total = vm.total / 1024**3
        return f"RAM {used:.1f}/{total:.0f} GB ({vm.percent}%)"

    # ----------------------------------------- main format
    def format(self, record: logging.LogRecord) -> str:
        level_colour = _LEVEL_COLOURS.get(record.levelno, "")
        timestamp = datetime.datetime.now().strftime(
            "%H:%M:%S" if not self.show_date else "%Y-%m-%d %H:%M:%S"
        )

        parts = [level_colour + timestamp + Palette.RESET]
        if self.show_rank:
            parts.append(f"rk:{os.getenv('RANK', 0)}")

        parts += [self._gpu_summary(), self._ram_summary(), record.getMessage()]
        return " | ".join(parts)


# ───────────────────────────── init_logger ─────────────────────────────
def init_logger(
    name: str = "app",
    *,
    show_rank: bool = False,
    show_date: bool = False,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a `logging.Logger` that prints coloured timestamps + GPU/RAM stats.

    Parameters
    ----------
    name:
        Logger name.
    show_rank:
        If True, include `RANK` env-var in each line (useful for DDP).
    show_date:
        If True, show full date; otherwise time only.
    level:
        Logging level threshold.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:  # avoid duplicate handlers in Jupyter etc.
        handler = logging.StreamHandler()
        handler.setFormatter(GPUFormatter(show_rank=show_rank, show_date=show_date))
        logger.addHandler(handler)

    logger.propagate = False
    return logger

# ────────────────────────────── global logger ───────────────────────────
logger = init_logger(
    name="server",
    show_rank=False,
    show_date=False,
    level=logging.INFO,
)
