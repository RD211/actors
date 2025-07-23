from __future__ import annotations

import datetime
import logging
import os
import re
import warnings

import colorama
import psutil
import pynvml
import torch

# ────────────────────────────── palette ────────────────────────────────
colorama.init(autoreset=True)
pynvml.nvmlInit()

# Custom logging levels
VERBOSE = 5  # Very detailed debug info with timing and extra details
NORMAL = 20  # Standard operation logs (same as INFO)
QUIET = 30  # Only important operations (same as WARNING)
SILENT = 50  # Only critical errors

# Add custom levels to logging module
logging.addLevelName(VERBOSE, "VERBOSE")
logging.addLevelName(NORMAL, "NORMAL")
logging.addLevelName(QUIET, "QUIET")
logging.addLevelName(SILENT, "SILENT")


class Palette:
    SUCCESS = colorama.Fore.GREEN + colorama.Style.BRIGHT
    WARNING = colorama.Fore.YELLOW + colorama.Style.BRIGHT
    ERROR = colorama.Fore.RED + colorama.Style.BRIGHT
    INFO = colorama.Fore.CYAN
    MUTED = colorama.Fore.BLUE
    RESET = colorama.Style.RESET_ALL
    BOLD = colorama.Style.BRIGHT
    CYAN = colorama.Fore.CYAN
    VERB = colorama.Fore.MAGENTA + colorama.Style.BRIGHT
    YELLOW = colorama.Fore.YELLOW + colorama.Style.BRIGHT


def colorize(text: str, style: str) -> str:
    """Wrap *text* in ANSI colour codes defined by *style*."""
    return f"{style}{text}{Palette.RESET}"


# ─────────────────────── GPU + RAM formatter ───────────────────────────
_LEVEL_COLOURS: dict[int, str] = {
    VERBOSE: colorama.Fore.LIGHTBLACK_EX,
    logging.DEBUG: colorama.Fore.CYAN,
    logging.INFO: colorama.Fore.GREEN,
    NORMAL: colorama.Fore.GREEN,
    logging.WARNING: colorama.Fore.YELLOW,
    QUIET: colorama.Fore.YELLOW,
    logging.ERROR: colorama.Fore.RED,
    logging.CRITICAL: colorama.Fore.MAGENTA,
    SILENT: colorama.Fore.MAGENTA,
}


class GPUFormatter(logging.Formatter):
    def __init__(self, *, show_rank: bool = False, show_date: bool = False) -> None:
        super().__init__()
        self.show_rank = show_rank
        self.show_date = show_date

        # Check if we should show system stats based on environment variable
        self.show_system_stats = (
            os.getenv("ACTORS_LOGGING_LEVEL", "").lower() == "verbose"
        )

    # ----------------------------------------- helpers
    @staticmethod
    def _gpu_summary() -> str:
        if not torch.cuda.is_available():
            return "GPU% N/A"
        percentages = []
        for idx in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            percentages.append(f"{int(mem.used / mem.total * 100)}%")
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

        # Only add GPU/RAM stats if environment flag is set
        if self.show_system_stats:
            parts += [self._gpu_summary(), self._ram_summary()]

        parts.append(record.getMessage())
        return " | ".join(parts)


# ───────────────────────────── convenience functions ─────────────────────────────
def verbose(logger: logging.Logger, message: str, *args, **kwargs):
    """Log a message with severity 'VERBOSE' (detailed debug info with timing)."""
    if logger.isEnabledFor(VERBOSE):
        logger._log(VERBOSE, message, args, **kwargs)


def normal(logger: logging.Logger, message: str, *args, **kwargs):
    """Log a message with severity 'NORMAL' (standard operation logs)."""
    if logger.isEnabledFor(NORMAL):
        logger._log(NORMAL, message, args, **kwargs)


def quiet(logger: logging.Logger, message: str, *args, **kwargs):
    """Log a message with severity 'QUIET' (only important operations)."""
    if logger.isEnabledFor(QUIET):
        logger._log(QUIET, message, args, **kwargs)


def silent(logger: logging.Logger, message: str, *args, **kwargs):
    """Log a message with severity 'SILENT' (only critical errors)."""
    if logger.isEnabledFor(SILENT):
        logger._log(SILENT, message, args, **kwargs)


# Add methods to Logger class
logging.Logger.verbose = lambda self, message, *args, **kwargs: verbose(
    self, message, *args, **kwargs
)
logging.Logger.normal = lambda self, message, *args, **kwargs: normal(
    self, message, *args, **kwargs
)
logging.Logger.quiet = lambda self, message, *args, **kwargs: quiet(
    self, message, *args, **kwargs
)
logging.Logger.silent = lambda self, message, *args, **kwargs: silent(
    self, message, *args, **kwargs
)


# ───────────────────────────── utility functions ─────────────────────────────
def should_use_tqdm() -> bool:
    """Check if tqdm should be enabled based on logging level."""
    return os.getenv("ACTORS_LOGGING_LEVEL", "normal").lower() == "verbose"


def get_logging_level() -> str:
    """Get the current logging level from environment."""
    return os.getenv("ACTORS_LOGGING_LEVEL", "normal").lower()


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

    Logging Levels:
    - VERBOSE (5): Very detailed debug info with timing, GPU/RAM stats, tqdm enabled
    - NORMAL (20): Standard operation logs including training metrics
    - QUIET (30): Only important operations (checkpoints, epochs, errors)
    - SILENT (50): Only critical errors

    Environment Variables:
    - ACTORS_LOGGING_LEVEL: "verbose", "normal" (default), "quiet", "silent"
    """
    # Override level based on environment variable if set
    env_level = os.getenv("ACTORS_LOGGING_LEVEL", "normal").lower()
    if env_level == "verbose":
        level = VERBOSE
        os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
    elif env_level == "normal":
        level = NORMAL
    elif env_level == "quiet":
        level = QUIET
        # Suppress vLLM logs in quiet mode
        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    elif env_level == "silent":
        level = SILENT
        # Suppress vLLM logs in silent mode
        os.environ["VLLM_LOGGING_LEVEL"] = "CRITICAL"
        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"

    # Mutes the logs when sleeping/waking up the model
    # Especially annoying since we offload and onload the model frequently
    if level > VERBOSE:
        for vllm_logger_name in (
            "vllm.executor.executor_base",
            "vllm.v1.core.block_pool",
            "vllm.v1.worker.gpu_worker",
        ):
            lg = logging.getLogger(vllm_logger_name)
            lg.setLevel(logging.WARNING)
            lg.propagate = False
            lg.handlers.clear()
            lg.addHandler(logging.NullHandler())

    # Mute warning about requires_grad.
    warnings.filterwarnings(
        action="ignore",
        message=re.escape(
            "None of the inputs have requires_grad=True. Gradients will be None"
        ),
        category=UserWarning,
        module=r"torch\.utils\.checkpoint",
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:  # avoid duplicate handlers in Jupyter etc.
        handler = logging.StreamHandler()
        handler.setFormatter(GPUFormatter(show_rank=show_rank, show_date=show_date))
        logger.addHandler(handler)

    logger.propagate = False

    # Only local main process should log
    if os.getenv("LOCAL_RANK", "0") != "0":
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger
    return logger


# ────────────────────────────── global logger ───────────────────────────
logger = init_logger(
    name="server",
    show_rank=False,
    show_date=False,
    level=NORMAL,
)
