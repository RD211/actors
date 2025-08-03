# ──────────────────────────────────────────────────
# Code to make it run with `python script.py` or in
# a notebook.
# ──────────────────────────────────────────────────

import functools
import os
import socket

import torch.distributed as dist


def free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


@functools.lru_cache(maxsize=1)
def init_distributed_one_gpu(backend: str = "nccl") -> None:
    if (
        not dist.is_available()
        or dist.is_initialized()
        or os.environ.get("RANK") is not None
        or os.environ.get("WORLD_SIZE") is not None
    ):
        return False
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", free_port())

    dist.init_process_group(backend=backend, rank=0, world_size=1)
    return True
