from __future__ import annotations

import asyncio
import atexit
import math
import os
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import ray
import torch
from vllm import RequestOutput, SamplingParams

from actors.inference.worker import DEFAULT_LORA, ModelWorker
from actors.utils.logger import Palette, colorize, logger

# ═══════════════════════════════════════════════════════════════════════
# Helpers & Types
# ═══════════════════════════════════════════════════════════════════════


def is_local_main() -> bool:
    for key in ("LOCAL_RANK", "LOCAL_PROCESS_INDEX", "ACCELERATE_LOCAL_PROCESS_INDEX"):
        if key in os.environ:
            return int(os.environ[key]) == 0
    return True


def main_process_only(return_value=None):
    """
    Decorator that turns any method into a no-op on non-local-main processes.
    """

    def _decorator(fn):
        @wraps(fn)
        def _wrapper(self, *args, **kwargs):
            if self._disabled:
                return return_value
            return fn(self, *args, **kwargs)

        return _wrapper

    return _decorator


@dataclass
class ModelStats:
    request_count: int = 0
    token_count: int = 0
    elapsed: float = 0.0

    @property
    def tps(self) -> float:
        return self.token_count / self.elapsed if self.elapsed else 0.0


@dataclass
class ModelRecord:
    name: str
    path: str
    is_v1: bool
    gpu_groups: list[list[int]]
    kwargs: dict[str, Any]
    workers: list[Any] = field(default_factory=list)
    stats: ModelStats = field(default_factory=ModelStats)


# ═══════════════════════════════════════════════════════════════════════
# ModelPool Class
# ═══════════════════════════════════════════════════════════════════════


class ModelPool:
    """RPC façade combining many ModelWorker processes."""

    _singleton: ModelPool | None = None
    _lock = threading.Lock()  # guard first construction

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._singleton is None:
                cls._singleton = super().__new__(cls)
        return cls._singleton

    _init_done = False

    def __init__(self) -> None:
        if self._init_done:  # prevent second-pass re-initialisation
            return
        self._disabled = not is_local_main()
        if self._disabled:
            return

        self.total_gpus = torch.cuda.device_count()
        self.models: dict[str, ModelRecord] = {}
        os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        ray.init(ignore_reinit_error=True)
        self._init_done = True
        # Register cleanup on exit
        atexit.register(self._cleanup_on_exit)

    def _cleanup_on_exit(self):
        """Clean up all models and Ray resources on program exit."""
        try:
            # Unload all models
            model_names = list(self.models.keys())
            for name in model_names:
                try:
                    self.unload_model(name)
                except Exception:
                    # Silently ignore individual model cleanup errors
                    pass

            # Shutdown Ray
            try:
                ray.shutdown()
            except Exception:
                # Silently ignore Ray shutdown errors
                pass
        except Exception:
            # Silently ignore all cleanup errors to prevent segfaults
            pass

    def list_models(self) -> list[str]:
        return list(self.models)

    def print_models(self) -> str:
        if not self.models:
            return "(no models loaded)"

        header = ["NAME", "GPUs", "REQ", "TOK", "TOK/s"]
        rows = [header]

        for rec in self.models.values():
            st = rec.stats
            rows.append(
                [
                    rec.name,
                    ",".join(map(str, rec.gpu_groups)),
                    str(st.request_count),
                    str(st.token_count),
                    f"{st.tps:,.0f}",
                ]
            )

        col_w = [max(len(r[i]) for r in rows) for i in range(len(header))]

        def pad(row, shade=""):
            return (
                shade
                + "  ".join(cell.ljust(col_w[i]) for i, cell in enumerate(row))
                + Palette.RESET
            )

        header_line = colorize(pad(header), Palette.SUCCESS)
        separator = "  ".join("-" * w for w in col_w)

        lines = [header_line, separator]

        for idx, row in enumerate(rows[1:], 1):
            shade = Palette.INFO if idx % 2 else Palette.MUTED
            lines.append(pad(row, shade))

        table = "\n".join(lines)
        logger.verbose("\n" + table)
        return table

    # ------------- model lifecycle --------------------------------
    def load_model(
        self,
        *,
        name: str,
        model_path: str,
        gpu_groups: list[list[int]] | int | None = None,
        use_v1_engine: bool = True,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if name in self.models:
            raise RuntimeError("Model already loaded")
        engine_kwargs = engine_kwargs or {}

        # compute groups
        if gpu_groups is None:
            gpu_groups = [[gid] for gid in range(self.total_gpus)]
        elif isinstance(gpu_groups, int):
            ids = list(range(self.total_gpus))
            size = math.ceil(len(ids) / gpu_groups)
            gpu_groups = [ids[i * size : (i + 1) * size] for i in range(gpu_groups)]

        logger.normal(
            colorize(
                f"⏳  Loading {name} on {len(gpu_groups)} GPU groups", Palette.INFO
            )
        )
        start = time.monotonic()
        workers = [
            ModelWorker.options(
                num_gpus=0,
                runtime_env={
                    "env_vars": {
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        "CUDA_VISIBLE_DEVICES": ",".join(map(str, grp)),
                        # All environment variables that we have.
                        # **os.environ,
                    }
                },
            ).remote(name, model_path, grp, use_v1_engine, engine_kwargs)
            for grp in gpu_groups
        ]

        # wait for readiness
        ray.get([w.ready.remote() for w in workers])
        logger.normal(
            colorize(
                f"✅  Ready {name} in {time.monotonic() - start:.1f}s", Palette.SUCCESS
            )
        )

        self.models[name] = ModelRecord(
            name, model_path, use_v1_engine, gpu_groups, engine_kwargs, workers
        )

    def unload_model(self, name: str) -> None:
        rec = self.models.pop(name, None)
        if not rec:
            raise RuntimeError("no such model")
        try:
            for w in rec.workers:
                try:
                    ray.kill(w)
                except Exception:
                    # Ignore individual worker kill errors
                    pass
            logger.quiet(colorize(f"✖️   Unloaded {name}", Palette.WARNING))
        except Exception:
            # Ignore overall unload errors but still log success
            logger.quiet(
                colorize(f"✖️   Unloaded {name} (with warnings)", Palette.WARNING)
            )

    # ------------- sleep / wake -----------------------------------
    def sleep(self, name: str, level: int = 1) -> None:
        rec = self.models[name]
        ray.get([w.sleep.remote(level) for w in rec.workers])
        logger.verbose(colorize(f"🛌  Sleep {name}", Palette.WARNING))

    def wake(self, name: str) -> None:
        rec = self.models[name]
        ray.get([w.wake.remote() for w in rec.workers])
        logger.verbose(colorize(f"⚡  Wake {name}", Palette.WARNING))

    # ------------- weights swap -----------------------------------
    def start_update(self, name: str) -> None:
        rec = self.models[name]
        # Remove the "Starting weight update" log - it's too verbose
        ray.get([w.start_update.remote() for w in rec.workers])

    def update_weights_batch(self, name: str, ipc_handles_batch: dict) -> None:
        rec = self.models[name]
        results = ray.get(
            [w.update_weights_batch.remote(ipc_handles_batch) for w in rec.workers]
        )
        for status, msg in results:
            if status == "ERROR":
                raise RuntimeError(msg)

    def finalize_update(self, name: str) -> None:
        rec = self.models[name]
        results = ray.get([w.finalize_update.remote() for w in rec.workers])
        for status, msg in results:
            if status == "ERROR":
                raise RuntimeError(msg)
        logger.normal(colorize(f"✅  Updated {name} weights", Palette.SUCCESS))

    # ------------- LoRA methods -----------------------------------

    def update_lora_weights(self, name: str) -> None:
        rec = self.models[name]
        results = ray.get([w.update_lora_weights.remote() for w in rec.workers])
        for status, msg in results:
            if status == "ERROR":
                raise RuntimeError(msg)
        logger.normal(colorize(f"✅ Updated LoRA weights for {name}", Palette.SUCCESS))

    def create_lora_if_not_present(self, name: str, lora_path: str) -> None:
        """Create and initialize LoRA adapter if not already present."""
        rec = self.models[name]
        results = ray.get(
            [w.create_lora_if_not_present.remote(lora_path) for w in rec.workers]
        )
        for status, msg in results:
            if status == "ERROR":
                raise RuntimeError(msg)
        logger.normal(
            colorize(f"🔧 Created/initialized LoRA for {name}", Palette.SUCCESS)
        )

    # ------------- inference internal -----------------------------
    @staticmethod
    def _scatter(batch: list[Any], workers: list[Any]) -> list[list]:
        shards: list[list] = [[] for _ in workers]
        for idx, item in enumerate(batch):
            shards[idx % len(workers)].append((idx, item))
        return shards

    @staticmethod
    def _gather(shards: list[list]) -> list[Any]:
        flat = [p for shard in shards for p in shard]
        flat.sort(key=lambda pair: pair[0])
        return [output for _, output in flat]

    def _infer(
        self,
        name: str,
        payload: list[Any],
        sampling_params: SamplingParams,
        msg_type: str,
        lora_request=DEFAULT_LORA,
    ) -> list[RequestOutput]:
        rec = self.models[name]

        # Handle LoRA request logic at pool level
        # Convert DEFAULT_LORA to actual LoRARequest if model has LoRA enabled
        if lora_request is DEFAULT_LORA and rec.kwargs.get("enable_lora", False):
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest(
                lora_name="lora", lora_int_id=1, lora_local_path="placeholder"
            )
        if lora_request is DEFAULT_LORA:
            lora_request = None

        shards = self._scatter(payload, rec.workers)

        start = time.monotonic()

        futures = []
        infer_fn_name = "generate" if msg_type == "GENERATE" else "chat"
        for worker, shard in zip(rec.workers, shards, strict=False):
            if shard:
                futures.append(
                    getattr(worker, infer_fn_name).remote(
                        shard, sampling_params, lora_request
                    )
                )

        collected: list[list] = ray.get(futures)

        duration = time.monotonic() - start
        merged = self._gather(collected)
        if not merged:
            return []
        produced_tokens = sum(len(o.outputs[0].token_ids) for o in merged)

        rec.stats.request_count += len(payload)
        rec.stats.token_count += produced_tokens
        rec.stats.elapsed += duration

        logger.normal(
            colorize(
                f"📝 {msg_type} {len(payload)} requests {produced_tokens} tokens generated in {duration:.2f}s",
                Palette.INFO,
            )
        )
        return merged

    async def _ainfer(
        self,
        name: str,
        payload: list[Any],
        sampling_params: SamplingParams,
        msg_type: str,
        lora_request=DEFAULT_LORA,
    ) -> list[RequestOutput]:
        rec = self.models[name]

        # Handle LoRA request logic at pool level
        # Convert DEFAULT_LORA to actual LoRARequest if model has LoRA enabled
        if lora_request is DEFAULT_LORA and rec.kwargs.get("enable_lora", False):
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest(
                lora_name="lora", lora_int_id=1, lora_local_path="placeholder"
            )
        if lora_request is DEFAULT_LORA:
            lora_request = None

        shards = self._scatter(payload, rec.workers)

        start = time.monotonic()

        futures = []
        infer_fn_name = "generate" if msg_type == "GENERATE" else "chat"
        for worker, shard in zip(rec.workers, shards, strict=False):
            if shard:
                futures.append(
                    getattr(worker, infer_fn_name).remote(
                        shard, sampling_params, lora_request
                    )
                )

        # Use asyncio.to_thread to run ray.get in a thread pool
        collected: list[list] = await asyncio.to_thread(ray.get, futures)

        duration = time.monotonic() - start
        merged = self._gather(collected)
        if not merged:
            return []
        produced_tokens = sum(len(o.outputs[0].token_ids) for o in merged)

        rec.stats.request_count += len(payload)
        rec.stats.token_count += produced_tokens
        rec.stats.elapsed += duration

        logger.normal(
            colorize(
                f"📝 async {msg_type} {len(payload)} requests {produced_tokens} tokens generated in {duration:.2f}s",
                Palette.INFO,
            )
        )
        return merged

    # ---------- RPC methods for clients ----------------------------
    def generate(
        self,
        name: str,
        prompts: list[str],
        sampling_params: SamplingParams,
        lora_request=DEFAULT_LORA,
    ) -> list[RequestOutput]:
        return self._infer(name, prompts, sampling_params, "GENERATE", lora_request)

    def chat(
        self,
        name: str,
        dialogs: list[list],
        sampling_params: SamplingParams,
        lora_request=DEFAULT_LORA,
    ) -> list[RequestOutput]:
        return self._infer(name, dialogs, sampling_params, "CHAT", lora_request)

    async def agenerate(
        self,
        name: str,
        prompts: list[str],
        sampling_params: SamplingParams,
        lora_request=DEFAULT_LORA,
    ) -> list[RequestOutput]:
        return await self._ainfer(
            name, prompts, sampling_params, "GENERATE", lora_request
        )

    async def achat(
        self,
        name: str,
        dialogs: list[list],
        sampling_params: SamplingParams,
        lora_request=DEFAULT_LORA,
    ) -> list[RequestOutput]:
        return await self._ainfer(name, dialogs, sampling_params, "CHAT", lora_request)


for name, member in list(ModelPool.__dict__.items()):
    if (
        callable(member)
        and not name.startswith("_")  # public API only
        and not isinstance(member, property)
    ):
        setattr(ModelPool, name, main_process_only(None)(member))
