from __future__ import annotations

import asyncio
import atexit
import hashlib
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
class SharedModelConfig:
    """Configuration for a shared base model that can serve multiple LoRA adapters."""

    model_path: str
    is_v1: bool
    gpu_groups: list[list[int]]
    base_kwargs: dict[str, Any]  # Engine kwargs without LoRA-specific settings
    base_model_id: str  # Unique identifier for this base model configuration
    max_lora_rank: int  # Capacity of the base for LoRA rank

    def __post_init__(self):
        # Ensure consistent ordering of GPU groups for hashing
        self.gpu_groups = [sorted(group) for group in self.gpu_groups]
        self.gpu_groups.sort()


@dataclass
class LoRAAdapterInfo:
    """Information about a LoRA adapter associated with an actor."""

    actor_name: str
    lora_path: str | None
    lora_rank: int
    max_loras: int
    adapter_id: int  # Unique ID for this adapter within the shared model


@dataclass
class ModelRecord:
    name: str
    path: str
    is_v1: bool
    gpu_groups: list[list[int]]
    kwargs: dict[str, Any]
    workers: list[Any] = field(default_factory=list)
    stats: ModelStats = field(default_factory=ModelStats)
    shared_config: SharedModelConfig | None = None
    lora_adapters: dict[str, LoRAAdapterInfo] = field(
        default_factory=dict
    )  # actor_name -> adapter_info
    is_shared: bool = False


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
        self.shared_models: dict[
            str, ModelRecord
        ] = {}  # base_model_id -> shared model record
        self.next_adapter_id = 1  # Global counter for adapter IDs
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

    def _generate_base_model_id(
        self,
        model_path: str,
        gpu_groups: list[list[int]],
        use_v1_engine: bool,
        base_kwargs: dict[str, Any],
        max_lora_rank: int,
    ) -> str:
        """Generate a unique identifier for a base model configuration."""
        # Normalize GPU groups for consistent hashing
        normalized_groups = [sorted(group) for group in gpu_groups]
        normalized_groups.sort()

        # Create a canonical representation for hashing
        config_data = {
            "model_path": model_path,
            "gpu_groups": normalized_groups,
            "use_v1_engine": use_v1_engine,
            "base_kwargs": sorted(base_kwargs.items()),
            "max_lora_rank": int(max_lora_rank),
        }

        # Generate hash
        config_str = str(config_data)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _extract_base_kwargs(self, engine_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract base engine kwargs, excluding LoRA-specific settings."""
        lora_specific_keys = {
            "enable_lora",
            "max_lora_rank",
            "max_loras",
            "lora_extra_vocab_size",
            "lora_vocab_padding_size",
            "allow_sharing",
            "expected_max_lora_rank",
        }
        return {k: v for k, v in engine_kwargs.items() if k not in lora_specific_keys}

    def _can_share_base_model(
        self,
        model_path: str,
        gpu_groups: list[list[int]],
        use_v1_engine: bool,
        engine_kwargs: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """
        Determine if a model can share a base model with existing models.

        Returns:
            (can_share, base_model_id): If can_share is True, base_model_id is the ID
            of the existing shared model. If False, base_model_id is None.
        """
        # Only share models that have LoRA enabled
        if not engine_kwargs.get("enable_lora", False):
            return False, None
        # Respect explicit opt-out
        if engine_kwargs.get("allow_sharing") is False:
            return False, None

        # Capacity required for this actor
        actor_required_rank = int(engine_kwargs.get("max_lora_rank", 16))

        base_kwargs = self._extract_base_kwargs(engine_kwargs)
        # Base id must include capacity to avoid padding overhead across incompatible ranks
        base_model_id = self._generate_base_model_id(
            model_path, gpu_groups, use_v1_engine, base_kwargs, actor_required_rank
        )

        # If we already have a base with IDENTICAL capacity, we can share directly
        if base_model_id in self.shared_models:
            return True, base_model_id

        # Otherwise, check if there exists a base with GREATER OR EQUAL capacity that matches base_kwargs and gpu settings
        # This allows merging smaller-rank actors into a larger-capacity base without reloading.
        for existing_id, rec in self.shared_models.items():
            if (
                rec.path == model_path
                and rec.is_v1 == use_v1_engine
                and sorted([sorted(g) for g in rec.gpu_groups])
                == sorted([sorted(g) for g in gpu_groups])
                and sorted(rec.shared_config.base_kwargs.items())
                == sorted(base_kwargs.items())
                and rec.shared_config.max_lora_rank >= actor_required_rank
            ):
                return True, existing_id

        return False, base_model_id

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

        # Check if we can share a base model
        can_share, base_model_id = self._can_share_base_model(
            model_path, gpu_groups, use_v1_engine, engine_kwargs
        )

        if can_share and base_model_id in self.shared_models:
            # Reuse existing shared model (must satisfy capacity)
            self._add_actor_to_shared_model(name, base_model_id, engine_kwargs)
        else:
            # Load new model (either standalone or new shared model)
            self._load_new_model(
                name,
                model_path,
                gpu_groups,
                use_v1_engine,
                engine_kwargs,
                base_model_id,
            )

    def _sanitize_engine_kwargs_for_worker(
        self, engine_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Remove pool-only control keys from engine kwargs before constructing workers."""
        sanitized = dict(engine_kwargs)
        for k in ("allow_sharing", "expected_max_lora_rank"):
            sanitized.pop(k, None)
        return sanitized

    def _add_actor_to_shared_model(
        self, name: str, base_model_id: str, engine_kwargs: dict[str, Any]
    ) -> None:
        """Add a new actor to an existing shared model."""
        shared_record = self.shared_models[base_model_id]

        # Enforce adapter capacity
        max_loras_cap = int(shared_record.kwargs.get("max_loras", 1))
        current_loras = len(shared_record.lora_adapters)
        if current_loras >= max_loras_cap:
            raise RuntimeError(
                f"Shared base {base_model_id} is at capacity: max_loras={max_loras_cap}. "
                f"Either increase capacity on the first actor that creates the base, "
                f"or set allow_sharing=False for this actor, or start a new base with a higher expected_max_lora_rank/max_loras."
            )

        # Create LoRA adapter info for this actor
        # Note: The actual LoRA rank will be set when the LoRA adapter is created
        # We use a default value here that will be updated later
        adapter_info = LoRAAdapterInfo(
            actor_name=name,
            lora_path=None,  # Will be set when LoRA is created
            lora_rank=16,  # Default, will be updated when LoRA is created
            max_loras=engine_kwargs.get("max_loras", 1),
            adapter_id=self.next_adapter_id,
        )
        self.next_adapter_id += 1

        # Add adapter to shared model
        shared_record.lora_adapters[name] = adapter_info

        # Create a reference in models dict that points to shared model
        self.models[name] = ModelRecord(
            name=name,
            path=shared_record.path,
            is_v1=shared_record.is_v1,
            gpu_groups=shared_record.gpu_groups,
            kwargs=engine_kwargs,
            workers=shared_record.workers,  # Share the same workers
            stats=ModelStats(),  # Individual stats per actor
            shared_config=shared_record.shared_config,
            lora_adapters={name: adapter_info},
            is_shared=True,
        )

        logger.normal(
            colorize(
                f"✅  Added {name} to shared model {base_model_id} (adapter_id={adapter_info.adapter_id})",
                Palette.SUCCESS,
            )
        )

    def _load_new_model(
        self,
        name: str,
        model_path: str,
        gpu_groups: list[list[int]],
        use_v1_engine: bool,
        engine_kwargs: dict[str, Any],
        base_model_id: str | None,
    ) -> None:
        """Load a new model (either standalone or the first instance of a shared model)."""
        # Determine if this should be a shared model
        should_be_shared = (
            engine_kwargs.get("enable_lora", False)
            and base_model_id is not None
            and engine_kwargs.get("allow_sharing", True)
        )

        if should_be_shared:
            # Ensure LoRA parameters are set for shared models
            final_engine_kwargs = engine_kwargs.copy()
            if "max_loras" not in final_engine_kwargs:
                final_engine_kwargs["max_loras"] = 8  # Allow multiple LoRA adapters
            # For shared base, set capacity to the requested capacity of the FIRST actor
            capacity = int(final_engine_kwargs.get("max_lora_rank", 16))
            final_engine_kwargs["max_lora_rank"] = capacity
            logger.normal(
                colorize(
                    f"⏳  Loading shared base model {name} on {len(gpu_groups)} GPU groups (capacity r={capacity})",
                    Palette.INFO,
                )
            )
        else:
            final_engine_kwargs = engine_kwargs
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
                    }
                },
            ).remote(
                name,
                model_path,
                grp,
                use_v1_engine,
                self._sanitize_engine_kwargs_for_worker(final_engine_kwargs),
            )
            for grp in gpu_groups
        ]

        # wait for readiness
        ray.get([w.ready.remote() for w in workers])

        load_time = time.monotonic() - start

        if should_be_shared:
            # Create shared model configuration
            base_kwargs = self._extract_base_kwargs(engine_kwargs)
            capacity = int(final_engine_kwargs.get("max_lora_rank", 16))
            shared_config = SharedModelConfig(
                model_path=model_path,
                is_v1=use_v1_engine,
                gpu_groups=gpu_groups,
                base_kwargs=base_kwargs,
                base_model_id=base_model_id,
                max_lora_rank=capacity,
            )

            # Create LoRA adapter info for this actor
            # Note: The actual LoRA rank will be set when the LoRA adapter is created
            adapter_info = LoRAAdapterInfo(
                actor_name=name,
                lora_path=None,
                lora_rank=16,  # Default, will be updated when LoRA is created
                max_loras=engine_kwargs.get("max_loras", 1),
                adapter_id=self.next_adapter_id,
            )
            self.next_adapter_id += 1

            # Create shared model record
            shared_record = ModelRecord(
                name=f"shared_{base_model_id}",
                path=model_path,
                is_v1=use_v1_engine,
                gpu_groups=gpu_groups,
                kwargs=final_engine_kwargs,
                workers=workers,
                shared_config=shared_config,
                lora_adapters={name: adapter_info},
                is_shared=True,
            )

            # Store in shared models registry
            self.shared_models[base_model_id] = shared_record

            # Create individual model record pointing to shared model
            self.models[name] = ModelRecord(
                name=name,
                path=model_path,
                is_v1=use_v1_engine,
                gpu_groups=gpu_groups,
                kwargs=engine_kwargs,
                workers=workers,
                stats=ModelStats(),
                shared_config=shared_config,
                lora_adapters={name: adapter_info},
                is_shared=True,
            )

            logger.normal(
                colorize(
                    f"✅  Ready shared model {name} (base_id={base_model_id}, capacity r={capacity}) in {load_time:.1f}s",
                    Palette.SUCCESS,
                )
            )
        else:
            # Create standalone model record
            self.models[name] = ModelRecord(
                name,
                model_path,
                use_v1_engine,
                gpu_groups,
                final_engine_kwargs,
                workers,
            )
            logger.normal(
                colorize(f"✅  Ready {name} in {load_time:.1f}s", Palette.SUCCESS)
            )

    def unload_model(self, name: str) -> None:
        rec = self.models.pop(name, None)
        if not rec:
            raise RuntimeError("no such model")

        try:
            if rec.is_shared and rec.shared_config:
                # Handle shared model unloading
                base_model_id = rec.shared_config.base_model_id
                shared_record = self.shared_models.get(base_model_id)

                if shared_record:
                    # Remove this actor from the shared model
                    shared_record.lora_adapters.pop(name, None)

                    # If no more actors are using this shared model, unload it completely
                    if not shared_record.lora_adapters:
                        for w in shared_record.workers:
                            try:
                                ray.kill(w)
                            except Exception:
                                # Ignore individual worker kill errors
                                pass
                        self.shared_models.pop(base_model_id, None)
                        logger.quiet(
                            colorize(
                                f"✖️   Unloaded shared model {base_model_id}",
                                Palette.WARNING,
                            )
                        )
                    else:
                        logger.quiet(
                            colorize(
                                f"✖️   Removed {name} from shared model {base_model_id}",
                                Palette.WARNING,
                            )
                        )
                else:
                    logger.quiet(
                        colorize(
                            f"✖️   Unloaded {name} (shared model not found)",
                            Palette.WARNING,
                        )
                    )
            else:
                # Handle standalone model unloading
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

        # Get the adapter info for this actor
        adapter_info = rec.lora_adapters.get(name)
        if not adapter_info:
            raise RuntimeError(f"No LoRA adapter info found for actor {name}")

        # Update the LoRA path in adapter info
        adapter_info.lora_path = lora_path

        # If this is a shared model, update the shared record as well
        if rec.is_shared and rec.shared_config:
            shared_record = self.shared_models[rec.shared_config.base_model_id]
            shared_record.lora_adapters[name].lora_path = lora_path

        # Create LoRA with proper adapter ID
        results = ray.get(
            [
                w.create_lora_if_not_present.remote(lora_path, adapter_info.adapter_id)
                for w in rec.workers
            ]
        )
        for status, msg in results:
            if status == "ERROR":
                raise RuntimeError(msg)
        logger.normal(
            colorize(
                f"🔧 Created LoRA for {name} (adapter_id={adapter_info.adapter_id})",
                Palette.SUCCESS,
            )
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
            import os

            from vllm.lora.request import LoRARequest

            # Get the adapter info for this actor
            adapter_info = rec.lora_adapters.get(name)
            if (
                adapter_info
                and adapter_info.lora_path
                and os.path.exists(adapter_info.lora_path)
            ):
                # Only create LoRA request if the path exists
                lora_request = LoRARequest(
                    lora_name=f"lora_{name}",
                    lora_int_id=adapter_info.adapter_id,
                    lora_local_path=adapter_info.lora_path,
                )
            else:
                # No LoRA request if adapter doesn't exist yet
                lora_request = None
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
            import os

            from vllm.lora.request import LoRARequest

            # Get the adapter info for this actor
            adapter_info = rec.lora_adapters.get(name)
            if (
                adapter_info
                and adapter_info.lora_path
                and os.path.exists(adapter_info.lora_path)
            ):
                # Only create LoRA request if the path exists
                lora_request = LoRARequest(
                    lora_name=f"lora_{name}",
                    lora_int_id=adapter_info.adapter_id,
                    lora_local_path=adapter_info.lora_path,
                )
            else:
                # No LoRA request if adapter doesn't exist yet
                lora_request = None
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
