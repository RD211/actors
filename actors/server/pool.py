from __future__ import annotations
import math, time, threading, uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from vllm import RequestOutput, SamplingParams

from actors.utils.logger import Palette, colorize, logger
from actors.server.worker import ModelWorker


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
    gpu_groups: List[List[int]]
    kwargs: Dict[str, Any]
    workers: List[ModelWorker] = field(default_factory=list)
    stats: ModelStats = field(default_factory=ModelStats)


class ModelPool:
    """RPC façade combining many ModelWorker processes."""

    DASH_INTERVAL = 60  # seconds

    _singleton: "ModelPool" | None = None
    _lock = threading.Lock()            # guard first construction

    DASH_INTERVAL = 60  # seconds

    # ---------- Pythonic singleton hook ----------------------------
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._singleton is None:
                cls._singleton = super().__new__(cls)
        return cls._singleton

    # ---------- normal initialisation (runs once) ------------------
    _init_done = False
    def __init__(self) -> None:
        if self._init_done:   # prevent second-pass re-initialisation
            return
        self.total_gpus = torch.cuda.device_count()
        self.models: Dict[str, ModelRecord] = {}
        threading.Thread(target=self._dashboard_loop, daemon=True).start()
        self._init_done = True
    # ------------- dashboard --------------------------------------
    def _render_dashboard(self) -> str:
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
        separator    = "  ".join("-" * w for w in col_w)

        prettified = [header_line, separator]

        for idx, row in enumerate(rows[1:], 1):
            shade = Palette.INFO if idx % 2 else Palette.MUTED
            prettified.append(pad(row, shade))

        return "\n".join(prettified)


    def _dashboard_loop(self) -> None:
        while True:
            time.sleep(self.DASH_INTERVAL)
            logger.info("\n" + self._render_dashboard())

    # ------------- RPC helpers ------------------------------------
    def list_models(self) -> List[str]:
        return list(self.models)

    def print_models(self) -> str:
        board = self._render_dashboard()
        logger.info("\n" + board)
        return board

    # ------------- model lifecycle --------------------------------
    def load_model(
        self,
        *,
        name: str,
        model_path: str,
        gpu_groups: List[List[int]] | int | None = None,
        use_v1_engine: bool = True,
        engine_kwargs: Dict[str, Any] | None = None,
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

        logger.info(colorize(f"⏳  Loading {name}", Palette.INFO))
        start = time.monotonic()
        workers = [
            ModelWorker(name, model_path, grp, use_v1_engine, engine_kwargs)
            for grp in gpu_groups
        ]
        for w in workers:
            w.start()

        # wait for readiness
        for w in workers:
            status, *detail = w.outbox.get()
            if status != "READY":
                raise RuntimeError(detail[0])
        logger.info(
            colorize(
                f"✅  Ready {name} in {time.monotonic()-start:.1f}s", Palette.SUCCESS
            )
        )

        self.models[name] = ModelRecord(
            name, model_path, use_v1_engine, gpu_groups, engine_kwargs, workers
        )

    def unload_model(self, name: str) -> None:
        rec = self.models.pop(name, None)
        if not rec:
            raise RuntimeError("no such model")
        for w in rec.workers:
            w.inbox.put(("QUIT", None))
            w.join()
        logger.warning(colorize(f"✖️   Unloaded {name}", Palette.WARNING))

    # ------------- sleep / wake -----------------------------------
    def sleep(self, name: str, level: int = 1) -> None:
        rec = self.models[name]
        for w in rec.workers:
            w.inbox.put(("SLEEP", level))
            w.outbox.get()
        logger.warning(colorize(f"🛌  Sleep {name}", Palette.WARNING))

    def wake(self, name: str) -> None:
        rec = self.models[name]
        for w in rec.workers:
            w.inbox.put(("WAKE", None))
            w.outbox.get()
        logger.warning(colorize(f"⚡  Wake {name}", Palette.WARNING))

    # ------------- weights swap -----------------------------------
    def update_weights(self, name: str, meta_blob: dict) -> None:
        rec = self.models[name]
        logger.info(colorize(f"♻️  Weights {name}", Palette.INFO))

        # We make sure the model is not sleeping
        self.wake(name)

        for w in rec.workers:
            w.inbox.put(("UPDATE_WEIGHTS", meta_blob))
            status, *msg = w.outbox.get()
            if status == "ERROR":
                raise RuntimeError(msg[0])
        logger.info(colorize(f"✅  Finished updating {name} weights", Palette.SUCCESS))

    # ------------- inference internal -----------------------------
    @staticmethod
    def _scatter(batch: List[Any], workers: List[ModelWorker]) -> List[list]:
        shards: List[list] = [[] for _ in workers]
        for idx, item in enumerate(batch):
            shards[idx % len(workers)].append((idx, item))
        return shards

    @staticmethod
    def _gather(shards: List[list]) -> List[Any]:
        flat = [p for shard in shards for p in shard]
        flat.sort(key=lambda pair: pair[0])
        return [output for _, output in flat]

    def _infer(
        self,
        name: str,
        payload: List[Any],
        sampling_params: SamplingParams,
        msg_type: str,
    ) -> List[RequestOutput]:
        rec = self.models[name]
        shards = self._scatter(payload, rec.workers)

        request_id = uuid.uuid4().hex
        start = time.monotonic()

        for worker, shard in zip(rec.workers, shards):
            worker.inbox.put((msg_type, (request_id, shard, sampling_params)))

        collected: List[list] = []
        for worker in rec.workers:
            status, *detail = worker.outbox.get()
            if status == "ERROR":
                raise RuntimeError(detail[0])
            _, shard = detail
            collected.append(shard)


        duration = time.monotonic() - start
        merged = self._gather(collected)
        produced_tokens = sum(len(o.outputs[0].token_ids) for o in merged)

        rec.stats.request_count += len(payload)
        rec.stats.token_count += produced_tokens
        rec.stats.elapsed += duration

        logger.info(
            colorize(
                f"📝 {msg_type} {len(payload)} requests {produced_tokens} tokens generated in {duration:.2f}s",
                Palette.INFO,
            )
        )
        return merged

    # ---------- RPC methods for clients ----------------------------
    def generate(
        self, name: str, prompts: List[str], sampling_params: SamplingParams
    ) -> List[RequestOutput]:
        return self._infer(name, prompts, sampling_params, "GENERATE")

    def chat(
        self, name: str, dialogs: List[list], sampling_params: SamplingParams
    ) -> List[RequestOutput]:
        return self._infer(name, dialogs, sampling_params, "CHAT")
