from __future__ import annotations
import os, traceback, torch
from multiprocessing import Process, Queue
from typing import Any, Dict, List
from vllm import LLM

from src.utils.shm_utils import load_shared_state_dict


def _v1_remote_load(worker_self, meta_blob: dict) -> None:
    state = load_shared_state_dict(meta_blob)
    worker_self.model_runner.model.load_weights(weights=state.items())


class ModelWorker(Process):
    """Runs one vLLM engine on the specified GPUs."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        gpu_ids: List[int],
        use_v1_engine: bool,
        engine_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(name=f"{model_name}-{gpu_ids}")
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.use_v1_engine = use_v1_engine
        self.engine_kwargs = engine_kwargs

        self.inbox: Queue = Queue()
        self.outbox: Queue = Queue()
        self.is_sleeping: bool = False
        self.sleep_level: int = 0

    # ---------------------------------------------------------------- run()
    def run(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        os.environ["VLLM_USE_V1"] = "1" if self.use_v1_engine else "0"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        try:
            engine = LLM(
                model=self.model_path,
                tensor_parallel_size=len(self.gpu_ids),
                trust_remote_code=True,
                enable_sleep_mode=True,
                **self.engine_kwargs,
            )
            self.outbox.put(("READY", self.gpu_ids))
        except Exception:
            self.outbox.put(("ERROR", traceback.format_exc()))
            return

        while True:
            msg_type, payload = self.inbox.get()
            if msg_type == "QUIT":
                break

            if msg_type == "SLEEP":
                level: int = payload
                if not self.is_sleeping:
                    engine.sleep(level=level)
                    self.is_sleeping, self.sleep_level = True, level
                self.outbox.put(("OK", None))
                continue

            if msg_type == "WAKE":
                if self.is_sleeping:
                    engine.wake_up()
                    self.is_sleeping, self.sleep_level = False, 0
                self.outbox.put(("OK", None))
                continue

            if msg_type == "UPDATE_WEIGHTS":
                meta_blob: Dict[str, Any] = payload
                try:
                    if self.use_v1_engine:
                        engine.collective_rpc(_v1_remote_load, args=(meta_blob,))
                    else:
                        weights = load_shared_state_dict(meta_blob)
                        engine.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
                            weights.items()
                        )
                    torch.cuda.empty_cache()
                    self.outbox.put(("OK", None))
                except Exception:
                    self.outbox.put(("ERROR", traceback.format_exc()))
                continue

            if msg_type in {"GENERATE", "CHAT"}:
                req_id, shard, sampling_dict = payload
                if self.is_sleeping:
                    self.outbox.put(("ERROR", f"asleep level {self.sleep_level}"))
                    continue
                if not shard:
                    self.outbox.put(("OK", req_id, []))
                    continue

                indices, inputs = zip(*shard)
                infer_fn = engine.generate if msg_type == "GENERATE" else engine.chat
                outputs = infer_fn(list(inputs), sampling_dict)
                self.outbox.put(("OK", req_id, list(zip(indices, outputs))))
