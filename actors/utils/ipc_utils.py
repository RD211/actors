from concurrent.futures import ThreadPoolExecutor

import deepspeed
import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import reduce_tensor
from vllm.platforms import current_platform


def _detach(named: tuple[str, torch.Tensor]):
    n, p = named
    return n.replace("module.", ""), p.detach()


def _tensors_to_ipc(tensors: dict[str, torch.Tensor], workers: int | None = None):
    def _one(item):
        name, t = item
        if t.device.type != "cuda":
            return None
        uuid = current_platform.get_device_uuid(t.device.index)
        return uuid, name, reduce_tensor(t)

    out: dict[str, dict[str, tuple]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_one, tensors.items()):
            if res is None:
                continue
            uuid, name, handle = res
            out.setdefault(uuid, {})[name] = handle
    return out


def gather_and_stream_state_dict(
    accelerator,
    logger,
    gpu_groups,
    model,
    callback,
    batch_size: int = 300,
    tie_word_embeddings: bool = False,
    lora_only: bool = False,
):
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed not initialised")

    # TODO: This is wrong on multi-node and will fail.
    g_rank = accelerator.process_index
    l_rank = accelerator.local_process_index  # noqa: F841
    local_world = torch.cuda.device_count()
    node_id = g_rank // local_world
    is_node_main = accelerator.is_local_main_process

    # ------------------------------------------------------------------ #
    # tensor‑parallel bookkeeping
    # ------------------------------------------------------------------ #
    rank_to_tp = {r: tp_id for tp_id, grp in enumerate(gpu_groups) for r in grp}
    if g_rank not in rank_to_tp:
        raise ValueError(f"Rank {g_rank} not present in gpu_groups")
    my_tp_id = rank_to_tp[g_rank]
    tp_members = gpu_groups[my_tp_id]
    tp_size = len(tp_members)
    tp_pos = tp_members.index(g_rank)
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # build parameter list
    # ------------------------------------------------------------------ #
    named = list(model.named_parameters())
    if lora_only:
        params = [(n, p) for n, p in named if "lora_A" in n or "lora_B" in n]
        if not params:
            logger.warning("No LoRA parameters found")
            return
    else:
        params = named
        if tie_word_embeddings:
            tied = next(p for n, p in named if "embed" in n)
            params.append(("lm_head.weight", tied))

    # ------------------------------------------------------------------ #
    # Streaming loop
    # ------------------------------------------------------------------ #

    for start in range(0, len(params), batch_size):
        chunk = params[start : start + batch_size]
        tensors = [p for _, p in chunk]

        with deepspeed.zero.GatheredParameters(tensors, modifier_rank=None):
            with ThreadPoolExecutor() as pool:
                detached = dict(pool.map(_detach, chunk))

            if not lora_only:
                idx0 = start
                detached = {
                    n: t
                    for i, (n, t) in enumerate(detached.items(), start=idx0)
                    if (i % tp_size) == tp_pos
                }

                if not detached:
                    accelerator.wait_for_everyone()
                    continue

            ipc_obj = _tensors_to_ipc(detached)

            gathered = accelerator.gather_for_metrics(
                [{"node_id": node_id, "tp_id": my_tp_id, "data": ipc_obj}]
            )

            if is_node_main:
                per_group = {}
                for obj in gathered:
                    if obj["node_id"] != node_id:  # ignore remotes
                        continue
                    tp = obj["tp_id"]
                    for uuid, d in obj["data"].items():
                        per_group.setdefault(tp, {}).setdefault(uuid, {}).update(d)
                for merged in per_group.values():
                    if merged:
                        callback(merged)

            del ipc_obj

        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
