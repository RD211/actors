import gc
import deepspeed
import torch
import random
import numpy as np
import ctypes
from multiprocessing import shared_memory
from concurrent.futures import ThreadPoolExecutor


def get_shareable_version(meta):
    return {
        key: {k: v for k, v in meta[key].items() if k != "_shm_obj"} for key in meta
    }


def create_shared_state_dict(state_dict, max_workers=200):

    def make_shm(key, tensor):
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        tensor = tensor.contiguous()

        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        nbytes = tensor.numel() * tensor.element_size()

        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        buf = shm.buf

        if dtype == torch.bfloat16:
            # one raw-bytes copy from the tensors data_ptr
            ptr = tensor.data_ptr()
            raw = ctypes.string_at(ptr, nbytes)
            buf[:nbytes] = raw
        else:
            # fallback via numpy
            np_arr = tensor.numpy()
            shm_arr = np.ndarray(shape, dtype=np_arr.dtype, buffer=buf)
            shm_arr[:] = np_arr

        del tensor
        if random.random() < 0.02:
            torch.cuda.empty_cache()
            gc.collect()

        return key, {
            "shm_name": shm.name,
            "shape": shape,
            "dtype": str(dtype),
            "nbytes": nbytes,
            "_shm_obj": shm,
        }

    shared_meta = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(make_shm, k, v) for k, v in state_dict.items()]
        for f in futures:
            k, m = f.result()
            shared_meta[k] = m
    return shared_meta


def load_shared_state_dict(meta):

    def load_shared_tensor(key, info):
        shm = shared_memory.SharedMemory(name=info["shm_name"])
        shape = tuple(info["shape"])
        dtype = info["dtype"]
        nbytes = info["nbytes"]
        try:
            if "bfloat16" in dtype or dtype.endswith("bf16"):
                # copy raw bytes out via a Python bytes object
                buf = shm.buf
                raw = buf[:nbytes].tobytes()
                tensor = torch.empty(shape, dtype=torch.bfloat16)
                ctypes.memmove(tensor.data_ptr(), raw, nbytes)
            else:
                buf = shm.buf
                np_arr = np.ndarray(shape, dtype=np.dtype(dtype), buffer=buf)
                tensor = torch.tensor(np_arr)
                del np_arr
            if random.random() < 0.1:
                gc.collect()
            return key, tensor
        finally:
            shm.close()
            shm.unlink()

    state_dict = {}
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [
            executor.submit(load_shared_tensor, k, info) for k, info in meta.items()
        ]
        for f in futures:
            k, t = f.result()
            state_dict[k] = t
    return state_dict

def gather_incremental_state_dict_to_cpu(accelerator, logger, model, batch_size=120):
    state_dict = {}
    
    def _copy_to_cpu(name_param):
        name, param = name_param
        return name, param.detach().cpu()
    
    params = list(model.named_parameters())
    total = len(params)
    logger.info(f"Gathering {total} parameters in batches of {batch_size}")

    #TODO: Could technically be improved by for each GPU gathering a part and then putting on shared memory and then
    # gathering the shared memory on the CPU.
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch = params[start:end]

        # Create a list of just the parameters in this batch.
        batch_params = [p for _, p in batch]
        
        # Use DeepSpeed's context manager to gather the partitioned parameters.
        with deepspeed.zero.GatheredParameters(batch_params, modifier_rank=None):
            logger.info(f"Gathering parameters {start} to {end}")
            if accelerator.is_local_main_process:
                with ThreadPoolExecutor() as executor:
                    # schedule all of the copies in parallel
                    futures = executor.map(_copy_to_cpu, batch)
                    for name, cpu_tensor in futures:
                        state_dict[name] = cpu_tensor
            
        torch.cuda.empty_cache()
        logger.info(f"Processed parameters {start} to {end}")
    
    # If there is a prefix of module. we remove it
    if any(name.startswith("module.") for name in state_dict.keys()):
        state_dict = {name.replace("module.", "", 1): tensor for name, tensor in state_dict.items()}
    return state_dict