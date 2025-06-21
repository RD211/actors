import deepspeed
import torch
from concurrent.futures import ThreadPoolExecutor
import logging

def gather_and_stream_state_dict(accelerator, logger: logging.Logger, model: torch.nn.Module, callback, batch_size: int=300):
    """
    Gathers the state dictionary from a model distributed with DeepSpeed ZeRO in batches,
    and calls a callback with each batch on the local main process.
    Tensors in the batch are on the GPU of the local main process.
    """
    
    # This will keep the tensor on its current device (GPU)
    def _copy_tensor(name_param):
        name, param = name_param
        return name, param.detach()
    
    params = list(model.named_parameters())
    total = len(params)

    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch = params[start:end]

        batch_params = [p for _, p in batch]
        
        # Gathers the full parameters on all processes, but we only process it on the local main process.
        with deepspeed.zero.GatheredParameters(batch_params, modifier_rank=None):
            if accelerator.is_local_main_process:
                batch_state_dict = {}
                with ThreadPoolExecutor() as executor:
                    futures = executor.map(_copy_tensor, batch)
                    for name, tensor in futures:
                        batch_state_dict[name] = tensor
                
                # The model on accelerator might have a "module." prefix.
                if any(name.startswith("module.") for name in batch_state_dict.keys()):
                    batch_state_dict = {name.replace("module.", "", 1): tensor for name, tensor in batch_state_dict.items()}

                callback(batch_state_dict)
            
        torch.cuda.empty_cache()
    
    accelerator.wait_for_everyone()
