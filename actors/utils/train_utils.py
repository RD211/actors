import gc
import torch


def disable_dropout_in_model(model: torch.nn.Module) -> None:
  # taken from: https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L847
  for module in model.modules():
      if isinstance(module, torch.nn.Dropout):
          module.p = 0

def free_memory() -> None:
  torch.cuda.empty_cache()
  torch.cuda.ipc_collect()
  gc.collect()