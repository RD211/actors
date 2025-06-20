import copy
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.accelerator import get_accelerator


def patch_deepspeed(config):
    if hasattr(config, "ds_config") \
            and "zero_optimization" in config.ds_config.keys() \
            and "offload_optimizer" in config.ds_config["zero_optimization"].keys() \
            and "pin_memory" in config.ds_config["zero_optimization"]["offload_optimizer"].keys() \
            and not config.ds_config["zero_optimization"]["offload_optimizer"]["pin_memory"]:
        get_accelerator().pin_memory = lambda x: x
    if hasattr(config, "ds_config") \
            and "zero_optimization" in config.ds_config.keys() \
            and "offload_param" in config.ds_config["zero_optimization"].keys() \
            and "pin_memory" in config.ds_config["zero_optimization"]["offload_param"].keys() \
            and not config.ds_config["zero_optimization"]["offload_param"]["pin_memory"]:
        get_accelerator().pin_memory = lambda x: x
    raw_init = copy.deepcopy(DeepSpeedZeroOptimizer.__init__)

    def safe_init(self, *args, **kwargs):
        while True:
            try:
                raw_init(self, *args, **kwargs)
                break
            except RuntimeError as e:
                continue

    DeepSpeedZeroOptimizer.__init__ = safe_init
    raw_initialize_optimizer_states = copy.deepcopy(DeepSpeedZeroOptimizer.initialize_optimizer_states)

    def safe_initialize_optimizer_states(self, *args, **kwargs):
        while True:
            try:
                raw_initialize_optimizer_states(self, *args, **kwargs)
                break
            except RuntimeError as e:
                continue

    DeepSpeedZeroOptimizer.initialize_optimizer_states = safe_initialize_optimizer_states
