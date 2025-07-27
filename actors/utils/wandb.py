def is_wandb_active() -> bool:
    try:
        import wandb

        return wandb.run is not None
    except ModuleNotFoundError:
        return False
