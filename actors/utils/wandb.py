def is_wandb_active() -> bool:
    """
    True  → wandb is importable *and* wandb.init() has already been called.
    False → otherwise (package missing, or no active run yet).
    """
    try:
        import wandb

        return wandb.run is not None
    except ModuleNotFoundError:
        return False
