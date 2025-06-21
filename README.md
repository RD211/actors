# Actors

A sophisticated framework for training reinforcement learning from human feedback (RLHF) models with distributed training, advanced logging, and comprehensive checkpointing.

## Features

- **Advanced Training**: Complete training loop with epoch management, configurable max steps, and deterministic data loading
- **Intelligent Logging**: Multi-level logging system with GPU/RAM monitoring and vLLM integration
- **Robust Checkpointing**: Automatic checkpoint management with configurable frequency and rotation
- **WandB Integration**: Comprehensive metrics logging and environment completion tables
- **Distributed Support**: Full support for multi-GPU and multi-node training
- **Modern UI**: Clean, colorized logging with progress tracking and ETA

## Documentation

- [Logging System](LOGGING.md) - Comprehensive guide to the logging levels and configuration
- [Training Guide](docs/training.md) - Training loop configuration and usage (coming soon)
- [Checkpointing](docs/checkpointing.md) - Checkpoint management and recovery (coming soon)

## Quick Start

```python
from actors.trainers.trainer import Trainer

# Initialize trainer with your environment
trainer = Trainer(env, use_wandb=True, log_every_n=10)

# Train with automatic checkpointing and progress tracking
trainer.train(
    data=your_dataset,
    epochs=3,
    checkpoint_every_n=100,
    max_checkpoints_to_keep=5
)
```

## Logging Levels

Control logging verbosity with the `ACTORS_LOGGING_LEVEL` environment variable:

- `verbose`: Full debugging with GPU stats and timing
- `normal`: Standard training progress (default)
- `quiet`: Important operations only
- `silent`: Critical errors only

See [LOGGING.md](LOGGING.md) for complete details.
