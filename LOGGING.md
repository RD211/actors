# Logging System

The actors framework includes a sophisticated logging system with multiple levels and automatic vLLM integration.

## Logging Levels

Set the `ACTORS_LOGGING_LEVEL` environment variable to control logging verbosity:

### `verbose`
- **Purpose**: Maximum detail for debugging and development
- **Shows**: All messages, GPU/RAM stats, detailed timing information
- **Features**: 
  - Enables tqdm progress bars in vLLM operations
  - Shows GPU profiler timing for training operations
  - Displays GPU memory usage and RAM statistics
  - Shows detailed operation logs (model loading, weight updates, etc.)
  - Full vLLM logging enabled

### `normal` (default)
- **Purpose**: Standard operation logging for training and inference
- **Shows**: Training metrics, model operations, detailed step information
- **Features**:
  - Training step progress with ETA and metrics
  - Sampling and backprop iteration details
  - Model loading and unloading notifications
  - Standard vLLM logging
  - No GPU/RAM stats or detailed timing

### `quiet`
- **Purpose**: Minimal logging for production or long-running jobs
- **Shows**: Training metrics, checkpoints, epochs, and errors
- **Features**:
  - Training step progress with ETA and metrics
  - Checkpoint save operations
  - Epoch boundaries
  - Warnings and errors
  - Suppressed vLLM logs (ERROR level only)
  - No detailed step information or weight updates

### `silent`
- **Purpose**: Critical errors only
- **Shows**: Only critical system failures
- **Features**:
  - Suppresses all routine operations
  - Only shows critical errors that require immediate attention
  - Suppressed vLLM logs (ERROR level only)

## Usage Examples

```bash
# Development/debugging with full details
export ACTORS_LOGGING_LEVEL=verbose
python train.py

# Normal training with progress updates
export ACTORS_LOGGING_LEVEL=normal  # or omit for default
python train.py

# Production with minimal output
export ACTORS_LOGGING_LEVEL=quiet
python train.py

# Only critical errors
export ACTORS_LOGGING_LEVEL=silent
python train.py
```

## Programmatic Usage

```python
from actors.utils.logger import init_logger, should_use_tqdm, get_logging_level

# Create a logger
logger = init_logger("my_component")

# Use appropriate logging levels
logger.verbose("Detailed debug information")
logger.normal("Standard operation")
logger.quiet("Important checkpoint saved")
logger.error("Something went wrong")

# Check settings
if should_use_tqdm():
    # Enable progress bars
    pass

current_level = get_logging_level()  # Returns: "verbose", "normal", "quiet", "silent"
```

## Integration with Training

The trainer automatically uses appropriate logging levels:
- **Training metrics**: Shown in `quiet` and above (always visible except `silent`)
- **Step details**: Sampling and backprop info shown in `normal` and above
- **Checkpoint operations**: Shown in `quiet` and above
- **Weight updates**: Only shown in `verbose` mode
- **Detailed timing**: Only in `verbose` mode
- **GPU profiler**: Timing shown in `verbose` mode

## vLLM Integration

The logging system automatically configures vLLM logging:
- `verbose`: Full vLLM debug output with tqdm progress bars
- `normal`: Standard vLLM logging
- `quiet`/`silent`: Suppressed vLLM output (ERROR level only)

## Testing

Test the logging system with:

```bash
python test_logging_levels.py
ACTORS_LOGGING_LEVEL=verbose python test_logging_levels.py
ACTORS_LOGGING_LEVEL=quiet python test_logging_levels.py
ACTORS_LOGGING_LEVEL=silent python test_logging_levels.py
```
