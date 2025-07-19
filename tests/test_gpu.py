import pytest
import torch

# Mark all tests in this file as slow tests (GPU-intensive)
pytestmark = pytest.mark.slow


@pytest.fixture
def check_gpu():
    """Check if GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    return True
