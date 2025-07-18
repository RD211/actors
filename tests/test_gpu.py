import pytest
import torch

# Mark all tests in this file as GPU-only
pytestmark = pytest.mark.gpu


@pytest.fixture
def check_gpu():
    """Check if GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    return True
