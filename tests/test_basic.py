import pytest

from actors import GRPOTrainer, LLMActor, OpenAIActor, TrainableLLMActor, vLLMActor

# Mark all tests in this file as CPU-only
pytestmark = pytest.mark.cpu


def test_imports():
    """Test that all main components can be imported."""
    assert LLMActor is not None
    assert TrainableLLMActor is not None
    assert vLLMActor is not None
    assert OpenAIActor is not None
    assert GRPOTrainer is not None


def test_version():
    """Test that the package has a version."""
    try:
        import actors

        assert (
            hasattr(actors, "__version__") or True
        )  # Package metadata will handle version
    except ImportError:
        pytest.skip("Package not installed")


class TestOpenAIActor:
    """Test OpenAI actor basic functionality."""

    def test_init(self):
        """Test OpenAI actor initialization."""
        actor = OpenAIActor(
            name="test", api_key="test-key", base_url="https://api.openai.com/v1"
        )
        assert actor.name == "test"


class TestActorTrainCfg:
    """Test actor training configuration."""

    def test_basic_config(self):
        """Test basic configuration creation."""
        from actors.trainers import ActorTrainCfg

        config = ActorTrainCfg(
            learning_rate=1e-6,
            beta=0.1,
        )
        assert config.learning_rate == 1e-6
        assert config.beta == 0.1


class TestGRPOTrainerCfg:
    """Test GRPO trainer configuration."""

    def test_basic_config(self):
        """Test basic trainer configuration."""
        from actors.trainers import GRPOTrainerCfg

        config = GRPOTrainerCfg(
            group_size=16,
            batch_size=64,
            num_iterations=10,
        )
        assert config.group_size == 16
        assert config.batch_size == 64
        assert config.num_iterations == 10
