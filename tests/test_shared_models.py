"""
Test suite for shared model functionality with LoRA actors.

This module tests that vLLM actors can share base models when they have
compatible configurations and use LoRA adapters, reducing VRAM and RAM usage.
"""

import pytest
import torch
from peft import LoraConfig, TaskType

from actors import ActorTrainCfg, vLLMActor
from actors.inference.pool import ModelPool


@pytest.fixture(scope="session", autouse=True)
def check_cuda():
    """Skip all tests if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for vLLM tests", allow_module_level=True)


@pytest.fixture(scope="function")
def clean_model_pool():
    """Ensure clean model pool state for each test."""
    pool = ModelPool()

    # Store original state
    original_models = dict(pool.models)
    original_shared_models = dict(pool.shared_models)
    original_next_adapter_id = pool.next_adapter_id

    # Clear for clean test
    pool.models.clear()
    pool.shared_models.clear()
    pool.next_adapter_id = 1

    yield pool

    # Cleanup after test
    try:
        # Unload any models created during test
        for model_name in list(pool.models.keys()):
            try:
                pool.unload_model(model_name)
            except Exception:
                pass  # Ignore cleanup errors

        # Restore original state
        pool.models = original_models
        pool.shared_models = original_shared_models
        pool.next_adapter_id = original_next_adapter_id
    except Exception:
        pass  # Ignore cleanup errors


def create_lora_config(rank=16, alpha=32, target_modules=None):
    """Create a LoRA configuration for testing."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
    )


def create_test_actor(name, model_path, engine_kwargs=None, lora_config=None):
    """Helper to create a test actor with LoRA configuration."""
    if engine_kwargs is None:
        engine_kwargs = {
            "gpu_memory_utilization": 0.2,
            "max_model_len": 512,
        }

    if lora_config is None:
        lora_config = create_lora_config()

    training_config = ActorTrainCfg(
        learning_rate=1e-5,
        peft_config=lora_config,
    )

    return vLLMActor(
        name=name,
        model_path=model_path,
        engine_kwargs=engine_kwargs,
        training_config=training_config,
        allow_sharing=True,
        expected_max_lora_rank=512,
    )


class TestSharedModels:
    """Test cases for shared model functionality."""

    def test_two_actors_share_same_base_model(self, clean_model_pool):
        pool = clean_model_pool
        model_path = "HuggingFaceTB/SmolLM2-135M-Instruct"

        # Create identical base configuration
        engine_kwargs = {
            "gpu_memory_utilization": 0.2,
            "max_model_len": 512,
        }

        # Create first actor
        create_test_actor("Actor1", model_path, engine_kwargs)

        # Verify initial state
        assert len(pool.models) == 1
        assert "Actor1" in pool.models

        # Create second actor with same configuration
        create_test_actor("Actor2", model_path, engine_kwargs)

        # Verify both actors are loaded
        assert len(pool.models) == 2
        assert "Actor1" in pool.models
        assert "Actor2" in pool.models

        # Check that models are shared
        actor1_record = pool.models["Actor1"]
        actor2_record = pool.models["Actor2"]

        assert actor1_record.is_shared, "Actor1 should be marked as shared"
        assert actor2_record.is_shared, "Actor2 should be marked as shared"

        # They should share the same base model ID
        assert actor1_record.shared_config is not None
        assert actor2_record.shared_config is not None
        assert (
            actor1_record.shared_config.base_model_id
            == actor2_record.shared_config.base_model_id
        ), "Actors should have same base model ID"

        # They should share the same worker instances (same memory)
        assert id(actor1_record.workers) == id(
            actor2_record.workers
        ), "Actors should share the same worker instances"

        # Verify shared model registry
        assert len(pool.shared_models) == 1
        base_model_id = actor1_record.shared_config.base_model_id
        assert base_model_id in pool.shared_models

        shared_record = pool.shared_models[base_model_id]
        assert len(shared_record.lora_adapters) == 2
        assert "Actor1" in shared_record.lora_adapters
        assert "Actor2" in shared_record.lora_adapters

        # Verify different adapter IDs
        adapter1 = shared_record.lora_adapters["Actor1"]
        adapter2 = shared_record.lora_adapters["Actor2"]
        assert (
            adapter1.adapter_id != adapter2.adapter_id
        ), "Actors should have different adapter IDs"

        # Cleanup
        pool.unload_model("Actor1")
        pool.unload_model("Actor2")

    def test_different_configs_dont_share(self, clean_model_pool):
        pool = clean_model_pool
        model_path = "HuggingFaceTB/SmolLM2-135M-Instruct"

        # Create actors with different engine kwargs
        engine_kwargs1 = {
            "gpu_memory_utilization": 0.2,
            "max_model_len": 512,
        }

        engine_kwargs2 = {
            "gpu_memory_utilization": 0.3,  # Different value
            "max_model_len": 512,
        }

        create_test_actor("Actor1", model_path, engine_kwargs1)
        create_test_actor("Actor2", model_path, engine_kwargs2)

        # Both should be shared models but with different base IDs
        actor1_record = pool.models["Actor1"]
        actor2_record = pool.models["Actor2"]

        assert actor1_record.is_shared
        assert actor2_record.is_shared

        # They should have different base model IDs
        assert (
            actor1_record.shared_config.base_model_id
            != actor2_record.shared_config.base_model_id
        ), "Actors with different configs should have different base model IDs"

        # They should NOT share worker instances
        assert id(actor1_record.workers) != id(
            actor2_record.workers
        ), "Actors with different configs should not share worker instances"

        # Should have two separate shared models
        assert len(pool.shared_models) == 2

        # Cleanup
        pool.unload_model("Actor1")
        pool.unload_model("Actor2")

    def test_non_lora_models_dont_share(self, clean_model_pool):
        pool = clean_model_pool
        model_path = "HuggingFaceTB/SmolLM2-135M-Instruct"

        engine_kwargs = {
            "gpu_memory_utilization": 0.2,
            "max_model_len": 512,
        }

        # Create actor without LoRA (no peft_config)
        vLLMActor(
            name="Actor1",
            model_path=model_path,
            engine_kwargs=engine_kwargs,
            allow_sharing=True,
            # No training_config with peft_config
        )

        # Create actor with LoRA
        create_test_actor("Actor2", model_path, engine_kwargs)

        actor1_record = pool.models["Actor1"]
        actor2_record = pool.models["Actor2"]

        # Non-LoRA actor should not be shared
        assert not actor1_record.is_shared, "Non-LoRA actor should not be shared"
        assert actor1_record.shared_config is None

        # LoRA actor should be shared (but alone)
        assert actor2_record.is_shared, "LoRA actor should be marked as shared"
        assert actor2_record.shared_config is not None

        # Should have one shared model (for Actor2 only)
        assert len(pool.shared_models) == 1

        # Cleanup
        pool.unload_model("Actor1")
        pool.unload_model("Actor2")

    def test_shared_model_cleanup(self, clean_model_pool):
        pool = clean_model_pool
        model_path = "HuggingFaceTB/SmolLM2-135M-Instruct"

        engine_kwargs = {
            "gpu_memory_utilization": 0.2,
            "max_model_len": 512,
        }

        # Create two actors sharing a model
        create_test_actor("Actor1", model_path, engine_kwargs)
        create_test_actor("Actor2", model_path, engine_kwargs)

        # Verify sharing
        assert len(pool.shared_models) == 1
        base_model_id = pool.models["Actor1"].shared_config.base_model_id
        assert len(pool.shared_models[base_model_id].lora_adapters) == 2

        # Unload first actor
        pool.unload_model("Actor1")

        # Shared model should still exist (Actor2 still using it)
        assert len(pool.shared_models) == 1
        assert len(pool.shared_models[base_model_id].lora_adapters) == 1
        assert "Actor2" in pool.shared_models[base_model_id].lora_adapters
        assert "Actor1" not in pool.shared_models[base_model_id].lora_adapters

        # Unload second actor
        pool.unload_model("Actor2")

        # Shared model should be completely removed
        assert len(pool.shared_models) == 0
        assert base_model_id not in pool.shared_models

    def test_adapter_id_uniqueness(self, clean_model_pool):
        """Test that adapter IDs are unique across all shared models."""
        pool = clean_model_pool
        model_path = "HuggingFaceTB/SmolLM2-135M-Instruct"

        engine_kwargs = {
            "gpu_memory_utilization": 0.2,
            "max_model_len": 512,
        }

        # Create multiple actors
        actors = []
        for i in range(3):
            actor = create_test_actor(f"Actor{i + 1}", model_path, engine_kwargs)
            actors.append(actor)

        # Get all adapter IDs
        adapter_ids = []
        for i in range(3):
            record = pool.models[f"Actor{i + 1}"]
            adapter_info = record.lora_adapters[f"Actor{i + 1}"]
            adapter_ids.append(adapter_info.adapter_id)

        # All adapter IDs should be unique
        assert len(set(adapter_ids)) == len(
            adapter_ids
        ), f"Adapter IDs should be unique: {adapter_ids}"

        # Adapter IDs should be sequential
        adapter_ids.sort()
        expected_ids = list(range(1, len(adapter_ids) + 1))
        assert (
            adapter_ids == expected_ids
        ), f"Adapter IDs should be sequential: {adapter_ids} vs {expected_ids}"

        # Cleanup
        for i in range(3):
            pool.unload_model(f"Actor{i + 1}")


class TestSharedModelsIntegration:
    """Integration tests for shared model functionality."""

    def test_collaborative_actors_sharing(self, clean_model_pool):
        pool = clean_model_pool
        model_path = "HuggingFaceTB/SmolLM2-135M-Instruct"

        # Simulate collaborative training setup
        base_engine_kwargs = {
            "gpu_memory_utilization": 0.3,
            "max_model_len": 1024,
        }

        # Create multiple collaborative actors
        actors = []
        for i in range(3):
            lora_config = create_lora_config(
                rank=2 ** (i + 4),
                alpha=2 ** (i + 5),
            )

            actor = create_test_actor(
                f"Collaborator{i + 1}", model_path, base_engine_kwargs, lora_config
            )
            actors.append(actor)

        # All should share the same base model
        base_model_ids = []
        for i in range(3):
            record = pool.models[f"Collaborator{i + 1}"]
            assert record.is_shared
            base_model_ids.append(record.shared_config.base_model_id)

        # All should have the same base model ID
        assert (
            len(set(base_model_ids)) == 1
        ), "All collaborators should share the same base model"

        # Should have exactly one shared model with 3 adapters
        assert len(pool.shared_models) == 1
        shared_record = list(pool.shared_models.values())[0]
        assert len(shared_record.lora_adapters) == 3

        # Cleanup
        for i in range(3):
            pool.unload_model(f"Collaborator{i + 1}")


def test_memory_usage_reduction():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    pool = ModelPool()
    model_path = "HuggingFaceTB/SmolLM2-135M-Instruct"

    try:
        # Create two actors that should share memory
        create_test_actor("MemTest1", model_path)
        create_test_actor("MemTest2", model_path)

        # Verify they're sharing workers
        record1 = pool.models["MemTest1"]
        record2 = pool.models["MemTest2"]

        assert record1.is_shared and record2.is_shared

        assert id(record1.workers) == id(
            record2.workers
        ), "Shared models should use the same worker instances (memory)"

    finally:
        # Cleanup
        try:
            pool.unload_model("MemTest1")
            pool.unload_model("MemTest2")
        except Exception:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
