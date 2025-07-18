"""
Actors: A reinforcement learning library for training and deploying LLM actors.

This library provides tools for training Large Language Models using reinforcement
learning techniques, with support for GRPO (Group Relative Policy Optimization),
vLLM integration, and PEFT/LoRA fine-tuning.
"""

__version__ = "0.1.0"

from .actors import LLMActor, OpenAIActor, TrainableLLMActor, vLLMActor
from .trainers import GRPOTrainer

__all__ = [
    "__version__",
    "LLMActor",
    "OpenAIActor",
    "TrainableLLMActor",
    "vLLMActor",
    "GRPOTrainer",
]
