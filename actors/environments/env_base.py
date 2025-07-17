from __future__ import annotations
import asyncio
import torch
from torch.utils.data import DataLoader, RandomSampler

import abc
from typing import Dict, Union, List, Any, Optional

from actors.actors.base import TrainableLLMActor
    
from actors.environments.types import EnvironmentOutput, GroupedEnvironmentOutput
from datasets import Dataset as HFDataset, DatasetDict



class Environment(abc.ABC):

    def __init__(
        self,
        train_data: Optional[Union[HFDataset, DatasetDict]] = None,
        eval_data: Optional[Union[HFDataset, DatasetDict, Dict[str, Union[HFDataset, DatasetDict]]]] = None,
    ) -> None:
        self._reg: Dict[str, TrainableLLMActor] = {}
        
        self.train_data = self._normalise_hf_splits(train_data) if train_data is not None else None
        
        if eval_data is not None:
            if isinstance(eval_data, dict):
                self.eval_datasets = {
                    name: self._normalise_hf_splits(data)
                    for name, data in eval_data.items()
                }
            else:
                self.eval_datasets = {"eval": self._normalise_hf_splits(eval_data)}
        else:
            self.eval_datasets = {}
        
        self._data_state = {"epoch": 0, "step_in_epoch": 0, "current_generator_seed": 0}
        self._rng = torch.Generator()
        self._dataloader = None

    @staticmethod
    def _normalise_hf_splits(
        data: Union[HFDataset, DatasetDict],
    ):
        if isinstance(data, DatasetDict):
            return data.get("train", next(iter(data.values())))
        return data

    def _build_dataloader(self, batch_size: int):
        """Create a DataLoader with the current RNG seed & generator state."""
        if self.train_data is None:
            return
            
        self._rng.manual_seed(self._data_state["current_generator_seed"])
        sampler = RandomSampler(self.train_data, generator=self._rng)

        def collate_fn(batch):
            if not batch:
                return {}
            keys = batch[0].keys()
            return {k: [d[k] for d in batch] for k in keys}

        self._dataloader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def get_dataloader(self, batch_size: int):
        """Get the current dataloader, building it if necessary."""
        if self._dataloader is None:
            self._build_dataloader(batch_size)
        return self._dataloader

    def get_data_state(self):
        """Get the current data state for checkpointing."""
        return self._data_state.copy()

    def set_data_state(self, state: Dict[str, Any], batch_size: int):
        """Set data state for resuming."""
        self._data_state = state.copy()
        if "rng_state" in state:
            self._rng.set_state(state["rng_state"])
        self._build_dataloader(batch_size)

    def get_rng_state(self):
        """Get RNG state for checkpointing."""
        return self._rng.get_state()

    def set_rng_state(self, state):
        """Set RNG state for resuming."""
        self._rng.set_state(state)

    def skip_to_step(self, target_step: int, batch_size: int):
        """Skip ahead to a specific step by advancing the data state."""
        if self.train_data is None:
            return
        
        # We need to build the dataloader first to know steps per epoch
        if self._dataloader is None:
            self._build_dataloader(batch_size)
            
        steps_per_epoch = len(self._dataloader)
        target_epoch = target_step // steps_per_epoch
        target_step_in_epoch = target_step % steps_per_epoch
        
        self._data_state.update(
            epoch=target_epoch,
            step_in_epoch=target_step_in_epoch,
            current_generator_seed=self._data_state["current_generator_seed"] + target_epoch,
        )
        self._build_dataloader(batch_size)

    def advance_epoch(self, batch_size: int):
        """Advance to the next epoch."""
        self._data_state.update(
            epoch=self._data_state["epoch"] + 1,
            step_in_epoch=0,
            current_generator_seed=self._data_state["current_generator_seed"] + 1,
        )
        self._build_dataloader(batch_size)

    def advance_step(self):
        """Advance to the next step within the current epoch."""
        self._data_state["step_in_epoch"] += 1

    def batches_left(self, batch_size: int) -> int:
        """Return the number of batches left in the current epoch."""
        if self._dataloader is None:
            self._build_dataloader(batch_size)
        if self.train_data is None:
            return 0
        
        steps_per_epoch = len(self._dataloader)
        current_step = self._data_state["step_in_epoch"]
        return max(0, steps_per_epoch - current_step)

    def get_next_batch(self, batch_size: int) -> Optional[Dict[str, List[Any]]]:
        """Get the next batch from the dataloader, handling edge cases."""
        if self.train_data is None:
            return None
        
        # Build dataloader if needed
        if self._dataloader is None or self._data_state["step_in_epoch"] == 0:
            self._build_dataloader(batch_size)
        
        dataloader_iter = iter(self._dataloader)
        
        # Skip to current position
        for _ in range(self._data_state["step_in_epoch"]):
            try:
                next(dataloader_iter)
            except StopIteration:
                return None
        
        # Get the next batch
        try:
            batch = next(dataloader_iter)
            self.advance_step()
            return batch
        except StopIteration:
            # Automatically advance to next epoch when we run out of batches
            self._data_state.update(
                epoch=self._data_state["epoch"] + 1,
                step_in_epoch=0,
                current_generator_seed=self._data_state["current_generator_seed"] + 1,
            )
            self._build_dataloader(batch_size)
            
            # Try to get batch from new epoch
            try:
                dataloader_iter = iter(self._dataloader)
                batch = next(dataloader_iter)
                self.advance_step()
                return batch
            except StopIteration:
                return None

    def expand_batch_for_groups(self, batch: Dict[str, List[Any]], group_size: int) -> Dict[str, List[Any]]:
        """Expand a batch by duplicating each item group_size times."""
        if not isinstance(batch, dict):
            raise ValueError("batch must be a dictionary")
        
        expanded = {}
        for k, v in batch.items():
            if isinstance(v, list):
                expanded[k] = [item for item in v for _ in range(group_size)]
            else:
                expanded[k] = v
        
        return expanded

    def register(self, actor: TrainableLLMActor) -> None:
        if actor.name in self._reg:
            raise ValueError(f"duplicate actor {actor.name}")
        if actor.training_config is None:
            raise ValueError(f"actor {actor.name} has no training config. Please set training_config before registering.")
        self._reg[actor.name] = actor


    # ------------------------------------------------------------------
    def get_trainable_actors(self) -> Dict[str, TrainableLLMActor]:
        return self._reg
    # ------------------------------------------------------------------
    
    def __call__(
        self, 
        batch_size: int, 
        group_size: int = 1
    ) -> GroupedEnvironmentOutput:
        """
        Get a batch from the data and run generation.
        
        Args:
            batch_size: Number of problems to include in batch
            group_size: Number of generations per problem
            
        Returns:
            GroupedEnvironmentOutput
        """
        # Get the next batch from data
        raw_batch = self.get_next_batch(batch_size)
        if raw_batch is None:
            raise StopIteration("No more batches available")
        
        # Expand batch for groups
        expanded_batch = self.expand_batch_for_groups(raw_batch, group_size)
        
        # Run generation
        env_output = asyncio.run(self.generate(expanded_batch))
        
        # Convert expanded batch back to original format for GroupedEnvironmentOutput
        original_batch = []
        first_key = next(iter(raw_batch.keys()))
        for i in range(len(raw_batch[first_key])):
            item = {k: v[i] for k, v in raw_batch.items()}
            original_batch.append(item)
        
        return GroupedEnvironmentOutput.from_environment_output(
            env_output, 
            original_batch,
            group_size
        )
    
    def eval(self, group_size: int = 1) -> Dict[str, GroupedEnvironmentOutput]:
        """
        Run evaluation on all eval datasets.
        
        Args:
            group_size: Number of generations per problem
            
        Returns:
            Dictionary mapping dataset names to their GroupedEnvironmentOutput
        """
        if not self.eval_datasets:
            return {}
        
        results = {}
        for eval_name, eval_data in self.eval_datasets.items():
            # Convert eval dataset to batch format
            eval_batch = {key: eval_data[key] for key in eval_data.column_names}
            
            # Always expand batch for groups (even if group_size=1)
            expanded_batch = self.expand_batch_for_groups(eval_batch, group_size)
            env_output = asyncio.run(self.generate(expanded_batch))
            
            # Convert to original format for GroupedEnvironmentOutput
            original_batch = []
            first_key = next(iter(eval_batch.keys()))
            for i in range(len(eval_batch[first_key])):
                item = {k: v[i] for k, v in eval_batch.items()}
                original_batch.append(item)
            
            results[eval_name] = GroupedEnvironmentOutput.from_environment_output(
                env_output,
                original_batch,
                group_size
            )
        
        return results
    
    @abc.abstractmethod
    async def generate(self, batch) -> EnvironmentOutput:
        """
        Generate outputs for a batch of inputs.
        
        Args:
            batch: Dictionary mapping column names to lists of values
            
        Returns:
            EnvironmentOutput containing actor outputs
        """
