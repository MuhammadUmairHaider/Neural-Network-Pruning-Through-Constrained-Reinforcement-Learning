import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import gym
import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces

from pruning.envs.base_class import OfflinePruningBaseClass
from pruning.envs.utils import InfiniteBatchSampler
from pruning.networks.create_network import get_network
from pruning.networks.masked_modules import MaskedConv2d, MaskedLinear
from pruning.prune import get_pruning_method


class NetworkWise(OfflinePruningBaseClass):
    def __init__(
            self,
            network: str,
            dataset: str,
            device: str,
            _seed: int = 718,
            batch_size: int = 128,
            finetune_iters: int = None,
            finetune_batch_size: int = None,
            optimizer: Optional[str] = None,
            lr: Optional[float] = None,
            use_train_data: bool = True,
            pruning_scheme: str = "MP",
            reward_type: str = "sparse",
            reward_on_masked_layers_only: bool = False,
            cost_scheme: str = "sparsity",
            cost_on_masked_layers_only: bool = False,
            max_timesteps: int = 10
        ):

        self.reward_type = reward_type
        self.reward_on_masked_layers_only = reward_on_masked_layers_only
        self.cost_scheme = cost_scheme
        self.cost_on_masked_layers_only = cost_on_masked_layers_only
        self.max_timesteps = max_timesteps
        self.done = True

        self.pruning_method = get_pruning_method(pruning_scheme)

        super(NetworkWise, self).__init__(
                network, dataset, device, _seed, batch_size, finetune_iters,
                finetune_batch_size, optimizer, lr, use_train_data
        )

    def _define_spaces(self):
        self.observation_space = gym.spaces.Box(
                low=0., high=1., shape=np.array((self.num_masked_layers,)),
                dtype=np.float32)
        self.action_space = gym.spaces.Box(
                low=0., high=1., shape=np.array((self.num_masked_layers,)),
                dtype=np.float32)

    def _define_reward_range(self):
        self.reward_range = (0,100*self.max_timesteps)

    def reset(self) -> np.ndarray:
        self.done = False
        self.timestep = 0
        self.reset_masks()
        return np.zeros(shape=self.observation_space.shape)

    def step(self, action: np.ndarray) -> Tuple[
            np.ndarray, float, bool, Dict]:

        assert not self.done
        #action = 0.1*np.ones_like(action)
        #action = np.array([.15]*8 + [.10])

        # Set masks
        masks = self.pruning_method(self.network, action, self.data,
                                    self.device, iterations=None)
        self.network.set_masks(masks)

        # Prepare returns
        obs = action
        self._finetune_network()
        _, reward = self._test_network(verbose=False)
        infos = {"mean_action": np.mean(action),
                 "max_action": np.max(action),
                 "min_action": np.min(action)}

        # Check if done
        self.timestep += 1
        if self.timestep >= self.max_timesteps:
            self.done = True

        # Get cost
        if self.cost_scheme == "null":
            infos["cost"] = 0
        elif self.cost_scheme == "sparsity":
            infos["cost"]= 1-self._get_network_sparsity(verbose=False)
        else:
            raise ValueError(f"Invalid cost scheme: {self.cost_scheme}")

        self.reset_masks()

        return obs, reward, self.done, infos
