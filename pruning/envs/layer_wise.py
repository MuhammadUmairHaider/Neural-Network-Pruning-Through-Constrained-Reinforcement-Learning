import os
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Union)

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


class LayerWise(OfflinePruningBaseClass):
    def __init__(
            self,
            network: str,
            dataset: str,
            device: str,
            _seed: int = 718,
            batch_size: int = 2048,
            reset_weights: bool = False,
            finetune_iters: Optional[Callable] = None,
            finetune_batch_size: int = None,
            optimizer: Optional[str] = None,
            lr: Optional[float] = None,
            use_train_data: bool = True,
            pruning_scheme: str = "MP",
            reward_type: str = "sparse",
            reward_on_masked_layers_only: bool = False,
            cost_scheme: str = "sparsity",
            cost_on_masked_layers_only: bool = False,
            action_clip_value: Callable = lambda t: 0.8,
            verbose: bool = False,
            **kwargs
        ):

        self.timesteps = 0
        self.action_clip_value = action_clip_value

        super(LayerWise, self).__init__(
                network, dataset, device, _seed, batch_size, finetune_iters,
                finetune_batch_size, optimizer, lr, use_train_data, reset_weights,
                verbose
        )

        self.reward_type = reward_type
        self.reward_on_masked_layers_only = reward_on_masked_layers_only
        self.cost_scheme = cost_scheme
        self.cost_on_masked_layers_only = cost_on_masked_layers_only
        self.done = True

        self.pruning_method = get_pruning_method(pruning_scheme)
        self.max_timesteps = sum([1 for layer in self.layers
                                    if isinstance(layer, MaskedConv2d)])
        self.pruning_method = get_pruning_method(pruning_scheme)

    def _define_spaces(self):
        self.observation_space = gym.spaces.Box(
                low=0., high=np.inf, shape=np.array((10,)), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
                low=0., high=1., shape=np.array((1,)), dtype=np.float32
        )

    def _define_reward_range(self):
        self.reward_range = (0, 100)

    def _define_cost_range(self):
        self.cost_range = (0, 1)

    def reset(self) -> np.ndarray:
        changed = self.finetune_iters is not None and self.finetune_iters(self.timesteps) > 0
        if self.reset_weights and changed:
            self._load_weights()
        self.done = False
        self.timestep = 0
        self.reset_masks()
        self.actions = []
        self.at_layer = 0

        # Iterate to first layer
        while not isinstance(self.layers[self.at_layer], MaskedConv2d):
            self.at_layer += 1
            if self.at_layer == self.num_layers-1:
                break

        return self._get_observations(self.at_layer)

    def step(self, action: np.ndarray) -> Tuple[
            np.ndarray, float, bool, Dict]:
        """action: percentage of weights to prune"""
        orig_action = action
        action = np.clip(action, self.action_space.low,
                         self.action_clip_value(self.timesteps))
        self.timesteps += 1

        self.actions.append(action[0])
        self.at_layer += 1

        while not isinstance(self.layers[self.at_layer], MaskedConv2d):
            self.at_layer += 1
            if self.at_layer == self.num_layers-1:
                break

        if self.at_layer < self.num_layers-1:
            obs = self._get_observations(self.at_layer)
            rew = 0
            infos = {'action': action[0], 'cost': 0}
            done = False
        else:
            obs = self._get_observations(0) # obs of next iteration
            self.set_masks(self.actions)
            self._finetune_network()
            rew = self._test_network(
                    verbose=self.verbose,
                    use_test_data=not self.use_train_data,
                    num_samples=self.batch_size
            )[1]
            sparsity = self._get_network_sparsity(verbose=self.verbose)
            if self.cost_scheme == 'null':
                cost = 0
            else:
                cost = len(self.actions) * (1-sparsity) * 100
            infos = {'action': action[0], 'cost': cost, 'sparsity': sparsity,
                     'max_action': np.max(self.actions), 'min_action': np.min(self.actions)}
            done = True

        infos['action_clip_value'] = self.action_clip_value(self.timesteps)
        infos['raw_action'] = orig_action

        return obs, rew, done, infos

    def set_masks(self, actions: list) -> None:
        assert len(self.actions) == self.max_timesteps
        masks = self.pruning_method(self.network, actions, self.data,
                                    self.device, iterations=None)
        self.network.set_masks(masks)

    def _get_observations(self, layer_idx: int) -> np.ndarray:
        layer = self.layers[layer_idx]
        sum_acs = sum(self.actions) if len(self.actions) > 0 else 0
        obs = np.array([layer_idx, layer.in_channels, layer.out_channels,
                        *layer.kernel_size, *layer.stride, *layer.padding,
                        sum_acs])
        return obs

def main():
    import random
    env = LayerWise('vgg11', 'cifar10', 'cuda:0', finetune_iters=10,
                    optimizer='adam', lr=0.3, finetune_batch_size=10)

    ac = env.action_space
    print(ac, ac.high)
    obs = env.reset()
    done = False
    #print(obs)
    while not done:
        obs, rew, done, info = env.step([0.6])
        #print(obs, rew, done, info)
    print(rew, info)
    print(ac, ac.high, env.action_space.high)
    exit()


    obs = env.reset()
    done = False
    #print(obs)

    while not done:
        obs, rew, done, info = env.step([random.uniform(0,1)])
        #print(obs, rew, done, info)
    print(rew)


    obs = env.reset()
    done = False
    #print(obs)

    while not done:
        obs, rew, done, info = env.step([0.1])
        #print(obs, rew, done, info)
    print(rew)
