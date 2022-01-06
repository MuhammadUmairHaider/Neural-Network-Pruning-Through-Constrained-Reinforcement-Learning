import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import gym
import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces

import pruning.utils as utils
from pruning.envs.utils import (InfiniteBatchSampler, test_network,
                                train_network)
from pruning.networks.create_network import get_network
from pruning.networks.masked_modules import MaskedConv2d, MaskedLinear


# Workaround for supporting gym's Monitor
class EnvSpec:
    id = None

class OfflinePruningBaseClass:
    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            network: str,
            dataset: str,
            device: str,
            _seed: int,
            batch_size: int,
            finetune_iters: Optional[Callable] = None,
            finetune_batch_size: Optional[int] = None,
            optimizer: Optional[str] = None,
            lr: Optional[float] = None,
            use_train_data: bool = True,
            reset_weights: bool = False,
            verbose: bool = False
        ):

        self.finetune_iters = finetune_iters
        self.finetune_batch_size = finetune_batch_size
        self.batch_size = batch_size
        self.device = device
        self.use_train_data = use_train_data
        self.verbose = verbose
        self.reset_weights = reset_weights

        self.train_data, self.test_data = utils.get_dataset(dataset)
        self.data = self.train_data if use_train_data else self.test_data
        self.network = get_network(network).to(self.device)
        self.network.eval()

        if self.finetune_iters is not None:
            assert optimizer is not None
            assert lr is not None
            assert finetune_batch_size is not None
            self.optimizer = utils.get_optimizer(optimizer)(
                    self.network.parameters(), lr=lr)

        # Restore network weights.
        self.weights_path = f"pruning/pretrained/{dataset}_{network}_{_seed}.pt"
        self._load_weights()

        # Print initial test accuracy
        self._test_network(verbose=True)

        self.layers = [module for module in self.network.modules()
                       if not isinstance(module, nn.Sequential)][1:]

        ## Old way - resnet does not support this
        #self.layers = [m for m in self.network.features.children()]
        #self.layers += [self.network.classifier]

        self.num_layers = len(self.layers)
        self.num_masked_layers = sum(
                [self._is_masked_module(layer) for layer in self.layers])

        self.data_generator = InfiniteBatchSampler(
                self.data, self.batch_size, self.device)

        self._define_spaces()
        self._define_reward_range()
        self.spec = EnvSpec

    def _load_weights(self):
        if os.path.exists(self.weights_path):
            state_dict = th.load(self.weights_path)
            self.network.load_state_dict(state_dict)
        else:
            raise ValueError(f"No model found at {self.weights_path}")

    def _test_network(self, verbose=True, use_test_data=True,
                      num_samples=None):
        data = self.test_data if use_test_data else self.train_data
        loss, acc = test_network(self.network, data, self.device,
                                 num_samples=num_samples)
        if verbose:
            print(utils.colorize(
                f"Network accuracy ({self.device}): {acc:.2f}%",
                "green", bold=True
                ), flush=True)
        return loss, acc

    def _finetune_network(self):
        if self.finetune_iters is not None and self.finetune_iters(self.timesteps) > 0:
            train_network(self.train_data, self.test_data, self.network, self.optimizer,
                          self.finetune_iters(self.timesteps), self.finetune_batch_size,
                          self.device, verbose=self.verbose)

    def _get_network_sparsity(self, verbose=False):
        sparsity = utils.get_network_sparsity(self.network)
        if verbose:
            print(utils.colorize(
                f"Network sparsity ({self.device}): {sparsity*100:.2f}%",
                "green", bold=True
                ), flush=True)

        return sparsity

    def _define_spaces(self) -> None:
        raise NotImplementedError

    def _define_reward_range(self) -> None:
        raise NotImplementedError

    def reset_masks(self) -> None:
        # Reset all layer masks to one
        for layer in self.layers:
            if self._is_masked_module(layer):
                layer.mask = th.ones(layer.mask.shape).to(self.device)
                #layer.weight.data *= layer.mask.to(self.device)

    def reset(self) -> None:
        raise NotImplementedError

    def step(self, action: Optional[np.ndarray]) -> Tuple[
            np.ndarray, float, List[bool], List[Dict]]:
        raise NotImplementedError

    def _is_masked_module(self, layer: nn.modules) -> bool:
        return utils.is_masked_module(layer)

    def seed(self, seed: Optional[int] = None) -> List[int]:
        return

    def close(self) -> None:
        pass

    def render(self, mode: str = "human") -> None:
        pass
