import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import gym
import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces

import pruning.utils as utils
from pruning.envs.utils import InfiniteBatchSampler
from pruning.networks.create_network import get_network
from pruning.networks.masked_modules import MaskedConv2d, MaskedLinear
from pruning.train import test, train


# Workaround for supporting gym's Monitor
class EnvSpec:
    id = None

class PruningBaseEnv:
    metadata = {"render.modes": ["human"]}
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
            cost_on_masked_layers_only: bool = False
        ):

        self.finetune_iters = finetune_iters
        self.finetune_batch_size = finetune_batch_size
        self.batch_size = batch_size
        self.device = device
        self.pruning_scheme = pruning_scheme
        self.reward_type = reward_type
        self.reward_on_masked_layers_only = reward_on_masked_layers_only
        self.cost_scheme = cost_scheme
        self.cost_on_masked_layers_only = cost_on_masked_layers_only

        self.train_data, self.test_data = utils.get_dataset(dataset)
        self.network = get_network(network).to(self.device)
        self.network.eval()

        if self.finetune_iters is not None:
            assert optimizer is not None, "Need to provide optimizer for finetuning"
            assert lr is not None, "Need to provide learning rate for finetuning"
            assert finetune_batch_size is not None, "Need to provide batch size for finetuning"
            self.optimizer = utils.get_optimizer(optimizer)(self.network.parameters(), lr=lr)

        # Restore network weights.
        self.weights_path = f"pruning/pretrained/{dataset}_{network}_{_seed}.pt"
        self._load_weights()

        # Print initial test accuracy
        self._test_network(verbose=True)

        self.layers = [m for m in self.network.features.children()]
        self.layers += [self.network.classifier]

        if use_train_data:
            self.data_generator = InfiniteBatchSampler(self.train_data, self.batch_size, self.device)
        else:
            self.data_generator = InfiniteBatchSampler(self.test_data, self.batch_size, self.device)

        self._define_spaces()
        self._define_reward_range()
        self.spec = EnvSpec

    def _load_weights(self):
        if os.path.exists(self.weights_path):
            state_dict = th.load(self.weights_path)
            self.network.load_state_dict(state_dict)
        else:
            raise ValueError(f"No pretrained model found at {self.weights_path}")

    def _test_network(self, verbose=True):
        loss, acc = test(self.network, self.test_data, self.device)
        if verbose:
            print(utils.colorize(f"Network accuracy ({self.device}): {acc:.2f}%", "green", bold=True), flush=True)
            return
        return loss, acc

    def _train_network(self):
        train(self.train_data, None, self.network, self.optimizer, self.finetune_iters, self.finetune_batch_size,
              self.device, print_every=None, group=None)

    def _define_spaces(self) -> None:
        raise NotImplementedError

    def _define_reward_range(self) -> None:
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> List[int]:
        return

    def close(self) -> None:
        pass

    def reset(self) -> None:
        self.current_layer_idx = 0
        self.done = False
        self.current_inputs, self.current_labels = next(self.data_generator)
        self.current_features = self.current_inputs
        self.last_action = None
        self.action = None

        # Reset all layer masks to one
        for layer in self.layers:
            if self._is_masked_module(layer):
                layer.mask = th.ones(layer.mask.shape).to(self.device)
                layer.weight.data *= layer.mask.to(self.device)

        observations, _, _, _ = self.step(self.action)

        return observations

    def step(self, action: Optional[np.ndarray]) -> Tuple[np.ndarray, float, List[bool], List[Dict]]:
        if "done" not in self.__dict__:
            assert False, "Reset needs to be called in the beginning"
        assert not self.done, "Env needs to be reset first"

        self.last_action = self.action
        self.action = action

        # Track reward, cost over all layers traversed
        reward, cost = 0., 0.

        # Set mask for current layer
        if self.action is not None:        # None action passed at reset
            layer = self.layers[self.current_layer_idx]
            # Get mask
            mask = self._get_mask(self.action, layer)
            layer.mask = mask.to(self.device)
            layer.weight.data *= layer.mask
            with th.no_grad():
                self.current_features = layer(self.current_features)
            # Get reward, cost
            reward += self._get_reward(self.current_layer_idx, len(self.layers), self.current_features, self.current_labels)
            cost += self._get_cost(self.current_layer_idx, len(self.layers), layer, self.action)
            # Increment layer idx
            self.current_layer_idx += 1

        # Iterate to next masked layer
        for layer in self.layers[self.current_layer_idx:]:
            if self._is_masked_module(layer):
                break
            with th.no_grad():
                if self.current_layer_idx == (len(self.layers)-1):
                    self.current_features = self.current_features.view(self.current_features.size(0), -1)
                self.current_features = layer(self.current_features)
            # Get reward, cost
            reward += self._get_reward(self.current_layer_idx, len(self.layers), self.current_features, self.current_labels)
            cost += self._get_cost(self.current_layer_idx, len(self.layers), layer, None)
            # Increment layer idx
            self.current_layer_idx += 1

        if self.current_layer_idx == len(self.layers):
            observations = np.zeros(self.observation_space.shape)
            self.done = True
        else:
            observations = self._get_observations(self.current_layer_idx,
                                                  self.layers[self.current_layer_idx],
                                                  self.current_features,
                                                  self.last_action)
            self.done = False

        # Prepare infos
        # TODO: Better way to do this
        if action is None: action = [0]
        info = {"cost": cost,
                "action_magnitude": np.abs(action[0])}

        if self.done:
            _, info["test_accuracy"] = self._test_network(verbose=False)
            info["sparsity"] = utils.get_network_sparsity(self.network, func=lambda l: self._is_masked_module(l))

        return observations, reward, self.done, info

    def _get_observations(self, layer_idx: int, layer: nn.modules, inputs: th.tensor) -> np.ndarray:
        raise NotImplementedError

    def _define_reward_range(self):
        if self.reward_type == "sparse":
            self.reward_range = (0,1)   # since reward=accuracy

    def _get_reward(self, layer_idx: int, num_layers: int, outputs: th.tensor, labels: th.tensor) -> float:
        if self.reward_type == "sparse":
            return self._sparse_reward(layer_idx, num_layers, outputs, labels)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

        return reward

    def _sparse_reward(self, layer_idx: int, num_layers: int, outputs: th.tensor, labels: th.tensor) -> float:
        if layer_idx == num_layers-1:
            # Finetune, if set
            if self.finetune_iters is not None:
                self._load_weights()         # reset weights from previous iteration
                self._train_network()

            # Calculate accuracy
            _, preds = outputs.max(1)
            reward = preds.eq(labels).detach().cpu().numpy().astype(np.float32).sum()
            reward /= labels.shape[0]
        else:
            reward = np.zeros(())

        return reward

    def _get_cost(self, layer_idx: int, num_layers: int, layer: nn.modules, action: Optional[np.ndarray]) -> float:
        if self.cost_scheme == "null":
            return 0.
        elif self.cost_scheme == "sparsity":
            if action is None or (self.cost_on_masked_layers_only and not self._is_masked_module(layer)):
                return 0.
            else:
                return 1.-np.linalg.norm(action)
        elif self.cost_scheme == "sparsity_at_end":
            if layer_idx == num_layers-1:
                return 1-utils.get_network_sparsity(self.network, func=lambda l: self._is_masked_module(l))
            else:
                return 0.
        else:
            raise ValueError(f"Unknown cost scheme {self.cost_scheme}")

    def _get_mask(self, actions: np.ndarray, layer: nn.modules)-> np.ndarray:
        raise NotImplementedError

    def _is_masked_module(self, layer: nn.modules) -> bool:
        return utils.is_masked_module(layer)

    def render(self, mode: str = "human"):
        pass


class PruningAMCEnv(PruningBaseEnv):
    """Environment based on the scheme in Automated Model Compression (AMC):
    https://arxiv.org/abs/1802.03494.
    """
    def _define_spaces(self):
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=np.array((12,)), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0., high=1., shape=np.array((1,)), dtype=np.float32)

    def _is_masked_module(self, layer: nn.modules) -> bool:
        # Will only learn masks for convolutional layers
        return isinstance(layer, MaskedConv2d)

    def _get_observations(self, layer_idx: int, layer: nn.modules, inputs: th.tensor,
                          last_actions: Optional[np.ndarray]) -> np.ndarray:
        if last_actions is None:
            last_actions = 0
        else:
            last_actions = last_actions[0]
        observations = np.array([layer_idx, layer.in_channels, layer.out_channels,
                                 *layer.kernel_size, *layer.stride, *layer.padding,
                                 *inputs.shape[-2:], last_actions],
                                dtype=np.float32)

        return observations

    def _get_mask(self, actions: np.ndarray, layer: nn.modules) -> th.tensor:
        mask = scoring_based_mask(layer, actions[0], scheme=self.pruning_scheme)
        return mask


def scoring_based_mask(layer, pruning_ratio, scheme="MP"):
    # Get scores
    if scheme == "MP":
        score = th.abs(layer.weight.data.cpu().detach())

    # Find mask according to score
    cutoff_index = round((1-pruning_ratio) * np.prod(layer.mask.shape))
    _, idx = th.sort(score.view(-1), descending=True)
    mask = th.ones(layer.mask.shape)
    mask_flat = mask.view(-1)
    mask_flat[idx[cutoff_index:]] = 0

    return mask

def main():
    from stable_baselines3.common import vec_env
    fn = lambda i: lambda: PruningAMCEnv("vgg19", "cifar10", f"cuda:{i}")
    env = vec_env.SubprocVecEnv([fn(i) for i in range(4)])
    env = vec_env.VecNormalize(env)
    env.reset()
    while True:
        r = env.step([[0.1],[0.1],[0.1],[0.1]])[1]
        print(type(r), r)
