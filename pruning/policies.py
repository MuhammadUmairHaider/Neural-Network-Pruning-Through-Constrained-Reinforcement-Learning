from itertools import zip_longest
from typing import Dict, List, Optional, Tuple, Type, Union

import stable_baselines3.common.policies as policies
import stable_baselines3.common.torch_layers as layers
import torch as th
import torch.nn as nn
from stable_baselines3.common.utils import get_device


class MlpExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        output_activation_fn: Optional[Type[nn.Module]] = None,
        device: Union[th.device, str] = "auto",
        create_cvf: bool = False
    ):
        super(MlpExtractor, self).__init__()
        device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        cost_value_only_layers = []  # Layer sizes of the network that only belongs to the cost value network
        last_layer_dim_shared = feature_dim

        self.create_cvf = create_cvf

        # If we also need to create a value function for cost
        if create_cvf:
            cost_value_net = []

        # Iterate through the shared layers and build the shared parts of the network
        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):  # Check that this is a shared layer
                layer_size = layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer_size
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]

                if create_cvf and "cvf" in layer:
                    assert isinstance(layer["cvf"], list), "Error: net_arch[-1]['cvf'] must contain a list of integers."
                    cost_value_only_layers = layer["cvf"]

                break  # From here on the network splits up in policy and value network

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared
        if create_cvf:
            last_layer_dim_cvf = last_layer_dim_shared

        # Build the non-shared part of the network
        for idx, (pi_layer_size, vf_layer_size, cvf_layer_size) in enumerate(
                zip_longest(policy_only_layers, value_only_layers, cost_value_only_layers
            )):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size

            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size

            if cvf_layer_size is not None:  # Will be none if cost_vf is False
                assert isinstance(cvf_layer_size, int), "Error: net_arch[-1]['cvf'] must only contain integers."
                cost_value_net.append(nn.Linear(last_layer_dim_cvf, cvf_layer_size))
                cost_value_net.append(activation_fn())
                last_layer_dim_cvf = cvf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        if create_cvf:
            self.latent_dim_cvf = last_layer_dim_cvf

        if len(policy_net) > 0 and output_activation_fn is not None:
            policy_net[-1] = output_activation_fn()

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)
        if create_cvf:
            self.cost_value_net = nn.Sequential(*cost_value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        #print(features)
        #print('\n\n\n\n\n')
        #print(self.policy_net(features))
        #print('\n\n\n\n\n')
        #print(self.policy_net(features).shape)
        #exit()
        if self.create_cvf:
            return self.policy_net(shared_latent), self.value_net(shared_latent), self.cost_value_net(shared_latent)
        else:
            return self.policy_net(shared_latent), self.value_net(shared_latent)


class SigmoidPolicy(policies.ActorTwoCriticsPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(self.features_dim, net_arch=self.net_arch, activation_fn=nn.Sigmoid,#self.activation_fn,
                                          output_activation_fn=nn.Sigmoid, create_cvf=True)



policies.register_policy("SigmoidPolicy", SigmoidPolicy)
