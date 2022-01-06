import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pruning.utils as utils


def get_pruning_method(method):
    if method.lower() == "mp":
        return MP
    if method.lower() == "rp":
        return RP
    if method.lower() == "sgp":
        return SGP
    if method.lower() == "sgwp":
        return SGWP
    if method.lower() == "sgwmp":
        return SGWMP
    if method.lower() == "sigwmp":
        return SIGWMP
    if method.lower() == "lgp":
        return LGP
    if method.lower() == "lgwp":
        return LGWP

    raise ValueError("Method %s not defined" % method)

# =====================================================================
# Utilities
# =====================================================================

def get_gradients(network, dataset, device, batch_size=512, iterations=None):
    # Put network in eval mode.
    network.eval()      # necessary?

    # Zero out previous gradients.
    dummy_optim = torch.optim.Adam(network.parameters())
    dummy_optim.zero_grad()

    # Prepare data iterator.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, drop_last=True)

    # Compute gradients.
    for itr, (x,y) in enumerate(loader):
        # Send to device.
        x = x.to(device)
        y = y.to(device)

        # Forward pass.
        out = network(x)
        loss = F.cross_entropy(out, y)

        # Compute gradients. Note that gradients are accumulated across
        # iterations.
        loss.backward()

        if iterations is not None and itr == iterations-1:
            break

    # Average gradients.
    gradients = []
    for m in network.modules():
        if utils.is_masked_module(m):
            gradients.append(m.weight.grad/(itr+1))

    return gradients

# =====================================================================
# Score-based pruning methods
# =====================================================================

def score_based_pruning(weights, masks, prune_ratios, score_func):
    """Abstract function for score-based pruning."""
    # Find new masks.
    new_masks = []
    for layer in range(len(prune_ratios)):
        score = score_func(weights, layer)
        new_mask = score_based_mask(score, masks[layer], prune_ratios[layer])
        new_masks.append(new_mask)

    # Update all masks.
    for layer in range(len(prune_ratios)):
        masks[layer] *= new_masks[layer]

    return new_masks

def score_based_mask(score, old_mask, prune_ratio):
    """Get mask for current layer."""
    assert (prune_ratio >= 0) and (prune_ratio <= 1)

    # Ensure deactivated neurons are not reactivated.
    score[old_mask <= 0] = float("-inf")

    # Find number of neurons to keep.
    surv_ratio = 1 - prune_ratio
    num_surv_weights = torch.sum(old_mask).item()
    cutoff_index = int(round(surv_ratio * num_surv_weights))

    # Find mask.
    _, idx = torch.sort(score.view(-1), descending=True)
    new_mask = torch.ones(old_mask.shape) * old_mask
    new_mask_flat = new_mask.view(-1)
    new_mask_flat[idx[cutoff_index:]] = 0

    return new_mask

# Note: The *args and **kwargs arguments are mainly to eliminate the need of
# handling different functions differently.

def MP(network, prune_ratios, *args, **kwargs):
    """Magnitude pruning."""
    weights = network.get_weights()
    mask = network.get_masks()

    def score_func(weights, layer):
        score = torch.abs(weights[layer])
        return score

    new_masks = score_based_pruning(network.get_weights(), network.get_masks(),
                                    prune_ratios, score_func)
    return new_masks

def RP(network, prune_ratios, *args, **kwargs):
    """Random pruning."""
    def score_func(weights, layer):
        score = torch.abs(torch.randn(weights[layer].size()))
        return score

    new_masks = score_based_pruning(network.get_weights(), network.get_masks(),
                                    prune_ratios, score_func)
    return new_masks

def SGP(network, prune_ratios, train_data, device, iterations=None):
    """Gradient pruning. Weights with smaller gradients are pruned."""
    gradients = get_gradients(network, train_data, device,
                              iterations=iterations)
    def score_func(weights, layer):
        score = torch.abs(gradients[layer])
        return score

    new_masks = score_based_pruning(network.get_weights(), network.get_masks(),
                                    prune_ratios, score_func)
    return new_masks

def SGWMP(network, prune_ratios, train_data, device, iterations=None):
    """Gradient pruning. Weights with smaller gradients-weight
    product are pruned."""
    gradients = get_gradients(network, train_data, device,
                              iterations=iterations)

    def score_func(weights, layer):
        score = torch.abs(gradients[layer].to("cpu") * weights[layer].to("cpu"))
        return score

    new_masks = score_based_pruning(network.get_weights(), network.get_masks(),
                                    prune_ratios, score_func)
    return new_masks

def SIGWMP(network, prune_ratios, train_data, device, iterations=None):
    """Gradient pruning. Weights with smaller gradients-weight
    product are pruned."""
    gradients = get_gradients(network, train_data, device,
                              iterations=iterations)

    def score_func(weights, layer):
        score = torch.abs(weights[layer].to("cpu")/(gradients[layer].to("cpu")+1e-10))
        return score

    new_masks = score_based_pruning(network.get_weights(), network.get_masks(),
                                    prune_ratios, score_func)
    return new_masks


def SGWP(network, prune_ratios, train_data, device, iterations=None):
    """Gradient pruning. Weights with smaller gradients-weight
    product are pruned."""
    gradients = get_gradients(network, train_data, device,
                              iterations=iterations)

    def score_func(weights, layer):
        score = gradients[layer].to("cpu") * weights[layer].to("cpu")
        return score

    new_masks = score_based_pruning(network.get_weights(), network.get_masks(),
                                    prune_ratios, score_func)
    return new_masks

def LGP(network, prune_ratios, train_data, device, iterations=None):
    """Gradient pruning. Weights with larger gradient magnitudes are
    pruned."""
    gradients = get_gradients(network, train_data, device,
                              iterations=iterations)
    def score_func(weights, layer):
        score = 1/(torch.abs(gradients[layer])+1e-10)
        return score

    new_masks = score_based_pruning(network.get_weights(), network.get_masks(),
                                    prune_ratios, score_func)
    return new_masks

def LGWP(network, prune_ratios, train_data, device, iterations=None):
    """Gradient pruning. Weights with larger gradients-weight product
    are pruned."""
    gradients = get_gradients(network, train_data, device,
                              iterations=iterations)
    def score_func(weights, layer):
        score = -gradients[layer].to("cpu") * weights[layer].to("cpu")
        return score

    new_masks = score_based_pruning(network.get_weights(), network.get_masks(),
                                    prune_ratios, score_func)
    return new_masks
