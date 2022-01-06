import argparse
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common import logger
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

import pruning.utils as utils
import wandb
from pruning.default_hparams import get_default_hparams
from pruning.networks.create_network import get_network
from pruning.prune import get_pruning_method

import torch_xla
import torch_xla.core.xla_model as xm

logger = logger.HumanOutputFormat(sys.stdout)

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(utils.colorize("Running on GPU", "green", bold=True))
        return device
    else:
        device = xm.xla_device()
        print(utils.colorize("Running on "+str(device), "green", bold=True))
        return device

def print_metrics(network, device, train_data=None, test_data=None,
                  heading=None, prepend={}, sparsity=False, group=None,
                  commit=True):
    metrics = prepend

    if train_data is not None:
        loss, acc = test(network, train_data, device)
        metrics.update({
            "TrainLoss": loss,
            "TrainAccuracy": acc,
        })
    if test_data is not None:
        loss, acc = test(network, test_data, device)
        metrics.update({
            "TestLoss": loss,
            "TestAccuracy": acc,
        })
    if sparsity:
        sparsity = utils.get_network_sparsity(network)
        metrics.update({
            "Sparsity": sparsity,
            "PercentPruned": sparsity*100,
            "CompressionRatio": 1/(1-sparsity)
        })

    if group is not None:
        updated_metrics = {}
        for key, val in metrics.items():
            updated_metrics[group+"/"+key] = val
        metrics = updated_metrics

    logger.write(metrics, {k: None for k in metrics.keys()})
    wandb.log(metrics, commit=commit)

class BatchSampler(Sampler):
    def __init__(self, data, num_iterations, batch_size):
        self.data = data
        self.num_iterations = num_iterations
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.num_iterations):
            indices = random.sample(range(len(self.data)), self.batch_size)
            yield indices

    def __len__(self):
        return self.num_iterations

def train(train_data, test_data, network, optimizer, num_iterations,
          batch_size, device, print_every=10000, group=None):
    """Trains a network on train_dataset for num_iterations."""
    # Activate train mode.
    network.train()

    # Prepare data iterator.
    batch_sampler = BatchSampler(train_data, num_iterations, batch_size)
    train_loader = DataLoader(train_data, batch_sampler=batch_sampler,
                              num_workers=0)

    # Train.
    start = time.time()
    for i, (x,y) in tqdm(enumerate(train_loader)):
        # Send to device.
        x = x.to(device)
        y = y.to(device)

        # Forward pass.
        out = network(x)
        loss = F.cross_entropy(out, y)

        # Backward pass.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print.
        if print_every is not None and i % print_every == 0:
            prepend = dict(Time=time.time()-start, Iteration=i)
            print_metrics(network, device, None, test_data, prepend=prepend,
                          group=group)
            network.train()

def test(network, data, device, batch_size=512):
    """Computes the network loss and accuracy on the given dataset."""
    # Prepare network for evaluation (puts layers such as batchnorm,
    # dropout into evaluation mode).
    network.eval()

    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        num_workers=0)

    correct, loss = 0., 0.
    for i, (x,y) in enumerate(loader):
        # Send data to device.
        x = x.to(device)
        y = y.to(device)

        # Feed data to network.
        with torch.no_grad():
            out = network(x)
            _, pred = out.max(1)

        # Compute loss and accuracy.
        loss += F.cross_entropy(out, y) * len(x)
        correct += pred.eq(y).sum().item()

    loss = loss/len(data)
    acc = correct/len(data) * 100.0

    return loss.item(), acc

def _main(cfg):
    wandb.init(
        project="Pruning",
        group='MP',
        name=f"{cfg.pruning_method}_{cfg.network}_{cfg.dataset}_{cfg.seed}",
        config=cfg
    )
    set_seeds(cfg.seed)
    device = get_device()

    # Directory to store models in.
    models_dir = os.path.join(wandb.run.dir, "models")
    os.mkdir(models_dir)

    # Load data and create network, optimizer.
    train_data, test_data = utils.get_dataset(cfg.dataset)
    network = get_network(cfg.network).to(device)
    optimizer = utils.get_optimizer(cfg.optimizer)(network.parameters(), lr=cfg.lr)

    # Restore network weights if they exist, otherwise pretrain.
    weights_path = f"pruning/pretrained/{cfg.dataset}_{cfg.network}_{cfg.seed}.pt"
    if os.path.exists(weights_path):
        print(utils.colorize("Loading pretrained network", "green", bold=True))
        state_dict = torch.load(weights_path)
        network.load_state_dict(state_dict)
    else:
        print(utils.colorize("Pretraining network", "green", bold=True))
        print_metrics(
                network, device, train_data, test_data,
                heading="BeforeTraining", group="Pretraining", commit=True
        )
        train(train_data, test_data, network, optimizer, cfg.pretrain_iters,
              cfg.batch_size, device, print_every=1000, group="Pretraining")
        print_metrics(
                network, device, train_data, test_data,
                heading="AfterTraining", group="Pretraining", commit=True
        )

        # Save network and logs.
        torch.save(network.state_dict(), weights_path)
        torch.save(
                network.state_dict(),
                os.path.join(models_dir, "pretrained_model.pt")
        )

    if cfg.pruning_method is not None:
        # Log metrics before training.
        print_metrics(network, device, train_data, test_data,
                      heading="BeforePruning", sparsity=True,
                      group="AFT", commit=True)

        # Prune.
        print(utils.colorize("Pruning network", "green", bold=True))
        start = time.time()
        pruning_method = get_pruning_method(cfg.pruning_method)
        for itr in range(cfg.pruning_iters):
            # Update masks.
            masks = pruning_method(network, cfg.prune_ratios, train_data, device,
                                   iterations=None)
            network.set_masks(masks)

            # Create new network.
            network = get_network(cfg.network).to(device)
            optimizer = utils.get_optimizer(cfg.optimizer)(network.parameters(), lr=cfg.lr)
            network.set_masks(masks)

            # Log metrics.
            prepend = dict(Time=time.time()-start, PruneIteration=itr)
            print_metrics(network, device, train_data, test_data,
                          heading="BeforeFinetuning", prepend=prepend,
                          sparsity=True, group="BFT", commit=False)

            # Finetune.
            train(train_data, test_data, network, optimizer, cfg.finetune_iters,
                  cfg.batch_size, device, print_every=None)

            # Log metrics.
            prepend = dict(Time=time.time()-start, PruneIteration=itr)
            print_metrics(network, device, train_data, test_data,
                          heading="AfterFinetuning", prepend=prepend,
                          sparsity=True, group="AFT", commit=True)

            # Save.
            torch.save(
                    network.state_dict(),
                    os.path.join(models_dir, "pruned_model_%03d.pt" % itr)
            )

def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", '-n', type=str, default="vgg11", help="vgg11|vgg16|vgg19")
    parser.add_argument("--dataset", '-d', type=str, default="cifar10", help="mnist|fmnist|cifar10|cifar100")
    parser.add_argument("--pruning_method", "-pm", type=str, default=None, help="mp|rp|sbp|lgp")
    parser.add_argument("--pruning_iters", "-pi", type=int, default=5)
    parser.add_argument("--seed", type=int, default=718, help="random seed")
    parser.add_argument("--prune_ratios", "-pr", type=float, nargs="*", default=None)

    args = vars(parser.parse_args())
    config = get_default_hparams(args["network"])
    config.update(args)
    config = SimpleNamespace(**config)

    _main(config)
    end = time.time()
    print("The entire script took %04.2f minutes" % ((end-start)/60))

if __name__=="__main__":
    main()
