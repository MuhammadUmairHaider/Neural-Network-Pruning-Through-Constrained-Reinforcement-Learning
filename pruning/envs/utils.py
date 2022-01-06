import random

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

# ==========================================================
# Data sampler
# ==========================================================

class BatchSampler(Sampler):
    def __init__(self, data_size, batch_size, num_iterations=None):
        self.data_size = data_size
        self.batch_size = batch_size or self.data_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = np.arange(self.data_size)
        np.random.shuffle(indices)
        itr = 0
        while True:
            if itr * self.batch_size >= self.data_size:
                break
            if self.num_iterations is not None and itr >= self.num_iterations:
                break
            yield indices[itr*self.batch_size:(itr+1)*self.batch_size]
            itr += 1

    def __len__(self):
        return np.ceil(self.data_size/self.batch_size)\
                if self.num_iterations is None else self.num_iterations


class InfiniteBatchSampler:
    def __init__(self, data, batch_size, device):
        self.data = data
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            batch_sampler = BatchSampler(len(self.data), self.batch_size)
            generator = DataLoader(self.data, batch_sampler=batch_sampler,
                                   num_workers=0)
            for datum in generator:
                _datum = [d.to(self.device) for d in datum]
                return _datum

class FiniteBatchSampler:
    def __init__(self, data_size, batch_size, num_iterations):
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        itr = 0
        while True:
            batch_sampler = BatchSampler(self.data_size, self.batch_size)
            for indices in batch_sampler:
                yield indices
                itr += 1
                if itr >= self.num_iterations:
                    break
            if itr >= self.num_iterations:
                break

    def __len__(self):
        return self.num_iterations


# =====================================================================
# Network
# =====================================================================

def train_network(train_data, test_data, network, optimizer, num_iterations,
                  batch_size, device, test_every=1000, verbose=True):
    """Trains a network on train_dataset for num_iterations."""
    # Activate train mode.
    network.train()

    # Prepare data iterator.
    batch_sampler = FiniteBatchSampler(len(train_data), batch_size, num_iterations)
    train_loader = DataLoader(train_data, batch_sampler=batch_sampler,
                              num_workers=0)

    # Train.
    if verbose:
        pbar = tqdm(total=num_iterations)
        to_set = {'train_loss': np.inf, 'test_loss': np.inf, 'test_acc': 0}
        pbar.set_postfix(**to_set)
    for i, (x,y) in enumerate(train_loader):
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

        if verbose:
            pbar.update(1)
            to_set['train_loss'] = '%05.3f' % loss
            if test_every is not None and i % test_every == 0:
                to_set['test_loss'], to_set['test_acc'] = test_network(network, test_data, device)
                network.train()
            pbar.set_postfix(**to_set)
    if verbose:
        pbar.close()

def test_network(network, data, device, batch_size=512, num_samples=None):
    """Computes the network loss and accuracy on the given dataset."""
    assert num_samples is None or num_samples >= batch_size
    # Prepare network for evaluation (puts layers such as batchnorm,
    # dropout into evaluation mode).
    network.eval()

    num_iters = None if num_samples is None else num_samples//batch_size

    batch_sampler = BatchSampler(len(data), batch_size, num_iters)
    loader = DataLoader(data, batch_sampler=batch_sampler, num_workers=0)
#    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
#                        num_workers=0)

    correct, loss = 0., 0.
    n_samples = 0
    for i, (x,y) in enumerate(loader):
        n_samples += x.shape[0]
        # Send data to device.
        x = x.to(device)
        y = y.to(device)

        # Feed data to network.
        with th.no_grad():
            out = network(x)
            _, pred = out.max(1)

        # Compute loss and accuracy.
        loss += F.cross_entropy(out, y) * len(x)
        correct += pred.eq(y).sum().item()

    #loss = loss/len(data)
    #acc = correct/len(data) * 100.0

    loss = loss/n_samples
    acc = correct/n_samples * 100.

    return loss.item(), acc
