import collections
import json
import math
import os
import pickle
import shutil
import subprocess
import types
from collections.abc import Callable

import gym
import numpy as np
import stable_baselines3.common.callbacks as callbacks
import stable_baselines3.common.vec_env as vec_env
import torch as th
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import Dataset
from stable_baselines3.common import logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import safe_mean, set_random_seed
from tqdm import tqdm

import pruning.envs as envs
from pruning.networks.masked_modules import MaskedConv2d, MaskedLinear

#==============================================================================
# Functions to handle parser.
#==============================================================================

def concat_nondefault_arguments(parser, ignore_keys=[], path_keys=[],
                                default_config=None, actual_config=None):
    """
    Given an instance of argparse.ArgumentParser, return a concatenation
    of the names and values of only those arguments that do not have the
    default value (i.e. alternative values were specified via command line).

    So if you run

        python file.py -abc 123 -def 456

    this function will return abc_123_def_456. (In case the value passed for
    'abc' or 'def' is the same as the default value, they will be ignored)

    If a shorter version of the argument name is specified, it will be
    preferred over the longer version.

    Arguments are sorted alphabetically.

    If you want this function to ignore some arguments, you can pass them
    as a list to the ignore_keys argument.

    If some arguments expect paths, you can pass in those as a list to the
    path_keys argument. The values of these will be split at '/' and only the
    last substring will be used.

    If the default config dictionary is specified then the default values in it
    are preferred ovr the default values in parser.

    If the actual_config dictionary is specified then the values in it are preferred
    over the values passed through command line.
    """
    sl_map = get_sl_map(parser)

    def get_default(key):
        if default_config is not None and key in default_config:
            return default_config[key]
        return parser.get_default(key)

    # Determine save dir based on non-default arguments if no
    # save_dir is provided.
    concat = ''
    for key, value in sorted(vars(parser.parse_args()).items()):
        if actual_config is not None:
            value = actual_config[key]

        # Skip these arguments.
        if key in ignore_keys:
            continue

        if type(value) == list:
            b = False
            if get_default(key) is None or len(value) != len(get_default(key)):
                b = True
            else:
                for v, p in zip(value, get_default(key)):
                    if v != p:
                        b = True
                        break
            if b:
                concat += '%s_' % sl_map[key]
                for v in value:
                    if type(v) not in [bool, int] and hasattr(v, "__float__"):
                        if v == 0:
                            valstr = 0
                        else:
                            valstr = round(v, 4-int(math.floor(math.log10(abs(v))))-1)
                    else: valstr = v
                    concat += '%s_' % str(valstr)

        # Add key, value to concat.
        elif value != get_default(key):
            # For paths.
            if value is not None and key in path_keys:
                value = value.split('/')[-1]

            if type(value) not in [bool, int] and hasattr(value, "__float__"):
                if value == 0:
                    valstr = 0
                else:
                    valstr = round(value, 4-int(math.floor(math.log10(abs(value))))-1)
            else: valstr = value
            concat += '%s_%s_' % (sl_map[key], valstr)

    if len(concat) > 0:
        # Remove extra underscore at the end.
        concat = concat[:-1]

    return concat

def get_sl_map(parser):
    """Return a dictionary containing short-long name mapping in parser."""
    sl_map = {}

    # Add arguments with long names defined.
    for key in parser._option_string_actions.keys():
        if key[1] == '-':
            options = parser._option_string_actions[key].option_strings
            if len(options) == 1:   # No short argument.
                sl_map[key[2:]] = key[2:]
            else:
                if options[0][1] == '-':
                    sl_map[key[2:]] = options[1][1:]
                else:
                    sl_map[key[2:]] = options[0][1:]

    # We've now processed all arguments with long names. Now need to process
    # those with only short names specified.
    known_keys = list(sl_map.keys()) + list(sl_map.values())
    for key in parser._option_string_actions.keys():
        if key[1:] not in known_keys and key[2:] not in known_keys:
            sl_map[key[1:]] = key[1:]

    return sl_map

def reverse_dict(x):
    """
    Exchanges keys and values in x i.e. x[k] = v ---> x[v] = k.
    Added Because reversed(x) does not work in python 3.7.
    """
    y = {}
    for k,v in x.items():
        y[v] = k
    return y

def merge_configs(config, parser, sys_argv):
    """
    Merge a dictionary (config) and arguments in parser. Order of priority:
    argument supplied through command line > specified in config > default
    values in parser.
    """

    parser_dict = vars(parser.parse_args())
    config_keys = list(config.keys())
    parser_keys = list(parser_dict.keys())

    sl_map = get_sl_map(parser)
    rev_sl_map = reverse_dict(sl_map)
    def other_name(key):
        if key in sl_map:
            return sl_map[key]
        elif key in rev_sl_map:
            return rev_sl_map[key]
        else:
            return key

    merged_config = {}
    for key in config_keys + parser_keys:
        if key in parser_keys:
            # Was argument supplied through command line?
            if key_was_specified(key, other_name(key), sys_argv):
                merged_config[key] = parser_dict[key]
            else:
                # If key is in config, then use value from there.
                if key in config:
                    merged_config[key] = config[key]
                else:
                    merged_config[key] = parser_dict[key]
        elif key in config:
            # If key was only specified in config, use value from there.
            merged_config[key] = config[key]

    return merged_config

def key_was_specified(key1, key2, sys_argv):
    for arg in sys_argv:
        if arg[0] == '-' and (key1 == arg.strip('-') or key2 == arg.strip('-')):
            return True
    return False

def get_name(parser, default_config, actual_config, mod_name):
    """Returns a name for the experiment based on parameters passed."""
    prefix = lambda x, y: x + '_'*(len(y)>0) + y

    name = actual_config["name"]
    if name is None:
        name = concat_nondefault_arguments(
                parser,
                ignore_keys=["config_file", "train_env_id", "eval_env_id", "seed",
                             "timesteps", "save_every", "eval_every", "n_iters",
                             "sync_wandb", "file_to_run"],
                path_keys=["expert_path"],
                default_config=default_config,
                actual_config=actual_config
        )
        if len(mod_name) > 0:
            name = prefix(mod_name.split('.')[-1], name)

        name = prefix(actual_config["train_env_id"], name)

    # Append seed and system id regardless of whether the name was passed in
    # or not
    if "wandb_sweep" in actual_config and not actual_config["wandb_sweep"]:
        sid = get_sid()
    else:
        sid = "-1"
    name = name + "_s_" + str(actual_config["seed"]) + "_sid_" + sid

    return name

# =====================================================================
# Dataset
# =====================================================================

# =====================================================================
# Imagenet Custom Class
class ImageNetTrain(Dataset):
  def __init__(self, root_dir, transform=None):
    self.transform = transform
    with open(root_dir+"train_data_batch_1", 'rb') as fo:
        train_batch1 = pickle.load(fo)
    with open(root_dir+"train_data_batch_2", 'rb') as fo:
        train_batch2 = pickle.load(fo)
    with open(root_dir+"train_data_batch_3", 'rb') as fo:
        train_batch3 = pickle.load(fo)
    with open(root_dir+"train_data_batch_4", 'rb') as fo:
        train_batch4 = pickle.load(fo)
    with open(root_dir+"train_data_batch_5", 'rb') as fo:
        train_batch5 = pickle.load(fo)
    with open(root_dir+"train_data_batch_6", 'rb') as fo:
        train_batch6 = pickle.load(fo)
    with open(root_dir+"train_data_batch_7", 'rb') as fo:
        train_batch7 = pickle.load(fo)
    with open(root_dir+"train_data_batch_8", 'rb') as fo:
        train_batch8 = pickle.load(fo)
    with open(root_dir+"train_data_batch_9", 'rb') as fo:
        train_batch9 = pickle.load(fo)
    with open(root_dir+"train_data_batch_10", 'rb') as fo:
        train_batch10 = pickle.load(fo)
    self.labels = th.cat((th.tensor(train_batch1['labels']),th.tensor(train_batch2['labels']),th.tensor(train_batch3['labels'])
    ,th.tensor(train_batch4['labels']),th.tensor(train_batch5['labels']),th.tensor(train_batch6['labels'])
    ,th.tensor(train_batch7['labels']),th.tensor(train_batch8['labels']),th.tensor(train_batch9['labels'])
    ,th.tensor(train_batch10['labels'])))
    self.train = th.cat((th.tensor(train_batch1['data']),th.tensor(train_batch2['data'])))
    train_batch1 = None
    train_batch2 = None
    self.train = th.cat((self.train,th.tensor(train_batch3['data'])))
    train_batch3 = None
    self.train = th.cat((self.train,th.tensor(train_batch4['data'])))
    train_batch4 = None
    self.train = th.cat((self.train,th.tensor(train_batch5['data'])))
    train_batch5 = None
    self.train = th.cat((self.train,th.tensor(train_batch6['data'])))
    train_batch6 = None
    self.train = th.cat((self.train,th.tensor(train_batch7['data'])))
    train_batch7 = None
    self.train = th.cat((self.train,th.tensor(train_batch8['data'])))
    train_batch8 = None
    self.train = th.cat((self.train,th.tensor(train_batch9['data'])))
    train_batch9 = None
    self.train = th.cat((self.train,th.tensor(train_batch10['data'])))
    train_batch10 = None
  
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    x = self.train[idx].type(th.float)
    x = x/255
    if self.transform:
      x = self.transform(x)
    return (x.reshape((3,32,32)),self.labels[idx]-1)

class ImageNetVal(Dataset):
  def __init__(self, root_dir, transform=None):
    self.transform = transform
    with open(root_dir+"val_data", 'rb') as fo:
        val_data = pickle.load(fo)
    self.val = th.tensor(val_data['data'])
    self.labels = th.tensor(val_data['labels'])
  
  def __len__(self):
    return len(self.labels)
    
  def __getitem__(self, idx):
    x = self.val[idx].type(th.float)
    x = x/255
    if self.transform:
      x = self.transform(x)
    return (x.reshape((3,32,32)),self.labels[idx]-1)
# =====================================================================

ROOT = "pruning/datasets/"

def get_dataset(dataset, normalize=True):
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
    save_dir = os.path.join(ROOT, dataset)

    if dataset == "mnist":
        if normalize:
            transform = T.Compose([
                T.ToTensor(), T.Normalize((0.1307,), (0.3081,))
                ])
        else:
            transform = T.ToTensor()

        train = datasets.MNIST(save_dir, train=True,
                               download=True, transform=transform)
        test = datasets.MNIST(save_dir, train=False,
                              download=True, transform=transform)

    elif dataset == "fmnist":
        transform = T.ToTensor()
        train = datasets.FashionMNIST(save_dir, train=True,
                                      download=True, transform=transform)
        test = datasets.FashionMNIST(save_dir, train=False,
                                     download=True, transform=transform)

    elif dataset == "cifar10":
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        train = datasets.CIFAR10(save_dir, train=True,
                                 download=True, transform=transform_train)
        test = datasets.CIFAR10(save_dir, train=False,
                                download=True, transform=transform_test)

    elif dataset == "cifar100":
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train = datasets.CIFAR100(save_dir, train=True,
                                 download=True, transform=transform_train)
        test = datasets.CIFAR100(save_dir, train=False,
                                download=True, transform=transform_test)

    elif dataset == "imagenet":
        train = ImageNetTrain(ROOT+"imagenet/train/",transform=None)
        test = ImageNetVal(ROOT+"imagenet/test/",transform=None)
    
    elif dataset == "tiny-imagenet":
        transform_train = T.Compose([
            T.RandomResizedCrop(64),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        train = datasets.ImageFolder('./pruning/datasets/tiny-imagenet-200/train',
                                     transform_train)
        test = datasets.ImageFolder('./pruning/datasets/tiny-imagenet-200/val',
                                    transform_test)

    else:
        raise ValueError('Unknown dataset %s.' % dataset)

    return train, test

# =====================================================================
# Networks
# =====================================================================

def is_base_module(m):
    return isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)

def is_masked_module(m):
    return isinstance(m, MaskedLinear) or isinstance(m, MaskedConv2d)

def is_batch_norm(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)

def is_masked_cnn(m):
    return isinstance(m, MaskedConv2d)

def get_network_sparsity(network, func=is_masked_module, threshold=0.5):
    nonzero = 0
    total = 0
    for name, m in network.named_modules():
        if func(m):
            p = m.mask
            nz_count = (p >= threshold).type(th.float32).sum().item()
            total_count = p.numel()
            nonzero += nz_count
            total += total_count

    return 1-nonzero/total

def get_optimizer(optimizer):
    if optimizer == "adam":
        fn = optim.Adam

    return fn

# =============================================================================
# Environment
# =============================================================================

def make_env(env_id, rank, log_dir, seed=0, use_cuda=True, **env_kwargs):
#    device = {0:0,1:1,2:3,3:1}[rank%3]
    device = {0:0,1:0,2:0,3:0}[rank%3]
    device = f"cuda:{device}" if use_cuda else "cpu"
    def _init():
        env = get_env_function(env_id)(device=device, **env_kwargs)
        env.seed(seed + rank)
        env = Monitor(env, log_dir, track_keywords=("cost",),
                      info_keywords=("sparsity","max_action","min_action"))
        #env = Monitor(env, log_dir, info_keywords=("test_accuracy", "sparsity"), track_keywords=("cost",))
        return env
    return _init

def make_train_env(env_id, save_dir, base_seed=0, num_threads=1, normalize=True,
                   use_cuda=True, env_kwargs={}, **kwargs):
    set_random_seed(base_seed, using_cuda=use_cuda)
    env = vec_env.SubprocVecEnv([make_env(env_id, i, save_dir, base_seed, use_cuda,
                                          **env_kwargs)
                                 for i in range(num_threads)])
    if normalize:
        #env = vec_env.VecNormalize(env)
        assert(all(key in kwargs for key in ['cost_info_str','reward_gamma','cost_gamma']))
        env = vec_env.VecNormalizeWithCost(env, training=True, norm_obs=True, norm_reward=True,
                                           norm_cost=True, cost_info_str=kwargs['cost_info_str'],
                                           reward_gamma=kwargs['reward_gamma'],
                                           cost_gamma=kwargs['cost_gamma'])
        return env

def make_eval_env(env_id, device, normalize=True, env_kwargs={}):
    env = [lambda: get_env_function(env_id)(device=device, **env_kwargs)]
    env = vec_env.SubprocVecEnv(env)
    if normalize:
        #env = vec_env.VecNormalize(env, training=False, norm_reward=False)
        env = vec_env.VecNormalizeWithCost(env, training=False, norm_obs=True, norm_reward=False,
                                           norm_cost=False)
    return env

def get_env_function(env_id):
    dic = {
            "NWOP": envs.NetworkWise,
            "LWOP": envs.LayerWise,
            "FP"  : envs.FilterPruning
        }
    return dic[env_id]

def eval_model(env, model, n_rollouts=3, deterministic=False):
    """This will also close the environment"""
    # Make a video
    mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_rollouts, deterministic=deterministic
    )
    print("Mean reward: %f +/- %f." % (mean_reward, std_reward))

    # Save the video
    env.close()

    return mean_reward, std_reward

# =============================================================================
# Callbacks
# =============================================================================

class AdjustedRewardCallback(callbacks.BaseCallback):
    """This callback computes the adjusted reward i.e. r - lambda*c"""
    def __init__(self, cost_fn=None):
        super(AdjustedRewardCallback, self).__init__()
        self.history = []        # Use for smoothing if needed
        self.cost_fn = cost_fn

    def _init_callback(self):
        pass

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        rewards = self.model.rollout_buffer.rewards.copy()
        costs = self.model.rollout_buffer.costs.copy()
        if isinstance(self.training_env, vec_env.VecNormalize):
            rewards = self.training_env.unnormalize_reward(rewards)
        adjusted_reward = (np.mean(rewards - self.model.dual.nu().item()*costs))
        self.logger.record("train/adjusted_reward", float(adjusted_reward))

        if self.cost_fn is not None:
            obs = self.model.rollout_buffer.orig_observations.copy()
            acs = self.model.rollout_buffer.actions.copy()
            cost = np.mean(self.cost_fn(obs, acs))
            self.logger.record("eval/true_cost", float(cost))

class ProgressBarCallback(callbacks.BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = int(self.num_timesteps)
        self._pbar.update(0)

    def _on_rollout_end(self):
        total_reward = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        average_cost = safe_mean(self.model.rollout_buffer.costs)
        total_cost = safe_mean([ep_info["cost"] for ep_info in self.model.ep_info_buffer])
        self._pbar.set_postfix(
                tr='%05.1f' % total_reward,
                tc='%05.1f' % total_cost,
                ac='%05.3f' % average_cost,
                nu='%05.1f' % self.model.dual.nu().item()
        )

# This callback should be used with the 'with' block, to allow for correct
# initialisation and destruction
class ProgressBarManager:
    def __init__(self, total_timesteps): # init object with total timesteps
        self.pbar = None
        self.total_timesteps = int(total_timesteps)

    def __enter__(self): # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps, dynamic_ncols=True)
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

class SaveEnvStatsCallback(callbacks.BaseCallback):
    def __init__(
            self,
            env,
            save_path,
            for_best,
    ):
        super(SaveEnvStatsCallback, self).__init__()
        self.env = env
        self.save_path = save_path
        self.for_best = for_best
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

    def _on_step(self):
        if isinstance(self.env, vec_env.VecNormalize):
            if self.for_best:
                self.env.save(os.path.join(self.save_path, "train_env_stats_best.pkl"))
            else:
                self.env.save(os.path.join(self.save_path, f"train_env_stats_{self.num_timesteps}.pkl"))

class LogEvalCost:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ep_mean_costs = []
        self.ep_mean_sparsity = []
        self.ep_cost = 0

    def __call__(self, loc, glob):
        self.ep_cost += loc['_info'][0]['cost']

        if loc['done'][0]:
            self.ep_mean_costs.append(np.mean(self.ep_cost))
            self.ep_mean_sparsity.append(loc['_info'][0]['sparsity'])
            self.ep_cost = 0

            if loc['i'] == loc['n_eval_episodes']-1:
                mean_cost = np.mean(self.ep_mean_costs)
                mean_sparsity = np.mean(self.ep_mean_sparsity)
                self.reset()
                logger.record("eval/mean_ep_cost", mean_cost)
                logger.record("eval/mean_ep_sparsity", mean_sparsity)

# =============================================================================
# File handlers
# =============================================================================

def save_dict_as_json(dic, save_dir, name=None):
    if name is not None:
        save_dir = os.path.join(save_dir, name+".json")
    with open(save_dir, 'w') as out:
        out.write(json.dumps(dic, separators=(',\n','\t:\t'),
                  sort_keys=True))

def load_dict_from_json(load_from, name=None):
    if name is not None:
        load_from = os.path.join(load_from, name+".json")
    with open(load_from, "rb") as f:
        dic = json.load(f)

    return dic

def save_dict_as_pkl(dic, save_dir, name=None):
    if name is not None:
        save_dir = os.path.join(save_dir, name+".pkl")
    with open(save_dir, 'wb') as out:
        pickle.dump(dic, out, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict_from_pkl(load_from, name=None):
    if name is not None:
        load_from = os.path.join(load_from, name+".pkl")
    with open(load_from, "rb") as out:
        dic = pickle.load(out)

    return dic

# =====================================================================
# Miscellaneous
# =====================================================================

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def del_and_make(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)

def dict_to_nametuple(dic):
    return collections.namedtuple("NamedTuple", dic)(**dic)

def dict_to_namespace(dic):
    return types.SimpleNamespace(**dic)

def get_net_arch(config):
    """
    Returns a dictionary with sizes of layers in policy network,
    value network and cost value network.
    """
    separate_layers = dict(pi=config.policy_layers,    # Policy Layers
                           vf=config.reward_vf_layers, # Value Function Layers
                           cvf=config.cost_vf_layers)  # Cost Value Function Layers
    if config.shared_layers is not None:
        return [*config.shared_layers, separate_layers]
    else:
        return [separate_layers]

def get_sid():
    try:
        sid = subprocess.check_output(['/bin/bash', '-i', '-c', "who_am_i"], timeout=2).decode("utf-8").split('\n')[-2]
        sid = sid.lower()
        if "system" in sid:
            sid = sid.strip("system")
        else:
            sid = -1
    except:
        sid = -1
    return str(sid)

def sync_wandb(folder, timeout=None):
    folder = folder.strip("/files")
    print(colorize("\nSyncing %s to wandb" % folder, "green", bold=True))
    run_bash_cmd("wandb sync %s" % folder, timeout)

def run_bash_cmd(cmd, timeout=None):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    try:
        output, error = process.communicate(timeout=timeout)
    except:
        pass


if __name__=='__main__':
    # Download all datasets. Tiny-imagenet needs to downloaded manually from
    # http://cs231n.stanford.edu/tiny-imagenet-200.zip
    for dataset in ["mnist", "fmnist", "cifar10"]:
        get_dataset(dataset)
