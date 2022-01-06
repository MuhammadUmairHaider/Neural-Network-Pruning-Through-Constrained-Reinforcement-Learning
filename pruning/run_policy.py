import argparse
import os
import shutil
import time

import numpy as np
from stable_baselines3 import PPOLagrangian
from stable_baselines3.common.vec_env import VecNormalize

import pruning.utils as utils
import wandb

USER="spiderbot"

def load_config(d):
    config = utils.load_dict_from_json(d, "config")
    config = utils.dict_to_namespace(config)
    return config

def _run_policy(args, new_dir):
    # Find which file to load
    if args.load_itr is not None:
        f = "models/rl_model_%s_steps" % args.load_itr
        e = "env_stats/train_env_stats_%d.pkl" % args.load_itr
    else:
        f = "best_model"
        e = "env_stats/train_env_stats_best.pkl"

    # Configure paths (restore from W&B server if needed)
    if args.remote:
        # Save everything in pruning/wandb/remote/<run_id>
        load_dir = os.path.join("pruning/wandb/remote/", args.load_dir.split('/')[-1])
        utils.del_and_make(load_dir)
        # Restore form W&B
        wandb.init(dir=load_dir)
        run_path = os.path.join(USER, args.load_dir)
        wandb.restore("config.json", run_path=run_path, root=load_dir)
        config = load_config(load_dir)
        if not config.dont_normalize:
            wandb.restore(e, run_path=run_path, root=load_dir)
        wandb.restore(f+".zip", run_path=run_path, root=load_dir)
    else:
        load_dir = os.path.join(args.load_dir, "files")
        config = load_config(load_dir)

    save_dir = os.path.join(load_dir, args.save_dir)
    if new_dir:
        utils.del_and_make(save_dir)
    model_path = os.path.join(load_dir, f)

    # Load model
    model = PPOLagrangian.load(model_path)

    # Create the vectorized environments
    env_args = dict(
            network=args.env_network,
            dataset=args.env_dataset,
            _seed=args.env_seed,
            batch_size=args.env_batch_size,
            reset_weights=True,
            finetune_iters=lambda t: args.env_finetune_iters,
            finetune_batch_size=args.env_finetune_batch_size,
            optimizer=args.env_optimizer,
            lr=args.env_learning_rate,
            use_train_data=not args.env_use_test_data,
            pruning_scheme=args.env_pruning_scheme,
            reward_type=args.env_reward_type,
            reward_on_masked_layers_only=args.env_reward_on_masked_layers_only,
            cost_scheme=args.env_cost_scheme,
            cost_on_masked_layers_only=args.env_cost_on_masked_layers_only,
            action_clip_value=lambda x: args.env_action_clip_value,
            soft_actions=not args.env_hard_actions,
            verbose=True
    )

    # Create env, model
    def make_env():
        env_id = args.env_id or config.eval_env_id
        env = utils.make_eval_env(env_id, 'cuda:0', normalize=False, env_kwargs=env_args)

        # Restore enviroment stats
        if not config.dont_normalize:
            env = VecNormalize.load(os.path.join(load_dir, e), env)
            env.norm_reward = False
            env.training = False

        return env

    # Evaluate
    env = make_env()
    mean_rew, std_rew = utils.eval_model(env, model, args.n_rollouts, deterministic=False)

    with open(os.path.join(save_dir, 'results.txt'), 'a') as f:
        sp = ' '*4
        if new_dir:
            f.write(f'Iteration{sp}Reward{sp}Std\n')
        s1 = str('best' if args.load_itr is None else args.load_itr).ljust(9)
        s2 = f'{mean_rew:04.3f}'
        s3 = f'{std_rew:03.3f}'
        f.write(f'{s1}{sp}{s2}{sp}{s3}\n')

def run_policy(args):
    if args.load_itr is None:
        _run_policy(args, not args.old_dir)
    else:
        for i, li in enumerate(args.load_itr):
            args.load_itr = li
            _run_policy(args, new_dir=(i==0 and not args.old_dir))


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", "-l", type=str, default="pruning/wandb/latest-run/")
    parser.add_argument("--old_dir", "-od", action="store_true")
    parser.add_argument("--remote", "-r", action="store_true")
    parser.add_argument("--save_dir", "-s", type=str, default="run_policy")
    parser.add_argument("--load_itr", "-li", type=int, nargs="*", default=None)
    parser.add_argument("--n_rollouts", "-nr", type=int, default=1)
    # ============================ Environment ========================================== #
    parser.add_argument("--env_id", "-e", type=str, default=None)
    parser.add_argument("--env_dataset", "-ed", type=str, default="cifar10")
    parser.add_argument("--env_network", "-en", type=str, default="vgg11")
    parser.add_argument("--env_seed", "-es", type=int, default=718)
    parser.add_argument("--env_batch_size", "-ebs", type=int, default=2048)
    parser.add_argument("--env_finetune_iters", "-efi", type=int, default=None)
    parser.add_argument("--env_finetune_batch_size", "-efbs", type=int, default=60)
    parser.add_argument("--env_optimizer", "-eo", type=str, default="adam")
    parser.add_argument("--env_learning_rate", "-elr", type=float, default=0.0003)
    parser.add_argument("--env_use_test_data", "-eutd", action="store_true")
    parser.add_argument("--env_pruning_scheme", "-eps", type=str, default="MP")
    parser.add_argument("--env_reward_type", "-ert", type=str, default="sparse")
    parser.add_argument("--env_reward_on_masked_layers_only", "-eromlo", action="store_true")
    parser.add_argument("--env_cost_scheme", "-ecs", type=str, default="sparsity")
    parser.add_argument("--env_cost_on_masked_layers_only", "-ecomlo", action="store_true")
    parser.add_argument("--env_hard_actions", "-eha", action="store_true")
    parser.add_argument("--env_action_clip_value", "-eacv", type=float, default=1.)
    args = parser.parse_args()

    run_policy(args)
    end = time.time()
    print(f'Time taken: {(end-start)/60:3.2f}m')

if __name__=="__main__":
    main()
