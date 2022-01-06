import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import stable_baselines3.common.callbacks as callbacks
from stable_baselines3 import PPOLagrangian
from stable_baselines3.common.vec_env import (VecNormalize,
                                              sync_envs_normalization)

import pruning.policies as policies
import pruning.utils as utils
import wandb


def cpg(config):
    # Create the vectorized environments
    # Train envs
    r = config.timesteps/config.num_threads
    # Action clip schedule
    m, c = config.train_env_action_clip_grad, config.train_env_action_clip_init
    ac_schedule = lambda t: min(m * t/r + c, 1.)
    # Finetune schedule
    f = config.train_env_finetune_iters
    if f is not None:
        s = int(r / len(f))
        def finetune_schedule(t):
            idx = np.clip(int(t / s), a_min=0, a_max=len(f)-1)
            return f[idx]
    else:
        finetune_schedule = None

    env_args = dict(
            network=config.env_network,
            dataset=config.env_dataset,
            _seed=config.env_seed,
            batch_size=config.env_batch_size,
            reset_weights=True,
            finetune_iters=finetune_schedule,
            finetune_batch_size=config.env_finetune_batch_size,
            optimizer=config.env_optimizer,
            lr=config.env_learning_rate,
            use_train_data=not config.env_use_test_data,
            pruning_scheme=config.env_pruning_scheme,
            reward_type=config.env_reward_type,
            reward_on_masked_layers_only=config.env_reward_on_masked_layers_only,
            cost_scheme=config.env_cost_scheme,
            cost_on_masked_layers_only=config.env_cost_on_masked_layers_only,
            action_clip_value=ac_schedule,
            soft_actions=not config.train_env_hard_actions,
            verbose=False
    )
    train_env = utils.make_train_env(config.train_env_id, config.save_dir, config.seed,
                                     config.num_threads, not config.dont_normalize,
                                     not config.dont_use_cuda, env_args, cost_info_str="cost",
                                     reward_gamma=config.reward_gamma, cost_gamma=config.cost_gamma)

    # Eval env
    if config.eval_every != -1:
        eval_device = "cpu" if config.num_threads == 4 else f"cuda:{config.num_threads}"
        c = config.eval_env_action_clip
        env_args['action_clip_value'] = lambda t: c
        env_args['finetune_iters'] = config.eval_env_finetune_iters
        env_args['use_train_data'] = False
        env_args['soft_actions'] = not config.eval_env_hard_actions
        env_args['verbose'] = True
        eval_env = utils.make_eval_env(config.eval_env_id, "cuda:3", not config.dont_normalize,
                                       env_args)

    # Define and train model
    model = PPOLagrangian(
                policy=config.policy_name,
                env=train_env,
                algo_type="lagrangian",
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                reward_gamma=config.reward_gamma,
                reward_gae_lambda=config.reward_gae_lambda,
                cost_gamma=config.cost_gamma,
                cost_gae_lambda=config.cost_gae_lambda,
                clip_range=config.clip_range,
                clip_range_reward_vf=config.clip_range_reward_vf,
                clip_range_cost_vf=config.clip_range_cost_vf,
                ent_coef=config.ent_coef,
                reward_vf_coef=config.reward_vf_coef,
                cost_vf_coef=config.cost_vf_coef,
                max_grad_norm=config.max_grad_norm,
                use_sde=config.use_sde,
                sde_sample_freq=config.sde_sample_freq,
                target_kl=config.target_kl,
                penalty_initial_value=config.penalty_initial_value,
                penalty_learning_rate=config.penalty_learning_rate,
                penalty_min_value=config.penalty_min_value,
                update_penalty_after=config.update_penalty_after,
                budget=config.budget,
                seed=config.seed,
                device=config.device,
                verbose=config.verbose,
                policy_kwargs=dict(net_arch=utils.get_net_arch(config))
    )

    # All callbacks
    env_stats_dir = os.path.join(config.save_dir, 'env_stats')
    save_env_stats = utils.SaveEnvStatsCallback(train_env, env_stats_dir, False)
    save_periodically = callbacks.CheckpointCallback(
            config.save_every,
            os.path.join(config.save_dir, "models"),
            verbose=0,
            callback_on_new_save=save_env_stats
    )
    adjusted_reward = utils.AdjustedRewardCallback()
    # Organize all callbacks in list
    all_callbacks = [save_periodically, adjusted_reward]

    if config.eval_every != -1:
        save_env_stats = utils.SaveEnvStatsCallback(train_env, env_stats_dir, True)
        save_best = callbacks.EvalCallback(
                eval_env,
                eval_freq=config.eval_every,
                best_model_save_path=config.save_dir,
                n_eval_episodes=1,
                verbose=0,
                callback_on_new_best=save_env_stats,
                callback_for_evaluate_policy=utils.LogEvalCost()
        )
        all_callbacks.append(save_best)

    # Train
    with utils.ProgressBarManager(config.timesteps) as callback:
        all_callbacks.append(callback)
        model.learn(total_timesteps=int(config.timesteps), cost_function="cost",
                    callback=all_callbacks)

    # Save normalization stats
    if isinstance(train_env, VecNormalize):
        train_env.save(os.path.join(config.save_dir, "train_env_stats.pkl"))

    if config.sync_wandb:
        utils.sync_wandb(config.save_dir, 120)

def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    # ========================== Setup ============================== #
    parser.add_argument("--config_file", "-cf", type=str, default=None)
    parser.add_argument("--project", "-p", type=str, default="Pruning")
    parser.add_argument("--name", "-n", type=str, default=None)
    parser.add_argument("--group", "-g", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="cpu")
    parser.add_argument("--dont_use_cuda", "-duc", action="store_true")
    parser.add_argument("--verbose", "-v", type=int, default=2)
    parser.add_argument("--wandb_sweep", "-ws", type=bool, default=False)
    parser.add_argument("--sync_wandb", "-sw", action="store_true")
    # ======================== Environment ========================== #
    parser.add_argument("--train_env_id", "-tei", type=str, default="LWOP")
    parser.add_argument("--eval_env_id", "-eei", type=str, default="LWOP")
    parser.add_argument("--env_dataset", "-ed", type=str, default="cifar10")
    parser.add_argument("--env_network", "-en", type=str, default="vgg11")
    parser.add_argument("--env_seed", "-es", type=int, default=718)
    parser.add_argument("--env_batch_size", "-ebs", type=int, default=2048)
    parser.add_argument("--train_env_finetune_iters", "-tefi", type=int, nargs="*", default=None)
    parser.add_argument("--eval_env_finetune_iters", "-eefi", type=int, default=None)
    parser.add_argument("--train_env_hard_actions", "-teha", action="store_true")
    parser.add_argument("--eval_env_hard_actions", "-eeha", action="store_true")
    parser.add_argument("--env_finetune_batch_size", "-efbs", type=int, default=60)
    parser.add_argument("--env_optimizer", "-eo", type=str, default="adam")
    parser.add_argument("--env_learning_rate", "-elr", type=float, default=0.0003)
    parser.add_argument("--env_use_test_data", "-eutd", action="store_true")
    parser.add_argument("--env_pruning_scheme", "-eps", type=str, default="MP")
    parser.add_argument("--env_reward_type", "-ert", type=str, default="sparse")
    parser.add_argument("--env_reward_on_masked_layers_only", "-eromlo", action="store_true")
    parser.add_argument("--env_cost_scheme", "-ecs", type=str, default="sparsity")
    parser.add_argument("--env_cost_on_masked_layers_only", "-ecomlo", action="store_true")
    parser.add_argument("--train_env_action_clip_init", "-teaci", type=float, default=0.8)
    parser.add_argument("--train_env_action_clip_grad", "-teacg", type=float, default=0.)
    parser.add_argument("--eval_env_action_clip", "-eeac", type=float, default=1.)
    parser.add_argument("--dont_normalize", "-dn", action="store_true")
    parser.add_argument("--seed", "-s", type=int, default=None)
    # ======================== Networks ============================= #
    parser.add_argument("--policy_name", "-pn", type=str, default="TwoCriticsMlpPolicy")
    parser.add_argument("--shared_layers", "-sl", type=int, default=None, nargs='*')
    parser.add_argument("--policy_layers", "-pl", type=int, default=[32,32], nargs='*')
    parser.add_argument("--reward_vf_layers", "-rl", type=int, default=[32,32], nargs='*')
    parser.add_argument("--cost_vf_layers", "-cl", type=int, default=[32,32], nargs='*')
    # ========================= Training ============================ #
    parser.add_argument("--timesteps", "-t", type=lambda x: int(float(x)), default=40000)
    parser.add_argument("--n_steps", "-ns", type=int, default=2500)
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--n_epochs", "-ne", type=int, default=10)
    parser.add_argument("--num_threads", "-nt", type=int, default=4)
    parser.add_argument("--save_every", "-se", type=float, default=2500)
    parser.add_argument("--eval_every", "-ee", type=float, default=2500)
    parser.add_argument("--plot_every", "-pe", type=float, default=2500)
    # =========================== MDP =============================== #
    parser.add_argument("--reward_gamma", "-rg", type=float, default=0.99)
    parser.add_argument("--reward_gae_lambda", "-rgl", type=float, default=0.95)
    parser.add_argument("--cost_gamma", "-cg", type=float, default=0.99)
    parser.add_argument("--cost_gae_lambda", "-cgl", type=float, default=0.95)
    parser.add_argument("--budget", "-b", type=float, default=0.0)
    # ========================= Losses ============================== #
    parser.add_argument("--clip_range", "-cr", type=float, default=0.2)
    parser.add_argument("--clip_range_reward_vf", "-crv", type=float, default=None)
    parser.add_argument("--clip_range_cost_vf", "-ccv", type=float, default=None)
    parser.add_argument("--ent_coef", "-ec", type=float, default=0.)
    parser.add_argument("--reward_vf_coef", "-rvc", type=float, default=0.5)
    parser.add_argument("--cost_vf_coef", "-cvc", type=float, default=0.5)
    parser.add_argument("--target_kl", "-tk", type=float, default=None)
    parser.add_argument("--max_grad_norm", "-mgn", type=float, default=0.5)
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-4)
    # ======================== Lagrangian =========================== #
    parser.add_argument("--penalty_initial_value", "-piv", type=float, default=1)
    parser.add_argument("--penalty_learning_rate", "-plr", type=float, default=0.1)
    parser.add_argument("--penalty_min_value", "-pmv", type=float, default=1)
    parser.add_argument("--update_penalty_after", "-upa", type=int, default=1)
    # =========================== SDE =============================== #
    parser.add_argument("--use_sde", "-us", action="store_true")
    parser.add_argument("--sde_sample_freq", "-ssf", type=int, default=-1)

    args = vars(parser.parse_args())

    # Get default config
    default_config, mod_name = {}, ''
    if args["config_file"] is not None:
        if args["config_file"].endswith(".py"):
            mod_name = args["config_file"].replace('/', '.').strip(".py")
            default_config = importlib.import_module(mod_name).config
        elif args["config_file"].endswith(".json"):
            default_config = utils.load_dict_from_json(args["config_file"])
        else:
            raise ValueError("Invalid type of config file")

    # Overwrite config file with parameters supplied through parser
    # Order of priority: supplied through command line > specified in config
    # file > default values in parser
    config = utils.merge_configs(default_config, parser, sys.argv[1:])

    # Choose seed
    if config["seed"] is None:
        config["seed"] = np.random.randint(0,100)

    # Get name by concatenating arguments with non-default values. Default
    # values are either the one specified in config file or in parser (if both
    # are present then the one in config file is prioritized)
    config["name"] = utils.get_name(parser, default_config, config, mod_name)

    # Initialize W&B project
    wandb.init(project=config["project"], name=config["name"], config=config,
               dir="./pruning", group=config["group"])
    wandb.config.save_dir = wandb.run.dir
    config = wandb.config

    print(utils.colorize("Configured folder %s for saving" % config.save_dir,
          color="green", bold=True))
    print(utils.colorize("Name: %s" % config.name, color="green", bold=True))

    # Save config
    utils.save_dict_as_json(config.as_dict(), config.save_dir, "config")

    # Train
    cpg(config)

    end = time.time()
    print(utils.colorize("Time taken: %05.2f minutes" % ((end-start)/60),
          color="green", bold=True))


if __name__=='__main__':
    main()
