#!/usr/bin/env python
import os, sys
import gym, logging
import tensorflow as tf
from baselines import bench
from baselines import logger
from datetime import datetime
from baselines.ppo1 import pposgd_simple
from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1.rnn_policy import RnnPolicy
from environments.real.environment import ExecuteEnvironment, LearningEnvironment
from environments.util.stacked_environment import StackedEnvWrapper
from mpi4py import MPI

print("Using tensorflow version: ", tf.__version__)

PARTICLES = 2
TIMESTEPS = 8e7 #3e7
DIRECTORY = 'models/ppo-real-train/'
VAR_REDUCTION = 2


def policy_fn(name, ob_space, ac_space):
    return RnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64, rnn_hid_units=64, num_hid_layers=2)


def train(env_id, num_timesteps, history_len, seed, render):
    U.single_threaded_session().__enter__()

    # We need to make sure the seed is different in each COMM world
    rank = MPI.COMM_WORLD.Get_rank()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    train_env = LearningEnvironment(num_particles=PARTICLES, disable_render=not render)
    train_env = StackedEnvWrapper(train_env, state_history_len=history_len)

    eval_env = LearningEnvironment(num_particles=PARTICLES, disable_render=not render)
    eval_env = StackedEnvWrapper(eval_env, state_history_len=history_len)
    eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), "monitor.json"))


    pposgd_simple.learn(train_env, eval_env, policy_fn,
            directory=DIRECTORY.format(history_len),
            max_timesteps=num_timesteps,
            timesteps_per_batch=1024*VAR_REDUCTION,
            clip_param=0.2,
            entcoeff=0.0001,
            optim_epochs=10,
            optim_stepsize=2e-4,
            optim_batchsize=64,
            gamma=0.995, lam=0.95, schedule='linear',
            render=render
        )
    env.close()


def evaluate(env_id, history_len, render):
    """Evaluate the policy in the simulation"""
    U.single_threaded_session().__enter__()
    env = LearningEnvironment(num_particles=PARTICLES, disable_render=not render)
    env = StackedEnvWrapper(env, state_history_len=history_len)
    directory = DIRECTORY.format(history_len)
    pposgd_simple.run_evaluation(env, policy_fn, directory=directory, render=render)
    env.close()


def execute(env_id, history_len, render):
    """Execute the policy on the real hardware"""
    from baselines.ppo1.rnn_policy import RnnPolicy
    from baselines.ppo1 import pposgd_simple
    U.single_threaded_session().__enter__()
    env = ExecuteEnvironment(num_particles=PARTICLES, disable_render=not render)
    env = StackedEnvWrapper(env, state_history_len=history_len)
    directory = DIRECTORY.format(history_len)
    pposgd_simple.run_evaluation(env, policy_fn, directory=directory, render=render)
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--train', help='Train Model', type=bool, default=False)
    parser.add_argument('--execute', help='Execute real solution', type=bool, default=False)
    parser.add_argument('--render', help='Render evaluation', type=bool, default=False)
    parser.add_argument('--gpu', help='Run on GPU', type=bool, default=False)
    parser.add_argument('--history', help='History steps to remember', type=int, default=4)
    args = parser.parse_args()

    # Configure the logger
    directory = DIRECTORY.format(args.history)
    logger.configure(dir=directory)

    # Disable the GPU
    if not args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if args.train:
        train(args.env, history_len=args.history, num_timesteps=TIMESTEPS, seed=args.seed, render=args.render)
    elif args.execute:
        execute(args.env, history_len=args.history, render=args.render)
    else:
        evaluate(args.env, history_len=args.history, render=args.render)



if __name__ == '__main__':
    main()

