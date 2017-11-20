#!/usr/bin/env python
import os, sys
import gym, logging
from baselines import bench
from baselines import logger
from baselines.ppo1.rnn_policy import RnnPolicy
from baselines.ppo1 import pposgd_simple
from baselines.common import set_global_seeds, tf_util as U
from environments.redis.environment import ExecuteEnvironment
from environments.redis.environment import LearningEnvironment
from mpi4py import MPI


PARTICLES = 1
TIMESTEPS = 6e7 #3e7
DIRECTORY = 'results/ppo-simple'


def policy_fn(name, ob_space, ac_space):
    return RnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64, rnn_hid_units=64, num_hid_layers=2)


def train(env_id, num_timesteps, seed, render):
    U.make_session(num_cpu=1).__enter__()

    # We need to make sure the seed is different in each COMM world
    rank = MPI.COMM_WORLD.Get_rank()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed + rank)

    disable_render = not render
    env = LearningEnvironment(num_particles=PARTICLES, disable_render=True)

    pposgd_simple.learn(env, policy_fn,
            directory=DIRECTORY,
            max_timesteps=num_timesteps,
            timesteps_per_batch=8192,
            clip_param=0.2,
            entcoeff=0.02,
            optim_epochs=10,
            optim_stepsize=1e-4,
            optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
            render=render
        )
    env.close()


def evaluate(env_id, render):
    """Evaluate the policy in the simulation"""
    U.make_session(num_cpu=4).__enter__()
    env = LearningEnvironment(num_particles=PARTICLES, disable_render=not render)
    pposgd_simple.run_evaluation(env, policy_fn, directory=DIRECTORY, render=render)
    env.close()


def execute(env_id, render):
    """Execute the policy on the real hardware"""
    from baselines.ppo1.rnn_policy import RnnPolicy
    from baselines.ppo1 import pposgd_simple
    U.make_session(num_cpu=4).__enter__()
    env = ExecuteEnvironment(num_particles=PARTICLES, disable_render=not render)
    pposgd_simple.run_evaluation(env, policy_fn, directory=DIRECTORY, render=render)
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--train', help='Train Model', type=bool, default=False)
    parser.add_argument('--execute', help='Execute real solution', type=bool, default=False)
    parser.add_argument('--render', help='Render evaluation', type=bool, default=False)
    args = parser.parse_args()

    if args.train:
        train(args.env, num_timesteps=TIMESTEPS, seed=args.seed, render=args.render)
    elif args.execute:
        execute(args.env, render=args.render)
    else:
        evaluate(args.env, render=args.render)



if __name__ == '__main__':
    main()

