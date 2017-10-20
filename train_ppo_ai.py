#!/usr/bin/env python
import os, sys
import gym, logging
from baselines import bench
from baselines import logger
from environments.redis.environment import LearningEnvironment
from baselines.common import set_global_seeds, tf_util as U


PARTICLES = 2


def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)

    train_env = LearningEnvironment(num_particles=PARTICLES, disable_render=True)
    eval_env = LearningEnvironment(num_particles=PARTICLES, disable_render=False)

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=128, num_hid_layers=3)
    #env = bench.Monitor(env, logger.get_dir() and
    #    os.path.join(logger.get_dir(), "monitor.json"))
    #env.seed(seed)
    #gym.logger.setLevel(logging.WARN)

    pposgd_simple.learn(train_env, eval_env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=1e-4, optim_batchsize=64,
            gamma=0.995, lam=0.95, schedule='linear'
        )
    train_env.close()
    eval_env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    train(args.env, num_timesteps=10e6, seed=args.seed)


if __name__ == '__main__':
    main()

