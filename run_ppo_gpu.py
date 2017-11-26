#!/usr/bin/env python
import gym
import tensorflow as tf
import argparse
from baselines import bench, logger
from environments.redis.environment import ExecuteEnvironment
from environments.redis.environment import LearningEnvironment


ENVIRONMENTS = 8
PARTICLES = 2
TIMESTEPS = 5e7 #3e7
DIRECTORY = 'results/ppo-gpu'

def train(env_id, num_timesteps, seed, render):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import PureLstmPolicy
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

    ncpu = 6
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env(rank):
        def env_fn():
            env = LearningEnvironment(num_particles=PARTICLES, disable_render=not render)
            env = bench.Monitor(env, logger.get_dir())
            return env
        return env_fn

    #env = DummyVecEnv([make_env])
    env = SubprocVecEnv([make_env(i) for i in range(ENVIRONMENTS)])
    #env = VecNormalize(env)

    set_global_seeds(seed)
    policy = PureLstmPolicy
    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=ENVIRONMENTS,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--render', type=bool, default=False)
    args = parser.parse_args()
    logger.configure(dir=DIRECTORY)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, render=args.render)


if __name__ == '__main__':
    main()

