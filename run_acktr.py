#!/usr/bin/env python
import argparse
import logging
import os
import tensorflow as tf
import gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction
from environments.redis.environment import LearningEnvironment, ObservationSpace

PARTICLES = 2

class AcktrEnv(LearningEnvironment):
    """
    Patch the environment so it works with acktr
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.observation_space = ObservationSpace(-1, 1, shape=(self.state_size))

    def _last_state(self,ob):
        return ob[-1,:]

    def reset(self,*args,**kwargs):
        ob = super().reset(*args,**kwargs)
        return self._last_state(ob)

    def step(self,*args,**kwargs):
        ob, rew, done, info = super().step(*args,**kwargs)
        return self._last_state(ob), rew, done, info



def train(env_id, num_timesteps, seed, render):
    #env=gym.make(env_id)
    env = AcktrEnv(num_particles=PARTICLES, disable_render=not render)
    #if logger.get_dir():
    #    env = bench.Monitor(env, os.path.join(logger.get_dir(), "monitor.json"))
    #set_global_seeds(seed)
    #env.seed(seed)
    #gym.logger.setLevel(logging.WARN)

    with tf.Session(config=tf.ConfigProto()) as session:
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        learn(env, policy=policy, vf=vf,
            gamma=0.99, lam=0.97, timesteps_per_batch=8000,
            desired_kl=0.0002,
            num_timesteps=num_timesteps,
            animate=True)

        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Mujoco benchmark.')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', help='environment ID', type=str, default="Reacher-v1")
    parser.add_argument('--render', help='Choose whether to render', type=bool, default=False)
    args = parser.parse_args()
    train(args.env, num_timesteps=5e7, seed=args.seed, render=args.render)
