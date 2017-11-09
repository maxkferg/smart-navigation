#!/usr/bin/env python
import random
import math
import os, sys
import gym, logging
import tensorflow as tf
from baselines import bench
from baselines import logger
from baselines.ppo1.pposgd_simple import *
from environments.redis.environment import LearningEnvironment
from baselines.common import set_global_seeds, tf_util as U
from mpi4py import MPI


PARTICLES = 2
TIMESTEPS = 4e7 #3e7


class DynamicsModel():

    def __init__(self, batch_size, sequence_length, num_hidden=10):
        num_inputs = 7
        num_outputs = 4

        # Input shape is a batch_size=1, sequence_length=None, input_size=8
        self.inputs = tf.placeholder(tf.float32, shape=(batch_size, sequence_length, num_inputs))
        self.labels = tf.placeholder(tf.float32, shape=(batch_size, num_outputs))

        # Unstack the steps along the history dimension
        x = tf.unstack(self.inputs, num=sequence_length, axis=1)

        # 1-layer Basic Cell with n_hidden units.
        cell = tf.nn.rnn_cell.LSTMCell(num_hidden, activation=tf.nn.elu)

        # Define RNN
        outputs, states = tf.nn.static_rnn(cell, x, dtype=tf.float32)

        self.predictions = tf.layers.dense(outputs[-1], units=num_outputs)
        self.loss_op = tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions)
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss_op)


class LinearDynamicsModel():

    def __init__(self, batch_size, sequence_length, num_hidden=10):
        num_inputs = 7
        num_outputs = 4

        # Input shape is a batch_size=1, sequence_length=None, input_size=8
        self.inputs = tf.placeholder(tf.float32, shape=(batch_size, sequence_length, num_inputs))
        self.labels = tf.placeholder(tf.float32, shape=(batch_size, num_outputs))

        # Unstack the steps along the history dimension
        x = tf.unstack(self.inputs, num=sequence_length, axis=1)

        # 1-layer Basic Cell with n_hidden units.
        cell = tf.nn.rnn_cell.LSTMCell(num_hidden, activation=tf.nn.elu)

        # Define RNN
        outputs, states = tf.nn.static_rnn(cell, x, dtype=tf.float32)

        self.predictions = tf.layers.dense(outputs[-1], units=num_outputs)
        self.loss_op = tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions)
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss_op)




def sample(env,pi,n=4,render=True):
    while True:
        X = []
        ob = env.reset()

        while True:
            stochastic = True
            ac, vpred = pi.act(stochastic, ob)
            inputs = [
                env.primary.x/800,
                env.primary.y/800,
                math.sin(env.primary.angle),
                math.cos(env.primary.angle),
                math.tan(env.primary.angle),
                ac[0],
                ac[1],
            ]
            ob, rew, done, _ = env.step(ac)

            # Discard remaininf data becuse the simulation is done
            if done:
                break

            labels = [
                env.primary.x/800,
                env.primary.y/800,
                math.sin(env.primary.angle),
                math.cos(env.primary.angle),
            ]
            # Add to our input/output buffers
            # Most recent data comes last
            X.append(inputs)

            if render:
                env.render()
                env.background = get_v_background(env, pi, stochastic)

            # Only return once we have four consecutive samples
            if len(X)==n:
                yield (X[:],labels)
                X.pop(0)


def train(num_timesteps, seed, evaluate, render):
    from baselines.ppo1.rnn_policy import RnnPolicy
    from baselines.ppo1 import pposgd_simple

    U.make_session(num_cpu=8).__enter__()

    # We need to make sure the seed is different in each COMM world
    rank = MPI.COMM_WORLD.Get_rank()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed + rank)

    env = LearningEnvironment(num_particles=PARTICLES, disable_render=not render)

    batch_size = 100
    optim_steps = 1000
    sequence_length = 4

    # Get a trained policy
    saver = Saver()
    pi = RnnPolicy(name="pi", ob_space=env.observation_space, ac_space=env.action_space, hid_size=64, rnn_hid_units=64, num_hid_layers=2)
    dynamics = DynamicsModel(sequence_length=sequence_length, batch_size=None)
    U.initialize()

    saver.restore_model("results/ppo-cloud") # Load weights
    sess = U.get_session()

    # Collect a set of samples
    inputs = []
    labels = []
    sampler = sample(env, pi, n=sequence_length, render=render)

    for k in range(500):
        # Get 100 samples
        for i in range(batch_size):
            X,Y = next(sampler)
            inputs.append(X)
            labels.append(Y)

        # Optimise heavily
        for i in range(optim_steps):
            feed = {dynamics.inputs: inputs, dynamics.labels: labels}
            _, loss, predictions = sess.run([dynamics.train_op, dynamics.loss_op, dynamics.predictions], feed_dict=feed)
            print(k, ':',i, '  ', loss)
        print('------------------')
        print('inputs:     ',inputs[0][-1][:4])
        print('labels:     ',labels[0])
        print('predictions:',predictions[0])
        print('------------------')

        # Test the model
        x,y = next(sampler)
        feed = {dynamics.inputs: [x], dynamics.labels: [y]}
        yhat = sess.run(dynamics.predictions, feed_dict=feed)
        print('y:   ',y)
        print('yhat:',yhat)



    env.close()



def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--eval', help='Evaluate solution', type=bool, default=False)
    parser.add_argument('--render', help='Render evaluation', type=bool, default=True)
    args = parser.parse_args()
    train(num_timesteps=TIMESTEPS, seed=args.seed, evaluate=args.eval, render=args.render)


if __name__ == '__main__':
    main()

