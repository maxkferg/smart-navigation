import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from learners.ddpg.ddpg import DDPG, REPLAY_START_SIZE
from environments.redis.environment import LearningEnvironment
from debug import *


t = datetime.now().strftime('%H-%M')
PATH = 'results/ddpg/checkpoints'.format(t)
LOGS = 'results/ddpg/logs/{0}'.format(t)
EPOCHS = 100000
EPISODES = 20
PARTICLES = 2
RENDER = True
STEP = 4



def fill_buffer(env, agent, epsilon):
    while agent.replay_buffer.count() <=  REPLAY_START_SIZE:
        done = False
        state = env.reset()
        rewards = 0
        # Training
        while not done:
            action = agent.noise_action(state, epsilon)
            next_state, reward, done, info = env.step(action, STEP)
            agent.perceive(state, action, reward, next_state, done)
            # Setup for next cycle
            state = next_state
            rewards += reward


def train(env, agent, epsilon):
    done = False
    state = env.reset()
    rewards = 0
    # Training
    while not done:
        #action = agent.noise_action(state, epsilon)
        action = agent.action(state)
        next_state, reward, done, info = env.step(action, STEP)
        agent.perceive(state, action, reward, next_state, done)
        # Setup for next cycle
        state = next_state
        rewards += reward
    return rewards

def test(env, agent, render=False):
    state = env.reset()
    done = False
    rewards = 0
    while not done:
        action = agent.action(state)
        next_state, reward, done, info = env.step(action, STEP)
        state = next_state
        rewards += reward
        if render:
            #env.background = get_q_background(env, agent, action)
            env.render()
    return rewards


if __name__=='__main__':
    # Setup
    epsilon = 0.4
    disabled = RENDER
    env = LearningEnvironment(num_particles=PARTICLES, disable_render=False)
    writer = tf.summary.FileWriter(LOGS, graph=tf.get_default_graph())
    agent = DDPG(env,writer)
    agent.restore_model(PATH)

    # Fill the buffer
    fill_buffer(env, agent, epsilon)

    # Train on a large number of epochs
    for epoch in range(EPOCHS):
        print("\nEPOCH: {0} epsilon={1:.3f}".format(epoch,epsilon))
        rewards = []

        # Run a few episodes
       # env.switch_backend("simulation")
        for episode in tqdm(range(EPISODES)):
            reward = train(env, agent, epsilon)
            rewards.append(reward)

        # Evaluate
        train_reward = np.mean(rewards)
        test_reward = np.mean([test(env, agent) for i in range(20)])
        print("Train Reward {0}, Test Reward {1}".format(train_reward, test_reward))

        #env.switch_backend("simulation")
        test(env, agent, render=RENDER)

        # Test in real environment
        #env.switch_backend("redis")
        test(env, agent, render=RENDER)

        # Save model
        agent.save_model(PATH,epoch)

        # Update parameters
        epsilon *= 0.99





