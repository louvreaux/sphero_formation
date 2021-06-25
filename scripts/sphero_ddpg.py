#!/usr/bin/env python3
"""
Main file
Original author - Samuel Matthew Koesnadi
https://github.com/samuelmat19/DDPG-tf2
"""
import logging
import random

import gym
import json
import time
from tqdm import trange
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from common_definitions import CHECKPOINTS_PATH, TOTAL_EPISODES, TF_LOG_DIR, UNBALANCE_P
from actor_critic import Brain
from utils import Tensorboard
from task_envs.sphero import sphero_world2


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    RL_TASK = 'SpheroWorld-v2'
    RENDER_ENV = False
    LEARN = True
    USE_NOISE = True
    # If we're learning we want to save weights, otherwise ignore
    if LEARN:
        SAVE_WEIGHTS = True
        WARM_UP = 0
    else:
        SAVE_WEIGHTS = False
        WARM_UP = 300
    EPS_GREEDY = 0.95

    # Step 1. create the gym environment
    env = gym.make(RL_TASK)
    env.pause()
 
    action_space_high = env.action_space.high#[0]
    action_space_low = env.action_space.low#[0]

    brain = Brain(env.observation_space.shape[0], env.action_space.shape[0], action_space_high,
                  action_space_low)
    tensorboard = Tensorboard(log_dir=TF_LOG_DIR)

    # load weights if available
    logging.info("Loading weights from %s*, make sure the folder exists", CHECKPOINTS_PATH)
    brain.load_weights(CHECKPOINTS_PATH)

    # all the metrics
    acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
    actions_squared = tf.keras.metrics.Mean('actions', dtype=tf.float32)
    Q_loss = tf.keras.metrics.Mean('Q_loss', dtype=tf.float32)
    A_loss = tf.keras.metrics.Mean('A_loss', dtype=tf.float32)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    startTime = time.time()

    # run iteration
    with trange(TOTAL_EPISODES) as t:
        for ep in t:
            env.unpause()
            prev_state = env.reset()
            # print("STATE: " + str(prev_state))
            env.pause()
            acc_reward.reset_states()
            actions_squared.reset_states()
            Q_loss.reset_states()
            A_loss.reset_states()
            brain.noise.reset()
            total_max_q = 0
            step = 0
            sum_reward = 0

            for _ in range(2500):
                step += 1
                if RENDER_ENV:  # render the environment into GUI
                    env.render()

                # Recieve state and reward from environment.
                cur_act, maxQ = brain.act(tf.expand_dims(prev_state, 0), _notrandom=(ep >= WARM_UP) and
                                    (random.random() < EPS_GREEDY+(1-EPS_GREEDY)*ep/TOTAL_EPISODES),
                                    noise=USE_NOISE)
                env.unpause()
                # print("ACTION: " + str(cur_act))
                state, reward, done, _ = env.step(cur_act)
                total_max_q += maxQ
                sum_reward += reward 
                # print("STATE: " + str(state))
                # print("----------------------------------")
                env.pause()
                brain.remember(prev_state, reward, state, int(done))

                # update weights
                if LEARN:
                    c, a = brain.learn(brain.buffer.get_batch(unbalance_p=UNBALANCE_P))
                    Q_loss(c)
                    A_loss(a)

                # post update for next step
                acc_reward(reward)
                actions_squared(np.square(cur_act/action_space_high))
                prev_state = state

                if done:
                    break

            ep_reward_list.append(acc_reward.result().numpy())
            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            avg_reward_list.append(avg_reward)

            # print the average reward
            t.set_postfix(r=avg_reward)
            tensorboard(ep, acc_reward, actions_squared, Q_loss, A_loss)

            m, s = divmod(int(time.time() - startTime), 60)
            h, m = divmod(m, 60)

            paramKeys = ['score', 'step', 'time', 'averageQ']
            paramValues = [sum_reward, step, h, total_max_q / step]
            paramDictionary = dict(zip(paramKeys, paramValues))

            # save weights
            if ep % 5 == 0 and SAVE_WEIGHTS:
                brain.save_weights(CHECKPOINTS_PATH)
                paramPath = CHECKPOINTS_PATH + str(_) + '.json'
                with open(paramPath, 'w') as outfile:
                    json.dump(paramDictionary, outfile)

    env.close()
    if SAVE_WEIGHTS:
        brain.save_weights(CHECKPOINTS_PATH)

    logging.info("Training done...")

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
