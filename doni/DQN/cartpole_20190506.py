"""
DQN
"""

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Model, Input

GAMMA = 0.99

env = gym.make('CartPole-v0')
OBS_SPACE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.n

EPISODE = 1000

class DQNAgent:
    def __init__(self, model):
        self.model = model

    def reset(self):
        pass

    def select_action(self, state):
        pass

    def train(self):
        pass

    def compile(self):
        self.model.compile(optimizer='adam', loss='mse')

def build_network():
    i = Input(shape=(OBS_SPACE,))
    h = Dense(4, activation='relu')(i)
    h = Dense(8, activation='relu')(h)
    o = Dense(ACTION_SPACE, activation='softmax')(h)
    return Model(inputs=[i], outputs=[])

main_model = build_network()
target_model = build_network()
main_agent = DQNAgent(main_model)
target_model = DQNAgent(target_model)
main_agent.compile()
target_model.compile()
