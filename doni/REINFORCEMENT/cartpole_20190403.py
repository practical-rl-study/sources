'''
REINFORCEMENT
'''

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Model, Input

GAMMA = 0.99

env = gym.make('CartPole-v0')
OBS_SPACE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.n

EPISODE = 10000

class REINFORCEMENTAgent:
    def __init__(self, model):
        self.model = model
        self.states = []
        self.rewards = []
        self.actions = []
        self.returns = []

    def reset(self):
        self.states = []
        self.rewards = []
        self.actions = []

    def select_action(self, state):
        action_probs = self.model.predict(np.expand_dims(state, axis=0))[0]
        return np.random.choice(2, 1, p=action_probs)[0]

    def train(self):
        gt = []
        for i, reward in enumerate(reversed(self.rewards)):
            if i == 0:
                gt.append(reward)
            else:
                gt.append(reward + GAMMA * gt[-1])
        returns = list(reversed(gt))
        actions_return = np.zeros(shape=(len(self.states),ACTION_SPACE))

        for i in range(len(self.states)):
            actions_return[i][self.actions[i]] = returns[i]

        self.model.fit(x=np.array(self.states), y=actions_return, batch_size=32, epochs=1)

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def compile(self, optimizer='adam', loss='categorical_crossentropy'):
        self.model.compile(optimizer=optimizer, loss=loss)

def build_network():
    i = Input(shape=(OBS_SPACE,))
    h = Dense(4, activation='relu')(i)
    h = Dense(8, activation='relu')(h)
    o = Dense(ACTION_SPACE, activation='softmax')(h)

    return Model(inputs=[i], outputs=[o])


model = build_network()
agent = REINFORCEMENTAgent(model)
agent.compile()
for e in range(EPISODE):

    s = env.reset()
    agent.reset()
    d = False
    step = 0
    while not d:
        # env.render()
        a = agent.select_action(s)
        ns, r, d, _ = env.step(a)
        agent.append_sample(s,a,r)
        s = ns
        step += 1
        if d:
            print('train~! e=', e+1, 'step=', step)
            r = 100
            agent.train()

