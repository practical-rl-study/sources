import gym
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input

env = gym.make('CartPole-v0')

OBS_SPACE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.n
EPISODE = 100

GAMMA = 0.3
discounted_factor = 0


class REINFORCEAgent:
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
        self.returns = []

    def select_action(self, state):
        action_probs = self.model.predict(np.expand_dims(state, axis=0))[0]
        return np.random.choice(2, 1, p=action_probs)[0]

    def train(self):
        # episodic sample --> reverse -> gt
        gt = []
        for i, reward in enumerate(reversed(self.rewards)):
            if i == 0:
                gt.append(reward)
            else:
                gt.append(reward + GAMMA * gt[-1])

        returns = list(reversed(gt))

        actions_return = np.zeros(shape=(len(self.states),ACTION_SPACE))

        for i, g_i in enumerate(range(len(self.states))):
            actions_return[i][self.actions[i]] = returns[i]

        # s,a,gt
        # model.fit
        assert len(self.states) == actions_return.shape[0]
        self.model.fit(x=np.asarray(self.states), y=actions_return, batch_size=64, epochs=1)

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def compile(self, optimizer='adam', loss='categorical_crossentropy'):
        self.model.compile(optimizer=optimizer, loss=loss)

def build_network():
    i = Input(shape=(OBS_SPACE,))
    h = Dense(2, activation='relu')(i)
    h = Dense(2, activation='relu')(h)
    o = Dense(ACTION_SPACE, activation='softmax')(h)

    return Model(inputs=[i], outputs=[o])


model = build_network()
agent = REINFORCEAgent(model)
agent.compile()

for e in range(EPISODE):
    s = env.reset()
    agent.reset()
    d = False
    while not d:
        # env.render()
        a = agent.select_action(s)
        ns, r, d, _ = env.step(a)
        agent.append_sample(s,a,r)
        s = ns

        if d:
            agent.train()
