import gym
import numpy as np
import math

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Input

def build_network():
    i = Input(shape=(OBS_SPACE, ))
    h = Dense(2, activation='linear')(i)
    #h = Dense(5, activation='relu')(h)
    o = Dense(ACTION_SPACE, activation='softmax')(h)

    return Model(inputs=[i], outputs=[o])


class REINFORCEAgent:
    def __init__(self, model):
        self.model = model
        self.states = []
        self.actions = []
        self.rewards = []
        self.returns = []

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.returns = []

    def compile(self, optimizer='adam', loss='categorical_crossentropy'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def select_action(self, state):
        action_probs = self.model.predict(np.expand_dims(state, axis=0))[0]

        return np.random.choice(2, 1, p=action_probs)[0]

    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def get_gt(self):
        self.rewards.reverse()

        gt = []

        for idx, r in enumerate(self.rewards):
            if idx == 0:
                gt.append(r)
            else:
                gt.append(r + gt[-1] * GAMMA)

        self.returns = list(reversed(gt))

    def train(self, step_cnt):
        self.get_gt()

        actions_return = np.zeros(shape=(len(self.states), ACTION_SPACE))

        for i in range(len(self.states)):
            actions_return[i][self.actions[i]] = self.returns[i]

        model.fit(x=np.array(self.states), y=actions_return, batch_size=1, epochs=int(min(10, max(step_cnt/20, 1))))


env = gym.make('CartPole-v0').env

'''
Type: Box(4)

Num	Observation	Min	Max
0	Cart Position	-2.4	2.4
1	Cart Velocity	-Inf	Inf
2	Pole Angle	~ -41.8°	~ 41.8°
3	Pole Velocity At Tip	-Inf	Inf
'''
OBS_SPACE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.n

EPISODE = 500
GAMMA = 0.99

model = build_network()
agent = REINFORCEAgent(model)
agent.compile(optimizer=Adam(lr=0.001))

succ_cnt = 0

for e in range(EPISODE):
    s = env.reset()
    agent.reset()
    d = False
    step_cnt = 0

    while not d:
        #env.render()
        a = agent.select_action(s)

        ns, r, d, _ = env.step(a)

        agent.append_sample(s, a, r if not d else -5)

        s = ns

        step_cnt+=1

        if step_cnt == 5000:
            break

    agent.train(step_cnt)
    print('episode', e)

    if step_cnt == 5000:
        break
    elif step_cnt == 200:
        succ_cnt+=1

        if succ_cnt == 10:
            break
    else:
        succ_cnt = 0

env.close()

env = gym.make('CartPole-v0').env

s = env.reset()

d = False
step_cnt = 0

while not d:
    step_cnt += 1
    env.render()
    a = agent.select_action(s)
    s, _ ,d , _ = env.step(a)

env.close()
print(step_cnt)
