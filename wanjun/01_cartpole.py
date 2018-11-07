import gym
import random
import numpy as np
from keras.models import Model, clone_model
from keras.layers import Input, Dense
from collections import deque


def build_network(n_action, n_observation):
    i = Input(shape=(n_observation,))
    h = Dense(40, activation='relu')(i)
    o = Dense(n_action, activation='linear')(h)
    model = Model(inputs=i, outputs=o)
    model.compile(optimizer='rmsprop', loss='mse')
    return model

env = gym.make('CartPole-v0')

n_action = env.action_space.n
n_observation = env.observation_space.shape[0]
Q = build_network(n_action, n_observation)
Q_target = clone_model(Q)

replay_memory = deque(maxlen=100000)

epsilon = 1.
min_epsilon = 0.
decay_duration = 3000
decay = (epsilon - min_epsilon) / decay_duration
batch_size = 64
train_start = 200
discount_factor = 0.99


for ep in range(500):
    observation = env.reset()
    done = False
    reward_sum = 0

    while not done:
        env.render()

        if epsilon < np.random.random():
            action = np.argmax(Q.predict(np.array([observation])))
        else:
            action = env.action_space.sample()
        epsilon = max(epsilon-decay, min_epsilon)

        next_observation, reward, done, info = env.step(action)
        replay_memory.append([observation, action, reward, next_observation, done])
        observation = next_observation
        reward_sum += reward

        if len(replay_memory) >= train_start:
            sample_batch = random.sample(replay_memory, batch_size)
            states, target = [], []
            for s, a, r, ns, d in sample_batch:
                q = Q.predict(np.array([s]))[0]
                q_target = Q_target.predict(np.array([ns]))[0]
                if d:
                    q[a] = r
                else:
                    q[a] = r + discount_factor * np.max(q_target)
                states.append(s)
                target.append(q)
            Q.fit(np.array(states), np.array(target), verbose=0)
    print(ep, round(epsilon, 3), int(reward_sum), len(replay_memory))
    Q_target.set_weights(Q.get_weights())






