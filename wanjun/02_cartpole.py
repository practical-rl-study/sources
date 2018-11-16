import gym
import random
import numpy as np
import copy
from keras.models import Model, clone_model
from keras.layers import Input, Dense
from collections import deque


def build_network(n_input, n_output):
    i = Input(shape=(n_input,))
    h = Dense(40, activation='relu', kernel_initializer='he_uniform')(i)
    h = Dense(40, activation='relu', kernel_initializer='he_uniform')(h)
    o = Dense(n_output, activation='linear', kernel_initializer='he_uniform')(h)
    model = Model(inputs=i, outputs=o)
    model.compile(optimizer='rmsprop', loss='mse')
    return model


def calc_reward(s, done):
    reward = -(abs(s[0]) * 0.1 + (s[2] * s[2] * 50))
    if done:
        reward -= 5 * (abs(s[0]) + abs(s[2]))
    return reward


env = gym.make('CartPole-v0')
n_action = env.action_space.n
n_observation = env.observation_space.shape[0]
Q = build_network(n_observation, n_action)
Q_target = clone_model(Q)
replay_memory = deque(maxlen=100000)

# settings

epsilon = 1.
min_epsilon = 0.
decay_duration = 200
decay = (epsilon - min_epsilon) / decay_duration
batch_size = 64
train_start = 200
discount_factor = 0.99


for ep in range(500):
    state = env.reset()
    done = False
    step = 0

    while not done:
        step += 1
        env.render()
        if epsilon < np.random.random():
            action = np.argmax(Q.predict(np.array([state])))
        else:
            action = env.action_space.sample()
        epsilon = max(epsilon-decay, min_epsilon)

        next_state, reward, done, info = env.step(action)

        for i in range(-3, 4):
            s, ns = copy.deepcopy(state), copy.deepcopy(next_state)
            s[0] += i/10
            ns[0] += i/10
            r = calc_reward(ns, done)
            a = action
            replay_memory.append([copy.deepcopy(s), a, r, copy.deepcopy(ns), done])

            for j in range(4):
                s[j] *= -1
                ns[j] *= -1
            a = 1 - action
            replay_memory.append([s, a, r, ns, done])

        state = next_state
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
    print('ep :', ep, 'step :', int(step))
    Q_target.set_weights(Q.get_weights())

