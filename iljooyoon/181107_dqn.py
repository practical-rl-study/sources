"""
DQN 구현

Deep Q Network

2015년 버젼

replay memory + target network
"""

import gym

from collections import deque
from keras.models import Model, clone_model
from keras.layers import Input, Dense
from keras import optimizers

import numpy as np
import random

"""
env

 init : 환경 전체 초기화
 reset : 에피소드 단위 별 초기화
 step : action 을 진행시킴
 render : 출력
"""
env = gym.make('CartPole-v0')
D = deque(maxlen=1000000)

n_observation = env.observation_space.shape[0]
n_action = env.action_space.n

input_layer = Input((n_observation,))
h = Dense(100, activation='relu')(input_layer)
h = Dense(100, activation='relu')(h)
h = Dense(100, activation='relu')(h)
output_layer = Dense(n_action, activation='linear')(h)

Q = Model(inputs=input_layer, outputs=output_layer)
Q.compile(optimizer='rmsprop', loss='mse')
Q_hat = clone_model(Q)
M = 500

start_epsilon = 1.0
end_epsilon = 0.01
decay_duration = 3000
decay_rate = (start_epsilon - end_epsilon) / decay_duration
discounted_factor = 0.99
epsilon = start_epsilon

batch_size = 32
total_step = 0
render = False

for e_num in range(M):
    state = env.reset()
    step = 0
    done = False
    while not done:
        if render:
            env.render()

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q.predict(np.reshape(state, [1, n_observation])))

        epsilon = max(epsilon - decay_rate, end_epsilon)

        next_state, reward, done, _ = env.step(action)
        step += 1
        D.append([state, action, reward, next_state, done, step])

        total_step += 1
        state = next_state

        """
        update network
        """
        if total_step % 80 == 0 and len(D) > batch_size:
            for _ in range(5):
                sample_batch = random.sample(D, batch_size)
                s_stack, y_stack = [], []

                for s, a, r, ns, d, st in sample_batch:
                    if d and step < 200:
                        y = -100
                    elif d and step == 200:
                        continue
                    else:
                        Q2 = Q_hat.predict(np.reshape(ns, [1, n_observation]))[0]
                        y = r + discounted_factor * np.max(Q2)

                    target_action_value = Q.predict(np.reshape(s, [1, n_observation]))
                    target_action_value[0][a] = y
                    s_stack.append(s)
                    y_stack.append(target_action_value[0])

                Q.fit(np.array(s_stack), np.array(y_stack), epochs=1, verbose=0)

            Q_hat.set_weights(Q.get_weights())

        if done:
            print(e_num, 'finished.', 'epsilon : ', epsilon, 'step : ', step)
            if step > 100:
                render = True
            else:
                render = False

