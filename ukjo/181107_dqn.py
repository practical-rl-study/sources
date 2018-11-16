"""
dqn을 구현해보는거에요

deep q network

2015년도 버젼으로 갈겁니다

replay memory + target network

"""

import gym  # 강화학습 돌리는 환경!!
from collections import deque
from keras.models import Model
from keras.layers import Dense, Input
from keras.models import clone_model
import numpy as np
import random
from keras import optimizers


env = gym.make('CartPole-v0')

n_action = env.action_space.n
n_observation = env.observation_space.shape[0]
n_max_step = env._max_episode_steps

M = 500

start_epsilon = 1.0
end_epsilon = 0.1
decay_duration = 1000000
epsilon = start_epsilon
decay_rate = (start_epsilon - end_epsilon) / decay_duration
batch_size = 32
discounted_factor = 0.99
total_step = 0
render = False
learning_rate = 0.001


D = deque(maxlen=1000000)

a = Input((n_observation,))
h = Dense(24, activation='relu')(a)
h = Dense(24, activation='relu')(h)
b = Dense(n_action, activation='linear')(h)
Q = Model(inputs=a, outputs=b) # Q
# Q.compile(optimizer=optimizers.rmsprop(lr=0.00025), loss='mse', metrics=['accuracy'])

Q.compile(loss='mse',optimizer=optimizers.Adam(lr=learning_rate))

Q_hat = clone_model(Q)
Q_hat.set_weights(Q.get_weights())

"""
env

 init : 초기화 환경전체
 reset : 에피소드 단위별 
 step  : 한단계 나가는 스탭 !!  action
 render : 눈에 보이기에!!
"""


def optimize():
    """
    update network
    """

    if len(D) < batch_size:
        pass
    else:
        # D size is bigger than batch_size
        sample_batch = random.sample(D, batch_size)

        for s, a, r, ns, d in sample_batch:
            target_action_value = None
            if d:
                y = r
                target_action_value = Q.predict(np.expand_dims(s, axis=0))
                target_action_value[0][a] = y
            else:
                target_action_value = Q_hat.predict(np.expand_dims(ns, axis=0), batch_size=1, verbose=0)  # [batch_size][action_space_n]
                y = r + discounted_factor * np.max(target_action_value[0])
                target_action_value[0][a] = y

            Q.fit(np.expand_dims(s, axis=0), target_action_value, epochs=1, verbose=0)


# episode loop
for e in range(M):  # loop until 500

    """episode start"""
    o = env.reset()

    R = 0
    # inside episode
    for step in range(200):

        total_step += 1

        # epsilon greedy
        a = np.random.rand()
        epsilon = epsilon - (total_step * decay_rate)
        if step < 10 or a < max(end_epsilon, epsilon):
            action = env.action_space.sample()
        else:
            action = np.argmax(Q.predict(np.expand_dims(o, axis=0)))

        if render:
            env.render()

        next_o, r, d, _ = env.step(action)

        if d and step < n_max_step-1:
            r = -100

        D.append([o, action, r, next_o, d])

        R += r
        o = next_o
        optimize()

        if d:
            if step < n_max_step-1:
                R += 100
            print('{} episode return : {}'.format(e, R))
            break

    if e % 5 == 0: # every C steps, where C = 10000
        Q_hat.set_weights(Q.get_weights())
