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

n_observation = env.observation_space.shape
n_action = env.action_space.n

print(n_observation, n_action)

a = Input(n_observation)
h = Dense(40)(a)
b = Dense(n_action)(h)

Q = Model(inputs=a, outputs=b)
rmsprop = optimizers.rmsprop(lr=0.00025, decay=1e-6, momentum=0.95)
Q.compile(optimizer=rmsprop, loss='mean_squared_error', metrics=['accuracy'])
Q_hat = clone_model(Q)
M = 500

start_epsilon = 1.0
end_epsilon = 0.1
decay_duration = 1000000
decay_rate = (start_epsilon - end_epsilon) / decay_duration
discounted_factor = 0.99

batch_size = 32
total_step = 0

for e_num in range(M):
    o = env.reset()
    epsilon = start_epsilon - decay_rate * e_num

    for _ in range(200):
        env.render()

        a = np.random.rand()
        if a > epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q.predict(o))

        next_o, r, d, _ = env.step(action)

        D.append([o, action, r, next_o, d])

        total_step += 1
        o = next_o

        """
        update network
        """

        if len(D) < batch_size:
            pass
        else:
            sample_batch = random.sample(D, batch_size)

        for s, a, r, ns, d in sample_batch:
            if d:
                y = r
            else:
                target_action_value = Q_hat.predict(s)[0]
                y = r + discounted_factor * np.max(target_action_value)

            target_action_value[0][a] = y

            Q.fit(s, target_action_value, epochs=1)

        if d:
            print(e_num, 'finished.', 'epsilon : ', epsilon)
            break

        if total_step % 10000 == 0:
            Q_hat.save_weights(Q.get_weights())