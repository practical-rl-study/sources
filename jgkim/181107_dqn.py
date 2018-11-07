"""
dqn 구현해보기
deep q network
2015년 끝판왕 버젼으로 구현 (replay memory + target network)

"""
import gym  # 강화학습 돌리는 환경
import collections import deque

# gym example googling
env = gym.make('CartPole-v0')

n_action = env.action_space.n
n_observation = env.action_space.n

D = deque(maxlen=1000000) # 양쪽에서 넣고 뺄수 있는 큐 구조임. # replay memory size

# keras.io 검색
#https://keras.io/models/model/

from keras.models import Model
from keras.layers import Dense, Input  # fully connected
from keras.models import clone_model
import numpy as np
import random
from keras import optimizers

# Q Network    Observation =>
a = Input(shape=(n_observation,))  # 4는 observation 개수
h = Dense(40)(a)
b = Dense(n_action)(h)   # b : output
Q = Model(inputs=a, outputs=b)
rmsprop = optimizers.rmsprop(lr=0.00025, decay=1e-6, momentum=0.95)
Q.compile(optimizer=rmsprop, loss='mse', metrics=['accuracy'])
# Q target Network
Q_hat = clone_model(Q)
M = 500

# 엡실론 값은? 논문 맨 뒤에 찾아보기
# initial exploration, final exploration, final exploration frame
start_epsilon = 1.0
end_epsilon = 0.1
epsilon = start_epsilon
decay_duration = 1000000
decay_rate = (start_epsilon - end_epsilon)/decay_duration
batch_size = 32
discount_factor = 0.99
total_step = 0

# _ 사용법
# _ : a = 2+5+7 해야하는데, 2+5+7만 했을 때 마지막 호출한 것을 불러주는 기능도 있음.

"""
env
init : 환경전체 초기화
reset : 에피소드 단위별 초기화
step : 한단계 나가는 스탭!! action
render : 화면에 보여줌.

"""

# 에피소드 단위
for _ in range(1000):

    observation = env.reset()


    # 에피소드 내
    for step in range(200):

        # epsilon greedy
        a = np.random.rand()    # (0,1)
        if step < 10 or a < epsilon:    # no_op max
            action = env.action_space.sample()
        else:
            action = np.argmax(Q.predict(observation))

        env.render()    # 에피소드 마다 뿌려준다.

        next_observation, rewards, done, _ = env.step(action)

        D.append([observation, action, rewards, next_observation, done])

        observation = next_observation

        # update network

        if len(D) < batch_size:
            pass
        else:   # D size is bigger than batch_size
            sample_batch = random.sample(D, batch_size)


            for s, a, r, ns, d in sample_batch:
                if d:
                    y = r
                else:
                    target_action_value = Q_hat.predict(s, batch_size=1)  #[batch_size][action_space_n]
                    y = r + discount_factor * np.max(target_action_value[0])

                target_action_value[0][a] = y
                Q.fit(s, target_action_value, epochs=1)

                # optimizer loss에서 mse로 이미 정의함.
                #(y - Q.predict(s)[a]) ** 2

        # hat and - = approximation
        if done:
            break
        if total_step % 10000 == 0:   # every C step, where C = 300
            Q_hat.save_weights(Q.get_weights())
