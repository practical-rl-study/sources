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
from keras.optimizers import Adam
from keras.models import Sequential

"""
env

 init : 초기화 환경전체
 reset : 에피소드 단위별 
 step  : 한단계 나가는 스탭 !!  action
 render : 눈에 보이기에!!
"""

class DQN:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        self.D = deque(maxlen=2000)


        # create main model and target model
        self.q = self.build_network()
        self.target_q = self.build_network()
        self.update_target()

    def update_target(self):
        self.target_q.set_weights(self.q.get_weights())

    def append_sample(self, s, a, r, ns, d):
        self.D.append([o, a, r, ns, d])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_val = self.q.predict(np.expand_dims(state, axis=0))
            return np.argmax(q_val[0])

    def build_network(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # def build_network(self):
    #     D = deque(maxlen=2000)
    #     a = Input((n_observation,))
    #     h = Dense(24, activation='relu', kernel_initializer='he_uniform')(a)
    #     h = Dense(24, activation='relu', kernel_initializer='he_uniform')(h)
    #     b = Dense(n_action, activation='linear', kernel_initializer='he_uniform')(h)
    #     Q = Model(inputs=a, outputs=b)  # Q
    #     Q.compile(loss='mse', optimizer=optimizers.RMSprop(lr=self.learning_rate))
    #     return Q

    def optimize(self):
        if len(self.D) < self.batch_size:
            return

        # D size is bigger than batch_size
        sample_batch = random.sample(self.D, self.batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for s, a, r, ns, d in sample_batch:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        q = self.q.predict(np.vstack(states), batch_size=self.batch_size)
        target = self.target_q.predict(np.vstack(next_states), batch_size=self.batch_size, verbose=0)

        for i in range(len(sample_batch)):
            y = rewards[i]
            if not dones[i]:
                y = rewards[i] + self.discount_factor * np.amax(target[i])
            q[i][actions[i]]  = y


        self.q.fit(x=np.vstack(states), y=q, epochs=1, verbose=0, batch_size=self.batch_size)


if __name__ == "__main__":

    EPISODES = 500
    TOTAL_STEP = 0

    env = gym.make('CartPole-v0')

    n_action = env.action_space.n
    n_observation = env.observation_space.shape[0]
    n_max_step = env._max_episode_steps

    agent = DQN(n_observation, n_action)

    # episode loop
    for e in range(EPISODES):  # loop until 500
        """episode start"""
        o = env.reset()
        R = 0
        d = False
        step = 0

        # inside episode
        while not d:
            TOTAL_STEP += 1
            step+=1

            if agent.render:
                env.render()

            action = agent.get_action(o)
            next_o, r, d, _ = env.step(action)

            if d and step < n_max_step-1:
                r = -100

            agent.append_sample(o, action, r, next_o, d)
            agent.optimize()

            R += r
            o = next_o

            if d:
                agent.update_target()

                if step < n_max_step-1:
                    R += 100
                print('{} episode, total steps : {}, return : {}'.format(e, TOTAL_STEP, R))
                break