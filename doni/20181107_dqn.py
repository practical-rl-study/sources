import gym
from collections import deque
from keras.models import Model
from keras.layers import Dense, Input
from keras.models import clone_model
from keras import optimizers
import numpy as np
import random

env = gym.make("CartPole-v0")

n_action = env.action_space.n
n_observation = env.observation_space.shape[0]

a = Input(shape=(4,))
h = Dense(40)(a)
b = Dense(2)(h)
Q = Model(inputs=a, outputs=b)

rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
Q.compile(optimizer=rmsprop, loss='mse', metrics=['accuracy'])

Q_hat = clone_model(Q)

D = deque(maxlen=1000000)
M = 500
start_epsilon = 1.0
end_epsilon = 0.1
decay_duration = 1000000
epsilon = start_epsilon
decay_rate = (start_epsilon - end_epsilon)/decay_duration

batch_size = 32
discount_factor = 0.99

total_step = 0

for _ in range(M):
    o = env.reset()

    for step in range(200):
        total_step = total_step + 1

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q.predict(np.array([o])))

        epsilon = max(epsilon-decay_rate, end_epsilon)
        env.render()
        next_o, r, d, _ = env.step(action)

        D.append([o, action, r, next_o, d])

        o = next_o

        if len(D) < batch_size:
            pass
        else:
            sample_batch = random.sample(D, batch_size)

            s_stack=[]
            y_stack=[]
            for s, a, r, ns, d in sample_batch:

                action_value = Q.predict(np.array([s]))

                if d:
                    y = r
                else:
                    target_action_value = Q_hat.predict(np.array([ns]), batch_size=1)
                    y = r + discount_factor * np.max(target_action_value[0])

                action_value[0][a] = y
                s_stack.append(s)
                y_stack.append(action_value)
                Q.fit(np.array(s_stack), np.array(y_stack), epochs=1)

        if total_step % 10000 == 0:
            Q_hat.save_weights(Q.get_weights())

env.env.close()