import gym
from collections import deque
from keras.models import Model
from keras.layers import Dense, Input
from keras.models import clone_model
import numpy as np
import random
from keras import optimizers


env = gym.make('CartPole-v0')

n_action = env.action_space.n
n_observation = env.observation_space.n

D = deque(maxlen=1000000)

a = Input((n_action,))
h = Dense(40)(a)
b = Dense(n_observation)(h)

Q = Model(inputs = a , outputs= b)
sgd = optimizers.SGD(lr=0.000025, decay=1e-6, momentum=0.95)
Q.compile(oss='mse', metrics=['accuracy'], optimizer=sgd)


Q_hat = clone_model(Q)

M = 500

discounted_factor = 0.99
batch_size = 32

start_episilon = 1.0
end_epsilon = 0.1
decay_duration = 1000000

episilon = start_episilon

decay_rate = (start_episilon - end_epsilon) / decay_duration

env.reset()

#Episode Loop
total_step = 0

for _ in range(M): #loop until 500
    o = env.reset()
    # inside episode
    for step in range(200):

        total_step += 1

        a = np.random.rand()
        if a < episilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q.predict(o))

        env.render()

        next_o, r, d, _ = env.step(action)

        D.append([o, action, r, next_o, d])
        o = next_o

        if len(D) < batch_size:
            pass
        else: # D size is bigger than batch_size
            sample_batch = random.sample(D, batch_size)


            for s, a, r, ns, d in sample_batch:
                if d:
                    y = r
                else:
                    target_action_value = Q_hat.predict(s) #[batchsize][action_space_n]
                    y = r + discounted_factor * np.max(target_action_value[0])

                target_action_value[0][a] = y

                Q.fit(s, target_action_value, epochs=1)

        if d:
            break

        if total_step % 10000 == 0:

            Q_hat.save_weights(Q.get_weights())

