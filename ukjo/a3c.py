"""
Asynchronous methods for deep reinforcement learning 논문 구현
https://arxiv.org/abs/1602.01783

우선 어제 내용 정정합니다. env를 스레드 별도로 띄울수가 없더군요 ㅜㅜ
"""

import threading
import gym
import numpy as np
import pylab
import time
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam

# dl
from keras.models import Model
from keras.layers import Dense, Activation, Input

MAX_EPISODE = 5000  # max episode

T = 0  # global episode time counter
THREADS = 8

scores = []
episode = 0


class GlobalNetwork:
    def __init__(self, state_size, action_size, env_name):
        # self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.env_name = env_name

        # hyper parameters
        self.n_hidden1, self.n_hideen2  = 20, 20
        self.actor_lr, self.critic_lr = 0.001, 0.001
        self.discounted_factor = 0.99

        self.actor_postfix = "_actor.h5"
        self.critic_postfix = "_critic.h5"

        self.actor, self.critic = self.build_models()

        self.actor_optim = self.actor_optimizer()
        self.critic_optim = self.critic_optimizer()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def run_threads(self): # make 8 threads and run!!!

        agents = [LocalNetwork(i, self.env_name, self.actor, self.critic, self.actor_optim, self.critic_optim, self.state_size, self.action_size, self.discounted_factor) for i in range(THREADS)]

        for agent in agents:
            agent.start()

        while True:
            time.sleep(20)

            plot = scores[:]
            pylab.plot(range(len(plot)), plot, 'b')
            pylab.savefig('cartpole_a3c.png')

            self.save_model('cartpole_a3c')


    def build_models(self): # Build actor, critic network
        shared_input = Input(shape=(state_size,))
        shared = Dense(self.n_hidden1, activation='relu')(shared_input)

        actor_hidden = Dense(self.n_hideen2, activation='relu', kernel_initializer='glorot_uniform')(shared)
        actor_out = Dense(action_size, activation='softmax')(actor_hidden)

        critic_hidden = Dense(self.n_hideen2, activation='relu', kernel_initializer='glorot_uniform')(shared)
        critic_output = Dense(1, activation='linear')(critic_hidden)

        actor = Model(inputs=shared_input, outputs=actor_out)
        critic = Model(inputs=shared_input, outputs=critic_output)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, self.action_size))
        advantages = K.placeholder(shape=(None, ))
        policy = self.actor.output

        action_prob = K.mean(action * policy, axis=1)
        eligibility = K.log(action_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        entropy = K.sum(policy * K.log(policy+ 1e-10), axis=1)

        actor_loss = loss + entropy * 0.01

        optimizer = Adam(lr=self.actor_lr)

        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        train = K.function([self.actor.input, action, advantages], [], updates=updates)

        return train

    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None,))
        value = self.critic.output
        loss = K.mean(K.square(discounted_reward-value))
        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)

        return train


    def save_model(self, name):
        self.actor.save_weights(name + self.actor_postfix)
        self.critic.save_weights(name + self.critic_postfix)

    def load_model(self, name):
        self.actor.load_weights(name + self.actor_postfix)
        self.critic.load_weights(name + self.critic_postfix)



class LocalNetwork(threading.Thread):

    def __init__(self, index, env_name, actor, critic, actor_optim, critic_optim, state_size, action_size, discounted_factor):
        threading.Thread.__init__(self)
        self.env_name = env_name
        self.actor = actor
        self.critic = critic
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.state_size = state_size
        self.action_size = action_size
        self.discounted_factor = discounted_factor
        self.index = index

        # param !! just define here
        self.actions = []
        self.states = []
        self.rewards =[]


    def take_action(self, state):
        # predict action using actor network!!
        action_probs = self.actor.predict(np.reshape(state, [1, self.state_size]))[0]
        return np.random.choice(self.action_size, 1, p=action_probs)[0]

    def get_discounted_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0

        if not done:
            running_add = self.critic.predict(np.reshape(self.states[-1], (1, self.state_size)))[0]

        for t in reversed(range(0, len(rewards))):
            running_add = self.rewards[t] +  self.discounted_factor * running_add  # Algorithm S3 R <- ri + gamma * R
            discounted_rewards[t] = running_add

        return discounted_rewards


    def train_episode(self, done):
        discounted_rewards = self.get_discounted_rewards(self.rewards, done)

        values = self.critic.predict(np.asarray(self.states))
        values = np.reshape(values, len(values))  # make it one-dimensional

        advantages = discounted_rewards - values

        self.actor_optim([self.states, self.actions, advantages])
        self.critic_optim([self.states, discounted_rewards])

        # param !! just define here
        self.actions = []
        self.states = []
        self.rewards =[]


    def run(self):  # overriding thread run !!!

        global episode

        env = gym.make(self.env_name)

        while episode < MAX_EPISODE:
            s = env.reset()
            score = 0

            while True:
                a = self.take_action(s)
                ns, r, d, _ = env.step(a)
                score +=r

                a_onehot = np.zeros(self.action_size)
                a_onehot[a] = 1

                self.actions.append(a_onehot)
                self.rewards.append(r)
                self.states.append(s)
                s = ns

                if d:
                    episode+=1
                    print('episode : {} / score : {}'.format(episode, score))
                    scores.append(score)
                    self.train_episode(score!=500)
                    break



if __name__ == '__main__':

    env_name = 'CartPole-v1'

    env = gym.make(env_name)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print('succeed loading state_size {}, action size {} after creating environment'.format(state_size, action_size))

    env.close()

    global_agent = GlobalNetwork(state_size, action_size, env_name)
    global_agent.run_threads()
