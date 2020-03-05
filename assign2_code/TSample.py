import numpy as np
import gym
import tensorflow as tf
import scipy.stats as sys
from scipy.special import rel_entr

from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from q1_schedule import LinearExploration, LinearSchedule
from q3_nature import NatureQN
from configs.q5_train_atari_nature import config


class TSampling(object):

    def _init_(self, bandit_num, model_path):

        # all variables need for Thompson Sampling
        self.bandit_num = bandit_num
        self.rnd = np.random.RandomState()
        self.probs = np.zeros((bandit_num, 2))
        self.samples = np.ones(bandit_num)
        self.entropy = np.zeros(bandit_num)
        self.standord = sys.beta(100, 100)

        # used to control when to stop the game
        self.win = 0
        self.lose = 0

        # make the env
        self.make_env()

        # load all the models
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.models = []
        for i in range(bandit_num):
            model = NatureQN(self.env, config)
            self.saver.restore(self.sess, model_path)
            self.models.append(model)

    def make_env(self):

        # get the env
        self.env = gym.make("Pong-v0")
        self.env = MaxAndSkipEnv(self.env, skip=config.skip_frame)
        self.env = PreproWrapper(self.env, prepro=greyscale, shape=(80, 80, 1),
                                 overwrite_render=config.overwrite_render)

    def run(self, game_num):

        t = 0
        while t < game_num:

            state = self.env.reset()

            while True:

                # use Thompson to get all samples for all models
                for i in range(self.bandit_num):
                    self.samples[i] = np.abs(self.rnd.beta(self.probs[i], self.probs[i]) - 0.5)
                    self.entropy[i] = self.kl_divergence(i)

                level = np.argmin(self.samples)[0]
                action = self.models[level].predict(state)

                new_state, reward, done, info = self.env.step(action)

                # when one side scores, make a record
                if reward == -1:

                    self.lose += 1
                    for i in range(0, level + 1):
                        self.probs[i][0] += 1

                elif reward == 1:

                    self.win += 1
                    for i in range(level, self.bandit_num + 1):
                        self.probs[i][1] += 1
                else:
                    pass

                state = new_state

                if done:
                    break

    def kl_divergence(self, bandit):

        a = self.probs[bandit][0]
        b = self.probs[bandit][1]

        x = np.linspace(0.01, 0.99, 99)

        p = sys.beta(a, b).pdf(x)
        q = self.standord.pdf(x)

        return rel_entr(p, q)
