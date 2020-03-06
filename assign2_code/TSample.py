import numpy as np
import gym
import logging
import tensorflow as tf
import scipy.stats as sys
import os
from scipy.special import rel_entr

from utils.preprocess import greyscale
from utils.wrappers import *
from utils.general import get_logger

from q1_schedule import LinearExploration, LinearSchedule
from q3_nature import NatureQN
from configs.ts_config import config


class TSampling(object):

    def __init__(self, bandit_num_upper, config):

        # used to control when to stop the game
        self.win = 0
        self.lose = 0

        self.config = config
        self.logger = get_logger(self.config.log_path)

        # make the env
        self.make_env()

        self.models = []
        for i in range(bandit_num_upper):
            model = NatureQN(self.env, config)
            if os.path.exists(self.config.checkpoint_path + str(i)):
                model = NatureQN(self.env, config)
                model.load(i, well_trained=True)
                print("---------------------",i)
                self.models.append(model)
                self.logger.info("loading model in level {}".format(i))

        self.env = MaxAndSkipEnvForTest(self.env)

        # all variables need for Thompson Sampling
        self.bandit_num = len(self.models)
        self.rnd = np.random.RandomState()
        self.probs = np.zeros((self.bandit_num, 2))
        self.samples = np.ones(self.bandit_num)
        self.entropy = np.zeros(self.bandit_num)
        self.standord = sys.beta(100, 100)

        self.logger.info("loading done, we have {} levels of model".format(self.bandit_num))

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
                action = self.models[level].predict(state)[0]

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

                self.env.render()

    def kl_divergence(self, bandit):

        a = self.probs[bandit][0]
        b = self.probs[bandit][1]

        x = np.linspace(0.01, 0.99, 99)

        p = sys.beta(a, b).pdf(x)
        q = self.standord.pdf(x)

        return sum(rel_entr(p, q))


if __name__ == '__main__':
    test = TSampling(8,config)
    test.run(1)



