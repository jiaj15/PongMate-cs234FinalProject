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
        # for bandit in range(bandit_num_upper):
        #     if os.path.exists(self.config.checkpoint_path + str(bandit)):
        #         model = NatureQN(self.env, config)
        #         model.load(bandit)
        #         print("---------------------", bandit)
        #         self.models.append(model)
        #         self.logger.info("loading model in level {}".format(bandit))

        # for test
        model = NatureQN(self.env, config)
        model.load(0, well_trained=True)
        self.models.append(model)

        self.env = MaxAndSkipEnvForTest(self.env)

        # all variables need for Thompson Sampling
        self.bandit_num = len(self.models)
        self.rnd = np.random.RandomState()
        self.probs = np.ones((self.bandit_num, 2))
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
            self.win = 0
            self.lose = 0

            while True:

                # use Thompson to get all samples for all models
                for i in range(self.bandit_num):
                    self.samples[i] = np.abs(self.rnd.beta(self.probs[i][0], self.probs[i][1]) - 0.5)
                    self.entropy[i] = self.kl_divergence(i)

                level = np.argmin(self.samples)
                action = self.models[level].predict(state)[0]

                new_state, reward, done, info = self.env.step(action)

                # when one side scores, make a record
                if reward == -1:

                    self.lose += 1
                    for j in range(0, level + 1):
                        self.probs[j][0] += 1

                elif reward == 1:

                    self.win += 1
                    for j in range(level, self.bandit_num):
                        self.probs[j][1] += 1
                else:
                    pass

                state = new_state

                if done:
                    self.logger.info("One game over, score is({},{})".format(self.lose, self.win))
                    break

                # self.env.render()
            i += 1

    def kl_divergence(self, bandit):

        a = self.probs[bandit][0]
        b = self.probs[bandit][1]

        x = np.linspace(0.01, 0.99, 99)

        p = sys.beta(a, b).pdf(x)
        q = self.standord.pdf(x)

        return sum(rel_entr(p, q))


if __name__ == '__main__':
    test = TSampling(8, config)
    test.run(3)
    # env = gym.make(config.env_name)
    # env = MaxAndSkipEnv(env, skip=config.skip_frame)
    # env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
    #                     overwrite_render=config.overwrite_render)
    #
    # model = NatureQN(env, config)
    # model.load(0, well_trained=True)
    #
    # env = MaxAndSkipEnvForTest(env)
    #
    # ob = env.reset()
    # for i in range(20):
    #     ob = env.reset()
    #     while True:
    #         action = model.predict(ob)[0]
    #         ob, r, done, info = env.step(action)
    #         if done:
    #             break
    #         #env.render()
    #
    # print(model.predict(ob))
