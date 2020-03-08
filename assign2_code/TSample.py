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

    def __init__(self, bandit_num_upper, config, env):

        # some parameters to tune
        self.GAP = 1
        self.EXPECTATION = 0.5
        self.OTHERUPDATE = 0.5
        self.HUMANLEVEL = 7

        # used to control when to stop the game
        self.win = 0
        self.lose = 0

        self.config = config
        self.logger = get_logger(self.config.log_path)

        # make the env
        # self.make_env()

        # self.models = []
        # for bandit in range(bandit_num_upper):
        #     if os.path.exists(self.config.checkpoint_path + str(bandit)):
        #         model = NatureQN(self.env, config)
        #         model.load(bandit)
        #         print("---------------------", bandit)
        #         self.models.append(model)
        #         self.logger.info("loading model in level {}".format(bandit))



        #self.model = NatureQN(self.env, config)


        self.levels = []
        for bandit in range(bandit_num_upper):
            if os.path.exists(self.config.checkpoint_path + str(bandit)):
                self.levels.append(bandit)


        # self.env = MaxAndSkipEnvForTest(self.env)


        # all variables need for Thompson Sampling
        # self.bandit_num = len(self.models)
        self.bandit_num = len(self.levels)
        self.rnd = np.random.RandomState()
        self.probs = np.ones((self.bandit_num, 2))
        self.init_probs_backup = np.ones((self.bandit_num, 2))
        self.samples = np.ones(self.bandit_num)
        self.entropy = np.zeros(self.bandit_num)
        self.standord = sys.beta(3, 3)

        self.logger.info("loading done, we have {} levels of model".format(self.bandit_num))

        self.level = self.bandit_num // 2

        self.g1 = tf.Graph()
        self.g2 = tf.Graph()

        with self.g1.as_default():
            self.model = NatureQN(env, config)
            self.model.load(self.levels[self.level])
        with self.g2.as_default():
            self.human_model = NatureQN(env, config)
            self.human_model.load(0, well_trained=True)



        self.win = 0
        self.lose = 0
        self.step = 0

    def make_env(self):

        # get the env
        self.env = gym.make("Pong-v0")
        self.env = gym.wrappers.Monitor(self.env, self.config.record_path, video_callable=lambda x: True, resume=True)
        self.env = MaxAndSkipEnv(self.env, skip=config.skip_frame)
        self.env = PreproWrapper(self.env, prepro=greyscale, shape=(80, 80, 1),
                                 overwrite_render=config.overwrite_render)

    def action_ts(self, state):
        """ when get a state, return the action according to the model it chooses"""
        with self.g1.as_default():
            action = self.model.predict(state)[0]
        self.logger.info("AI use model in level {}".format(self.levels[self.level]))
        return action

    def action_human(self, state, human_level_episilon):
        self.logger.info("human use model in level {}".format(human_level_episilon))
        with self.g2.as_default():
            best_action = self.human_model.predict(state)[0]
        if np.random.random() < human_level_episilon:
            return np.random.choice(range(6))
        else :
            return best_action


    def updateBelief(self, reward, done=False):
        """ every step update the belief about the human player """
        self.step += 1

        if reward == -1:

            self.lose += 1
            self.probs[self.level][0] += 1
            for j in range(0, self.level):
                self.probs[j][0] += self.OTHERUPDATE

        elif reward == 1:

            self.win += 1
            self.probs[self.level][1] += 1
            for j in range(self.level + 1, self.bandit_num):
                self.probs[j][1] += self.OTHERUPDATE
        else:
            pass

        if reward == -1 or reward == 1:
            for i in range(self.bandit_num):
                self.samples[i] = np.abs(self.rnd.beta(self.probs[i][0], self.probs[i][1]) - self.EXPECTATION)
            level = np.argmin(self.samples)

            if level != self.level:
                self.level = level
                with self.g1.as_default():
                    self.model.load(self.levels[self.level])
            # self.model.load(self.levels[self.level])

            if done:
                self.logger.info("One game over, score is({},{}, whole steps are {})".format(self.lose, self.win, self.step))
                for i in range(self.bandit_num):
                    self.entropy[i] = self.kl_divergence(i)
                guess = np.argmin(self.entropy)
                self.init_probs_backup[guess][0] += 1
                self.init_probs_backup[guess][1] += 1
                if self.lose > self.win + self.GAP and guess + 1 < self.bandit_num:
                    self.init_probs_backup[guess + 1][0] += 1
                    self.init_probs_backup[guess + 1][1] += 1
                if self.win > self.lose + self.GAP and guess - 1 >= 0:
                    self.init_probs_backup[guess - 1][0] += 1
                    self.init_probs_backup[guess - 1][1] += 1

                self.probs[:, :] = self.init_probs_backup[:, :]
                self.win = 0
                self.lose = 0
                self.step = 0


    def cleanMemory(self):
        """ call this method when we change to a new human player """

        self.probs = np.ones((self.bandit_num, 2))
        self.init_probs_backup = np.ones((self.bandit_num, 2))

        self.level = self.bandit_num // 2
        self.model.load(self.levels[self.level])

        self.win = 0
        self.lose = 0




    def run(self, game_num):

        t = 0
        self.level = self.bandit_num//2
        self.model.load(self.levels[self.level])

        while t < game_num:

            state = self.env.reset()
            self.win = 0
            self.lose = 0
            step = 0
            self.probs[:,:] = self.init_probs_backup[:,:]
            print(self.probs,self.init_probs_backup)

            while True:

                step += 1
               # print(self.init_probs_backup)

                action = self.model.predict(state)[0]
                new_state, reward, done, info = self.env.step(action)

                # when one side scores, make a record
                if reward == -1:

                    self.lose += 1
                    for j in range(0, self.level + 1):
                        self.probs[j][0] += 1

                elif reward == 1:

                    self.win += 1
                    for j in range(self.level, self.bandit_num):
                        self.probs[j][1] += 1
                else:
                    pass

                if reward == -1 or reward == 1:
                    for i in range(self.bandit_num):
                        self.samples[i] = np.abs(self.rnd.beta(self.probs[i][0], self.probs[i][1]) - self.EXPECTATION)
                    level = np.argmin(self.samples)

                    if level != self.level:
                        self.level = level
                        self.model.load(self.levels[self.level])
                    self.logger.info("use model in level {}".format(self.levels[self.level]))

                state = new_state
                # self.env.render()
                if done:
                    self.logger.info("One game over, score is({},{}, whole steps are {})".format(self.lose, self.win, step))
                    for i in range(self.bandit_num):
                        self.entropy[i] = self.kl_divergence(i)
                    guess = np.argmin(self.entropy)
                    self.init_probs_backup[guess][0] += 1
                    self.init_probs_backup[guess][1] += 1
                    if self.lose > self.win + self.GAP and guess + 1 < self.bandit_num:
                        self.init_probs_backup[guess + 1][0] += 1
                        self.init_probs_backup[guess + 1][1] += 1
                    if self.win > self.lose + self.GAP and guess - 1 >= 0:
                        self.init_probs_backup[guess - 1][0] += 1
                        self.init_probs_backup[guess - 1][1] += 1
                    break

                # self.env.render()
            t += 1

    def kl_divergence(self, bandit):

        a = self.probs[bandit][0]
        b = self.probs[bandit][1]

        x = np.linspace(0.01, 0.99, 99)

        p = sys.beta(a, b).pdf(x)
        q = self.standord.pdf(x)

        return sum(rel_entr(p, q))


if __name__ == '__main__':
    # test = TSampling(8, config)
    # test.run(10)
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

    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                        overwrite_render=config.overwrite_render)
    test = TSampling(4, config, env)
    env = MaxAndSkipEnvForTest(env)
    state = env.reset()
    while True:
        a1 = test.action_ts(state)
        a2 = test.action_human(state,0.9)
        print(a1, a2)
        new_state, reward, done, info = env.step(a1)
        state = new_state



