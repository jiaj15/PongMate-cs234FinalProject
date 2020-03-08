import gym
import retro
from two_agent import *
from utils.preprocess import greyscale
from utils.wrappers import *

from q1_schedule import LinearExploration, LinearSchedule
from q3_nature import NatureQN

from configs.q5_train_atari_nature import config, config_short

import matplotlib.pyplot as plt


"""
Use deep Q network for the Atari game. Please report the final result.
Feel free to change the configurations (in the configs/ folder). 
If so, please report your hyperparameters.

You'll find the results, log and video recordings of your agent every 250k under
the corresponding file in the results folder. A good way to monitor the progress
of the training is to use Tensorboard. The starter code writes summaries of different
variables.

To launch tensorboard, open a Terminal window and run 
tensorboard --logdir=results/
Then, connect remotely to 
address-ip-of-the-server:6006 
6006 is the default port used by tensorboard.
"""
if __name__ == '__main__':
    # make env
#     config = config_short
#     env = gym.make(config.env_name)
#     env = MaxAndSkipEnv(env, skip=config.skip_frame)
#     env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
#                         overwrite_render=config.overwrite_render)

    # env = retro.make(game='Pong-Atari2600', use_restricted_actions=retro.Actions.DISCRETE) 
    # env = retroActionWrapper(env)
    # env = retroWrapper(env)

    # # env = gym.make('Pong-v0')
    # env = MaxAndSkipEnv(env, skip=config.skip_frame)
    # env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
    #                     overwrite_render=config.overwrite_render)
    
    # # exploration strategy
    # exp_schedule = LinearExploration(env, config.eps_begin,
    #         config.eps_end, config.eps_nsteps)
    
    # # learning rate schedule
    # lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
    #         config.lr_nsteps)
    
    # # train model
    # model = NatureQN(env, config)
    # model.run(exp_schedule, lr_schedule)
    
    # for test

    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                        overwrite_render=config.overwrite_render)

    model = NatureQN(env, config)
    model.load(0, well_trained=True)

    env = MaxAndSkipEnvForTest(env)
    # env.ale.setDifficulty(1)
    
    ob = env.reset()
    for i in range(20):
        ob = env.reset()
        while True:
                action = model.predict(ob)[0]
                if np.random.random() < 0.05:
                    d = 1
                else:
                    d = 0
                # d = np.random.choice(range(2), 1)[0]
                env.ale.setDifficulty(d)
                ob, r, done, info = env.step(action)
                if done:
                        break
                env.render()
        
        
    print(model.predict(ob))
# 