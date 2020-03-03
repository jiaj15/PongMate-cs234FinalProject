import argparse
import sys
import time
import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt
import numpy as np
# import cv2
from utils.wrappers import TennisPreproWrapper, MaxAndSkipEnv
from configs.tennis_training_config import config
# def greyscale_tennis(state):
#     """
#     Preprocess state image 
#     # [168, 48, 143] # back ground
#             # [240 128 128] # pink player
#             # [117 231 194] # green-blue
#             # [236 236 236] # ball
#             # [214 214 214] # net
#             # [74 74 74] # black
#     """
#     state = np.reshape(state, [250, 160, 3]).astype(np.float32)
#     patterns = np.array([[240,128,128],[117,231,194],[236,236,236], [214,214,214],[74,74,74],[168, 48, 143]])
#     # bgpattern = np.array([168, 48, 143]) 
    
#     # ppp = np.array([240,128,128])
#     # bpp = np.array([117,231,194])

#     # bp = np.array([236,236,236])
#     # netp = np.array([214,214,214])
#     # blp = np.array([74,74,74])

#     # grey scale
#     # DO NOT grey scale because players may exchange sides
#     # state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114
#     # state = state[:, :, 0] * 0.2 + state[:, :, 1] * 0.7 + state[:, :, 2] * 0.1

#     # karpathy
#     state = state[45:215, 10:150]  # crop
#     # state[state==194] = 1
#     # state[state==231] = 100
#     # state[state==236] = 255

#     # state = state[::2,::2] # downsample by factor of 2
    
#     state = state[:, :, 0] * 0.15 + state[:, :, 1] * 0.7 + state[:, :, 2] * 0.15
#     # patterns = patterns[:, 0] * 0.15 + patterns[:, 1] * 0.7 + patterns[ :, 2] * 0.15
#     # print(patterns)
#     state = cv2.pyrDown(state)

#     state = state[:, :, np.newaxis]
#     return state.astype(np.uint8)

    # return state.astype(np.uint8)
def isFire(action):
    """
    action: int
    env: Pong-v0
    return true if it is a FIRE action
    """
    ACTION_MEANINGS = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    action_mean = ACTION_MEANINGS[action]
    if action_mean.find('FIRE') == -1:
        return False
    else:
        return True
    

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='Tennis-v0', help='Select the environment to run')
    args = parser.parse_args()
    
    logger.set_level(logger.INFO)

    # env = gym.make(args.env_id)
    # env = MaxAndSkipEnv(env, skip=config.skip_frame)
    # env = TennisPreproWrapper(env, prepro=greyscale_tennis, shape=config.shape, 
    #                     overwrite_render=config.overwrite_render)
    # env.reset()
    # # env.render()

    # # You provide the directory to write to (can be an existing
    # # directory, including one with existing data -- all monitor files
    # # will be namespaced). You can also dump to a tempdir if you'd
    # # like: tempfile.mkdtemp().
    # # outdir = '~/tmp/random-agent-results'
    # # env = wrappers.Monitor(env, directory=outdir, force=True)
    # env.seed(0)
    # agent = RandomAgent(env.action_space)
    # #print(env.ale.getAvailableDifficulties()) [0, 1, 2, 3]
    # env.ale.setDifficulty(1)
    # # print(env.ale.getAvailableModes()) [0, 2] // single/double agent
    # env.ale.setMode(0)

    env = gym.make('Pong-v0')
    for action in range(6):
        st  = env.get_action_meanings()[action] 
        if st.find('FIRE') == -1:
            print(st)

    # get_action_meanings()



    # episode_count = 1
    # reward = 0
    # done = False
    # pixels = []
    # pixels.append([168, 48, 143])
    
    # for i in range(episode_count):
    #     ob = env.reset()
    #     while True:
    #         action = agent.act(ob, reward, done)
    #         ob, reward, done, _ = env.step(action)
    #         # plt.matshow(ob[:,:, 2])
    #         # plt.imshow(ob)
    #         # plt.show()
    #         # # print(ob.shape)
    #         env._render()
    #         print(reward)
    #         time.sleep(0.5)
    #         # for i in range(ob.shape[-1]):
            
    #         # [168, 48, 143] 
    #         # [240 128 128] # pink player
    #         # [117 231 194] # green-blue
    #         # [236 236 236] # ball
    #         # [214 214 214] # net
    #         # [74 74 74] # black
    #         # for i in range(ob.shape[0]):
    #         #     for j in range(ob.shape[1]):
    #         #         f = False
    #         #         for pix in pixels:
    #         #             if ob[i,j,0] == pix[0] and ob[i, j, 1] == pix[1] and ob[i, j, 2] == pix[2]:
    #         #                 f = True
    #         #                 break
    #         #         if not f:
    #         #             print(ob[i,j,:])
    #         #             pixels.append(ob[i,j,:])
    #         # unique, counts = np.unique(ob, return_counts=True)
    #         # print(dict(zip(unique, counts)))

    #         # B: 143 214 194

            
    #         if done:
    #             break
    #         # env.render()
    #         # break
    #         # break


    # # Close the env and write monitor result info to disk
    # env.close()

