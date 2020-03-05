import numpy as np
import gym
from gym import spaces
from utils.viewer import SimpleImageViewer
from collections import deque

import os
try:
    from stable_baselines.deepq import DQN, wrap_atari_dqn, CnnPolicy                    
except ImportError:
    print("IGNORE IF YOU ARE NOT USING FIREOTHERSKIPENV ")


class MaxAndSkipEnv(gym.Wrapper):
    """
    Wrapper from Berkeley's Assignment
    Takes a max pool over the last n states
    """
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip

    def step(self, action):
        total_info = {"delta_score": 0.0}
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward          
        
            if done:
                break
        # print(total_reward)
        total_info["delta_score"] = total_reward
        if abs(total_reward) < 0.1:
            total_reward = 0.01
        else :
            total_reward = 0.1*total_reward
            # np.random.choice([-1.0, 1.0], replace=True)

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, total_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs



def isFire(action):
    """
    action: int
    env: Pong-v0
    return true if it is a FIRE action
    """
    # ACTION_MEANINGS = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    # action_mean = ACTION_MEANINGS[action]
    # if action_mean.find('FIRE') == -1:
    #     return False
    # else:
    #     return True
    return (action==1) or (action==4) or (action==5)
 

class FireOtherSkipEnv(gym.Wrapper):
    """
    skip frames not producing FIRE action in DDQN
    """
    def __init__(self, env=None):
        """Return only every `skip`-th frame"""
        super(FireOtherSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self.env = wrap_atari_dqn(self.env)
        model_output='/home/jingjia16/stable-baselines/scripts/deepq_pong.zip' 
        if os.path.exists(model_output):
            self.model = DQN.load(model_output)
        else:
            print("failed to load the model")





class PreproWrapper(gym.Wrapper):
    """
    Wrapper for Pong to apply preprocessing
    Stores the state into variable self.obs
    """
    def __init__(self, env, prepro, shape, overwrite_render=True, high=255):
        """
        Args:
            env: (gym env)
            prepro: (function) to apply to a state for preprocessing
            shape: (list) shape of obs after prepro
            overwrite_render: (bool) if True, render is overwriten to vizualise effect of prepro
            grey_scale: (bool) if True, assume grey scale, else black and white
            high: (int) max value of state after prepro
        """
        super(PreproWrapper, self).__init__(env)
        self.overwrite_render = overwrite_render
        self.viewer = None
        self.prepro = prepro
        self.observation_space = spaces.Box(low=0, high=high, shape=shape, dtype=np.uint8)
        self.high = high


    def step(self, action):
        """
        Overwrites _step function from environment to apply preprocess
        """
        obs, reward, done, info = self.env.step(action)
        self.obs = self.prepro(obs)
        return self.obs, reward, done, info


    def reset(self):
        self.obs = self.prepro(self.env.reset())
        return self.obs


    def _render(self, mode='human', close=False):
        """
        Overwrite _render function to vizualize preprocessing
        """

        if self.overwrite_render:
            if close:
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                return
            img = self.obs
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = SimpleImageViewer()
                self.viewer.imshow(img)

        else:
            super(PreproWrapper, self)._render(mode, close)


class TennisPreproWrapper(gym.Wrapper):
    """
    Wrapper for Tennis to apply preprocessing
    Stores the state into variable self.obs
    """
    def __init__(self, env, prepro, shape, overwrite_render=True, high=255):
        """
        Args:
            env: (gym env)
            prepro: (function) to apply to a state for preprocessing
            shape: (list) shape of obs after prepro
            overwrite_render: (bool) if True, render is overwriten to vizualise effect of prepro
            grey_scale: (bool) if True, assume grey scale, else black and white
            high: (int) max value of state after prepro
        """
        super(TennisPreproWrapper, self).__init__(env)
        self.overwrite_render = overwrite_render
        self.viewer = None
        self.prepro = prepro
        self.observation_space = spaces.Box(low=0, high=high, shape=shape, dtype=np.uint8)
        self.high = high


    def step(self, action):
        """
        Overwrites _step function from environment to apply preprocess
        """
        obs, reward, done, info = self.env.step(action)
        self.obs = self.prepro(obs)
        return self.obs, reward, done, info


    def reset(self):
        self.obs = self.prepro(self.env.reset())
        return self.obs


    def _render(self, mode='human', close=False):
        """
        Overwrite _render function to vizualize preprocessing
        """

        if self.overwrite_render:
            if close:
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                return
            img = self.obs
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = SimpleImageViewer()
                self.viewer.imshow(img)
                return self.viewer.isopen

        else:
            super(TennisWrapper, self)._render(mode, close)
