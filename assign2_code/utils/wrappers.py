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
        # if abs(total_reward) < 0.1:
        #     total_reward = 0.01
        # else :
        #     total_reward = 0.1*total_reward
            # np.random.choice([-1.0, 1.0], replace=True)

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, total_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class MaxAndSkipEnvForTest(gym.Wrapper):
    """
    Wrapper from Berkeley's Assignment
    Takes a max pool over the last n states
    """
    def __init__(self, env=None):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnvForTest, self).__init__(env)
        init_frame = self.env.reset()
        # most recent raw observations (for max pooling across time steps)
        self._frame_buffer = deque(maxlen=4)
        for i in range(4):
            self._frame_buffer.append(np.zeros((80,80,1)))

    def step(self, action):        
        obs, reward, done, info = self.env.step(action)
        self._frame_buffer.append(obs)
        frame = np.concatenate(self._frame_buffer, 2)      

        return frame, reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        obs = self.env.reset()
        for i in range(4):
            self._frame_buffer.append(np.zeros((80,80,1)))
        self._frame_buffer.append(obs)
        frame = np.concatenate(self._frame_buffer, 2) 
        return frame



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
            
    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        total_reward = reward
        interval_length = 0

        # counting for later rewards
        while True:
            dqn_action, _states = self.model.predict(obs)
            # print(dqn_action)
            if isFire(dqn_action):
                # print("--------------{}".format(dqn_action))
                break
            obs, reward, done, info = self.env.step(dqn_action)
            self._obs_buffer.append(np.array(obs[:,:,:]))
            total_reward += reward
            interval_length += 1
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        interval_reward = total_reward*0.5 + interval_length*0.1            
        return max_frame, interval_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs





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
