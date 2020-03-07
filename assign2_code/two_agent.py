import retro
from configs.q5_train_atari_nature import config, config_short
from q3_nature import NatureQN
import numpy as np
import gym
import time
import retro
from utils.wrappers import *
from utils.preprocess import greyscale

try:
    from stable_baselines.deepq import DQN, wrap_atari_dqn, CnnPolicy
    from stable_baselines.common.atari_wrappers import LazyFrames                  
except ImportError:
    print("IGNORE IF YOU ARE NOT USING STABLEBASELINEAGENT ")

ACTION_MEANINGS = {0:0, 1:1, 2:2, 3:4, 4:3, 5:5}

class retroActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(retroActionWrapper, self).__init__(env)
        self.action_space = spaces.Discrete(6)
    def action(self, action):        
        return ACTION_MEANINGS[action]

class retroWrapper(gym.Wrapper):
    def __init__(self, env, frameskip=[2, 5]):
        super(retroWrapper, self).__init__(env)
        self.frameskip= frameskip

    def step(self, a):
        reward = 0.0
        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            ob, r, done, info = self.env.step(a)
            reward += r
        return ob, reward, done, info

class StableBaselineAgent(object):
    """
    mimic a player with different levels
    """
    def __init__(self, env, epsilon=0.5):
        self.epsilon = epsilon
        model_output='/home/jingjia16/stable-baselines/scripts/deepq_pong.zip' 
        self.action_space = env.action_space
        if os.path.exists(model_output):
            self.model = DQN.load(model_output)
        else:
            print("failed to load the model")
            raise NotImplementedError
    def act(self, observation):
        # observation = LazyFrames(observation)
        best_action = self.model.predict(observation)

        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else :
            return best_action

class WellTrainedAgent(object):
    def __init__(self, env=None, epsilon=0.5):
        self.epsilon = epsilon
        
        # env = gym.make('Pong-v0')

        # env = MaxAndSkipEnv(env, skip=config.skip_frame)
        # env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
        #                     overwrite_render=config.overwrite_render)
        self.model = NatureQN(env, config)
        self.action_space = env.action_space
        self.model.load(0, well_trained=True)
        
    def act(self, observation):
        best_action, qvales = self.model.predict(observation)
        print((best_action,qvales))
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else :
            return best_action
    def test(self, env):
        env = MaxAndSkipEnvForTest(env)
        ob = env.reset()
        while True:
            act = self.act(ob)
            ob, r, done, info = env.step(act)
            env.render()
            if done:
                break
            time.sleep(0.01)









class HumanAgent(object):
    def __init__(self, env):
        self.human_agent_action = 0
        self.human_wants_restart = False
        self.human_sets_pause = False
        self.env = env
        self.env.render()
        self.env.unwrapped.viewer.window.on_key_press = self.key_press
        self.env.unwrapped.viewer.window.on_key_release = self.key_release
        



    def key_press(self, key, mod):
        if key==32: self.human_sets_pause = not self.human_sets_pause
        a = int( key - ord('0') )
        if a <= 0 or a >= 6: return
        self.human_agent_action = a

    def key_release(self, key, mod):
        a = int( key - ord('0') )
        if a <= 0 or a >= 6: return
        if self.human_agent_action == a:
            self.human_agent_action = 0

    def main(self):
        ob = self.env.reset()
        total_reward = 0
        while True:
            ob, r, done, info = self.env.step(self.human_agent_action)
            if r != 0:
                print("reward %0.3f" % r)
            total_reward += r
            self.env.render()
            if done: break
            while self.human_sets_pause:
                self.env.render()
                time.sleep(0.1)
            time.sleep(0.1)
if __name__ == "__main__":
    # print(retro.data.list_games())
   
    env = retro.make(game='Pong-Atari2600', use_restricted_actions=retro.Actions.DISCRETE) 
    env = retroActionWrapper(env)
    env = retroWrapper(env)

    # env = gym.make('Pong-v0')
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
                        overwrite_render=config.overwrite_render)
    
    # for action in range(env.action_space.n):
    #     print(action)
    #     print(list(env.get_action_meaning(action)))
        # print(env.action_to_array(action))
        # print(env.action_space.sample())
        # lists.append(env.action_space.sample())
    # print(list(genv.ale.getAvailableModes()))
    # genv.ale.setMode(2)
    # human = HumanAgent(genv)
    # human.main()
    wt = WellTrainedAgent(env)
    wt.test(env)
        

    