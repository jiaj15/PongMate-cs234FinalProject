import retro
from configs.q5_train_atari_nature import config, config_short
from q3_nature import NatureQN
import numpy as np
import gym
import time
import retro
from utils.wrappers import *
from utils.preprocess import greyscale
from utils.replay_buffer import ReplayBuffer
from TSample import TSampling
from configs.ts_config import config

try:
    from stable_baselines.deepq import DQN, wrap_atari_dqn, CnnPolicy
    from stable_baselines.common.atari_wrappers import LazyFrames                  
except ImportError:
    print("IGNORE IF YOU ARE NOT USING STABLEBASELINEAGENT ")

#ACTION_MEANINGS = {0:0, 1:1, 2:2, 3:4, 4:3, 5:5}
ACTION_MEANINGS = {0:1, 1:1, 2:2, 3:4, 4:3, 5:5}

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        print(buttons)
        buttons = []
        buttons.append(['BUTTON', None, 'SELECT', 'RESET', 'UP', 'DOWN', 'LEFT', 'RIGHT'])
        buttons.append(['UP', 'DOWN',  'SELECT', 'RESET', 'NONSENSE', None,'LEFT', 'RIGHT', 'NONSENSE', 'BUTTON'])

        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for i,button_two in enumerate(combo):
                if len(button_two)==0:
                    arr[6*i + buttons[i].index(None)] = True

                for button in button_two:
                    arr[6*i+ buttons[i].index(button)] = True
            self._decode_discrete_action.append(arr.copy())
            print((combo, arr, len(self._decode_discrete_action)))
        # arr = np.array([False] * env.action_space.n)
        # arr[14] = True
        # self._decode_discrete_action.append(arr.copy())
        # arr[14] = False
        # arr[15] = True
        # self._decode_discrete_action.append(arr.copy())
        # arr[14] = True
        # self._decode_discrete_action.append(arr.copy())

        

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        # print(self._decode_discrete_action[act].copy())
        return self._decode_discrete_action[act].copy()


class PongDiscretizer(Discretizer):
    """
    Use Sonic-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env, players=2):
        lens1 = [[], ['BUTTON'], ['UP'], ['DOWN'], ['BUTTON','UP'], ['BUTTON','DOWN']]
        # lens2 = [[], ['BUTTON'], ['LEFT'], ['RIGHT'], ['BUTTON','LEFT'], ['BUTTON','RIGHT']]
        # lens1 = [[], ['BUTTON'], ['UP'], ['DOWN'], ['RESET'], ['SELECT']]
        lens2 = [[], ['BUTTON'], ['UP'], ['DOWN'], ['RESET'], ['SELECT']]
        combos = []
        for i in range(6):
            for j in range(6):
                combos.append((lens1[i], lens1[j]))           
        super().__init__(env=env, combos=combos)



class retroActionWrapper(gym.ActionWrapper):
    def __init__(self, env, agent_num=1):
        super(retroActionWrapper, self).__init__(env)
        if agent_num == 1:
            self.action_space = spaces.Discrete(6)
        else:
            self.action_space = spaces.Discrete(324)
        self.agent_num = agent_num
    def action(self, action):
        print(action)
        if self.agent_num == 1:     
            return (ACTION_MEANINGS[action])
        else:
            combined_act = ACTION_MEANINGS[action[0]] + 18 * ACTION_MEANINGS[action[1]]
            print(combined_act)
            print(env.get_action_meaning(combined_act))
            return combined_act




class retroWrapper(gym.Wrapper):
    """
    used only for training!
    """
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

class retroOneWrapper(gym.Wrapper):
    def __init__(self, env, frameskip=[2, 5]):
        super(retroOneWrapper, self).__init__(env)
        self.frameskip= frameskip

    def step(self, a):
        reward = [0.0, 0.0]
        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            ob, r, done, info = self.env.step(a)
            reward[0] += r[0]
            reward[1] += r[1]
        return ob, reward[0], done, info

class retroTwoWrapper(gym.Wrapper):
    def __init__(self, env, frameskip=[2, 5]):
        super(retroTwoWrapper, self).__init__(env)
        self.frameskip= frameskip

    def step(self, a):
        reward = [0.0, 0.0]
        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            ob, r, done, info = self.env.step(a)
            reward[0] += r[0]
            reward[1] += r[1]
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

class randomAgent(object):
    def __init__(self):
        pass
        
    def act(self):
        return np.random.choice(range(6), 1)[0]

class WellTrainedAgent(object):
    def __init__(self, env=None, epsilon=0.5):
        self.epsilon = epsilon
        self.model = NatureQN(env, config)
        self.model.load(0, well_trained=True)
        
    def act(self, observation):
        best_action, qvales = self.model.predict(observation)
        # print((best_action,qvales))
        if np.random.random() < self.epsilon:
            return np.random.choice(range(6))
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
    def __init__(self, env=None):
        self.human_agent_action = 0
        self.human_wants_restart = False
        self.human_sets_pause = False
        env = retro.make(game='Pong-Atari2600', players=2)

        test_env = PongDiscretizer(env, players=2)
        #test_env = retroActionWrapper(env, agent_num=2)
        print(test_env.action_space.n)
        test_env = retroTwoWrapper(test_env)
        test_env = MaxAndSkipEnv(test_env, skip=config.skip_frame, agent_num=2)
        test_env = PreproWrapper(test_env, prepro=greyscale, shape=(80, 80, 1),
                            overwrite_render=config.overwrite_render)
        self._frame_buffer = deque(maxlen=4)
        for i in range(2):
            self._frame_buffer.append(np.zeros((80,80,1)))

        test_env = MaxAndSkipEnvForTest(test_env)

        config_env = PongDiscretizer(env, players=1)
        #self.sd = StableBaselineAgent(config_env, epsilon=0.00001)
        config_env = retroActionWrapper(env, agent_num=1)
        config_env = retroOneWrapper(config_env) 
        config_env = MaxAndSkipEnv(config_env, skip=config.skip_frame)
        config_env = PreproWrapper(config_env, prepro=greyscale, shape=(80, 80, 1),
                            overwrite_render=config.overwrite_render)
        # self.sd = StableBaselineAgent(config_env, epsilon=0.00001)

        self.ts = TSampling(10, config, config_env)
        # self.wt = WellTrainedAgent(config_env, epsilon=0.01)
        self.env = test_env
        self.env.render()
        self.env.unwrapped.viewer.window.on_key_press = self.key_press
        self.env.unwrapped.viewer.window.on_key_release = self.key_release
        



    def key_press(self, key, mod):
        if key==32: self.human_sets_pause = not self.human_sets_pause
        a = int( key - ord('0') )
        if a <= 0 or a >= 10: return
        self.human_agent_action = a
        if self.human_agent_action >= 7:
            self.human_agent_action += 29


    def key_release(self, key, mod):
        a = int( key - ord('0') )
        if a <= 0 or a >= 10: return
        # if self.human_agent_action >= 7:
        #     self.human_agent_action += 29
        if self.human_agent_action == a:
            self.human_agent_action = 0
        if self.human_agent_action - 29 == a:
            self.human_agent_action = 0


    def main(self):
        ob = self.env.reset()
        total_reward = 0
        self.rollout = 0
        while True:
            
            #a1 = self.wt.act(ob)
            a1 = self.ts.action(ob)
            if self.rollout == 0:
                action = 6*a1+self.human_agent_action
                ob, r, done, info = self.env.step(action)
            elif self.rollout < 0:
                if a1 == 1 or a1 == 4 or a1 == 5:
                    action = 6 *a1 + self.human_agent_action
                else:
                    action = 6 * 1 + self.human_agent_action
                print("right: serve ball")
                # action = 6 * 1 + self.human_agent_action
                ob, r, done, info = self.env.step(action)
            elif self.rollout > 0:
                print("left: serve ball")
                action = 6*a1+ np.random.choice([1,4,5],1)[0]
                ob, r, done, info = self.env.step(action)
            
            if r[0] != 0:
                print("reward {}".format(r))
            total_reward += r[0]
            self.env.render()
            self.ts.updateBelief(r[1], done)

            if done: break
            if r[0] > 0:
                self.rollout = -10
            elif r[1] > 0:
                self.rollout = 10
            else:
                self.rollout -= np.sign(self.rollout)
            
            

            while self.human_sets_pause:
                self.env.render()
                time.sleep(0.1)
            time.sleep(0.1)


# def make_config_env():
#     config_env = retro.make(game='Pong-Atari2600', use_restricted_actions=retro.Actions.DISCRETE) 
#     print(config_env.action_space.n)
#     config_env = retroActionWrapper(config_env)
#     config_env = retroWrapper(config_env)

#     # env = gym.make('Pong-v0')
#     config_env = MaxAndSkipEnv(config_env, skip=config.skip_frame)
#     config_env = PreproWrapper(config_env, prepro=greyscale, shape=(80, 80, 1),
#                         overwrite_render=config.overwrite_render)
#     return config_env

# def make_test_env(agent_num=2):
#     env = retro.make(game='Pong-Atari2600', use_restricted_actions=retro.Actions.DISCRETE, players=agent_num)
#     print(env.action_space.n) 
#     env = retroActionWrapper(env, agent_num=2)
#     env = retroWrapper(env)

#     # env = gym.make('Pong-v0')
#     env = MaxAndSkipEnv(env, skip=config.skip_frame)
#     env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1),
#                         overwrite_render=config.overwrite_render)
#     env = MaxAndSkipEnvForTest(env)
#     return env






if __name__ == "__main__":
    # env = gym.make('Pong-v0')

    # # print(retro.data.list_games())
    env = retro.make(game='Pong-Atari2600', players=2)

    test_env = PongDiscretizer(env, players=2)
    #test_env = retroActionWrapper(env, agent_num=2)
    print(test_env.action_space.n)
    test_env = retroTwoWrapper(test_env)
    test_env = MaxAndSkipEnv(test_env, skip=config.skip_frame, agent_num=2)
    test_env = PreproWrapper(test_env, prepro=greyscale, shape=(80, 80, 1),
                        overwrite_render=config.overwrite_render)
    test_env = MaxAndSkipEnvForTest(test_env)

    config_env = PongDiscretizer(env, players=1)
    config_env = retroActionWrapper(env, agent_num=1)
    config_env = retroOneWrapper(config_env) 
    config_env = MaxAndSkipEnv(config_env, skip=config.skip_frame)
    config_env = PreproWrapper(config_env, prepro=greyscale, shape=(80, 80, 1),
                         overwrite_render=config.overwrite_render)
    ts = TSampling(10, config, config_env)
    # wt = WellTrainedAgent(config_env, epsilon=0.01)

    # # test_env = PongDiscretizer(env, players=2)
    # # #test_env = retroActionWrapper(env, agent_num=2)
    # # print(test_env.action_space.n)
    # # test_env = retroTwoWrapper(test_env)
    # # test_env = MaxAndSkipEnv(test_env, skip=config.skip_frame, agent_num=2)
    # # test_env = PreproWrapper(test_env, prepro=greyscale, shape=(80, 80, 1),
    # #                     overwrite_render=config.overwrite_render)
    # # test_env = MaxAndSkipEnvForTest(test_env)
    # # print('retro.Actions.DISCRETE action_space', env.action_space)
    # # for i in range(env.action_space.n):

    # #     print((i,env.get_action_meaning(i)))

    

    # rd = randomAgent()

    ob = test_env.reset()
    for level in levels:
    for i in range(10):
        ob = test_env.reset()
        print("game:{}".format(i))
        rollout = 0
        
        while True:
            a1 = ts.action_human(ob,5)
            ob = np.fliplr(ob)
            a2 = ts.action_ts(ob)
            if rollout == 0:
                action = 6*a1+a2
                ob, r, done, info = test_env.step(action)
            elif rollout < 0:
                print("right: serve ball")
                if a1 == 1 or a1 == 4 or a1 == 5:
                    action = 6 *a1 + a2
                else:
                    action = 6 * 1 + a2
                ob, r, done, info = test_env.step(action)
            elif rollout > 0:
                print("left: serve ball")
                action = 6*a1 + 1
                ob, r, done, info = test_env.step(action)
            
            if r[0] != 0:
                print("reward {}".format(r))
            ts.updateBelief(r[1], done)
            test_env.render()

            if done: break
            if r[0] > 0:
                rollout = -6
            elif r[1] > 0:
                rollout = 6
            else:
                rollout -= np.sign(rollout)
            
            if done:
                test_env.reset()
                break
            time.sleep(0.1)
            test_env.render()
    test_env.close()
    # kb = HumanAgent()
    # kb.main()






   
        

    