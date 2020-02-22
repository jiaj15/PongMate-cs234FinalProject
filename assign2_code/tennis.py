import gym

if __name__ == "__main__":
    env = gym.make("Tennis-v0")
    print(env.action_space.n)