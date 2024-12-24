
import gymnasium as gym
import stable_baselines as sb3


env = gym.make("LunarLander-v2")

model = PPO("MlpPolicy", env, verbose = 1)


for _ in range(10):  # training phase
    model.learn(total_timesteps = 1000)


episodes = 10 

for ep in episodes:
    obs = env.reset()
    done = False

    while not done:
         action, _  = env.preditct(obs)
         obs, reward, done, info, _ = env.step(action)
         env.render()

env.close()


