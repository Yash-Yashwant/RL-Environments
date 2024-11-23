
import gymnasium as gym 
from stable_baselines3 import A2C
import os
import tensorboard

# env = gym.make("LunarLander-v2", render_mode = "human")``
# env.reset()


# model_path = f"{models}/270000.zip"

# model = PPO.load(model_path, env=env)

# episodes = 10

# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:
#         env.render()
#         action, _ = model.predict(obs)
#         obs, reward, done, infor = env.step(aciton)
# env.close()




#creating the environment

env = gym.make("LunarLander-v2", render_mode="human")
env.reset()

# trianing the agent

model = A2C("MlpPolicy", env, verbose=2) 


for i in range(30):
    model.learn(total_timesteps=10000) # Im gonna let the model learn, but for how much time, so I pass the timestep argumet.


# evaluation  phase

episode = 10

for ep in episode:
    obs = env.reset()
    done = False

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, info, done, _ = env.step(action)
        # obs, info, done, _ = env.step(env.action_space.sample())
    env.close()