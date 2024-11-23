import gymnasium as gym 
from stable_baselines3 import A2C
import os
import tensorboard

######## PPO ###############


# model_dr = "models/PPO" # this will be the directory path to save the checkpoint files
# logdir = "logs" # will be saving all the logs

# # os.makedirs("model_dr")
# # os.makedirs("logs")                                           

# env = gym.make("LunarLander-v2") # generating the environment
# obs, info = env.reset() # reset the env after every episode

# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log= logdir) # PPO is an RL algo to train agent. 
# # the overall focus of these algorithms is to maximise the cumilative reward. How does that happen? By improving the agent predicitons ability for the future actions

# # MlpPolicy, its a policy architecture
# # env, working environment
# # verbose, it's verbosity parameter, used for logging, 0 means logging of, 1 - standard logging, 2 - detailed logging.

# # question for me, PPO's underlying principle is neural network, have to clarify myself about this topic. 
# for i in range(30):
#     model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name= "PPO")  # at every iteration, the time_step resets.
#     model.save(f"{model_dr}/{10000*i}")

####### A2C #############
import gymnasium as gym 
from stable_baselines3 import A2C
import os
import numpy as np 
import tensorboard
model_dr = "models/A2C" # this will be the directory path to save the checkpoint files
logdir = "logs" # will be saving all the logs

# os.makedirs("model_dr")
# os.makedirs("logs")                                           

env = gym.make("LunarLander-v2") # generating the environment
obs, info = env.reset() # reset the env after every episode

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log= logdir) # PPO is an RL algo to train agent. 
# the overall focus of these algorithms is to maximise the cumilative reward. How does that happen? By improving the agent predicitons ability for the future actions

# MlpPolicy, its a policy architecture
# env, working environment
# verbose, it's verbosity parameter, used for logging, 0 means logging of, 1 - standard logging, 2 - detailed logging.

# question for me, PPO's underlying principle is neural network, have to clarify myself about this topic. 

for i in range(30):
    model.learn(total_timesteps=10000, reset_num_timesteps= False, tb_log_name= "A2C")  # at every iteration, the time_step resets.
    model.save(f"{model_dr}/{10000*i}")



######################### using the saved model #####################

# import gymnasium as gym 
# from stable_baselines3 import A2C
# import os
# import tensorboard


# env = gym.make("LunarLander-v2", render_mode = "human")

# model_dir = "models/A2C"
# model_path = f"{model_dir}/250000.zip"

# episodes = 10

# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:
#         env.render()
#         obs, reward, done, info,_ = env.step(env.action_space.sample())
# env.close()
