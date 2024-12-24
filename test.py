
import gymnasium as gym
import ale_py
# import shimmy
import os
import wandb
from stable_baselines3 import PPO
# from stable_baselines3 import A2C
import tensorboard
model_dir = "models_boxing/A2C"
log_dir = "logs_boxing"

# os.makedirs(model_dir, exist_ok=True)
# os.makedirs(log_dir, exist_ok=True)

env = gym.make("ALE/Boxing-v5", render_mode = "human")

# model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Train the agent
for i in range(30):
    model.learn(total_timesteps=10000, reset_num_timesteps= False, tb_log_name="PPO") # 10k timesteps per iteration, and for 30 iterations
    # model.save("{}/{}".format(model_dir, 10000*i))
    model.save(f"{model_dir}/{10000*i}")


# Evaluate the agent

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

env.close()




