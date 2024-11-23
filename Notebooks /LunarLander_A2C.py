#creating an A2C environment
from stable_baselines3 import PPO
import gymnasium as gym


print("Updated Environment: Substituted predict instead of sample")
env = gym.make("LunarLander-v2", render_mode = "human")
env.reset()


#Training Loop
model = PPO("MlpPolicy", env, verbose=1)  # Corrected policy name
for i in range(5):
    model.learn(total_timesteps=1000)
    # wandb.log() have to implement wandb log

#Evaluation Loop
episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        print("here")
        env.render()
        # add predict() 
        action, _ = model.predict()
        obs, reward, info,done, _ = env.step(action)

        # obs, reward, done, info = env.step(env.action_space.sample())
        # action, _states = model.predict(obs)
        # obs, reward, done, info = env.step(action)

env.close()


# episodes mean reward is really bad
# things I want to change, increase learning rate, and reduce output log time.




