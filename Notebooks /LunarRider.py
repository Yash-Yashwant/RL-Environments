############################ Introduction ###################################################

import gym
env = gym.make("LunarLander-v2", render_mode = "human")
# env.reset()
# for step in range(200):
#     env.render()  # Add parentheses to call the render function
#     env.step(env.action_space.sample())
# env.close()


# while True:
#     env.reset()
#     done = False


#     while not done:
#         action = env.action_space.sample()
#         observation, reward, terminated, truncated, info = env.step(action)
#         env.render()

#         if terminated:
#             print("Succeffuly Landed")
#             done = True

#         else:
#             print("Needs re-work")
#             break
# env.close()


import gym

env = gym.make("LunarLander-v2", render_mode="human")

# # Run until successfully landing
# while True:
#     env.reset()  # Reset the environment without capturing the observation or info
#     done = False
    
#     while not done:
#         action = env.action_space.sample()  # Choose a random action
#         _, reward, terminated, truncated, _ = env.step(action)  # Perform the action
        
#         env.render()  # Render the environment

#         if terminated:  # Check if the episode ended (either success or failure)
#             # If the reward is positive, we assume it's a successful landing
#             if reward > 0:
#                 print("Successfully landed! Ending after first success.")
#                 done = True  # Stop the loop on success
#             else:
#                 print("Failed attempt. Resetting environment...")  # Failed landing
#                 break  # Reset the environment for a new attempt

# env.close()
max_steps = 1000  # Set a limit for steps in each episode to prevent endless loops

while True:
    env.reset()  # Capture the observation for possible use
    done = False
    steps = 0  # Counter for steps within an episode
    
    while not done and steps < max_steps:
        action = env.action_space.sample()  # Choose a random action
        observation, reward, terminated, truncated, _ = env.step(action)  # Perform the action
        env.render()  # Render the environment
        steps += 1
        
        if terminated or truncated:
            if reward > 0:  # positive reward implies successful landing
                print("Successfully landed! Ending after first success.")
                done = True  # Stop the loop on success
            else:
                print("Failed attempt. Resetting environment...")
            break  # Exit the loop to reset environment after success or failure

    if done:  # Exit the outer loop after first successful landing
        break

env.close()




