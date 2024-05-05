from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
from robotEnv import CustomEnv
import pygame
import time
import json
import os
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Initialize a dictionary to store episode times
episode_times = {}


screen = pygame.display.set_mode((800, 600))

env = CustomEnv(screen)

model_path = "/home/two-intel/Documents/Two/ait/reinforcement-learning/2d-robot-arm-2DoF-DRL/stb3/report/A2C_MLP_Robot2DoF/model/80000.zip"

folder_path = '/home/two-intel/Documents/Two/ait/reinforcement-learning/2d-robot-arm-2DoF-DRL/stb3/report/A2C_MLP_Robot2DoF/model/'
files = os.listdir(folder_path)

for file in files:
    # print(folder_path+file)

    # Load the trained model
    model = A2C.load(folder_path+file, env=env)

    text = model_path.split("/")

    last_text = text[-1].split(".")

    # Lists to store rewards and episode lengths
    episode_rewards = []
    episode_lengths = []

    time_recoard = []

    torch.cuda.empty_cache()



    # Inference loop
    episodes = 10
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        start_time = time.time()
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            env.render(mode='human')  # Render the environment in human mode during inference
            print(f"reward : {reward}")
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Calculate and print time taken
        end_time = time.time()
        time_taken = end_time - start_time
        time_recoard.append(time_taken)

    time_average = sum(time_recoard) / len(time_recoard)



    print(f"Episode {ep+1} took {time_average:.2f} seconds")


            # Save frames as a video
    # video_path = f"report/video_{text[10]}_{last_text[0]}.mp4"
    # with imageio.get_writer(video_path, fps=30) as video:
    #     for frame in frames:
    #         video.append_data(frame)


    # Plotting rewards
    plt.plot(np.arange(1, episodes + 1), episode_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Episode Reward Plot model {last_text[0]} Time : {time_average:.2f}')
    # plt.grid(True)
    # plt.show()

    # plt.tight_layout()
    plt.savefig(f"report/img1_{text[10]}_{last_text[0]}.jpg")
    # plt.show()

    env.close()  # Close the environment after inference
    torch.cuda.empty_cache()