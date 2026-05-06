import argparse
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="MiniGrid-Empty-8x8-v0")
parser.add_argument("--steps", type=int, default=100000)
args = parser.parse_args()

env = gym.make(args.env)
env = ImgObsWrapper(env)

obs, info = env.reset()
episode_reward = 0.0

for step in range(args.steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    if terminated or truncated:
        print(f"episode done reward={episode_reward}")
        obs, info = env.reset()
        episode_reward = 0.0

print("training finished")
