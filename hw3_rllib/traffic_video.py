import ray
from ray import tune
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
from ray.rllib.algorithms.ppo import PPOConfig
import os

# 1. Start Ray and Re-register the environment
ray.init(ignore_reinit_error=True)

def env_creator(env_config):
    import highway_env
    import gymnasium as gym
    return gym.make("intersection-v0", config=env_config)

tune.register_env("my_intersection", env_creator)

# 2. Reconstruct the exact same config used in training
config = (
    PPOConfig()
    .environment(
        env="my_intersection",
        env_config={
            "initial_vehicle_count": 5,
            "duration": 15,
            "offroad_terminal": True,
        }
    )
    .framework("torch")
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
)

# 3. Build and Restore the Agent
algo = config.build_algo()
# Ensure this path points to your saved checkpoint folder
algo.restore("/content/traffic_checkpoint") 

# 4. Create a dedicated environment for recording
# We use 'rgb_array' so the RecordVideo wrapper can capture frames
video_path = "/content/videos"
os.makedirs(video_path, exist_ok=True)

eval_env = gym.make("intersection-v0", render_mode="rgb_array")
eval_env = RecordVideo(
    eval_env, 
    video_folder=video_path, 
    name_prefix="trained_traffic_agent"
)

# 5. Run a single "Inference" episode
obs, info = eval_env.reset()
done = False
truncated = False
total_reward = 0

print("Starting video recording...")

while not (done or truncated):
    # The brain picks the action based on current observation
    action = algo.compute_single_action(obs)
    obs, reward, done, truncated, info = eval_env.step(action)
    total_reward += reward

eval_env.close()
print(f"Finished! Video saved to {video_path}. Final Reward: {total_reward}")

