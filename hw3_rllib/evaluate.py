import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import ray
from ray.rllib.algorithms.algorithm import Algorithm

# 1. Initialize Ray
ray.init(ignore_reinit_error=True)

# 2. Path to your checkpoint
# This should be the folder created by algo.save()
checkpoint_path = "/content/my_checkpoints"

# 3. Restore the Algorithm directly from the checkpoint
# This loads the config, the policy weights, and the optimizer state
algo = Algorithm.from_checkpoint(checkpoint_path)

print(f"Successfully restored checkpoint from: {checkpoint_path}")

# 4. Setup the Recording Environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)

# 5. Run the Evaluation
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    # Since we restored with the 'Old API Stack' settings from your train script,
    # compute_single_action will work perfectly here.
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Evaluation finished! Total Reward: {total_reward}")
env.close()
