import ray
import os
from ray.rllib.algorithms.ppo import PPOConfig

# 1. Start Ray
ray.init(ignore_reinit_error=True)

# 2. Setup your experiment
config = (
    PPOConfig()
    .environment("CartPole-v1")
    .framework("torch")
    .resources(num_gpus=1)
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
)

# 3. Build the algorithm
algo = config.build()

# 4. Training Loop
for i in range(35):
    results = algo.train()
    
    # Since you disabled the new API stack, metrics are at the top level
    env_metrics = results.get("env_runners", ())
    reward = env_metrics.get("episode_reward_mean", "N/A")
    total_steps = results.get("agent_timesteps_total", 0)

    print(f"Iter: {i} | Steps: {total_steps} | Avg Reward: {reward}")

# 5. Save the Checkpoint
checkpoint_dir = algo.save(checkpoint_dir="/content/my_checkpoints")
print(f"\nCheckpoint saved at: {checkpoint_dir}")

# Optional: How to restore later
# from ray.rllib.algorithms.algorithm import Algorithm
# restored_algo = Algorithm.from_checkpoint(checkpoint_dir)

