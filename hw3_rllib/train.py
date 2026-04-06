import ray
from ray.rllib.algorithms.ppo import PPOConfig

# 1. Start Ray
ray.init()

# 2. Setup your experiment
config = (
    PPOConfig()
    .environment("CartPole-v1")
    .framework("torch")  # or "tf2"
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
)

# 3. Build the algorithm and run a few loops
algo = config.build()

for i in range(10): # Increased iterations to give it time
    results = algo.train()
    
    # Check the new API stack structure
    env_metrics = results.get("env_runners", {})
    reward = env_metrics.get("episode_reward_mean", "N/A")
    total_steps = results.get("num_env_steps_sampled_lifetime", 0)
    
    print(f"Iter: {i} | Steps: {total_steps} | Avg Reward: {reward}")

