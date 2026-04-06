import ray
from ray import tune
import gymnasium as gym
import highway_env
from ray.rllib.algorithms.ppo import PPOConfig

# 1. Start Ray
ray.init(ignore_reinit_error=True)

# 2. FIXED Creator Function
def env_creator(env_config):
    import highway_env
    import gymnasium as gym
    # IMPORTANT: Pass the env_config dictionary to the 'config' parameter
    # instead of using **env_config
    return gym.make("intersection-v0", config=env_config)

# Register it (only need to do this once)
tune.register_env("my_intersection", env_creator)

# 3. Configure PPO
config = (
    PPOConfig()
    .environment(
        env="my_intersection",
        env_config={
            "initial_vehicle_count": 5,
        }
    )
    .framework("torch")
    .env_runners(
        num_env_runners=1,
        # This is the secret to speed!
        num_envs_per_env_runner=4, 
    )
    .training(
        # Standard PPO parameter names
        train_batch_size=1000
    )
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
)

algo = config.build_algo()

# 4. Training Loop
for i in range(10):
    results = algo.train()

    env_metrics = results.get("env_runners", {})
    reward = env_metrics.get("episode_reward_mean", "N/A")
    # 'timesteps_total' is the best way to see progress
    total_steps = results.get("timesteps_total", 0)
    #reward = results.get("episode_reward_mean", "N/A")
    
    print(f"Iter: {i} | Total Sim Steps: {total_steps} | Avg Reward: {reward}")

