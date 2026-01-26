"""
TO EDIT: Some miscellaneous utility functions

Functions to edit:
    1. sample_trajectory (line 19)
    2. sample_trajectories (line 67)
    3. sample_n_trajectories (line 83)
"""
import numpy as np
import time

############################################
############################################

MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]
MJ_ENV_KWARGS = {name: {"render_mode": "rgb_array"} for name in MJ_ENV_NAMES}
MJ_ENV_KWARGS["Ant-v4"]["use_contact_forces"] = True

# 
def sample_trajectory(env, policy, max_path_length, render=False):
    # ----- reset (new Gym API compatible) -----
    reset_out = env.reset()
    ob = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    obs, acs, rewards, next_obs, terminals = [], [], [], [], []
    image_obs = []
    steps = 0

    while True:

        # ----- rendering (FIXED SHAPE) -----
        if render:
            if hasattr(env.unwrapped, "sim"):
                if "track" in env.unwrapped.model.camera_names:
                    frame = env.unwrapped.sim.render(
                        camera_name="track", height=500, width=500
                    )[::-1]
                else:
                    frame = env.unwrapped.sim.render(
                        height=500, width=500
                    )[::-1]
            else:
                frame = env.render()

            # CRITICAL FIX: add camera dimension
            image_obs.append(frame[None, ...])   # (1, H, W, 3)

        obs.append(ob)

        # policy expects (obs,) or (1, obs)
        ac = policy.get_action(ob)
        ac = ac[0]
        acs.append(ac)

        # ----- step (new Gym API compatible) -----
        step_out = env.step(ac)
        if len(step_out) == 4:
            ob_next, rew, done, _ = step_out
        else:
            ob_next, rew, terminated, truncated, _ = step_out
            done = terminated or truncated

        rewards.append(rew)
        next_obs.append(ob_next)

        rollout_done = done or steps >= max_path_length
        terminals.append(int(rollout_done))

        ob = ob_next
        steps += 1

        if rollout_done:
            break

    # ----- stack arrays -----
    image_obs = np.array(image_obs, dtype=np.uint8) if render else None

    return Path(
        obs,
        image_obs,      # shape: (T, 1, H, W, 3)
        acs,
        rewards,
        next_obs,
        terminals,
    )

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    """
        Collect rollouts until we have collected `min_timesteps_per_batch` steps.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        # TODO
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):
    """
        Collect `ntraj` rollouts.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    """
    paths = []
        
    for _ in range(ntraj):
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)

    return paths

############################################
############################################

# def Path(obs, image_obs, acs, rewards, next_obs, terminals):
#     """
#         Take information (separate arrays) from a single rollout
#         and return it in a single dictionary
#     """
#     if image_obs != []:
#         image_obs = np.stack(image_obs, axis=0)
#     return {"observation" : np.array(obs, dtype=np.float32),
#             "image_obs" : np.array(image_obs, dtype=np.uint8),
#             "reward" : np.array(rewards, dtype=np.float32),
#             "action" : np.array(acs, dtype=np.float32),
#             "next_observation": np.array(next_obs, dtype=np.float32),
#             "terminal": np.array(terminals, dtype=np.float32)}

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    if isinstance(image_obs, list):
        image_obs = np.array(image_obs, dtype=np.uint8) if len(image_obs) > 0 else None

    path = {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": image_obs,  # None or np.ndarray
        "action": np.array(acs, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }
    return path

def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])
