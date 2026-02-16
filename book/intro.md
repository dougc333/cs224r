# cs224r 

These homeworks represent long horizon dense reward RL tasks. They are a starting point for the more difficult long horizon sparse reward tasks in LLMs and Agents. 


There are 4 Mujoco envs with expert data or trajectories and policies. Aant, halfcheetah, hopper, and walker2d.

Some notes:  
What this checkpoint contributes:

  - dont run PPO through a notebook cell. Copy to a colab terminal and run using python cli. 
  - visualizing ant during training: SubprocVecEnv is a parallel vectored envs. Creates a separate env in a separate process. Set to 32/64 for 51G in high-ram colab CPU. GPU has no effect on performance.  Cant view all 64. Sample and view ant in a panel with graphs of kl, entropy and vertical height.
  -ipy widget to step through at each _on_rollout_end. This allows correlation between ant stability and graphs of kl, entropy, etc. Good for specification for react UI in distributed training.  
  - mujoco vertical height is used to tell if the ant is flipped over and should be terminated. This signal can be used as a reward signal to  a video_llm to detect failures visually.

  - testing of langrange contraints and pomdp.








