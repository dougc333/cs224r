# cs224r 

 

There are 4 provided envs, ant, halfcheetah, hopper, walker2d. 

1) How to add data and policy for a Mujoco env using stable baselines 3. 

2) How to do quality eval of the provided policies and trajectories. The trajectories are sampled from the policies and you can create mp4 videos to see the behavior of the expert policies. This is important because BC can achieve performance at most equal to the expert. To surpass expert level you need a policy gradient. 

3) Behavior cloning and DAgger. BC and DAgger are supervided learning methods. They are not RL

4) PPO, can train PPO without a baseline. This isn' always possible. The robot control policies are small, 2 layer MLPs so random initialization is probably ok. For larger policies this wont work. This corresponds to the fine tuninng after Supervised training

5) Visualizations: use videos and tensorboard charts. Add React App when running outside Colab. Visualizations show the failure modes; if the robot is on its back flailing about. Hard to see from line charts. Reward hacking modes like dead leg also show up on videos. 

6) How to tune; sweep learning rate. 

7) Graph KL. 

8) show KL between 2 gaussian distributions is equal to subtraction of mean and variances. 

9) this isn't close to production. Colab workbooks have to be refactored to python libs which can be downloaded and run on GPU nodes. 







