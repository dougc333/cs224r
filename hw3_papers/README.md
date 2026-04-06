envpool parallel env for atari simulators
atari envs arent supported in mujoco_xla which is a physics engine first
 For example, DQN takes eight days and 200 million frames to train an agent to play a single
Atari game [22], while IMPALA [7] shortens this process to a few hours and Seed RL [6] continues
to push the boundary of training throughput.


We accelerate the environment execution by implementing a general
C++ threadpool-based executor engine that can run multiple environments in parallel.

 
RLLib: 

We chose to build RLlib on top of the Ray framework, which
allows Python tasks to be distributed across large clusters.
Ray’s distributed scheduler is a natural fit for the hierarchical
control model, as nested computation can be implemented
in Ray with no central task scheduling bottleneck.


