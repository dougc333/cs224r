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

The RayLIb paper is talking about a related but different kind of zero-copy. It is the same high-level goal—avoid copying large data—but Ray’s object-store approach is for shared immutable objects, while your design is a manual mutable shared-memory buffer with slot assignment and flags.

RLlib separates the implementation of algorithms into the
declaration of the algorithm-specific policy graph and the
choice of an algorithm-independent policy optimizer. The
policy optimizer is responsible for the performance-critical
tasks of distributed sampling, parameter updates, and managing replay buffers. To distribute the computation, the
optimizer operates over a set of policy evaluator replicas.
To complete the example, the developer chooses a policy
optimizer and creates it with references to existing evaluators. The async optimizer uses the evaluator actors to
compute gradients in parallel on many CPUs (Figure 4(c)).
Each optimizer.step() runs a round of remote tasks to
improve the model. Between steps, policy graph replicas
can be queried directly, e.g., to print out training statistics:


