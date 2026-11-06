---
title: "Reinforcement Learning: From Math to Code"
subtitle: "CS224R notes: MDPs, imitation learning, policy gradients, actor-critic, DDPG/TD3/SAC, DQN, CQL, IQL"
---

This document combines two sets of notes, *Actor Critic* (policy gradients, REINFORCE, actor-critic, the start of DQN) and *DQN, IQL, CQL* (MDPs, Bellman equations, DQN), and adds two CS224R lectures they didn't cover: imitation learning (Lecture 2) and off-policy actor-critic (Lecture 5). Sections that were missing or unfinished in the originals are marked **[new]**: computing V(s), imitation learning, the advantage actor-critic update, DDPG/TD3/SAC, the rest of the DQN worked example, DQN with a neural network, CQL, and IQL.

Notation used throughout: $s$ is a state, $a$ an action, $r$ a reward, $\gamma$ the discount factor, $\pi_\theta$ a policy with parameters $\theta$, and $G_t$ the return from time $t$.

# Part 1. MDP foundations

## 1.1 The Markov property, MDPs and POMDPs

All Q-learning methods start from the optimal Bellman equation. These equations define the optimal value functions of an MDP. The MDP assumption is the Markov property: the future depends only on the current state and action. This applies to the environment. A point of confusion is that the environment can be Markov while the observation is not. An observation may be only partial, as in a POMDP, a partially observable MDP.

**Example: a web browser.** The full browser state includes everything that defines the current state of a modern browser:

- the current DOM tree
- element values
- scroll position
- open tabs
- the history stack
- cookies and session state
- network-loaded data
- hidden menus and modals
- JavaScript app state
- cursor and focus state

The observation is partial. A human uses only the screenshot, while a computer agent has to filter the relevant properties, such as:

- the screenshot
- the accessibility tree
- the visible text
- the current URL

That observation can miss important state. For example:

- a menu may be off-screen
- a form field may have hidden validation state
- a page may have data in collapsed sections
- two tabs may look the same but have different history or context
- a button may be disabled because of earlier unseen actions
- async loading may still be in progress

So the observation $o_t$ does not fully determine the future:

$$P(o_{t+1}, r_t \mid o_t, a_t, \text{history}) \neq P(o_{t+1}, r_t \mid o_t, a_t)$$

**Example: Atari.** Using the same thought pattern, is an Atari environment an MDP? The complete Atari simulator for Pong contains the screen, the game state (ball velocity, the score, and the position of the paddles). That is Markov.

A single screenshot is not Markov: an agent cannot infer the velocity of the ball from one frame. Stacking a sequence of 4 frames into a single observation lets the agent infer the velocity of the ball, so both the environment and the observation are Markov. This is one of the reasons the Atari benchmarks can exceed human performance if tuned correctly.

**What is an episode in Pong?** Once the ball is launched in a new game, there is a fixed number of 4-frame states before the ball is either hit or missed by the paddle. A miss ends the point, giving a reward and a termination condition. A hit sends the ball back with a reward of 0, and the game continues to the next paddle hit or miss. This continues until someone misses.

Colab cell that records Pong frames and creates an MP4:

```python
# Colab cell: record Pong frames and create MP4 at the end

!pip -q install "gymnasium[atari]" ale-py matplotlib pillow imageio imageio-ffmpeg

import os
import gymnasium as gym
import ale_py
import numpy as np
from PIL import Image
import imageio
from IPython.display import Video, display

gym.register_envs(ale_py)

out_dir = "/content/pong_debug_frames"
os.makedirs(out_dir, exist_ok=True)

mp4_path = "/content/pong_noop_debug.mp4"

env = gym.make(
    "ALE/Pong-v5",
    render_mode="rgb_array",
    frameskip=1,
    repeat_action_probability=0.0,
)

obs, info = env.reset(seed=42)


# FIRE once to start the game
FIRE_ACTION = 1
obs, reward, terminated, truncated, info = env.step(FIRE_ACTION)
print("After FIRE:", reward, terminated, truncated)

if terminated or truncated:
    obs, info = env.reset(seed=42)

frames = []

for t in range(60):
    # NOOP so paddle does not move
    obs, reward, terminated, truncated, info = env.step(0)
    frame = obs.copy()
    frames.append(frame)

    png_path = os.path.join(out_dir, f"frame_{t:03d}.png")
    Image.fromarray(frame).save(png_path)

env.close()

print(f"Saved {len(frames)} PNG frames to {out_dir}")

# Write MP4
with imageio.get_writer(mp4_path, fps=15) as writer:
    for frame in frames:
        writer.append_data(frame)

print("Saved MP4:", mp4_path)

# Display inline in Colab
display(Video(mp4_path, embed=True, width=640))
```

## 1.2 MDP setup

An MDP has:

- states $s \in \mathcal{S}$
- actions $a \in \mathcal{A}$
- a transition and reward law $P(s', r \mid s, a)$
- a discount $\gamma \in [0, 1)$

The return from time $t$ is

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

For a policy $\pi$, the state value is

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$$

and the action value **[new]** is the same expectation, also conditioned on the first action:

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a], \qquad V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}[Q^\pi(s, a)]$$

## 1.3 The policy is a distribution

There are two choices: a continuous distribution or a discrete distribution. A continuous distribution like a Gaussian is specified by a mean and a variance; these two real numbers uniquely define the Gaussian. A discrete distribution is represented as a frequency histogram or a probability vector. The probability vector $[0.2, 0.4, 0.4]$ represents three values with probabilities 0.2, 0.4 and 0.4. The convention is to use the category index $[0, 1, 2]$: there is a 20% probability of drawing a 0, 40% of drawing a 1, and 40% of drawing a 2.

PyTorch code for a random draw from a probability vector:

```python
import torch
from torch.distributions import Categorical

probs = torch.tensor([0.2, 0.5, 0.3])
dist = Categorical(probs=probs)

sample = dist.sample()
print("sample:", sample.item())
print("log_prob(1):", dist.log_prob(torch.tensor(1)))
print("entropy:", dist.entropy())
```

One run printed `sample: 1`. A single draw can be any index. If you rerun the program $N$ times, the frequencies approach the probability vector, and $N$ has to be large enough for the law of large numbers to apply. This matters when debugging and looking at single sampled results from code.

## 1.4 The Bellman equations

There are two optimal Bellman equations:

$$V^*(s) = \max_a \mathbb{E}\big[r + \gamma V^*(s') \mid s, a\big]$$

and

$$Q^*(s, a) = \mathbb{E}\big[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\big]$$

where the next state $s' \sim P(s' \mid s, a)$, the reward $r \sim R(r \mid s, a, s')$, $s$ is the current state, $a$ the current action, $r$ the immediate reward, $s'$ the next state, $\gamma$ the discount factor, and $a'$ ranges over the next possible actions.

## 1.5 How to calculate V(s) **[new]**

For a fixed policy $\pi$, $V^\pi$ satisfies the Bellman expectation equation (no max, because the policy picks the action):

$$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}\,\mathbb{E}_{s', r}\big[r + \gamma V^\pi(s')\big]$$

There are three standard ways to compute it.

**1. Dynamic programming (known model, small tables).** Start from $V_0 = 0$ and repeat the backup until it stops changing:

$$V_{k+1}(s) = \sum_a \pi(a \mid s) \sum_{s', r} P(s', r \mid s, a)\big[r + \gamma V_k(s')\big]$$

**2. Monte Carlo (no model).** Run full episodes and average the observed returns. This is unbiased, but has high variance:

$$V(s) \approx \frac{1}{N}\sum_{i=1}^{N} G_t^{(i)} \quad \text{for every visit } S_t = s$$

**3. Temporal difference, TD(0) (no model, one step).** Bootstrap from the current estimate of the next state. This is lower variance but biased:

$$V(s) \leftarrow V(s) + \alpha\big(r + \gamma V(s') - V(s)\big)$$

With a neural network $V_\phi$, each of these becomes regression onto a target: the target is $G_t$ for Monte Carlo (what REINFORCE with a baseline uses below) or $r + \gamma V_\phi(s')$ for TD (what actor-critic uses).

```python
import numpy as np

def mc_values(episodes, gamma, n_states):
    """episodes: list of [(s, r), ...] for one policy."""
    total = np.zeros(n_states); count = np.zeros(n_states)
    for ep in episodes:
        G = 0.0
        for s, r in reversed(ep):
            G = r + gamma * G
            total[s] += G; count[s] += 1
    return total / np.maximum(count, 1)

def td0_update(V, s, r, s_next, done, alpha, gamma):
    target = r + gamma * V[s_next] * (1 - done)
    V[s] += alpha * (target - V[s])
    return V
```

# Part 2. Imitation learning **[new]**

Imitation learning (IL) learns a policy from expert demonstrations instead of from a reward. No reward function is needed, but IL needs a lot of data. It is the supervised-learning stage that usually comes before RL fine-tuning, the same way an LLM is trained with SFT before RL. These notes follow CS224R Lecture 2, your lecture notes, the DAgger paper notes, and HW1 (`hw1/notebooks/bc_daggar.ipynb`).

## 2.1 Setup and notation

A set of expert trajectories is collected as demonstrations:

$$\mathcal{D} := \{\tau_1, \dots, \tau_N\}, \qquad \tau_i = (s_1, a_1, s_2, a_2, \dots, s_T)$$

The expert follows an unknown policy $\pi^*$. In HW1 the expert is the `GaussianPolicy` stored in `Ant.pkl`, a 2-layer tanh MLP with observation normalization and a learned `log_std`. The demonstrations are in `expert_data_Ant-v4.pkl`: a list of trajectory dicts with `observation` (111-dimensional with contact forces), `action` (8-dimensional), `reward`, `next_observation` and `terminal`.

## 2.2 Version 0: behavior cloning with an L2 loss

The simplest version of IL is to train a policy $\hat a = \pi_\theta(s)$ that minimizes the error between its action and the demonstrated action for each state, over all the trajectories in the demonstrations:

$$\min_\theta \sum_{(s, a) \in \mathcal{D}} \big\| \pi_\theta(s) - a \big\|^2$$

The policy is an MLP and the loss is MSE, so this is plain supervised learning. Deploy the policy and it should mimic the expert. It is a deterministic policy with no exploration.

## 2.3 Behavior cloning as maximum likelihood

The general form treats the policy as a distribution and maximizes the likelihood of the expert's actions:

$$\theta^* = \arg\max_\theta\ \mathbb{E}_{(s, a) \sim \mathcal{D}}\big[\log \pi_\theta(a \mid s)\big]$$

With a Gaussian policy whose variance is fixed, this is exactly version 0. Using the Gaussian log-probability from Part 3:

$$\log \pi_\theta(a \mid s) = -\frac{\|a - \mu_\theta(s)\|^2}{2\sigma^2} + \text{const}$$

so maximizing the log-likelihood is minimizing $\|a - \mu_\theta(s)\|^2$. The L2 loss **is** a Gaussian assumption. Learning $\sigma_\theta(s)$ as well gives the full negative log-likelihood loss:

$$L(\theta) = -\frac{1}{|\mathcal{D}|}\sum_{(s,a) \in \mathcal{D}} \log \mathcal{N}\big(a;\ \mu_\theta(s),\ \sigma_\theta^2(s)\big)$$

This is the same `dist.log_prob(action).sum(-1)` used in the policy gradient, but here the actions come from the expert and there is no return weighting. Behavior cloning is policy gradient with every weight $\hat A_t$ set to 1 on expert data.

## 2.4 What goes wrong with version 0: multimodal actions

The expert may not be deterministic. Suppose some drivers in the dataset go straight around an obstacle and some go right. The action distribution for that state has two modes. An L2 loss fit to two Gaussians returns the **mean** of the two, which is a low-probability action. It isn't the center of the road; it is the center of the actions between the two distributions, which can drive straight into the obstacle.

Worked example: half the demonstrations steer $a = -1$ (left), half steer $a = +1$ (right).

$$\arg\min_{\hat a}\ \tfrac{1}{2}(\hat a - (-1))^2 + \tfrac{1}{2}(\hat a - 1)^2 = 0$$

The L2 policy goes straight, an action no expert ever took.

**Making the network bigger doesn't fix this.** There are two separate kinds of expressive power:

1. **Network expressivity:** how many functions from $s$ to output parameters the network can represent. A bigger network increases this.
2. **Distribution expressivity:** how many different action distributions those output parameters can describe. A Gaussian head can only ever describe one mode, however big the network is, so a larger network trained with L2 still outputs the mean.

Distribution expressivity increases along Gaussian → mixture of Gaussians → discretized/autoregressive → diffusion.

## 2.5 Expressive policy distributions

**Mixture of Gaussians (MoG).** Output $K$ means, $K$ standard deviations and $K$ mixture weights:

$$\pi_\theta(a \mid s) = \sum_{k=1}^{K} w_k(s)\,\mathcal{N}\big(a;\ \mu_k(s),\ \sigma_k^2(s)\big), \qquad \sum_k w_k(s) = 1$$

Train it with the same negative log-likelihood. The policy can output a unimodal distribution in some states and a multimodal one in others. The limitation is that the number of components $K$ is a hyperparameter fixed ahead of time.

**Discretization and autoregressive models.** Turn the continuous problem into a discrete one: split each action dimension into $B$ bins, make the output a categorical (multinomial) distribution, and train with cross-entropy, exactly the categorical policy from Part 3:

$$L(\theta) = -\sum_{(s,a) \in \mathcal{D}}\sum_{d=1}^{D}\log p_\theta\big(\text{bin}(a_d) \mid s\big)$$

There are two reasons to discretize. First, the number of bins can be large, so the histogram can model complex distributions that need more expressiveness than a MoG. Second, some problems are already discrete in their states and actions. Predicting the dimensions one at a time, each conditioned on the previous ones, $p(a \mid s) = \prod_d p(a_d \mid s, a_{<d})$, makes it autoregressive. This is how LLMs are used over actions: tokenize the actions and predict the next token. OpenVLA discretizes actions and decodes them autoregressively, and Waymo also uses discretized, autoregressive models.

**Diffusion.** Diffusion is a good choice for continuous actions. Instead of diffusing over images, the model learns to denoise actions conditioned on the observation, $p(a \mid s)$. It starts from noise $a^K \sim \mathcal{N}(0, I)$ and repeatedly removes predicted noise:

$$a^{k-1} = \frac{1}{\sqrt{\alpha_k}}\left(a^k - \frac{1 - \alpha_k}{\sqrt{1 - \bar\alpha_k}}\,\epsilon_\theta(a^k, k, s)\right) + \sigma_k z$$

trained with the simple noise-prediction loss $\big\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_k}\,a + \sqrt{1 - \bar\alpha_k}\,\epsilon,\ k,\ s)\big\|^2$. This is Diffusion Policy. In the lecture's results, GMM policies degraded more than diffusion policies, and tasks like hanging shirts with robot arms needed the more expressive model.

Diffusion and autoregressive models are very general ways to learn $p(\text{action} \mid \text{observation})$. When training any of these generative models we are increasing the probability of the expert's actions relative to others, which is different from version 0's regression to a single point. A good resource is [Berkeley CS294-158 Deep Unsupervised Learning](https://sites.google.com/view/berkeley-cs294-158-sp20/home).

## 2.6 Compounding errors and distribution shift

Behavior cloning can only do as well as the expert, never better, and it has a deeper problem: compounding errors.

The policy is trained on states from the expert's distribution, $s \sim d_{\pi^*}$, but at test time it visits states from its own distribution, $s \sim d_{\pi_\theta}$. A small mistake takes the agent to a state slightly off the expert's path. The expert never visited that state, so the policy has no data for it and makes a bigger mistake, which takes it further off. The car drifts off the road; the Ant flips over or ends up with a dead leg (see `hw1/videos/ant_flipover.mp4` and `ant_deadleg.mp4`).

The standard analysis: if the policy makes a mistake with probability $\epsilon$ on states from the expert's distribution, then over a horizon of $T$ steps

$$J(\pi^*) - J(\pi_\theta) = O(\epsilon T^2) \quad \text{(behavior cloning)}$$

because each mistake can put the agent into states where it keeps making mistakes for the rest of the episode. The error is quadratic in the horizon, not linear.

## 2.7 DAgger: Dataset Aggregation

DAgger fixes distribution shift by collecting expert labels **on the states the learned policy actually visits**. It is online: data is recollected as the algorithm runs.

1. Train $\pi_1$ on the expert demonstrations $\mathcal{D}$.
2. For $i = 1, 2, \dots, N$:
   a. Run $\pi_i$ in the environment to collect states $\{s\} \sim d_{\pi_i}$.
   b. Ask the expert to label each visited state: $a = \pi^*(s)$.
   c. Aggregate: $\mathcal{D} \leftarrow \mathcal{D} \cup \{(s, \pi^*(s))\}$.
   d. Train $\pi_{i+1}$ on all of $\mathcal{D}$.

The labels are the "corrective actions": what the expert would do from the off-road state the student drove into. The DAgger paper frames the goal as minimizing the **observed surrogate loss** under the policy's own induced state distribution:

$$\hat\pi = \arg\min_\pi\ \mathbb{E}_{s \sim d_\pi}\big[\ell(s, \pi)\big]$$

It is a surrogate loss because it is computed only on the states the policy visits, not on all possible states. The data is not i.i.d.: the policy determines the states, the states determine the loss, and the loss determines the policy. The objective is non-stationary, and the paper notes that this interaction makes the optimization non-convex even if $\ell(s, \cdot)$ is convex in $\pi$ for every state. DAgger handles this by aggregating all data from every iteration (a no-regret online learning approach), and its error grows linearly with the horizon:

$$J(\pi^*) - J(\pi_{\text{DAgger}}) = O(\epsilon T)$$

**Costs.** DAgger needs an expert that can be queried at any state. That is easy with a scripted or learned expert like the HW1 `Ant.pkl` policy, but hard with human experts, since it can be tricky to get the corrective actions. Running an untrained policy to collect data may also be unsafe on a real robot.

## 2.8 Code: behavior cloning with a Gaussian policy

Plain PyTorch, trained with negative log-likelihood. The data is the HW1 expert pickle flattened into $(s, a)$ pairs.

```python
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianMLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))   # state-independent std

    def forward(self, obs):
        h = self.net(obs)
        return Normal(self.mu(h), self.log_std.clamp(-5, 2).exp())

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        dist = self(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        a = dist.mean if deterministic else dist.sample()
        return a.squeeze(0).numpy()


def load_demos(path="cs224r/expert_data/expert_data_Ant-v4.pkl"):
    with open(path, "rb") as f:
        trajs = pickle.load(f)                      # list of trajectory dicts
    obs = np.concatenate([t["observation"] for t in trajs]).astype(np.float32)
    acts = np.concatenate([t["action"] for t in trajs]).astype(np.float32)
    return obs, acts


def train_bc(policy, obs, acts, epochs=50, batch_size=256, lr=1e-3):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    obs_t, acts_t = torch.as_tensor(obs), torch.as_tensor(acts)
    n = len(obs_t)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            dist = policy(obs_t[idx])
            loss = -dist.log_prob(acts_t[idx]).sum(dim=-1).mean()   # NLL; sum over action dims
            opt.zero_grad()
            loss.backward()
            opt.step()
    return policy
```

Replacing the loss with `((dist.mean - acts_t[idx]) ** 2).sum(-1).mean()` gives version 0 (L2). In HW1 the `imitation` library's `BC` trainer with a 128×128 tanh `ActorCriticPolicy` reached a return of about 4497 from only 2 demonstrations, close to the expert's 4553.

## 2.9 Code: a mixture-of-Gaussians policy head

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal


class MoGPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, n_components=5, hidden=256):
        super().__init__()
        self.K, self.D = n_components, act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.logits = nn.Linear(hidden, n_components)              # mixture weights w_k(s)
        self.mu = nn.Linear(hidden, n_components * act_dim)        # K means
        self.log_std = nn.Linear(hidden, n_components * act_dim)   # K stds

    def forward(self, obs):
        h = self.net(obs)
        B = obs.shape[0]
        mu = self.mu(h).view(B, self.K, self.D)
        std = self.log_std(h).view(B, self.K, self.D).clamp(-5, 2).exp()
        components = Independent(Normal(mu, std), 1)               # each is a D-dim Gaussian
        return MixtureSameFamily(Categorical(logits=self.logits(h)), components)


# Training is the same NLL as before; log_prob already sums over action dims:
# loss = -policy(obs).log_prob(acts).mean()
```

On the left/right example above, the MoG policy puts one component near $-1$ and one near $+1$, and sampling from it picks one of the two real behaviors instead of their average.

## 2.10 Code: DAgger

`expert_fn(obs)` is anything that returns the expert action for a state, such as `AntExpertPolicy().act` from the HW1 notebook, which reconstructs the expert MLP from `Ant.pkl`.

```python
import gymnasium as gym
import numpy as np


def rollout_states(env, policy, n_episodes, max_steps=1000, seed=0):
    """Run the student and record the states it visits."""
    states, returns = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        G = 0.0
        for _ in range(max_steps):
            states.append(obs)
            obs, r, terminated, truncated, _ = env.step(policy.act(obs))
            G += float(r)
            if terminated or truncated:
                break
        returns.append(G)
    return np.asarray(states, dtype=np.float32), float(np.mean(returns))


def dagger(env, policy, expert_fn, demo_obs, demo_acts,
           n_iters=10, episodes_per_iter=2, epochs_per_iter=20):
    # 1. start from behavior cloning on the expert demos
    data_obs, data_acts = demo_obs.copy(), demo_acts.copy()
    train_bc(policy, data_obs, data_acts, epochs=epochs_per_iter)

    for i in range(n_iters):
        # 2a. run the current student: states come from d_{pi_i}, not from the expert
        new_obs, student_return = rollout_states(env, policy, episodes_per_iter, seed=1000 * i)

        # 2b. expert labels the states the student actually visited (corrective actions)
        new_acts = np.stack([expert_fn(s) for s in new_obs]).astype(np.float32)

        # 2c. aggregate all data so far
        data_obs = np.concatenate([data_obs, new_obs])
        data_acts = np.concatenate([data_acts, new_acts])

        # 2d. retrain on the aggregated dataset
        train_bc(policy, data_obs, data_acts, epochs=epochs_per_iter)
        print(f"iter {i}: student return {student_return:8.1f}, dataset size {len(data_obs)}")
    return policy


# Usage for HW1 Ant:
# env = gym.make("Ant-v4", use_contact_forces=True)
# obs, acts = load_demos()
# student = GaussianMLPPolicy(obs.shape[1], acts.shape[1])
# expert = AntExpertPolicy()
# dagger(env, student, lambda s: expert.act(s, deterministic=True), obs, acts)
```

Your HW1 videos of the three policies are in `hw1/videos_expert/`, `hw1/videos_bc/` and `hw1/videos_dagger/`, with a side-by-side in `hw1/videos_compare/`.

## 2.11 Summary

| Method | Policy distribution | Loss | Handles multimodal actions | Handles distribution shift |
|---|---|---|---|---|
| BC, version 0 | deterministic / fixed-variance Gaussian | L2 (MSE) | No, averages the modes | No, error $O(\epsilon T^2)$ |
| BC, Gaussian NLL | Gaussian with learned $\sigma$ | NLL | No | No |
| BC, MoG | mixture of $K$ Gaussians | NLL | Yes, up to $K$ modes | No |
| BC, discretized / autoregressive | categorical per dimension or token | cross-entropy | Yes | No |
| BC, diffusion | implicit, via denoising | noise-prediction MSE | Yes | No |
| DAgger | any of the above | same, on aggregated data | Depends on the head | Yes, error $O(\epsilon T)$ |

# Part 3. Policy gradients

## 3.1 The policy gradient

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat A_t\right]$$

$\hat A_t$ is the advantage estimate and is a function of $t$. There are no $\theta$ parameters in it, so it is treated as a constant.

## 3.2 From math to code

The notation $\pi_\theta(a \mid s)$ refers to a distribution. We start with two different distributions: a Gaussian and a categorical distribution.

## 3.3 Gaussian policy for control

We assume a Gaussian distribution for the policy:

$$\pi_\theta(a \mid s) = \mathcal{N}\big(a;\ \mu_\theta(s),\ \sigma_\theta^2(s)\big)$$

A Gaussian is defined by a mean and a variance:

$$\pi_\theta(a \mid s) = \frac{1}{\sqrt{2\pi}\,\sigma_\theta(s)}\exp\!\left(-\frac{(a - \mu_\theta(s))^2}{2\sigma_\theta^2(s)}\right)$$

Take the log of both sides to match the policy gradient. Convert the log of a product to a sum using $\log ab = \log a + \log b$, and use $\log(\exp(x)) = x$:

$$\log \pi_\theta(a \mid s) = -\frac{1}{2}\log(2\pi) - \log \sigma_\theta(s) - \frac{(a - \mu_\theta(s))^2}{2\sigma_\theta^2(s)}$$

This is a function of the state, so for each state there is a separate mean and variance.

We have two terms left, a $\log \pi_\theta$ term and an expectation. Starting from the objective

$$J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]$$

take the gradient of both sides:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t\big]$$

Replace the expectation with two summations, a Monte Carlo estimate over time $T$ with batch size $N$:

$$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t})\,\hat Q(s_{i,t}, a_{i,t})$$

For the policy gradient, $\hat Q(s_t, a_t)$ is $G_t$. We don't need a neural network to estimate the rewards; we can calculate them exactly from the environment. $T$ is the number of time steps, or the horizon of the environment.

$$G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k+1}$$

$\gamma$ guarantees the infinite series converges. Replace the infinite horizon with a finite $T$:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots + \gamma^{T-t-1} r_T$$

### Calculating log probs of a Gaussian

```python
from torch.distributions import Normal
dist = Normal(mean, std)
logp = dist.log_prob(action)
```

Start with the normal distribution:

$$p(a) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(a - \mu)^2}{2\sigma^2}\right)$$

We want $\log p(a)$, so take the log of both sides and use the product rule $\log(xy) = \log x + \log y$:

$$\log p(a) = \log\frac{1}{\sigma\sqrt{2\pi}} + \log \exp\left(-\frac{(a - \mu)^2}{2\sigma^2}\right)$$

Using $\log e^x = x$, the second term becomes $-\dfrac{(a-\mu)^2}{2\sigma^2}$. For the first term, rewrite $\dfrac{1}{\sigma\sqrt{2\pi}} = (\sigma\sqrt{2\pi})^{-1}$, so

$$\log\frac{1}{\sigma\sqrt{2\pi}} = -\log(\sigma\sqrt{2\pi}) = -\log\sigma - \log\sqrt{2\pi} = -\log\sigma - \frac{1}{2}\log(2\pi)$$

Combine both parts:

$$\log p(a) = -\frac{1}{2}\log(2\pi) - \log\sigma - \frac{(a-\mu)^2}{2\sigma^2}$$

How many action dimensions are there in Ant?

```python
import gymnasium as gym
env = gym.make("Ant-v4")
print(env.action_space)
print(env.action_space.shape)
```

There are 8 action dimensions:

```python
mean = actor(obs)          # shape: [8]
std = torch.exp(log_std)   # shape: [8]
dist = Normal(mean, std)
action = dist.sample()     # shape: [8]
log_prob = dist.log_prob(action).sum(dim=-1)
```

Sum over the 8 action dimensions to get a single scalar, because the dimensions are independent and the joint log-probability is the sum of the per-dimension log-probabilities.

### Calculating the gradient of the log probs

$$\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,G_t$$

This means: change the policy parameters so that good actions become more likely and bad actions become less likely. In PyTorch:

```python
loss = -log_prob * return_t
loss.backward()
```

`loss.backward()` takes the gradient of `log_prob * return_t`. The code now looks like:

```python
mean = actor(obs)              # depends on θ
std = torch.exp(log_std)       # may depend on θ
dist = Normal(mean, std)
action = dist.sample()
log_prob = dist.log_prob(action).sum(dim=-1)

loss = -log_prob * G_t
loss.backward()
optimizer.step()
```

- If $G_t > 0$, minimizing $-\log\pi \cdot G_t$ increases the log-probability, so the action becomes more likely.
- If $G_t < 0$, minimizing $-\log\pi \cdot G_t$ decreases the log-probability, so the action becomes less likely.

Important detail: detach the action before sending it to the environment.

```python
env_action = action.detach().cpu().numpy()
```

## 3.4 Categorical policy

For a simple gridworld MDP, the policy is a categorical distribution. The mental model is a coin or die with one trial. Assume $K$ possible actions per state, $a \in \{1, \dots, K\}$, with probabilities $p = [p_1, p_2, \dots, p_K]$ such that

$$p_i \ge 0, \qquad \sum_{i=1}^{K} p_i = 1$$

These two rules are the axioms of probability: probabilities are never negative and the total probability sums to 1.

The categorical distribution is defined by a PMF, the probability mass function, which is the amount of probability allocated to each category:

$$P(y) = \prod_{i=1}^{K} p_i^{y_i}$$

This is an unusual product notation: $p$ is the probability vector that sums to 1, and $y$ is a one-hot vector. For example,

$$p = [0.1, 0.2, 0.6, 0.1], \quad y = [0, 0, 1, 0]$$

$$P(y) = 0.1^0 \cdot 0.2^0 \cdot 0.6^1 \cdot 0.1^0 = 0.6$$

Taking the log of both sides turns the product into a sum and removes the exponent:

$$\log P(y) = \log\Big[\prod_{i=1}^{K} p_i^{y_i}\Big] = \sum_{i=1}^{K} y_i \log p_i$$

To calculate the gradient we need a representation we can differentiate, so we use the softmax definition of categorical probability:

$$p_i = \frac{e^{z_i}}{\sum_{m=1}^{K} e^{z_m}}$$

A softmax turns $K$ logits $z = [z_1, \dots, z_K]$ into a PMF. The logits don't have to be normalized; any real numbers will do. For $z = [1, 2, 0]$:

$$e^z = [e^1, e^2, e^0] \approx [2.72, 7.39, 1], \qquad \text{sum} = 11.11$$

$$p = [2.72/11.11,\ 7.39/11.11,\ 1/11.11] \approx [0.245, 0.665, 0.090]$$

Now take the log. $\log$ and $e$ cancel:

$$\log p_i = \log e^{z_i} - \log\Big(\sum_{m=1}^{K} e^{z_m}\Big) = z_i - \log\Big(\sum_{m=1}^{K} e^{z_m}\Big)$$

Differentiate with respect to $z_j$:

$$\frac{\partial \log p_i}{\partial z_j} = \frac{\partial z_i}{\partial z_j} - \frac{\partial}{\partial z_j}\log\Big(\sum_{m=1}^{K} e^{z_m}\Big)$$

The first term is $\dfrac{\partial z_i}{\partial z_j} = \mathbf{1}[j = i]$. For the second term, use the chain rule:

$$\frac{\partial}{\partial z_j}\log\Big(\sum_m e^{z_m}\Big) = \frac{1}{\sum_m e^{z_m}}\cdot\frac{\partial}{\partial z_j}\Big(\sum_m e^{z_m}\Big)$$

Only the $m = j$ term depends on $z_j$, so the derivative of the sum is $e^{z_j}$, and

$$\frac{\partial}{\partial z_j}\log\Big(\sum_m e^{z_m}\Big) = \frac{e^{z_j}}{\sum_m e^{z_m}} = p_j$$

Putting it together:

$$\frac{\partial \log p_i}{\partial z_j} = \mathbf{1}[j = i] - p_j$$

If $j = i$ this is $1 - p_i$; otherwise it is $-p_j$. If $y$ is the one-hot vector for the chosen class $i$, then

$$\nabla_z \log \pi(a = i) = y - p$$

Suppose the logits produce $p = [0.1, 0.2, 0.6, 0.1]$ and the sampled action is class 3, with one-hot $y = [0, 0, 1, 0]$:

$$\nabla_z \log p_3 = [0, 0, 1, 0] - [0.1, 0.2, 0.6, 0.1] = [-0.1, -0.2, 0.4, -0.1]$$

The gradient says to increase class 3 by 0.4 and decrease the others by 0.1, 0.2 and 0.1. The gradients sum to 0.

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)

# Example
policy = CategoricalPolicy(obs_dim=4, act_dim=4)
obs = torch.randn(3, 4)   # batch of 3 states

dist = policy(obs)
actions = dist.sample()
log_probs = dist.log_prob(actions)

print("actions:", actions)
print("log_probs:", log_probs)
print("probs:", dist.probs)
```

Sample output:

```text
actions: tensor([0, 0, 3])
log_probs: tensor([-1.4774, -1.5225, -1.3155], grad_fn=<SqueezeBackward1>)
probs: tensor([[0.2282, 0.2188, 0.2772, 0.2757],
        [0.2182, 0.2304, 0.2786, 0.2729],
        [0.2534, 0.2232, 0.2551, 0.2683]], grad_fn=<SoftmaxBackward0>)
```

**Probabilities.** Row 1 holds the action probabilities for state 1, row 2 for state 2, row 3 for state 3. Row 1 says action 0 is 22.82%, action 1 is 21.88%, action 2 is 27.72% and action 3 is 27.57%. Sampling picked action 0. This isn't argmax, which would be action 2; it is a sample, and at 22.82% action 0 is a reasonable draw.

**Actions.** The number of actions is set by `act_dim=4`. `actions: tensor([0, 0, 3])` says the first sample chose class 0, the second class 0, and the third class 3. `Categorical.sample()` returns an index into the distribution. With `actions_list = ["left", "right", "up", "down"]`, 0 means left and 3 means down. The list itself is never specified, only the number of actions.

**Log probabilities.** `log_probs: tensor([-1.4774, -1.5225, -1.3155])` are the log-probabilities of the sampled actions: row 1, action 0, probability 0.2282; row 2, action 0, probability 0.2182; row 3, action 3, probability 0.2683. Check: $\log(0.2282) \approx -1.4774$, $\log(0.2182) \approx -1.5225$, $\log(0.2683) \approx -1.3155$.

## 3.5 Derivation: REINFORCE

REINFORCE is the vanilla policy gradient, here first without baseline variance reduction.

**1. Objective.** Let a trajectory be

$$\tau = (s_0, a_0, s_1, a_1, \dots, s_T, a_T)$$

and define its discounted return as

$$R(\tau) = \sum_{t=0}^{T}\gamma^t r_t$$

The policy objective is

$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}[R(\tau)]$$

For discrete trajectories, write this as

$$J(\theta) = \sum_\tau p_\theta(\tau)\,R(\tau)$$

**2. Differentiate the objective.** Note that this is different from supervised deep learning: we aren't setting the derivative to 0, we are following it uphill.

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_\tau p_\theta(\tau)\,R(\tau) = \sum_\tau \nabla_\theta p_\theta(\tau)\,R(\tau)$$

$R(\tau)$ does not depend directly on $\theta$; it comes from the environment as a sum of rewards from 0 to $T$. Use the log-derivative trick:

$$\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)$$

so

$$\nabla_\theta J(\theta) = \sum_\tau p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)\,R(\tau) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\big[\nabla_\theta \log p_\theta(\tau)\,R(\tau)\big]$$

**3. Expand the trajectory probability.**

$$p_\theta(\tau) = p(s_0)\prod_{t=0}^{T}\pi_\theta(a_t \mid s_t)\,p(s_{t+1} \mid s_t, a_t)$$

Take the log:

$$\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T}\log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T}\log p(s_{t+1} \mid s_t, a_t)$$

Differentiate with respect to $\theta$. The environment dynamics and initial-state distribution do not depend on $\theta$, so they drop out:

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

Substitute back:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\Big(\sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t \mid s_t)\Big) R(\tau)\right]$$

**4. Use the return-to-go instead of the full trajectory return.** An action can't affect rewards that came before it, so replace $R(\tau)$ with

$$G_t = \sum_{k=t}^{T}\gamma^{k-t} r_k$$

The REINFORCE gradient becomes

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,G_t\right]$$

**5. Monte Carlo estimator.** Sample $N$ trajectories $\tau^{(1)}, \tau^{(2)}, \dots, \tau^{(N)}$ from the current policy. For each, compute the Monte Carlo return-to-go

$$G_t^{(i)} = \sum_{k=t}^{T_i}\gamma^{k-t} r_k^{(i)}$$

and replace the expectation with a sample average:

$$\hat\nabla_\theta J(\theta) = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T_i}\nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})\,G_t^{(i)}$$

This is the Monte Carlo REINFORCE estimator. It is unbiased, but often high variance.

**6. Single-trajectory version.** With one sampled episode, there is only one summation, from $t = 0$ to $T$:

$$\hat\nabla_\theta J(\theta) = \sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,G_t$$

This is often how the simplest REINFORCE algorithm is presented.

**7. The loss used in practice.** Deep learning optimizers minimize losses, so define

$$L(\theta) = -\sum_{t=0}^{T}\log \pi_\theta(a_t \mid s_t)\,G_t$$

or the same quantity averaged over sampled trajectories:

$$L(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T_i}\log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})\,G_t^{(i)}$$

Minimizing $L$ is equivalent to maximizing $J$.

**8. With a baseline.** To reduce variance, subtract a baseline $b(s_t)$. Because $\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi_\theta(a \mid s)] = 0$, any baseline that depends only on the state leaves the gradient unbiased:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\big(G_t - b(s_t)\big)\right]$$

A common choice is the state value, $b(s_t) = V^\pi(s_t)$. Then the advantage is

$$\hat A_t = G_t - V_\phi(s_t)$$

and the update becomes

$$\nabla_\theta J(\theta) \approx \sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat A_t$$

## 3.6 REINFORCE with a learned baseline

What changed from plain REINFORCE:

- a `ValueNetwork` was added
- advantages are computed as $\hat A_t = G_t - V_\phi(s_t)$
- the actor loss becomes $L_\pi = -\sum_t \log \pi_\theta(a_t \mid s_t)\,\hat A_t$
- the critic (value) loss is MSE: $L_V = \sum_t \big(V_\phi(s_t) - G_t\big)^2$

Full code for Pendulum-v1 with a Gaussian policy, a value network, and MP4 evaluation recording:

```python
import os
import glob
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from gymnasium.wrappers import RecordVideo

try:
    from IPython.display import Video, display
    IN_IPYTHON = True
except Exception:
    IN_IPYTHON = False


# ============================================================
# REINFORCE with learned baseline for Pendulum-v1
# Gaussian policy + value network + MP4 eval recording
# ============================================================

@dataclass
class Config:
    env_name: str = "Pendulum-v1"
    seed: int = 0
    hidden_dim: int = 128
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    gamma: float = 0.99
    num_episodes: int = 1000
    max_steps_per_episode: int = 200
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    normalize_advantages: bool = True

    eval_every: int = 20
    eval_episodes: int = 5

    video_dir: str = "./pendulum_pg_videos"
    video_name_prefix: str = "pendulum_eval"


CFG = Config()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_limit: float,
        hidden_dim: int = 128,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.act_limit = act_limit
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mu = self.act_limit * torch.tanh(self.mu_layer(h))
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        return dist, mu, std

    def sample_action(self, obs: np.ndarray, device: str):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist, mu, std = self.forward(obs_t)

        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        action = raw_action.squeeze(0).detach().cpu().numpy()
        action = np.clip(action, -self.act_limit, self.act_limit)
        return action, log_prob.squeeze(0)

    @torch.no_grad()
    def mean_action(self, obs: np.ndarray, device: str):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist, mu, std = self.forward(obs_t)
        action = mu.squeeze(0).cpu().numpy()
        return np.clip(action, -self.act_limit, self.act_limit)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def compute_returns(rewards, gamma: float):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns


@torch.no_grad()
def evaluate_policy(env_name: str, policy: GaussianPolicy, device: str, episodes: int = 5):
    env = gym.make(env_name)
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=CFG.seed + 1000 + ep)
        done = False
        ep_ret = 0.0
        steps = 0

        while not done and steps < CFG.max_steps_per_episode:
            action = policy.mean_action(obs, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward
            steps += 1

        returns.append(ep_ret)

    env.close()
    return float(np.mean(returns))


@torch.no_grad()
def record_eval_video(
    policy: GaussianPolicy,
    env_name: str,
    device: str,
    video_dir: str,
    episode_name: str,
    seed: int,
    max_steps: int,
):
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True,
        name_prefix=episode_name,
    )

    obs, _ = env.reset(seed=seed)
    done = False
    ep_return = 0.0
    steps = 0

    while not done and steps < max_steps:
        action = policy.mean_action(obs, device)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_return += reward
        steps += 1

    env.close()

    mp4_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    latest_video = mp4_files[-1] if mp4_files else None

    print(f"Recorded eval return: {ep_return:.2f}")
    print(f"Saved video: {latest_video}")
    return latest_video


def train():
    set_seed(CFG.seed)

    env = gym.make(CFG.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    policy = GaussianPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        hidden_dim=CFG.hidden_dim,
        log_std_min=CFG.log_std_min,
        log_std_max=CFG.log_std_max,
    ).to(CFG.device)

    value_net = ValueNetwork(
        obs_dim=obs_dim,
        hidden_dim=CFG.hidden_dim,
    ).to(CFG.device)

    policy_optimizer = optim.Adam(policy.parameters(), lr=CFG.policy_lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=CFG.value_lr)

    episode_returns = []

    for episode in range(1, CFG.num_episodes + 1):
        obs, _ = env.reset(seed=CFG.seed + episode)

        observations = []
        log_probs = []
        rewards = []

        done = False
        steps = 0

        while not done and steps < CFG.max_steps_per_episode:
            action, log_prob = policy.sample_action(obs, CFG.device)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            observations.append(obs.copy())
            log_probs.append(log_prob)
            rewards.append(float(reward))

            obs = next_obs
            steps += 1

        # Returns G_t
        returns = compute_returns(rewards, CFG.gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=CFG.device)

        # States
        obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32, device=CFG.device)

        # Baseline V(s_t)
        values = value_net(obs_tensor)

        # Advantage = return - baseline
        advantages = returns - values.detach()

        if CFG.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = torch.stack(log_probs)

        # -------------------------
        # Policy update
        # -------------------------
        policy_loss = -(log_probs * advantages).sum()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # -------------------------
        # Value / baseline update
        # -------------------------
        value_loss = nn.functional.mse_loss(values, returns)

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        ep_return = float(np.sum(rewards))
        episode_returns.append(ep_return)

        if episode % CFG.eval_every == 0:
            eval_return = evaluate_policy(
                CFG.env_name,
                policy,
                CFG.device,
                episodes=CFG.eval_episodes,
            )
            recent_train = float(np.mean(episode_returns[-CFG.eval_every:]))
            print(
                f"episode={episode:4d} "
                f"train_return={recent_train:9.2f} "
                f"eval_return={eval_return:9.2f} "
                f"policy_loss={policy_loss.item():9.2f} "
                f"value_loss={value_loss.item():9.2f}"
            )

    env.close()
    return policy, value_net


def main():
    trained_policy, trained_value = train()

    video_path = record_eval_video(
        policy=trained_policy,
        env_name=CFG.env_name,
        device=CFG.device,
        video_dir=CFG.video_dir,
        episode_name=CFG.video_name_prefix,
        seed=999,
        max_steps=CFG.max_steps_per_episode,
    )

    if video_path is not None and IN_IPYTHON:
        display(Video(video_path, embed=True))


if __name__ == "__main__":
    main()
```

# Part 4. Actor-critic

The **actor** outputs the policy. The **critic** outputs $V(s)$ or $Q(s, a)$. The actor picks actions; the critic evaluates them.

Start from the policy gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,Q^\pi(s_t, a_t)\big]$$

Subtract a baseline:

$$\nabla_\theta J(\theta) = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\big(Q^\pi(s_t, a_t) - V^\pi(s_t)\big)\big]$$

Approximate $V^\pi$ with a critic $V_\phi$:

$$\hat A_t = G_t - V_\phi(s_t)$$

so the actor loss becomes

$$L_\pi(\theta) = -\sum_t \log \pi_\theta(a_t \mid s_t)\,\hat A_t$$

and the critic is trained by regression:

$$L_V(\phi) = \sum_t \big(V_\phi(s_t) - y_t\big)^2$$

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# Example batch of states
states = torch.randn(5, 3)   # 5 states, obs_dim=3

# Example regression targets (returns or TD targets)
targets = torch.tensor([1.2, 0.5, -0.7, 2.0, 1.1], dtype=torch.float32)

value_net = ValueNet(obs_dim=3)
optimizer = optim.Adam(value_net.parameters(), lr=1e-3)

# Forward pass
pred_values = value_net(states)

# MSE regression loss
critic_loss = F.mse_loss(pred_values, targets)

optimizer.zero_grad()
critic_loss.backward()
optimizer.step()

print("pred_values:", pred_values.detach())
print("targets:", targets)
print("critic_loss:", critic_loss.item())
```

## 4.1 One-step advantage actor-critic (A2C) **[new]**

REINFORCE with a baseline still waits for the end of the episode to compute $G_t$. Actor-critic replaces $G_t$ with a one-step bootstrapped target, using the critic for everything after the next reward:

$$y_t = r_t + \gamma V_\phi(s_{t+1})(1 - d_t)$$

where $d_t$ is 1 if the episode terminated at step $t$. The advantage becomes the TD error:

$$\hat A_t = \delta_t = r_t + \gamma V_\phi(s_{t+1})(1 - d_t) - V_\phi(s_t)$$

Since $\mathbb{E}[r_t + \gamma V^\pi(s_{t+1})] = Q^\pi(s_t, a_t)$, $\delta_t$ estimates $Q^\pi - V^\pi$ with much lower variance than $G_t - V$, at the cost of bias when $V_\phi$ is wrong. The actor and critic losses are:

$$L_\pi(\theta) = -\sum_t \log \pi_\theta(a_t \mid s_t)\,\delta_t, \qquad L_V(\phi) = \sum_t \big(V_\phi(s_t) - y_t\big)^2$$

Generalized advantage estimation (GAE), used by PPO, interpolates between the two extremes with a parameter $\lambda$: $\hat A_t = \sum_{l \ge 0}(\gamma\lambda)^l\delta_{t+l}$. $\lambda = 0$ gives the one-step TD error and $\lambda = 1$ gives $G_t - V(s_t)$.

```python
import torch
import torch.nn.functional as F

def a2c_update(policy, value_net, opt_pi, opt_v, batch, gamma=0.99):
    obs, actions, rewards, next_obs, dones = batch   # tensors, shape [B, ...]

    values = value_net(obs)                           # V(s_t)
    with torch.no_grad():
        targets = rewards + gamma * value_net(next_obs) * (1 - dones)
    advantages = (targets - values).detach()          # TD error, no grad into actor

    dist = policy(obs)                                # Categorical or Normal
    log_probs = dist.log_prob(actions)
    if log_probs.dim() > 1:                           # Gaussian: sum action dims
        log_probs = log_probs.sum(dim=-1)

    actor_loss = -(log_probs * advantages).mean()
    critic_loss = F.mse_loss(values, targets)

    opt_pi.zero_grad(); actor_loss.backward(); opt_pi.step()
    opt_v.zero_grad(); critic_loss.backward(); opt_v.step()
    return actor_loss.item(), critic_loss.item()
```

# Part 5. Off-policy actor-critic: DDPG, TD3, SAC **[new]**

This follows CS224R Lecture 5 and your Lecture 5 notes. It connects the actor-critic methods in Part 4 to the Q-learning in Part 6: DDPG, TD3 and SAC are "DQN for continuous actions".

## 5.1 Why off-policy

Every method so far in Parts 3 and 4 is **on-policy**. The gradient $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a \mid s)\,\hat A]$ is an expectation under the *current* policy, so it takes a batch, computes one gradient, throws the batch away, and collects a new one. That is very sample-inefficient: millions of environment steps for MuJoCo tasks like Ant.

**Off-policy** methods learn from data collected by *any* policy, including older versions of themselves stored in a replay buffer. Each transition can be reused for many gradient steps. There are two ways to get there.

**Option 1: importance sampling.** Your Lecture 5 notes contrast two equations: "the first equation is policy gradient… the second equation uses off-policy data and you can take more than one gradient step per batch". The second one corrects for the data coming from an old policy $\pi_{\theta_{\text{old}}}$ by reweighting:

$$\nabla_\theta J(\theta) = \mathbb{E}_{a \sim \pi_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}\,\nabla_\theta \log \pi_\theta(a \mid s)\,\hat A(s, a)\right]$$

The ratio blows up when $\pi_\theta$ drifts far from $\pi_{\theta_{\text{old}}}$, so this only works for data from a policy that is close to the current one. PPO clips the ratio to keep it near 1. This is "slightly off-policy": a few gradient steps per batch, not a large replay buffer.

**Option 2: learn $Q^\pi(s, a)$ from a replay buffer.** This is the approach of DDPG, TD3 and SAC, and what the rest of this part covers. The key fact is that the Bellman equation for $Q^\pi$ holds for **any** transition $(s, a, r, s')$, no matter which policy chose $a$:

$$Q^\pi(s, a) = \mathbb{E}_{s'}\Big[r + \gamma\,\mathbb{E}_{a' \sim \pi(\cdot \mid s')}\big[Q^\pi(s', a')\big]\Big]$$

Only the next action $a'$ has to come from the current policy, and we can sample that ourselves without touching the environment. So $Q^\pi$ can be fit from old data.

## 5.2 Value, Q and advantage (recap)

From your Lecture 5 notes:

- $V^\pi(s)$ is the expected sum of future rewards starting at $s$ and following $\pi$. For a browser agent: how promising is this browser situation overall?
- $Q^\pi(s, a)$ is the expected future reward for taking action $a$ in $s$ and then following $\pi$. For a browser agent: how good is this exact next action (click Search vs. type the destination)?
- $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ is how much better $a$ is than the policy's average action.

For a discrete policy $V^\pi(s) = \sum_a \pi(a \mid s)\,Q^\pi(s, a)$; for a continuous one $V^\pi(s) = \int \pi(a \mid s)\,Q^\pi(s, a)\,da$.

## 5.3 The problem with DQN for continuous actions

DQN's target and greedy policy both use $\max_{a'} Q(s', a')$. With a few discrete actions this is just the largest output of the network. With continuous actions, like Ant's 8 joint torques, the max is an optimization problem over $\mathbb{R}^8$ at every step.

The fix: learn an **actor** $\mu_\theta(s)$ that approximately solves the max, $\mu_\theta(s) \approx \arg\max_a Q_\phi(s, a)$, and train it by gradient ascent on the critic.

## 5.4 DDPG: Deep Deterministic Policy Gradient

DDPG uses a deterministic actor $a = \mu_\theta(s)$ and a Q-critic $Q_\phi(s, a)$.

**Critic.** Same as DQN, with the actor in place of the max. The target uses target copies of both networks:

$$y = r + \gamma\,(1 - d)\,Q_{\bar\phi}\big(s',\ \mu_{\bar\theta}(s')\big), \qquad L(\phi) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{B}}\big[(Q_\phi(s, a) - y)^2\big]$$

**Actor.** The deterministic policy gradient. Push the action in the direction that increases Q, using the chain rule through the critic:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \mathcal{B}}\Big[\nabla_a Q_\phi(s, a)\big|_{a = \mu_\theta(s)}\ \nabla_\theta \mu_\theta(s)\Big]$$

In code this is just `actor_loss = -Q(s, actor(s)).mean()`. Autograd applies the chain rule, and only the actor's optimizer steps.

Compare with the stochastic policy gradient: there is no $\log \pi$ and no advantage. The gradient comes from the critic's slope $\nabla_a Q$, which is lower variance but only as good as the critic.

**Exploration.** A deterministic policy doesn't explore, so add noise when acting: $a = \mu_\theta(s) + \mathcal{N}(0, \sigma^2)$, clipped to the action bounds.

**Target networks.** Soft (Polyak) updates after every step, with $\tau \approx 0.005$:

$$\bar\phi \leftarrow \tau\phi + (1 - \tau)\bar\phi, \qquad \bar\theta \leftarrow \tau\theta + (1 - \tau)\bar\theta$$

DDPG works but is brittle. The actor exploits any place where the critic overestimates Q, which pushes the critic's errors upward over time.

## 5.5 TD3: Twin Delayed DDPG

TD3 keeps DDPG's structure and adds three fixes aimed at Q overestimation.

**1. Clipped double Q-learning.** Learn two critics $Q_{\phi_1}, Q_{\phi_2}$ and use the **minimum** in the target, which counteracts the upward bias (the same problem Double DQN addresses in Part 6):

$$y = r + \gamma\,(1 - d)\min_{j=1,2} Q_{\bar\phi_j}(s', \tilde a')$$

Both critics regress onto this same $y$.

**2. Target policy smoothing.** Add clipped noise to the target action, so the target is an average over nearby actions and the actor can't exploit a narrow spike in Q:

$$\tilde a' = \operatorname{clip}\big(\mu_{\bar\theta}(s') + \operatorname{clip}(\epsilon, -c, c),\ a_{\text{low}},\ a_{\text{high}}\big), \qquad \epsilon \sim \mathcal{N}(0, \tilde\sigma^2)$$

**3. Delayed policy updates.** Update the actor and target networks only every $d = 2$ critic updates, so the actor follows a critic that has had time to settle. The actor uses $Q_{\phi_1}$ only:

$$L(\theta) = -\mathbb{E}_{s \sim \mathcal{B}}\big[Q_{\phi_1}(s, \mu_\theta(s))\big]$$

## 5.6 SAC: Soft Actor-Critic

SAC uses a **stochastic** policy and changes the objective itself: maximize reward plus the entropy of the policy.

$$J(\pi) = \mathbb{E}\left[\sum_t \gamma^t\Big(r_t + \alpha\,\mathcal{H}\big(\pi(\cdot \mid s_t)\big)\Big)\right], \qquad \mathcal{H}(\pi(\cdot \mid s)) = \mathbb{E}_{a \sim \pi}\big[-\log \pi(a \mid s)\big]$$

The temperature $\alpha$ trades reward against randomness. The entropy bonus gives exploration for free (no added noise), keeps the policy from collapsing onto one action early, and lets it keep several good behaviors, the same multimodality issue as in imitation learning.

**Soft Bellman target.** The value of the next state includes the entropy bonus, $V(s') = \mathbb{E}_{a' \sim \pi}[Q(s', a') - \alpha \log \pi(a' \mid s')]$. Combined with TD3's twin critics:

$$y = r + \gamma\,(1 - d)\Big(\min_{j=1,2} Q_{\bar\phi_j}(s', a') - \alpha\log \pi_\theta(a' \mid s')\Big), \qquad a' \sim \pi_\theta(\cdot \mid s')$$

Note $a'$ is sampled from the **current** policy, not a target actor; SAC has no target policy network.

**Actor.** Minimize the KL divergence to the Boltzmann distribution $\exp(Q/\alpha)$, which works out to:

$$L(\theta) = \mathbb{E}_{s \sim \mathcal{B},\ a \sim \pi_\theta}\Big[\alpha\log \pi_\theta(a \mid s) - \min_{j} Q_{\phi_j}(s, a)\Big]$$

**The reparameterization trick.** To differentiate through the sampled $a$, write the sample as a deterministic function of noise: $a = \tanh\big(\mu_\theta(s) + \sigma_\theta(s) \odot \xi\big)$, $\xi \sim \mathcal{N}(0, I)$. Gradients flow through $\mu_\theta$ and $\sigma_\theta$ into $Q$, like DDPG's chain rule but with a stochastic policy. In PyTorch this is `dist.rsample()` instead of `dist.sample()`. This is lower variance than the score-function ($\log \pi$) gradient from Part 3.

**Tanh squashing and the log-prob correction.** The tanh keeps actions in $[-1, 1]$. Squashing changes the density, so the Gaussian log-probability from Part 3 needs a change-of-variables correction for each action dimension:

$$\log \pi(a \mid s) = \log \mathcal{N}(u;\ \mu, \sigma^2) - \sum_{i=1}^{D}\log\big(1 - \tanh^2(u_i)\big), \qquad a = \tanh(u)$$

**Automatic temperature.** Instead of hand-tuning $\alpha$, pick a target entropy $\bar{\mathcal{H}} = -\dim(\mathcal{A})$ (that's $-8$ for Ant) and adjust $\alpha$ to hit it:

$$L(\alpha) = \mathbb{E}_{a \sim \pi}\big[-\alpha\,(\log \pi(a \mid s) + \bar{\mathcal{H}})\big]$$

If the policy's entropy falls below the target, $\alpha$ grows and pushes entropy back up, and vice versa. Optimize $\log\alpha$ so $\alpha$ stays positive.

## 5.7 Comparison

| | DDPG | TD3 | SAC |
|---|---|---|---|
| Policy | deterministic $\mu_\theta(s)$ | deterministic $\mu_\theta(s)$ | stochastic, tanh-squashed Gaussian |
| Critics | 1 | 2, take the min | 2, take the min |
| Target for $a'$ | target actor $\mu_{\bar\theta}(s')$ | target actor + clipped noise | sample from the current policy |
| Actor gradient | $\nabla_a Q \cdot \nabla_\theta\mu$ | same, every $d$ steps | reparameterized, plus entropy |
| Exploration | Gaussian noise on actions | Gaussian noise on actions | the entropy bonus |
| Extra | | delayed actor updates | learned temperature $\alpha$ |

SAC is the usual default for continuous control and is the base CQL is built on (Part 7). Compared with PPO (on-policy), these methods typically need 5–10× fewer environment steps on MuJoCo, but each step does more compute and they have more hyperparameters.

## 5.8 Code: TD3 and SAC on Pendulum

One file with a shared replay buffer and both algorithms. DDPG is TD3 with one critic, no target smoothing and `policy_delay=1`.

```python
import copy
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def mlp(inp, out, hidden=256):
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, out))


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=200_000):
        self.o = np.zeros((size, obs_dim), np.float32); self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros(size, np.float32); self.o2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros(size, np.float32); self.ptr, self.n, self.size = 0, 0, size

    def add(self, o, a, r, o2, d):
        self.o[self.ptr], self.a[self.ptr], self.r[self.ptr], self.o2[self.ptr], self.d[self.ptr] = o, a, r, o2, d
        self.ptr = (self.ptr + 1) % self.size; self.n = min(self.n + 1, self.size)

    def sample(self, batch_size=256):
        i = np.random.randint(0, self.n, batch_size)
        return [torch.as_tensor(x[i]) for x in (self.o, self.a, self.r, self.o2, self.d)]


class Critic(nn.Module):                       # Q(s, a): state and action go in together
    def __init__(self, obs_dim, act_dim):
        super().__init__(); self.q = mlp(obs_dim + act_dim, 1)
    def forward(self, o, a):
        return self.q(torch.cat([o, a], dim=-1)).squeeze(-1)


def soft_update(net, target, tau):
    with torch.no_grad():
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.mul_(1 - tau).add_(tau * p)


# ------------------------------------------------------------------ TD3
class TD3:
    def __init__(self, obs_dim, act_dim, gamma=0.99, tau=0.005, lr=3e-4,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.actor = nn.Sequential(mlp(obs_dim, act_dim), nn.Tanh())     # actions in [-1, 1]
        self.q1, self.q2 = Critic(obs_dim, act_dim), Critic(obs_dim, act_dim)
        self.actor_t, self.q1_t, self.q2_t = map(copy.deepcopy, (self.actor, self.q1, self.q2))
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=lr)
        self.gamma, self.tau, self.noise, self.clip, self.delay = gamma, tau, policy_noise, noise_clip, policy_delay
        self.it = 0

    @torch.no_grad()
    def act(self, o, explore_std=0.1):
        a = self.actor(torch.as_tensor(o, dtype=torch.float32)).numpy()
        return np.clip(a + explore_std * np.random.randn(*a.shape), -1, 1)

    def update(self, batch):
        o, a, r, o2, d = batch
        self.it += 1
        with torch.no_grad():
            # target policy smoothing: clipped noise on the target action
            eps = (torch.randn_like(a) * self.noise).clamp(-self.clip, self.clip)
            a2 = (self.actor_t(o2) + eps).clamp(-1, 1)
            # clipped double-Q: min of the two target critics
            y = r + self.gamma * (1 - d) * torch.min(self.q1_t(o2, a2), self.q2_t(o2, a2))
        q_loss = F.mse_loss(self.q1(o, a), y) + F.mse_loss(self.q2(o, a), y)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        if self.it % self.delay == 0:            # delayed actor + target updates
            pi_loss = -self.q1(o, self.actor(o)).mean()     # deterministic policy gradient
            self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()
            for net, tgt in ((self.actor, self.actor_t), (self.q1, self.q1_t), (self.q2, self.q2_t)):
                soft_update(net, tgt, self.tau)


# ------------------------------------------------------------------ SAC
class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__(); self.net = mlp(obs_dim, 2 * act_dim)

    def forward(self, o, deterministic=False):
        mu, log_std = self.net(o).chunk(2, dim=-1)
        std = log_std.clamp(-20, 2).exp()
        dist = Normal(mu, std)
        u = mu if deterministic else dist.rsample()        # reparameterization trick
        a = torch.tanh(u)
        # Gaussian log-prob minus the tanh change-of-variables correction
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, log_prob


class SAC:
    def __init__(self, obs_dim, act_dim, gamma=0.99, tau=0.005, lr=3e-4):
        self.actor = SquashedGaussianActor(obs_dim, act_dim)
        self.q1, self.q2 = Critic(obs_dim, act_dim), Critic(obs_dim, act_dim)
        self.q1_t, self.q2_t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=lr)
        self.log_alpha = torch.zeros(1, requires_grad=True)          # learn log(alpha) so alpha > 0
        self.a_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(act_dim)
        self.gamma, self.tau = gamma, tau

    @torch.no_grad()
    def act(self, o, deterministic=False):
        a, _ = self.actor(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0), deterministic)
        return a.squeeze(0).numpy()

    def update(self, batch):
        o, a, r, o2, d = batch
        alpha = self.log_alpha.exp().detach()

        # critic: soft Bellman target, a' sampled from the CURRENT policy
        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            q_t = torch.min(self.q1_t(o2, a2), self.q2_t(o2, a2))
            y = r + self.gamma * (1 - d) * (q_t - alpha * logp2)
        q_loss = F.mse_loss(self.q1(o, a), y) + F.mse_loss(self.q2(o, a), y)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        # actor: alpha * log pi - min Q, gradients flow through rsample
        a_new, logp = self.actor(o)
        q_new = torch.min(self.q1(o, a_new), self.q2(o, a_new))
        pi_loss = (alpha * logp - q_new).mean()
        self.pi_opt.zero_grad(); pi_loss.backward(); self.pi_opt.step()

        # temperature: push entropy toward the target
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.a_opt.zero_grad(); alpha_loss.backward(); self.a_opt.step()

        soft_update(self.q1, self.q1_t, self.tau); soft_update(self.q2, self.q2_t, self.tau)


# ------------------------------------------------------------------ training loop
def train(algo_name="sac", env_id="Pendulum-v1", total_steps=20_000, start_steps=1_000, seed=0):
    np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make(env_id)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    scale = float(env.action_space.high[0])            # networks output [-1, 1]; scale to env
    agent = SAC(obs_dim, act_dim) if algo_name == "sac" else TD3(obs_dim, act_dim)
    buf = ReplayBuffer(obs_dim, act_dim)

    o, _ = env.reset(seed=seed); ep_ret, rets = 0.0, []
    for t in range(total_steps):
        a = np.random.uniform(-1, 1, act_dim) if t < start_steps else agent.act(o)
        o2, r, terminated, truncated, _ = env.step(a * scale)
        buf.add(o, a, r, o2, float(terminated))         # store the [-1, 1] action
        o, ep_ret = o2, ep_ret + r
        if terminated or truncated:
            rets.append(ep_ret); o, _ = env.reset(); ep_ret = 0.0
        if t >= start_steps:
            agent.update(buf.sample())                  # one gradient step per env step, reusing old data
        if (t + 1) % 2_000 == 0 and rets:
            print(f"{algo_name} step {t+1:6d}  avg return (last 5) {np.mean(rets[-5:]):8.1f}")
    return agent


if __name__ == "__main__":
    train("sac")
    train("td3")
```

On Pendulum the return starts around −1500 (random torques) and a good policy reaches about −150. In a test run, SAC first reached −125 at 6k steps and TD3 reached −127 at 8k steps. The training returns keep bouncing between about −150 and −350 because they include exploration (sampling for SAC, added noise for TD3), so evaluate with `act(o, deterministic=True)` for SAC or `explore_std=0` for TD3 to see the learned policy. For Ant, change `env_id` to `"Ant-v4"` and `total_steps` to about 1M.

# Part 6. Q-learning: DQN

DQN only learns a Q-function. The policy is implicit: act greedily, $a = \arg\max_a Q(s, a)$, with some exploration.

DQN is online: it stores transitions in a replay buffer and samples from it to get experience tuples $(s, a, r, s', \text{done})$, or `{obs, action, reward, next_obs, terminal}`. These are used to fit $Q(s, a)$ to an approximation of the optimal Bellman equation, which is an expectation.

## 6.1 From the Bellman equation to a target

The action-value function is defined by the optimal Bellman equation:

$$Q^*(s, a) = \mathbb{E}\big[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\big]$$

We replace the expectation with a sampling procedure. Define the target from one sampled transition:

$$y = r + \gamma \max_{a'} Q(s', a')$$

$y$ is an unbiased estimator of the expectation (if $Q = Q^*$):

$$\mathbb{E}[y] = \mathbb{E}\big[r + \gamma \max_{a'} Q^*(s', a')\big] = Q^*(s, a)$$

so we can use supervised regression to fit $Q(s, a)$ to $y$.

## 6.2 Tabular Q-learning

For a small number of states and discrete actions, $Q(s, a)$ is a table:

```text
           action 0   action 1   action 2
state s1     0.2        1.4       -0.7
state s2     0.9        0.1        2.0
state s3    -0.3        0.8        0.5
```

Move each entry toward its target with step size $\alpha$:

$$Q(s, a) \leftarrow Q(s, a) + \alpha\big(y - Q(s, a)\big) = Q(s, a) + \alpha\big(r + \gamma \max_{a'} Q(s', a') - Q(s, a)\big)$$

Define the TD error as

$$\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$$

and the update becomes an error term added to the current Q value: $Q(s, a) \leftarrow Q(s, a) + \alpha\,\delta$.

**Worked example [completed].** Let $Q(s_1, a_1) = 1$. `env.step()` returns reward $r = 2$ and next state $s_3$, and the best value at the next state is $\max_{a'} Q(s_3, a') = 4.0$. Take $\gamma = 0.9$ and $\alpha = 0.5$ (the original notes didn't set these values):

$$y = 2 + 0.9 \times 4.0 = 5.6$$

$$\delta = 5.6 - 1 = 4.6$$

$$Q(s_1, a_1) \leftarrow 1 + 0.5 \times 4.6 = 3.3$$

The entry moved halfway toward the target. Repeating this over many sampled transitions converges to $Q^*$ in the tabular case.

```python
import numpy as np

def q_learning_update(Q, s, a, r, s_next, done, alpha=0.5, gamma=0.9):
    target = r + gamma * np.max(Q[s_next]) * (1 - done)
    td_error = target - Q[s, a]
    Q[s, a] += alpha * td_error
    return td_error

def epsilon_greedy(Q, s, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    return int(np.argmax(Q[s]))
```

## 6.3 Replacing the table with a neural network **[new]**

We can replace the table with a neural network $Q_\theta(s, a)$ that takes a state and outputs one value per action. Images (Atari) and continuous states (CartPole, robots) are too large for a table, and a network generalizes between similar states. The tabular update turns into a regression loss:

$$L(\theta) = \big(Q_\theta(s, a) - y\big)^2, \qquad y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')(1 - d)$$

The $r$ comes from the replay buffer. Two changes make this stable:

1. **Replay buffer.** Consecutive transitions are highly correlated. Storing them in a buffer and sampling random minibatches makes the batches closer to i.i.d. and reuses each transition many times.
2. **Target network.** If $y$ is computed with the same network being trained, every gradient step also moves the target. DQN computes $y$ with a frozen copy $Q_{\text{target}}$ and copies the weights over every $C$ steps (or uses a soft update $\bar\theta \leftarrow \tau\theta + (1-\tau)\bar\theta$). The target is computed under `torch.no_grad()` so no gradient flows through it.

The $(1 - d)$ mask removes the bootstrap term at terminal states, where there is no future value.

**Double DQN.** $\max_{a'} Q_{\text{target}}$ tends to overestimate, because the max picks up noise. Double DQN selects the action with the online network and evaluates it with the target network: $y = r + \gamma\, Q_{\text{target}}\big(s', \arg\max_{a'} Q_\theta(s', a')\big)(1 - d)$.

The DQN target in PyTorch:

```python
with torch.no_grad():
    next_q_values = q_net_target(next_obs)
    next_q_max = next_q_values.max(dim=1).values
    y = rewards + gamma * next_q_max * (1 - dones)
```

## 6.4 Full DQN on CartPole **[new]**

```python
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)                 # [B, n_actions]: one Q value per action


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buf = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nxt, done = map(np.array, zip(*batch))
        return (torch.tensor(obs, dtype=torch.float32),
                torch.tensor(act, dtype=torch.int64),
                torch.tensor(rew, dtype=torch.float32),
                torch.tensor(nxt, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


def train_dqn(total_steps=50_000, gamma=0.99, lr=1e-3, batch_size=64,
              target_update_every=1_000, learning_starts=1_000,
              eps_start=1.0, eps_end=0.05, eps_decay_steps=20_000, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = QNetwork(obs_dim, n_actions)
    q_net_target = QNetwork(obs_dim, n_actions)
    q_net_target.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    obs, _ = env.reset(seed=seed)
    ep_return, returns = 0.0, []

    for step in range(total_steps):
        # epsilon-greedy exploration, linearly decayed
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = int(q.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        # store terminated (not truncated): a time-limit cut still has future value
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        ep_return += reward

        if terminated or truncated:
            returns.append(ep_return)
            ep_return = 0.0
            obs, _ = env.reset()

        if step >= learning_starts and len(buffer) >= batch_size:
            o, a, r, o2, d = buffer.sample(batch_size)

            # Q_theta(s, a): pick the column for the action that was taken
            q_sa = q_net(o).gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_max = q_net_target(o2).max(dim=1).values
                y = r + gamma * next_q_max * (1 - d)

            loss = F.smooth_l1_loss(q_sa, y)   # Huber loss; F.mse_loss also works
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

        if step % target_update_every == 0:
            q_net_target.load_state_dict(q_net.state_dict())

        if step % 5_000 == 0 and returns:
            print(f"step={step:6d} eps={eps:.2f} avg_return(last 20)={np.mean(returns[-20:]):.1f}")

    env.close()
    return q_net


if __name__ == "__main__":
    train_dqn()
```

For Atari, the network becomes a CNN over 4 stacked 84×84 grayscale frames (the frame stacking from Part 1 is what makes the observation Markov), the buffer holds about 1M transitions, and the target network is updated every 10k steps.

# Part 7. Offline RL: CQL and IQL **[new]**

DQN is online: it collects its own data. In offline RL the agent only gets a fixed dataset $\mathcal{D} = \{(s, a, r, s', d)\}$ collected by some other behavior policy $\pi_\beta$, and it can't interact with the environment.

Running DQN on a fixed dataset fails because of **out-of-distribution actions**. The target $\max_{a'} Q(s', a')$ queries actions that never appear in $\mathcal{D}$. Q-values for those actions are never corrected by data, so errors there tend to be overestimates, the max picks them, and bootstrapping spreads the overestimate everywhere. CQL and IQL are two answers to this problem.

## 7.1 CQL: Conservative Q-Learning

CQL keeps the usual Bellman error and adds a regularizer that pushes Q down on actions the current policy would choose and pushes it up on actions actually in the dataset:

$$L(\theta) = \alpha\,\mathbb{E}_{s \sim \mathcal{D}}\Big[\log\sum_{a}\exp Q_\theta(s, a) - \mathbb{E}_{a \sim \mathcal{D}}\big[Q_\theta(s, a)\big]\Big] + \frac{1}{2}\,\mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\Big[\big(Q_\theta(s, a) - y\big)^2\Big]$$

with the same target as DQN, $y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')(1 - d)$.

- $\log\sum_a \exp Q(s, a)$ is a soft maximum over all actions. Minimizing it lowers the Q-values of whichever actions currently look best, including out-of-distribution ones.
- $-\mathbb{E}_{a \sim \mathcal{D}}[Q(s, a)]$ raises the Q-values of actions in the data, so in-distribution actions are not pushed down.
- $\alpha$ trades off conservatism against fitting the Bellman target. The result is a Q-function that lower-bounds the true value of the learned policy, so the policy prefers actions supported by the data.

For continuous actions the log-sum-exp is estimated by sampling actions from a uniform distribution and from the current policy, and CQL is usually built on top of SAC (Part 5).

```python
import torch
import torch.nn.functional as F

def cql_dqn_loss(q_net, q_net_target, batch, gamma=0.99, alpha=1.0):
    obs, actions, rewards, next_obs, dones = batch      # fixed offline dataset

    q_all = q_net(obs)                                   # [B, n_actions]
    q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_max = q_net_target(next_obs).max(dim=1).values
        y = rewards + gamma * next_q_max * (1 - dones)
    bellman = 0.5 * F.mse_loss(q_sa, y)

    # conservative term: push down soft-max over all actions, push up data actions
    cql_term = (torch.logsumexp(q_all, dim=1) - q_sa).mean()

    return bellman + alpha * cql_term
```

## 7.2 IQL: Implicit Q-Learning

IQL avoids querying out-of-distribution actions entirely: no $\max_{a'}$ is taken over actions. It only ever evaluates $Q$ on $(s, a)$ pairs from the dataset. It learns three networks.

**1. Value function by expectile regression.** Fit $V_\psi(s)$ to an upper expectile of $Q(s, a)$ over dataset actions:

$$L_V(\psi) = \mathbb{E}_{(s,a) \sim \mathcal{D}}\Big[L_2^\tau\big(Q_{\bar\theta}(s, a) - V_\psi(s)\big)\Big], \qquad L_2^\tau(u) = \big|\tau - \mathbf{1}(u < 0)\big|\,u^2$$

With $\tau = 0.5$ this is ordinary MSE and $V$ is the mean of Q over the data's actions. As $\tau \to 1$ the loss weights positive errors more heavily, so $V(s)$ approaches the max of Q over actions that **appear in the data**. That gives an in-distribution approximation of $\max_a Q(s, a)$ without ever evaluating an unseen action. Typical values are $\tau = 0.7$ to $0.9$.

**2. Q-function by regression onto V.** The target uses $V(s')$ in place of $\max_{a'} Q(s', a')$:

$$L_Q(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\Big[\big(r + \gamma V_\psi(s')(1 - d) - Q_\theta(s, a)\big)^2\Big]$$

**3. Policy extraction by advantage-weighted regression.** Q and V never needed a policy. Afterward, extract one by behavior cloning that is weighted toward high-advantage dataset actions:

$$L_\pi(\phi) = -\mathbb{E}_{(s,a) \sim \mathcal{D}}\Big[\exp\big(\beta\,(Q_{\bar\theta}(s, a) - V_\psi(s))\big)\log \pi_\phi(a \mid s)\Big]$$

$\beta$ is an inverse temperature. Small $\beta$ is close to plain behavior cloning; large $\beta$ concentrates on the best actions in the data. The weight is usually clipped (for example at 100) for stability.

```python
import torch
import torch.nn.functional as F

def expectile_loss(diff, tau=0.7):
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()

def iql_losses(q_net, q_target, v_net, policy, batch,
               gamma=0.99, tau=0.7, beta=3.0):
    obs, actions, rewards, next_obs, dones = batch       # actions: continuous, [B, act_dim]

    # 1) V: expectile regression toward target Q on dataset actions
    with torch.no_grad():
        q_t = q_target(obs, actions)                      # Q(s, a) for a in D only
    v = v_net(obs)
    v_loss = expectile_loss(q_t - v, tau)

    # 2) Q: regress onto r + gamma * V(s'), no max over actions
    with torch.no_grad():
        y = rewards + gamma * v_net(next_obs) * (1 - dones)
    q_loss = F.mse_loss(q_net(obs, actions), y)

    # 3) policy: advantage-weighted behavior cloning
    with torch.no_grad():
        adv = q_t - v.detach()
        weights = torch.clamp(torch.exp(beta * adv), max=100.0)
    log_prob = policy(obs).log_prob(actions).sum(dim=-1)
    pi_loss = -(weights * log_prob).mean()

    return v_loss, q_loss, pi_loss
```

## 7.3 Summary

| Method | Data | Target for $Q(s, a)$ | How it handles unseen actions |
|---|---|---|---|
| DQN | online, replay buffer | $r + \gamma \max_{a'} Q_{\text{target}}(s', a')$ | It doesn't need to: it can try them and correct the estimate |
| CQL | fixed dataset | same as DQN | Penalizes high Q on actions outside the data |
| IQL | fixed dataset | $r + \gamma V(s')$ | Never queries them; $V$ is an expectile over dataset actions |
