"""
TO EDIT: Defines a pytorch policy as the agent's actor.

Functions to edit:
    1. get_action (line 96)
    2. forward (line 110)
    3. update (line 126)
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.policies.base_policy import BasePolicy


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to actions

    Attributes
    ----------
    logits_na: nn.Sequential
        A neural network that outputs dicrete actions
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    get_action:
        Calls the actor forward function
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # Initialize variables for environment (action/observation dimension, number of layers, etc.)
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        # NOTE: This works for a continuous action space. All our environments use a continuous action space.
        self.logits_na = None
        self.mean_net = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(

            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    ##################################

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        :param obs: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        if obs.ndim == 1:
            obs = obs[None] # [1, ob_dim]

        obs_t = ptu.from_numpy(obs) # numpy -> torch on correct device

        with torch.no_grad():
            act_t = self.mean_net(obs_t)   # [1, ac_dim]

        act = ptu.to_numpy(act_t).astype(np.float32)  # keep [1, ac_dim]
        return act        #raise NotImplementedError

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!
        return self.mean_net(observation)
        #raise NotImplementedError

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss. Recall that to update the policy
        # you need to backpropagate the gradient and step the optimizer.
         # Convert numpy -> torch if needed
        if isinstance(observations, np.ndarray):
            observations = ptu.from_numpy(observations)
        if isinstance(actions, np.ndarray):
            actions = ptu.from_numpy(actions)

        pred_actions = self.mean_net(observations) # [batch_size, ac_dim]
        loss = F.mse_loss(pred_actions, actions)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {"Training Loss": ptu.to_numpy(loss)}

