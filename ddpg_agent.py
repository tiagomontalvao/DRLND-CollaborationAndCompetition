"""DDPG Agent"""

import numpy as np
import random
import torch
from torch.optim import Adam

from model import Actor, Critic
from normal_noise import NormalNoise

LR_ACTOR = 5e-4      # learning rate of the actor
LR_CRITIC = 1e-3     # learning rate of the critic
WEIGHT_DECAY = 0     # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent():
    def __init__(self, state_size, action_size, n_agents, seed=0):
        super(DDPGAgent, self).__init__()

        self.action_size = action_size
        self.state_size = state_size
        self.seed = seed

        random.seed(seed)

        # Actor Network (w/ Target Network)
        self.actor = Actor(state_size, action_size, seed).to(device)
        self.target_actor = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic = Critic(state_size*n_agents, action_size*n_agents, seed).to(device)
        self.target_critic = Critic(state_size*n_agents, action_size*n_agents, seed).to(device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = NormalNoise(action_size, seed)

    def act(self, obs, noise=0.0):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).unsqueeze(0).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs).cpu().data.numpy() + noise*self.noise.sample()
        self.actor.train()
        return action[0]

    def target_act(self, obs, noise=0.0):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(device)
        action = self.target_actor(obs).cpu().data.numpy() + noise*self.noise.sample()
        return action
