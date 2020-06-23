"""Implements experience replay buffer"""

import random
import numpy as np
import torch
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, states, actions, rewards, next_states, dones):
        """Add new experiences to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # states.shape = (n_agents, batch_size, state_size)
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None], axis=1)).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None], axis=1)).float().to(device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None], axis=1)).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None], axis=1)).float().to(device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None], axis=1).astype(np.uint8)).float().to(device)

        # states_full.shape = (batch_size, n_agents*state_size)
        states_full = torch.from_numpy(np.stack([e.state.reshape(-1) for e in experiences if e is not None])).float().to(device)
        actions_full = torch.from_numpy(np.stack([e.action.reshape(-1) for e in experiences if e is not None])).float().to(device)

        return (states, states_full, actions, actions_full, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)