"""MADDPG Agent"""

import numpy as np
import torch
import torch.nn.functional as F

from ddpg_agent import DDPGAgent
from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e6)    # replay buffer size
BATCH_SIZE = 256          # minibatch size
GAMMA = 0.99              # discount factor
TAU = 1e-3                # for soft update of target parameters
UPDATE_EVERY = 5          # how often to update the network
N_UPDATES_PER_STEP = 5    # number of updates to perform at each learning step
INITIAL_NOISE = 1.0       # initial value for noise multiplier
NOISE_DECAY = 0.99925     # value to multiply noise after each step

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPGAgent():
    def __init__(self, n_agents, state_size, action_size, seed=0):
        super(MADDPGAgent, self).__init__()

        self.n_agents = n_agents
        self.maddpg_agent = [DDPGAgent(state_size, action_size, n_agents, seed+i) for i in range(n_agents)]

        # Replay memory
        self.memory = ReplayBuffer((n_agents, action_size), BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Noise parameters with exponential decay
        self.noise = INITIAL_NOISE
        self.noise_decay = NOISE_DECAY

    def reset(self):
        """Reset agent before each episode"""
        self.noise = INITIAL_NOISE

    def act(self, states, noise=False):
        """
        Get actions from all agents in the MADDPG object
        Expects states to be (n_agents, state_size)
        """
        # Set noise = self.noise if parameter noise=True, else set it to 0.0
        noise *= self.noise
        actions = [
            agent.act(state, noise)
            for agent, state in zip(self.maddpg_agent, states)
        ]
        return np.clip(actions, -1, 1)

    def target_act(self, states, noise=0.0):
        """
        Get actions from all agents in the MADDPG object using target network
        Expects states to be (batch_size, n_agents, state_size)
        """
        target_actions = [
            agent.target_act(state, noise)
            for agent, state in zip(self.maddpg_agent, states)
        ]
        return np.clip(target_actions, -1, 1)

    def preprocess_tensor_to_critic(self, tensor):
        """Transform tensor with shape (n_agents, batch_size, object_size) to (batch_size, n_agents*object_size)"""
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor).float().to(device)
        return tensor.permute(1, 0, 2).reshape(BATCH_SIZE, -1)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        self.noise *= self.noise_decay

        # If enough samples are available in memory, get random subset and learn
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(N_UPDATES_PER_STEP):
                for idx_agent in range(self.n_agents):
                    experiences = self.memory.sample()
                    self.learn(experiences, idx_agent)

    def learn(self, experiences, idx_agent):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, s_full, a, a_full, r, s', done) tuples
            idx_agent (int): index of agent to update
        """
        states, states_full, actions, actions_full, rewards, next_states, dones = experiences

        agent = self.maddpg_agent[idx_agent]

        rewards_agent = rewards[idx_agent].view(-1, 1)
        dones_agent = dones[idx_agent].view(-1, 1)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.target_act(next_states)
        Q_targets_next = agent.target_critic(
            self.preprocess_tensor_to_critic(next_states),
            self.preprocess_tensor_to_critic(actions_next)
        )
        # Compute Q targets for current states (y_i)
        Q_targets = rewards_agent + (GAMMA * Q_targets_next * (1 - dones_agent))
        # Compute critic loss
        Q_expected = agent.critic(states_full, actions_full)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping to avoid exploding gradient
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # Detach the other agents to save computation
        actions_pred = torch.stack([
            self.maddpg_agent[i].actor(state)
                if i == idx_agent
                else self.maddpg_agent[i].actor(state).detach()
            for i, state in enumerate(states)
        ]).float().to(device)
        actor_loss = -agent.critic(
            states_full,
            self.preprocess_tensor_to_critic(actions_pred)
        ).mean()
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(agent.critic, agent.target_critic, TAU)
        self.soft_update(agent.actor, agent.target_actor, TAU)

    def load_model(self, prefix_name='model'):
        """Load models for all the agents"""
        for i in range(self.n_agents):
            self.maddpg_agent[i].actor.load_state_dict(torch.load('{}_actor_{}.pt'.format(prefix_name, i)))
            self.maddpg_agent[i].critic.load_state_dict(torch.load('{}_critic_{}.pt'.format(prefix_name, i)))

    def save_model(self, prefix_name='model'):
        """Save models of all the agents"""
        for i in range(self.n_agents):
            torch.save(self.maddpg_agent[i].actor.state_dict(), '{}_actor_{}.pt'.format(prefix_name, i))
            torch.save(self.maddpg_agent[i].critic.state_dict(), '{}_critic_{}.pt'.format(prefix_name, i))

    def soft_update(self, source_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            source_model (torch.nn.Module): weights will be copied from
            target_model (torch.nn.Module): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(tau*source_param.data + (1.0-tau)*target_param.data)
