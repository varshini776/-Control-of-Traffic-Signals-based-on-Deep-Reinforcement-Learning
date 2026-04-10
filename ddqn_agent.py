import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ddqn_model import DDQNNet
from replay_buffer import ReplayBuffer

class DDQNAgent:
    def __init__(
        self,
        obs_dim,
        n_actions=4,
        lr=1e-3,
        gamma=0.99,
        tau=0.005,            # soft update blend rate
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10_000,
        batch_size=64,
        learn_start=512,      # steps before first update
    ):
        self.n_actions   = n_actions
        self.gamma       = gamma
        self.tau         = tau
        self.epsilon     = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size  = batch_size
        self.learn_start = learn_start

        self.online_net = DDQNNet(obs_dim, n_actions)
        self.target_net = DDQNNet(obs_dim, n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)
        self.steps     = 0

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.online_net(state_t)
        return int(q_vals.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.learn_start:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states)
        actions_t     = torch.LongTensor(actions).unsqueeze(1)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1)

        # current Q
        current_q = self.online_net(states_t).gather(1, actions_t)

        # DDQN target: online net picks action, target net scores it
        with torch.no_grad():
            best_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
            target_q     = self.target_net(next_states_t).gather(1, best_actions)
            td_target    = rewards_t + self.gamma * target_q * (1 - dones_t)

        loss = nn.MSELoss()(current_q, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        # soft update target network
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1 - self.tau) * tp.data)

        # decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps  += 1
        return loss.item()