# agent.py

import torch
import random
import numpy as np

from collections import deque

from dqn_model import DQNNet
from double_dqn import ddqn_update


class ReplayBuffer:

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        s, a, r, s_next, d = zip(*batch)

        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s_next)),
            torch.FloatTensor(d)
        )

    def __len__(self):
        return len(self.buffer)


class DDQNAgent:

    def __init__(self, state_dim=10, action_dim=4):

        self.online_net = DQNNet(state_dim, action_dim)

        self.target_net = DQNNet(state_dim, action_dim)

        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=1e-3
        )

        self.buffer = ReplayBuffer()

        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.target_update_freq = 500

        self.step_count = 0

        self.action_dim = action_dim

    def select_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_net(state)

        return q_values.argmax().item()

    def learn(self, batch_size=64):

        if len(self.buffer) < batch_size:
            return None

        batch = self.buffer.sample(batch_size)

        loss = ddqn_update(
            self.online_net,
            self.target_net,
            self.optimizer,
            batch,
            self.gamma
        )

        self.step_count += 1

        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(
                self.online_net.state_dict()
            )

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

        return loss