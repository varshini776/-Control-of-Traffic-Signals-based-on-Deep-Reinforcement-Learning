import torch
import torch.nn as nn

class DDQNNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # dueling streams
        self.value_stream  = nn.Linear(hidden, 1)
        self.adv_stream    = nn.Linear(hidden, n_actions)

    def forward(self, x):
        feat = self.net(x)
        value = self.value_stream(feat)
        adv   = self.adv_stream(feat)
        return value + adv - adv.mean(dim=1, keepdim=True)  # dueling aggregation