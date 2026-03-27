import torch

from env_config import medium_traffic
from env import TrafficEnv
from dqn_model import DQNNet


def encode_state(obs):

    import numpy as np

    return np.array([

        obs["N"]["straight"],
        obs["N"]["turn"],

        obs["S"]["straight"],
        obs["S"]["turn"],

        obs["E"]["straight"],
        obs["E"]["turn"],

        obs["W"]["straight"],
        obs["W"]["turn"],

        obs["phase"],

        obs["totals"]["NS"] - obs["totals"]["EW"]

    ], dtype=np.float32)


model = DQNNet()

model.load_state_dict(torch.load("model.pth"))

model.eval()

cfg = medium_traffic()

env = TrafficEnv(cfg)

obs = env.reset()

while True:

    state = encode_state(obs)

    state = torch.FloatTensor(state)

    with torch.no_grad():
        action = model(state).argmax().item()

    obs, done = env.step(action)