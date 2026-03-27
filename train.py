# train.py

import numpy as np
import torch

from env_config import medium_traffic
from env import TrafficEnv

from agent import DDQNAgent
from reward import DelayAwareReward


def encode_state(obs):

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


def train():

    cfg = medium_traffic()

    env = TrafficEnv(cfg)

    agent = DDQNAgent()

    reward_fn = DelayAwareReward()

    episodes = 50

    max_steps = 2000

    for ep in range(episodes):

        obs = env.reset()

        state = encode_state(obs)

        total_reward = 0

        for step in range(max_steps):

            action = agent.select_action(state)

            next_obs, done = env.step(action)

            reward = reward_fn.compute(next_obs)

            next_state = encode_state(next_obs)

            agent.buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            loss = agent.learn()

            state = next_state

            total_reward += reward

            if done:
                break

        print(f"Episode {ep} Reward {total_reward:.2f}")

    torch.save(agent.online_net.state_dict(), "model.pth")

    print("Training complete. Model saved as model.pth")


if __name__ == "__main__":
    train()