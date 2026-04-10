from traffic_env import TrafficEnv
from ddqn_agent import DDQNAgent
import numpy as np

EPISODES   = 50
MAX_STEPS  = 500
OBS_DIM    = 9   # 8 queue counts + 1 phase

def train():
    env   = TrafficEnv()
    agent = DDQNAgent(obs_dim=OBS_DIM)

    best_reward = -np.inf

    for ep in range(EPISODES):
        obs        = env.reset()
        total_reward = 0
        total_loss   = 0
        loss_count   = 0
        last_action  = obs[-1]

        for t in range(MAX_STEPS):
            action = agent.act(obs)

            # phase switch penalty
            switch_penalty = -0.5 if action != last_action else 0
            next_obs, reward, done, info = env.step(action)
            reward += switch_penalty

            agent.store(obs, action, reward, next_obs, done)
            loss = agent.update()

            if loss is not None:
                total_loss += loss
                loss_count += 1

            obs = next_obs
            last_action = action
            total_reward += reward

            if done:
                break

        avg_loss = total_loss / loss_count if loss_count else 0
        print(
            f"ep={ep:>3} | reward={total_reward:>8.1f} | "
            f"throughput={env.total_throughput:>5} | "
            f"eps={agent.epsilon:.3f} | loss={avg_loss:.4f}"
        )

        if total_reward > best_reward:
            best_reward = total_reward
            import torch
            torch.save(agent.online_net.state_dict(), "best_ddqn.pth")
            print(f"         ^ new best saved")

if __name__ == "__main__":
    train()