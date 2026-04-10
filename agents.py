import numpy as np
from traffic_env import TrafficEnv

class FixedAgent:
    def __init__(self, interval=10):
        self.interval = interval
        self.timer = 0
        self.current_phase = 0

    def act(self, obs):
        self.timer += 1
        if self.timer >= self.interval:
            self.timer = 0
            self.current_phase = (self.current_phase + 1) % 4
        return self.current_phase


class EpsilonGreedyAgent:
    def __init__(self, epsilon=0.3, alpha=0.1, gamma=0.9):
        self.q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def _q(self, state):
        if state not in self.q:
            self.q[state] = np.zeros(4)
        return self.q[state]

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        return int(np.argmax(self._q(state)))

    def update(self, s, a, r, s_next):
        q_now = self._q(s)[a]
        q_next = np.max(self._q(s_next))
        self._q(s)[a] += self.alpha * (r + self.gamma * q_next - q_now)
        self.epsilon = max(0.05, self.epsilon * 0.9995)


def run(agent_type="fixed", episodes=3):
    env = TrafficEnv()
    agent = FixedAgent() if agent_type == "fixed" else EpsilonGreedyAgent()

    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        for t in range(500):
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            if isinstance(agent, EpsilonGreedyAgent):
                agent.update(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
            if t % 100 == 0:
                print(f"  ep={ep} t={t} phase={info['phase_name'][:12]} "
                      f"wait={info['total_wait']} cleared={info['cleared']}")
        print(f"Episode {ep} | total_reward={total_reward:.1f} "
              f"throughput={env.total_throughput}")

if __name__ == "__main__":
    print("=== Fixed agent ===")
    run("fixed")
    print("\n=== ε-greedy agent ===")
    run("egreedy")