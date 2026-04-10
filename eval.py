import numpy as np
import torch
from traffic_env import TrafficEnv
from ddqn_model import DDQNNet
from agents import FixedAgent, EpsilonGreedyAgent
import sys
OBS_DIM   = 9
N_ACTIONS = 4
EPISODES  = 10
MAX_STEPS = 500
LANES     = ["N","NE","E","SE","S","SW","W","NW"]

class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file     = open(filepath, "w")

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()

sys.stdout = Tee("eval_results.txt")
# ── loaders ────────────────────────────────────────────────────────────────

def load_ddqn(path="best_ddqn.pth"):
    net = DDQNNet(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net

def ddqn_act(net, obs):
    t = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        return int(net(t).argmax(dim=1).item())

# ── single episode runner ──────────────────────────────────────────────────

def run_episode(agent_type, net=None, seed=42):
    env = TrafficEnv()
    np.random.seed(seed)

    if agent_type == "fixed":
        agent = FixedAgent(interval=10)
    elif agent_type == "greedy":
        agent = EpsilonGreedyAgent(epsilon=0.0)   # pure greedy, no explore
    else:
        agent = None

    obs = env.reset()
    total_reward  = 0
    total_wait    = 0
    phase_counts  = [0, 0, 0, 0]
    last_action   = obs[-1]
    switches      = 0

    for t in range(MAX_STEPS):
        if agent_type == "ddqn":
            action = ddqn_act(net, obs)
        else:
            action = agent.act(obs)

        if action != last_action:
            switches += 1
        last_action = action

        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        total_wait   += info["total_wait"]
        phase_counts[action] += 1
        obs = next_obs
        if done:
            break

    return {
        "reward":     round(total_reward, 1),
        "throughput": env.total_throughput,
        "avg_wait":   round(total_wait / MAX_STEPS, 2),
        "switches":   switches,
        "phase_dist": [round(p/MAX_STEPS*100, 1) for p in phase_counts],
    }

# ── main eval ──────────────────────────────────────────────────────────────

def evaluate():
    net = load_ddqn()
    agents = ["fixed", "greedy", "ddqn"]
    results = {a: [] for a in agents}

    print(f"\nRunning {EPISODES} episodes per agent...\n")

    for ep in range(EPISODES):
        seed = ep * 7
        for a in agents:
            r = run_episode(a, net=net, seed=seed)
            results[a].append(r)
            print(f"  ep={ep:>2} [{a:>6}] reward={r['reward']:>8.1f}  "
                  f"thru={r['throughput']:>5}  "
                  f"avg_wait={r['avg_wait']:>5}  "
                  f"switches={r['switches']:>4}")

    # ── summary ────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'METRIC':<20} {'FIXED':>10} {'GREEDY':>10} {'DDQN':>10}")
    print("="*65)

    metrics = [
        ("Avg reward",    "reward",     False),
        ("Avg throughput","throughput", False),
        ("Avg wait",      "avg_wait",   True),   # lower is better
        ("Avg switches",  "switches",   True),
    ]

    for label, key, lower_better in metrics:
        vals = {a: np.mean([r[key] for r in results[a]]) for a in agents}
        best = min(vals, key=vals.get) if lower_better else max(vals, key=vals.get)
        row  = f"{label:<20}"
        for a in agents:
            marker = " *" if a == best else "  "
            row   += f"{vals[a]:>9.1f}{marker}"
        print(row)

    print("="*65)
    print("* = best\n")

    # ── phase usage ────────────────────────────────────────────────────────
    print("Phase distribution (% of steps):")
    print(f"{'':20} {'P0':>6} {'P1':>6} {'P2':>6} {'P3':>6}")
    for a in agents:
        avg_dist = [
            round(np.mean([r["phase_dist"][i] for r in results[a]]), 1)
            for i in range(4)
        ]
        print(f"  {a:<18} {avg_dist[0]:>5}% {avg_dist[1]:>5}% "
              f"{avg_dist[2]:>5}% {avg_dist[3]:>5}%")

    # ── win rate ───────────────────────────────────────────────────────────
    ddqn_wins_reward = sum(
        results["ddqn"][i]["reward"] > max(
            results["fixed"][i]["reward"],
            results["greedy"][i]["reward"]
        ) for i in range(EPISODES)
    )
    print(f"\nDDQN beats both baselines in {ddqn_wins_reward}/{EPISODES} episodes (reward)")
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print("Results saved to eval_results.txt")
    
if __name__ == "__main__":
    evaluate()