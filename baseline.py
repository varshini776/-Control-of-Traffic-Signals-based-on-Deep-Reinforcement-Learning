"""
baseline.py
===========

Baseline control policies for the traffic simulator.

POLICIES
--------
Random      — pure chance (lower bound)
Greedy      — serve largest queue (smart heuristic)
Fixed-Time  — cycle through phases on timer (real-world baseline)

ACTIONS
-------
0 → NS STRAIGHT
1 → NS TURN
2 → EW STRAIGHT
3 → EW TURN
"""

import numpy as np


def greedy_policy(obs):
    """Serve the phase with the largest queue."""
    ns_straight = obs["N"]["straight"] + obs["S"]["straight"]
    ns_turn = obs["N"]["turn"] + obs["S"]["turn"]
    ew_straight = obs["E"]["straight"] + obs["W"]["straight"]
    ew_turn = obs["E"]["turn"] + obs["W"]["turn"]

    values = [ns_straight, ns_turn, ew_straight, ew_turn]
    return int(np.argmax(values))


def random_policy():
    """Choose a random phase."""
    return np.random.randint(0, 4)


def fixed_time_policy(step, cycle=10):
    """Cycle through phases on a fixed timer."""
    return (step // cycle) % 4


def run_episode(env, policy: str = "greedy"):
    """Run a single episode and return metrics."""
    valid = {"greedy", "random", "fixed"}
    if policy not in valid:
        raise ValueError(f"policy must be one of {valid}")

    obs = env.reset()

    for step in range(env.cfg.max_steps):
        if policy == "greedy":
            action = greedy_policy(obs)
        elif policy == "fixed":
            action = fixed_time_policy(step)
        else:
            action = random_policy()

        obs, done = env.step(action)

        if done:
            break

    m = env.metrics()

    return {
        "avg_queue": round(m["avg_queue"], 4),
        "throughput": round(m["throughput"], 4),
        "steps": step + 1,
    }


def evaluate_baseline(policy: str, arrival_rate: float = 0.3, n_seeds: int = 5):
    """Evaluate a policy across multiple seeds."""
    from env_config import EnvConfig
    from env_enhanced import TrafficEnv

    queues = []
    throughputs = []

    for seed in range(n_seeds):
        cfg = EnvConfig(arrival_rate=arrival_rate, seed=seed)
        env = TrafficEnv(cfg)

        result = run_episode(env, policy=policy)

        queues.append(result["avg_queue"])
        throughputs.append(result["throughput"])

    def stats(x):
        return float(np.mean(x)), float(np.std(x))

    mq, sq = stats(queues)
    mt, st = stats(throughputs)

    return {
        "policy": policy,
        "arrival_rate": arrival_rate,
        "n_seeds": n_seeds,
        "mean_queue": round(mq, 4),
        "std_queue": round(sq, 4),
        "mean_throughput": round(mt, 4),
        "std_throughput": round(st, 4),
    }


if __name__ == "__main__":
    from env_config import EnvConfig
    from env_enhanced import TrafficEnv

    print("=" * 60)
    print("Baseline Comparison (λ=0.3)")
    print("=" * 60)

    for pol in ("random", "fixed", "greedy"):
        cfg = EnvConfig(arrival_rate=0.3, seed=42)
        env = TrafficEnv(cfg)

        r = run_episode(env, policy=pol)

        print(
            f"{pol:<10} | "
            f"avg_queue={r['avg_queue']:.2f} | "
            f"throughput={r['throughput']:.2f} | "
            f"steps={r['steps']}"
        )