# reward.py

import numpy as np


class DelayAwareReward:
    """
    Multi-objective delay-aware reward function.

    reward = -(alpha * queue + beta * imbalance + gamma * phase_switch_penalty)
    """

    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prev_phase = None

    def compute(self, obs):

        queue_ns = obs["totals"]["NS"]
        queue_ew = obs["totals"]["EW"]

        total_queue = queue_ns + queue_ew

        imbalance = abs(queue_ns - queue_ew)

        phase = obs["phase"]

        phase_switch_penalty = 0
        if self.prev_phase is not None and phase != self.prev_phase:
            phase_switch_penalty = 1

        self.prev_phase = phase

        reward = -(
            self.alpha * total_queue +
            self.beta * imbalance +
            self.gamma * phase_switch_penalty
        )

        return float(reward)