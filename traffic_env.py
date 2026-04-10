import numpy as np

PHASES = [
    {"name": "NS through+right", "ns": True,  "ew": False, "left": False},
    {"name": "EW through+right", "ns": False, "ew": True,  "left": False},
    {"name": "NS left turn",     "ns": True,  "ew": False, "left": True},
    {"name": "EW left turn",     "ns": False, "ew": True,  "left": True},
]

LANES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
NS_LANES = {"N", "S", "NE", "SW"}
EW_LANES = {"E", "W", "SE", "NW"}

class TrafficEnv:
    def __init__(self, arrival_rate=0.4, max_queue=12):
        self.arrival_rate = arrival_rate
        self.max_queue = max_queue
        self.reset()

    def reset(self):
        self.queues = {lane: np.random.randint(1, 5) for lane in LANES}
        self.phase = 0
        self.step_count = 0
        self.total_throughput = 0
        return self._get_obs()

    def _get_obs(self):
        return tuple(min(self.queues[l], 5) for l in LANES) + (self.phase,)

    def step(self, action):
        ph = PHASES[action]
        self.phase = action
        cleared = 0

        for lane in LANES:
            can_go = (ph["ns"] and lane in NS_LANES) or \
                     (ph["ew"] and lane in EW_LANES)
            if can_go and self.queues[lane] > 0:
                flow = min(self.queues[lane], 1 if ph["left"] else 2)
                self.queues[lane] -= flow
                cleared += flow

        # arrivals
        for lane in LANES:
            if np.random.random() < self.arrival_rate:
                self.queues[lane] = min(self.queues[lane] + 1, self.max_queue)

        self.total_throughput += cleared
        self.step_count += 1
        total_wait = sum(self.queues.values())
        reward = cleared - 0.1 * total_wait

        done = self.step_count >= 500
        info = {"cleared": cleared, "total_wait": total_wait,
                "phase_name": ph["name"]}
        return self._get_obs(), reward, done, info