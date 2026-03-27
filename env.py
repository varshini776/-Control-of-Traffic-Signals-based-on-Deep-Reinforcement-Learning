import numpy as np
from env_config import EnvConfig
from vehicle import Vehicle


class TrafficEnv:
    """
    4-phase traffic intersection simulator with individual vehicle tracking.

    PHASES
    -------
    0 → NS STRAIGHT  (N,S straight only)
    1 → NS TURN      (N,S turning only)
    2 → EW STRAIGHT  (E,W straight only)
    3 → EW TURN      (E,W turning only)
    4 → YELLOW (all stop)

    ACTIONS
    -------
    action ∈ {0,1,2,3}
    Each action requests a phase.
    """

    PHASE_NS_STRAIGHT = 0
    PHASE_NS_TURN = 1
    PHASE_EW_STRAIGHT = 2
    PHASE_EW_TURN = 3
    PHASE_YELLOW = 4

    def __init__(self, cfg: EnvConfig, width=800, height=800):
        self.cfg = cfg
        self.width = width
        self.height = height
        self.cx = width // 2
        self.cy = height // 2
        
        np.random.seed(cfg.seed)
        self.reset()

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def reset(self):
        np.random.seed(self.cfg.seed)
        self._init_state()
        return self._get_obs()

    def step(self, action: int):
        self._step_count += 1

        self._arrivals()
        self._update_phase(action)
        self._update_vehicles()
        self._departures()
        self._update_stats()

        done = self._step_count >= self.cfg.max_steps
        return self._get_obs(), done

    def metrics(self):
        steps = max(self._step_count, 1)

        return {
            "avg_queue": round(self._total_queue_sum / steps, 3),
            "throughput": round(self._total_departed / steps, 3),
            "departed": int(self._total_departed),
        }

    def get_active_vehicles(self):
        """Return list of active vehicles."""
        return [v for v in self._vehicles if v.state != "done"]

    # --------------------------------------------------
    # INTERNAL STATE
    # --------------------------------------------------

    def _init_state(self):
        # queues: each arm has straight + turn lanes
        self._q = {
            arm: {"straight": 0.0, "turn": 0.0}
            for arm in ["N", "S", "E", "W"]
        }

        self._current_phase = self.PHASE_NS_STRAIGHT
        self._next_phase = None

        self._yellow_timer = 0
        self._green_held = 0

        self._step_count = 0
        self._total_departed = 0.0
        self._total_queue_sum = 0.0

        # Vehicle tracking
        self._vehicles = []
        self._vehicle_id_counter = 0

    # --------------------------------------------------
    # ARRIVALS
    # --------------------------------------------------

    def _arrivals(self):
        cap = self.cfg.max_queue_capacity
        lam = self.cfg.arrival_rate
        turn_ratio = self.cfg.turn_ratio

        for arm in self._q:
            arrivals = np.random.poisson(lam)

            turning = arrivals * turn_ratio
            straight = arrivals - turning

            # Update queues
            self._q[arm]["turn"] += turning
            self._q[arm]["straight"] += straight

            # Create actual vehicles
            for _ in range(int(straight)):
                self._create_vehicle(arm, "straight")

            for _ in range(int(turning)):
                self._create_vehicle(arm, "turn")

            # Enforce capacity
            total = self._total_queue(arm)
            if total > cap:
                overflow = total - cap
                self._q[arm]["straight"] = max(
                    0, self._q[arm]["straight"] - overflow
                )

    def _create_vehicle(self, arm, lane_type):
        """Spawn a vehicle at the edge based on arm and lane type."""
        spacing = 18
        queue_count = int(self._q[arm][lane_type])

        if arm == "N":
            x = self.cx - (20 if lane_type == "straight" else 10)
            y = 50 + queue_count * spacing
            direction = "N"
        elif arm == "S":
            x = self.cx + (10 if lane_type == "straight" else 20)
            y = self.height - 50 - queue_count * spacing
            direction = "S"
        elif arm == "W":
            x = 50 + queue_count * spacing
            y = self.cy + (10 if lane_type == "straight" else 20)
            direction = "W"
        elif arm == "E":
            x = self.width - 50 - queue_count * spacing
            y = self.cy - (20 if lane_type == "straight" else 10)
            direction = "E"

        vehicle = Vehicle(x, y, direction, lane_type)
        self._vehicles.append(vehicle)
        self._vehicle_id_counter += 1

    # --------------------------------------------------
    # PHASE CONTROL
    # --------------------------------------------------

    def _update_phase(self, action: int):
        """
        Handles:
        - yellow transition
        - minimum green duration
        - switching between 4 phases
        """

        # If in yellow → countdown
        if self._yellow_timer > 0:
            self._yellow_timer -= 1

            if self._yellow_timer == 0:
                self._current_phase = self._next_phase
                self._green_held = 0

            return

        self._green_held += 1

        # If requesting a different phase → switch (after min green)
        if action != self._current_phase:
            if self._green_held >= self.cfg.min_green_steps:
                self._next_phase = action
                self._current_phase = self.PHASE_YELLOW
                self._yellow_timer = self.cfg.yellow_duration
                self._green_held = 0

    # --------------------------------------------------
    # VEHICLE UPDATES
    # --------------------------------------------------

    def _update_vehicles(self):
        """Update all vehicle positions based on current phase."""
        # Determine which arms can move
        can_move_ns_straight = self._current_phase == self.PHASE_NS_STRAIGHT
        can_move_ns_turn = self._current_phase == self.PHASE_NS_TURN
        can_move_ew_straight = self._current_phase == self.PHASE_EW_STRAIGHT
        can_move_ew_turn = self._current_phase == self.PHASE_EW_TURN

        for vehicle in self._vehicles:
            can_move = False

            if vehicle.direction in ("N", "S"):
                if vehicle.lane_type == "straight":
                    can_move = can_move_ns_straight
                else:
                    can_move = can_move_ns_turn
            else:  # E, W
                if vehicle.lane_type == "straight":
                    can_move = can_move_ew_straight
                else:
                    can_move = can_move_ew_turn

            vehicle.update(can_move, self.cx, self.cy, self.width, self.height)

    # --------------------------------------------------
    # DEPARTURES
    # --------------------------------------------------

    def _departures(self):
        sat = self.cfg.saturation_flow

        # 🟡 YELLOW → nobody moves
        if self._current_phase == self.PHASE_YELLOW:
            return

        # Determine allowed movement
        if self._current_phase == self.PHASE_NS_STRAIGHT:
            arms = ("N", "S")
            move_type = "straight"

        elif self._current_phase == self.PHASE_NS_TURN:
            arms = ("N", "S")
            move_type = "turn"

        elif self._current_phase == self.PHASE_EW_STRAIGHT:
            arms = ("E", "W")
            move_type = "straight"

        elif self._current_phase == self.PHASE_EW_TURN:
            arms = ("E", "W")
            move_type = "turn"

        else:
            return

        # Apply departures
        for arm in arms:
            departed = min(self._q[arm][move_type], sat)
            self._q[arm][move_type] -= departed
            self._total_departed += departed

    # --------------------------------------------------
    # STATS
    # --------------------------------------------------

    def _update_stats(self):
        total = sum(self._total_queue(a) for a in ["N", "S", "E", "W"])
        self._total_queue_sum += total

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------

    def _total_queue(self, arm):
        return self._q[arm]["straight"] + self._q[arm]["turn"]

    def _get_obs(self):
        return {
            "N": self._q["N"],
            "S": self._q["S"],
            "E": self._q["E"],
            "W": self._q["W"],
            "phase": self._current_phase,
            "step": self._step_count,
            "totals": {
                "NS": self._total_queue("N") + self._total_queue("S"),
                "EW": self._total_queue("E") + self._total_queue("W"),
            }
        }

    def get_light_state(self, arm):
        """
        Get traffic light state for a specific arm.
        Returns: "red", "yellow", or "green"
        """
        if self._current_phase == self.PHASE_YELLOW:
            return "yellow"

        arm_to_phase = {
            "N": [self.PHASE_NS_STRAIGHT, self.PHASE_NS_TURN],
            "S": [self.PHASE_NS_STRAIGHT, self.PHASE_NS_TURN],
            "E": [self.PHASE_EW_STRAIGHT, self.PHASE_EW_TURN],
            "W": [self.PHASE_EW_STRAIGHT, self.PHASE_EW_TURN],
        }

        if self._current_phase in arm_to_phase[arm]:
            return "green"
        else:
            return "red"