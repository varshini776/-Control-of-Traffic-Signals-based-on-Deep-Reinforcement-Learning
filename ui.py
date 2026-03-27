import pygame
import sys
import numpy as np

WIDTH, HEIGHT = 600, 600
ROAD_W = 120
CAR_SIZE = 8
INTERSECTION_SIZE = 60

COLORS = {
    "bg": (20, 20, 20),
    "road": (50, 50, 50),
    "lane": (200, 200, 0),
    "car": (0, 200, 255),
    "car_turning": (100, 200, 255),
    "red": (255, 50, 50),
    "green": (50, 255, 50),
    "yellow": (255, 220, 50),
    "white": (255, 255, 255),
    "dark_gray": (40, 40, 40),
}


class TrafficUI:
    def __init__(self, env, policy_name="greedy"):

        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        pygame.display.set_caption("🚦 Advanced Traffic Intersection Simulator")

        self.clock = pygame.time.Clock()

        self.env = env

        self.frame = 0

        self.policy_name = policy_name

        self.paused = False

        if policy_name == "ddqn":
            self.load_ddqn_model()
    # --------------------------------------------------
    # DRAWING INFRASTRUCTURE
    # --------------------------------------------------

    def draw_roads(self):
        """Draw horizontal and vertical roads."""
        c = WIDTH // 2

        # Vertical road
        pygame.draw.rect(
            self.screen, COLORS["road"],
            (c - ROAD_W // 2, 0, ROAD_W, HEIGHT)
        )

        # Horizontal road
        pygame.draw.rect(
            self.screen, COLORS["road"],
            (0, c - ROAD_W // 2, WIDTH, ROAD_W)
        )

    def draw_lane_markings(self):
        """Draw yellow dashed lane markings."""
        c = WIDTH // 2

        # Vertical dashed line
        for i in range(0, HEIGHT, 40):
            pygame.draw.line(
                self.screen, COLORS["lane"],
                (c, i), (c, i + 20), 2
            )

        # Horizontal dashed line
        for i in range(0, WIDTH, 40):
            pygame.draw.line(
                self.screen, COLORS["lane"],
                (i, c), (i + 20, c), 2
            )

    def draw_intersection_box(self):
        """Draw the intersection area."""
        c = WIDTH // 2
        size = INTERSECTION_SIZE
        pygame.draw.rect(
            self.screen, COLORS["dark_gray"],
            (c - size // 2, c - size // 2, size, size), 2
        )

    def draw_individual_vehicles(self, obs):
        """Draw each vehicle individually with color based on lane type."""
        vehicles = self.env.get_active_vehicles()

        for vehicle in vehicles:
            color = COLORS["car_turning"] if vehicle.lane_type == "turn" else COLORS["car"]
            vehicle.draw_on_surface(self.screen, color, CAR_SIZE)

    def draw_traffic_lights(self, obs):
        """Draw 4 traffic lights (one for each direction)."""
        c = WIDTH // 2
        light_dist = 70
        light_radius = 12

        # Light positions for N, S, E, W
        light_positions = {
            "N": (c, c - light_dist),
            "S": (c, c + light_dist),
            "E": (c + light_dist, c),
            "W": (c - light_dist, c),
        }

        for arm, (x, y) in light_positions.items():
            state = self.env.get_light_state(arm)

            if state == "green":
                color = COLORS["green"]
            elif state == "yellow":
                color = COLORS["yellow"]
            else:
                color = COLORS["red"]

            # Draw light circle
            pygame.draw.circle(self.screen, color, (int(x), int(y)), light_radius)

            # Draw border
            pygame.draw.circle(self.screen, COLORS["white"], (int(x), int(y)), light_radius, 2)

            # Draw arm label
            font = pygame.font.SysFont("Arial", 16, bold=True)
            label = font.render(arm, True, COLORS["white"])
            offset = 25
            if arm == "N":
                self.screen.blit(label, (int(x) - 8, int(y) - offset))
            elif arm == "S":
                self.screen.blit(label, (int(x) - 8, int(y) + offset - 15))
            elif arm == "E":
                self.screen.blit(label, (int(x) + offset - 20, int(y) - 8))
            elif arm == "W":
                self.screen.blit(label, (int(x) - offset, int(y) - 8))

    def draw_queue_counts(self, obs):
        """Display queue counts for each arm."""
        font = pygame.font.SysFont("Arial", 22, bold=True)
        c = WIDTH // 2
        offset = 90

        def total(arm):
            return int(obs[arm]["straight"] + obs[arm]["turn"])

        positions = {
            "N": (c - 50, 10),
            "S": (c - 50, HEIGHT - 35),
            "W": (10, c - 40),
            "E": (WIDTH - 120, c - 40),
        }

        for arm, (x, y) in positions.items():
            text = font.render(f"{arm}: {total(arm)}", True, COLORS["white"])
            self.screen.blit(text, (x, y))

    def draw_phase_info(self, obs):
        """Display current phase and phase name."""
        font = pygame.font.SysFont("Arial", 24, bold=True)
        small_font = pygame.font.SysFont("Arial", 18)

        phase_map = {
            0: "NS STRAIGHT",
            1: "NS TURN",
            2: "EW STRAIGHT",
            3: "EW TURN",
            4: "🟡 YELLOW",
        }

        phase = obs["phase"]
        phase_name = phase_map.get(phase, "UNKNOWN")

        # Draw phase display
        text = font.render(f"Phase: {phase_name}", True, COLORS["white"])
        self.screen.blit(text, (20, 20))

        # Draw step counter
        step_text = small_font.render(f"Step: {obs['step']}", True, COLORS["white"])
        self.screen.blit(step_text, (20, 50))

    def draw_policy_display(self):
        """Display current policy name prominently."""
        font = pygame.font.SysFont("Arial", 20, bold=True)
        policy_color = (100, 255, 150) if self.policy_name != "random" else (255, 150, 100)

        policy_text = font.render(f"Policy: {self.policy_name.upper()}", True, policy_color)
        self.screen.blit(policy_text, (WIDTH - 250, 20))

    def draw_metrics(self, obs):
        """Display real-time metrics."""
        font = pygame.font.SysFont("Arial", 16)
        metrics = self.env.metrics()

        y_offset = HEIGHT - 100
        texts = [
            f"Avg Queue: {metrics['avg_queue']:.2f}",
            f"Throughput: {metrics['throughput']:.2f}",
            f"Total Departed: {metrics['departed']}",
        ]

        for i, text_str in enumerate(texts):
            text = font.render(text_str, True, COLORS["white"])
            self.screen.blit(text, (20, y_offset + i * 25))

    def draw_pause_indicator(self):
        """Show if simulation is paused."""
        if self.paused:
            font = pygame.font.SysFont("Arial", 28, bold=True)
            pause_text = font.render("⏸ PAUSED", True, (255, 100, 100))
            text_rect = pause_text.get_rect(center=(WIDTH // 2, HEIGHT - 30))
            self.screen.blit(pause_text, text_rect)

    
    def load_ddqn_model(self):

        import torch
        from dqn_model import DQNNet

        self.model = DQNNet()

        self.model.load_state_dict(torch.load("model.pth"))

        self.model.eval()

    def encode_state(self, obs):

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

    def ddqn_action(self, obs):

        import torch

        state = self.encode_state(obs)

        state = torch.FloatTensor(state)

        with torch.no_grad():

            action = self.model(state).argmax().item()

        return action

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------

    def run(self, policy="greedy"):
        """Run the simulation with the given policy."""
        from baseline import greedy_policy, random_policy, fixed_time_policy

        obs = self.env.reset()
        self.policy_name = policy

        running = True
        while running:
            self.screen.fill(COLORS["bg"])

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        obs = self.env.reset()
                        self.paused = False

            # Step simulation (unless paused)
            if not self.paused:
                # Compute action based on policy
                if policy == "greedy":
                    action = greedy_policy(obs)

                elif policy == "fixed":
                    action = fixed_time_policy(self.env._step_count)

                elif policy == "random":
                    action = random_policy()

                elif policy == "ddqn":
                    action = self.ddqn_action(obs)
                obs, done = self.env.step(action)
                # Continuous environment - no reset on done

            # Draw everything
            self.draw_roads()
            self.draw_lane_markings()
            self.draw_intersection_box()
            self.draw_individual_vehicles(obs)
            self.draw_traffic_lights(obs)
            self.draw_queue_counts(obs)
            self.draw_phase_info(obs)
            self.draw_policy_display()
            self.draw_metrics(obs)
            self.draw_pause_indicator()

            pygame.display.flip()
            self.clock.tick(10)  # 10 FPS

            # Print metrics periodically
            if self.env._step_count % 20 == 0:
                print(self.env.metrics())

            self.frame += 1

        pygame.quit()
        sys.exit()