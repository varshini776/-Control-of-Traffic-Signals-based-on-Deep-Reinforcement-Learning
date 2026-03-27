import math


class Vehicle:
    """
    Individual vehicle that follows traffic rules on a 2-way road system.
    
    2-Way Road System:
    - Each road (N/S/E/W) has 2 lanes: INCOMING and OUTGOING
    - INCOMING: vehicles approaching the intersection
    - OUTGOING: vehicles leaving the intersection (after green light)
    
    States:
    - approaching: moving toward intersection on incoming lane
    - waiting: stopped at red light on incoming lane
    - crossing: passing through intersection
    - exiting: leaving intersection on outgoing lane
    - done: completely exited scene
    """

    def __init__(self, x, y, direction, lane_type="straight"):
        """
        Initialize a vehicle.
        
        Args:
            x, y: starting position
            direction: current direction (N, S, E, W) - which road entering from
            lane_type: "straight" or "turn" - indicates vehicle's intended path
        """
        self.x = x
        self.y = y
        self.direction = direction  # Incoming from which direction
        self.lane_type = lane_type  # "straight" or "turn"
        
        # Determine destination based on lane type
        if lane_type == "straight":
            self.destination = direction  # Continue in same direction
        else:
            # Turn vehicles go to the right
            right_turns = {"N": "E", "S": "W", "E": "S", "W": "N"}
            self.destination = right_turns.get(direction, direction)
        
        self.speed = 2
        self.size = 8

        self.state = "approaching"  # approaching, waiting, crossing, exiting, done
        self.turned = False
        self.stopped_distance = 60  # distance at which to stop for red light
        self.crossing_threshold = 30  # distance at which vehicle is in intersection
        self.exit_threshold = 50  # distance from center when moving to outgoing lane

    def distance_to_center(self, cx, cy):
        """Calculate Euclidean distance to intersection center."""
        return math.hypot(self.x - cx, self.y - cy)

    def move_forward(self):
        """Move vehicle one step forward in its current direction."""
        if self.direction == "N":
            self.y += self.speed
        elif self.direction == "S":
            self.y -= self.speed
        elif self.direction == "E":
            self.x -= self.speed
        elif self.direction == "W":
            self.x += self.speed

    def turn_to_destination(self):
        """
        Change direction to match destination.
        Executed when vehicle crosses intersection.
        """
        if self.destination == self.direction:
            # Straight through - no change
            return
        
        # Execute turn (right turn logic)
        right_turns = {"N": "E", "S": "W", "E": "S", "W": "N"}
        left_turns = {"N": "W", "S": "E", "E": "N", "W": "S"}
        
        if self.destination == right_turns.get(self.direction):
            self.direction = self.destination
            self.turned = True
        elif self.destination == left_turns.get(self.direction):
            # Left turn (via right-turn rule in some jurisdictions)
            self.direction = self.destination
            self.turned = True

    def is_out_of_bounds(self, width, height):
        """Check if vehicle has completely exited the scene."""
        return self.x < -50 or self.x > width + 50 or self.y < -50 or self.y > height + 50

    def update(self, can_move, cx, cy, width, height):
        """
        Update vehicle state and position based on traffic signal.
        
        Args:
            can_move: bool - True if traffic signal allows movement for this vehicle
            cx: intersection center x
            cy: intersection center y
            width: scene width
            height: scene height
        
        Returns:
            bool: True if vehicle moved, False otherwise
        """
        dist = self.distance_to_center(cx, cy)

        # Stop at red light (when can_move is False)
        if not can_move:
            # Only stop if approaching intersection and far enough away
            if self.state == "approaching" and dist < self.stopped_distance and dist > self.crossing_threshold:
                self.state = "waiting"
                return False
            # If already crossing, continue through
            elif self.state in ["crossing", "exiting"]:
                pass  # Continue moving
            else:
                return False

        # When traffic signal allows (can_move is True)
        if can_move:
            # Resume from waiting
            if self.state == "waiting":
                self.state = "approaching"
            
            # Transition to crossing when entering intersection
            if self.state == "approaching" and dist < self.crossing_threshold:
                self.state = "crossing"
                if not self.turned:
                    self.turn_to_destination()
            
            # Transition to exiting when passing through intersection
            if self.state == "crossing" and dist > self.exit_threshold:
                self.state = "exiting"

        # Move forward in current direction
        self.move_forward()

        # Check exit condition
        if self.is_out_of_bounds(width, height):
            self.state = "done"
            return False

        return True

    def draw_on_surface(self, surface, color, size):
        """Draw vehicle as a rectangle."""
        import pygame
        pygame.draw.rect(
            surface,
            color,
            (int(self.x), int(self.y), size, size)
        )