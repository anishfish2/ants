import functools
import gymnasium
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import re
import math
from ant import ant  # Ensure your ant class has attributes: location, current_angle, cargo, and max_cargo.
from food import food
import shutil

# Default constants
DEFAULT_NUM_STEPS = 100
DEFAULT_SIZE = 50          # Environment extends from -size to size in each dimension.
DEFAULT_NUM_ANTS = 1
DEFAULT_NUM_FOOD = 20      # For example, 20 food items.
DEFAULT_RANGE_RADIUS = 10  # Sensor range for food detection.
DEFAULT_MAX_CARGO = 1      # Agent can carry one food item at a time.

def euclidean_distance(pos_a, pos_b):
    """Calculate Euclidean distance between two positions (each a dict with 'x' and 'y')."""
    return math.sqrt((pos_b['x'] - pos_a['x']) ** 2 + (pos_b['y'] - pos_a['y']) ** 2)

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human", "video"], "name": "home_foraging_env_v1"}

    def __init__(self, render_mode=None, num_ants=DEFAULT_NUM_ANTS, num_food=DEFAULT_NUM_FOOD,
                 size=DEFAULT_SIZE, num_steps=DEFAULT_NUM_STEPS, range_radius=DEFAULT_RANGE_RADIUS,
                 max_cargo=DEFAULT_MAX_CARGO):
        """
        Environment in which the agent must forage by picking up food and then returning it
        to the home base (located at (0,0)). The agent can only carry up to max_cargo (default 1).
        """
        self.num_food = num_food
        self.size = size
        self.num_steps = num_steps
        self.ant_range_radius = range_radius
        self.max_cargo = max_cargo

        # Define home base (fixed at (0,0))
        self.home_base = {'x': 0, 'y': 0}

        # Create agents and food.
        self.ants = [ant() for _ in range(num_ants)]
        for a in self.ants:
            a.max_cargo = max_cargo
            a.cargo = 0  # Start with no food carried.
        self.food = [food(self.size) for _ in range(num_food)]
        self.step_count = 0

        # Logging for video rendering.
        self.ant_location_log = {a: [] for a in self.ants}
        self.ant_angle_log = {a: [] for a in self.ants}
        self.ant_cargo_log = {a: [] for a in self.ants}  # Log the cargo at each timestep.
        self.food_location_log = {f: [] for f in self.food}

        # Attribute for tracking delivered food per episode.
        self.food_delivered_count = 0

        self.render_mode = render_mode
        if self.render_mode == 'human':

            plt.ion()
            self.figure, self.ax = plt.subplots()
            self.ax.set_aspect('equal')

            # Create the ant scatter; initial color will be red.
            self.ant_scatter = self.ax.scatter([], [], s=100, color='red')
            # Create the food scatter.
            self.food_scatter = self.ax.scatter([], [], s=100, color='green')
            # Draw home base in blue.
            self.base_scatter = self.ax.scatter([self.home_base['x']], [self.home_base['y']], s=150, color='blue')
            # Quiver to show the direction the ant is facing.
            self.quiver = self.ax.quiver([], [], [], [], scale=50, color='red')
            self.ax.set_xlim(-self.size, self.size)
            self.ax.set_ylim(-self.size, self.size)
            # Prepare a list for observation range circles.
            self.obs_circles = []

    def observation_space(self, agent):
        # Observation vector has 11 elements:
        # [8 food sensor readings, cargo flag, normalized distance to base, normalized relative angle to base]
        low = np.zeros(11, dtype=np.float32)
        high = np.ones(11, dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)

    def action_space(self, agent):
        # Three actions: 0 = turn left, 1 = turn right, 2 = move forward.
        return Discrete(3)

    def get_observation(self, a):
        """
        Build the observation for an agent:
          - 8 normalized food sensor readings.
          - Cargo flag (0 if not carrying food, 1 if carrying).
          - Normalized distance to home base.
          - Normalized relative angle between agent’s heading and the direction to home.
        """
        sensor_readings = [self.ant_range_radius] * 8
        section_angle = 360 / 8
        for f in self.food:
            if f.picked_up:
                continue
            d = euclidean_distance(a.location, f.location)
            if d <= self.ant_range_radius:
                angle_to_food = math.degrees(math.atan2(
                    f.location['y'] - a.location['y'],
                    f.location['x'] - a.location['x']
                )) % 360
                adjusted_angle = (angle_to_food - math.degrees(a.current_angle)) % 360
                section_index = min(7, int(adjusted_angle // section_angle))
                if d < sensor_readings[section_index]:
                    sensor_readings[section_index] = d
        # Normalize sensor readings: 1 means food is right there; 0 means no food detected.
        normalized_sensors = [1.0 - (d / self.ant_range_radius) for d in sensor_readings]

        # Cargo flag.
        cargo_flag = a.cargo / a.max_cargo

        # Home base information.
        dx = self.home_base['x'] - a.location['x']
        dy = self.home_base['y'] - a.location['y']
        dist_to_base = math.sqrt(dx**2 + dy**2)
        max_dist = 2 * self.size * math.sqrt(2)
        normalized_dist_to_base = dist_to_base / max_dist

        # Relative angle from agent’s current heading to the home base.
        angle_to_base = math.atan2(dy, dx)
        rel_angle = angle_to_base - a.current_angle
        rel_angle = ((rel_angle + math.pi) % (2 * math.pi)) - math.pi
        normalized_rel_angle = (rel_angle + math.pi) / (2 * math.pi)

        observation = normalized_sensors + [cargo_flag, normalized_dist_to_base, normalized_rel_angle]
        return np.array(observation, dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn("Render called without a render mode.")
            return
        elif self.render_mode == "human":
            # Update ant positions and headings.
            ant_x = [a.location['x'] for a in self.ants]
            ant_y = [a.location['y'] for a in self.ants]
            # Change ant color if holding something: orange if cargo > 0, red otherwise.
            ant_colors = ['orange' if a.cargo > 0 else 'red' for a in self.ants]
            self.ant_scatter.set_offsets(np.c_[ant_x, ant_y])
            self.ant_scatter.set_color(ant_colors)
            # Update the quiver for agent headings.
            u = [np.cos(a.current_angle) for a in self.ants]
            v = [np.sin(a.current_angle) for a in self.ants]
            self.quiver.set_offsets(np.c_[ant_x, ant_y])
            self.quiver.set_UVC(u, v)
            # Update food scatter (only show food that hasn't been picked up).
            food_x = [f.location['x'] for f in self.food if not f.picked_up]
            food_y = [f.location['y'] for f in self.food if not f.picked_up]
            self.food_scatter.set_offsets(np.c_[food_x, food_y])
            # Remove previous observation range circles.
            for circ in self.obs_circles:
                circ.remove()
            self.obs_circles = []
            # For each agent, draw a dashed circle representing the observation range.
            for a in self.ants:
                circ = plt.Circle((a.location['x'], a.location['y']), self.ant_range_radius,
                                  color='gray', fill=False, linestyle='--')
                self.obs_circles.append(circ)
                self.ax.add_patch(circ)
            plt.draw()
            plt.pause(0.1)
        elif self.render_mode == "video":
            # Video rendering is handled by make_video().
            pass

    def close(self):
        if self.render_mode == "human":
            plt.ioff()
            plt.close(self.figure)

    def reset(self, seed=None, options=None):
        """
        Reset the environment: reinitialize the agent at the home base and respawn food.
        Also reset the logs and the delivered-food counter.
        """
        self.ants = [ant() for _ in range(len(self.ants))]
        for a in self.ants:
            a.cargo = 0
            a.max_cargo = self.max_cargo
            a.location = {'x': self.home_base['x'], 'y': self.home_base['y']}
            a.current_angle = 0
        self.food = [food(self.size) for _ in range(self.num_food)]
        self.step_count = 0
        self.ant_location_log = {a: [a.location.copy()] for a in self.ants}
        self.ant_angle_log = {a: [a.current_angle] for a in self.ants}
        self.ant_cargo_log = {a: [a.cargo] for a in self.ants}
        self.food_location_log = {f: [f.location.copy()] for f in self.food}
        observations = {a: self.get_observation(a) for a in self.ants}
        infos = {a: {} for a in self.ants}
        self.state = observations
        # Reset the delivered-food counter for the new episode.
        self.food_delivered_count = 0
        return observations, infos

    def step(self, actions):
        """
        Process one timestep:
          - Update agent based on its chosen action.
          - Check for food pickup and delivery.
          - Apply shaping rewards.
          - Log location, angle, and cargo for video recording.
        """
        self.step_count += 1
        env_truncation = self.step_count >= self.num_steps
        if not actions:
            return {}, {}, {}

        for f in self.food:
            self.food_location_log[f].append(f.location.copy())
        terminations = {a: False for a in self.ants}
        rewards = {a: 0 for a in self.ants}
        previous_positions = {a: a.location.copy() for a in self.ants}

        # Process actions for each agent.
        for a in self.ants:
            self.ant_location_log[a].append(a.location.copy())
            self.ant_angle_log[a].append(a.current_angle)
            self.ant_cargo_log[a].append(a.cargo)
            action = np.argmax(actions[a])
            if action == 0:
                a.update_angle(np.pi / 4)  # Turn left.
            elif action == 1:
                a.update_angle(-np.pi / 4)  # Turn right.
            elif action == 2:
                a.update_location(1)  # Move forward.
            # Keep the agent in bounds.
            if not (-self.size <= a.location['x'] <= self.size and -self.size <= a.location['y'] <= self.size):
                a.location = previous_positions[a]

            # Interaction: if not carrying food, try to pick up a food item.
            for f in self.food:
                if a.cargo < a.max_cargo:
                    if not f.picked_up and euclidean_distance(a.location, f.location) < 5:
                        f.picked_up = True
                        a.cargo += 1
                        rewards[a] += 50  # Reward for pickup.

            # If carrying food, check for delivery at home base.
            if a.cargo > 0:
                if euclidean_distance(a.location, self.home_base) < 5:
                    rewards[a] += 200  # Reward for delivery.
                    self.food_delivered_count += 1
                    a.cargo = 0

            shaping_factor = 0.5
            if a.cargo == 0:
                # When not carrying food, reward moving closer to the nearest available food.
                available_food = [f for f in self.food if not f.picked_up]
                if available_food:
                    prev_dist = min(euclidean_distance(previous_positions[a], f.location) for f in available_food)
                    curr_dist = min(euclidean_distance(a.location, f.location) for f in available_food)
                    rewards[a] += (prev_dist - curr_dist) * shaping_factor
            else:
                # When carrying food, reward moving closer to the home base.
                prev_dist = euclidean_distance(previous_positions[a], self.home_base)
                curr_dist = euclidean_distance(a.location, self.home_base)
                rewards[a] += (prev_dist - curr_dist) * shaping_factor
        observations = {a: self.get_observation(a) for a in self.ants}
        self.state = observations

        if env_truncation:
            print('Delivered food:', self.food_delivered_count)
            if self.render_mode == "video":
                self.make_video()
            for a in self.ants:
                terminations[a] = True
            self.ants = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations

    def make_video(self):
        """
        Generate a video from the logged frames.
        For each timestep, draw:
        - The home base (blue) at the center.
        - Each agent: marker (red if not carrying; orange if carrying),
            quiver for heading, and a dashed circle showing the observation range.
        - Food items (green) from their logged positions.
        The resulting video will visually match the human render.
        """
        print('Making video')
        plt.ioff()
        if self.render_mode == "video":
            frames_dir = "frames"

            if os.path.exists(frames_dir):
                try:
                    shutil.rmtree(frames_dir)
                    print(f"Folder '{frames_dir}' deleted successfully.")
                except OSError as e:
                    print(f"Error deleting folder '{frames_dir}': {e}")
            else:
                print(f"Folder '{frames_dir}' does not exist.")

            os.makedirs(frames_dir, exist_ok=False)
            # Assume all ants have the same number of logged steps.
            num_frames = len(next(iter(self.ant_location_log.values())))
            for i in range(num_frames):
                fig, ax = plt.subplots()
                # Draw home base.
                ax.scatter(self.home_base['x'], self.home_base['y'], s=150, color='blue')
                # Draw each agent.
                for a in self.ants:
                    loc = self.ant_location_log[a][i]
                    angle = self.ant_angle_log[a][i]
                    cargo = self.ant_cargo_log[a][i]
                    # Choose color based on cargo (orange if carrying, red otherwise).
                    color = 'orange' if cargo > 0 else 'red'
                    ax.scatter(loc['x'], loc['y'], s=100, color=color)
                    # Draw a quiver for the agent's heading.
                    ax.quiver(loc['x'], loc['y'], np.cos(angle), np.sin(angle),
                              scale=20, color=color)
                    # Draw the observation range as a dashed circle.
                    circ = plt.Circle((loc['x'], loc['y']), self.ant_range_radius,
                                      color='gray', fill=False, linestyle='--')
                    ax.add_patch(circ)
                # Draw food positions.
                for f in self.food:
                    if i < len(self.food_location_log[f]):
                        food_loc = self.food_location_log[f][i]
                        ax.scatter(food_loc['x'], food_loc['y'], s=100, color='green')
                ax.set_xlim(-self.size, self.size)
                ax.set_ylim(-self.size, self.size)
                plt.savefig(f"{frames_dir}/frame_{i:05d}.png")
                plt.close(fig)
            # Create video from frames.
            frame_files = sorted(
                os.listdir(frames_dir),
                key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
            )
            frame_paths = [os.path.join(frames_dir, file) for file in frame_files]
            frame = cv2.imread(frame_paths[0])
            height, width, _ = frame.shape
            video_path = "retrieval_env.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                video.write(frame)
            video.release()
            # Clean up temporary frames.
            for frame_path in frame_paths:
                os.remove(frame_path)
            os.rmdir(frames_dir)
            return video_path
        else:
            return None
