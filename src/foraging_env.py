import functools
import gymnasium
from gymnasium.spaces import Box, Discrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from ant import ant
import matplotlib.pyplot as plt
import numpy as np 
import cv2
import os
import re
import math
from food import food

# Default Constants
num_steps = 100
size = 50
num_ants = 1
num_food = 1
range_radius = 10

def distance(pos_a, pos_b):
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((pos_b['x'] - pos_a['x']) ** 2 + (pos_b['y'] - pos_a['y']) ** 2)

def get_observations(a, food_list, sensor_range):
    """
    For the given ant, return an 8-element list.
    Each element is the distance to the closest (unpicked) food in that 45° sector.
    If no food is detected in a sector, the value defaults to sensor_range.
    """
    # Initialize all eight sensor readings to the maximum range.
    observations = [sensor_range] * 8  
    section_angle = 360 / 8  
    for f in food_list:
        if f.picked_up:
            continue
        dist = distance(a.location, f.location)
        if dist <= sensor_range:
            angle_to_food = math.degrees(math.atan2(
                f.location['y'] - a.location['y'],
                f.location['x'] - a.location['x']
            )) % 360
            adjusted_angle = (angle_to_food - math.degrees(a.current_angle)) % 360
            section_index = min(7, int(adjusted_angle // section_angle))
            if dist < observations[section_index]:
                observations[section_index] = dist
    # Normalize: 1 means food is right there; 0 means no food detected.
    normalized_obs = [1.0 - (d / sensor_range) for d in observations]
    return normalized_obs

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "foraging_env_v2"}

    def __init__(self, render_mode=None, num_ants=1, num_food=1, size=50, num_steps=100, range_radius=10):
        """
        Initialize the environment with ants and food.
        """
        self.num_food = num_food
        self.size = size
        self.num_steps = num_steps
        self.ant_range_radius = range_radius
        
        self.ants = [ant() for _ in range(num_ants)]
        self.food = [food(self.size) for _ in range(num_food)]
        self.step_count = 0
        
        self.ant_name_mapping = dict(zip(self.ants, list(range(len(self.ants)))))
        self.render_mode = render_mode
        if self.render_mode == 'human':
            plt.ion() 
            self.figure, self.ax = plt.subplots()
            self.ant_scatter = self.ax.scatter([], [], s=100)
            self.food_scatter = self.ax.scatter([], [], s=100, color='green')
            self.quiver = self.ax.quiver([], [], [], [], scale=50, color='red')
            self.ax.set_xlim(-self.size, self.size)
            self.ax.set_ylim(-self.size, self.size)

    def observation_space(self, agent):
        # Eight continuous sensor readings, each between 0 and sensor_range.
        low = np.zeros(8, dtype=np.float32)
        high = np.ones(8, dtype=np.float32) * self.ant_range_radius
        return Box(low=low, high=high, dtype=np.float32)

    def action_space(self, agent):
        # Four discrete actions: 0: turn left, 1: turn right, 2: move forward, 3: move backward.
        return Discrete(4)

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn("Render called without a render mode.")
            return
        elif self.render_mode == "human":
            x = [a.location['x'] for a in self.ants]
            y = [a.location['y'] for a in self.ants]
            # Use current_angle (in radians) directly for computing direction.
            u = [np.cos(a.current_angle) for a in self.ants] 
            v = [np.sin(a.current_angle) for a in self.ants]
            food_x = [f.location['x'] for f in self.food if not f.picked_up]
            food_y = [f.location['y'] for f in self.food if not f.picked_up]
            self.ant_scatter.set_offsets(np.c_[x, y])
            self.quiver.set_offsets(np.c_[x, y])
            self.quiver.set_UVC(u, v)
            self.food_scatter.set_offsets(np.c_[food_x, food_y])
            plt.draw()
            plt.pause(0.1)
        elif self.render_mode == "video":
            pass
   
    def close(self):
        if self.render_mode == "human":
            plt.ioff()  
            plt.close(self.figure)

    def reset(self, seed=None, options=None):
        """
        Reinitialize ants and food, along with logging variables.
        """
        self.ants = [ant() for _ in range(num_ants)]
        self.food = [food(self.size) for _ in range(self.num_food)]
        self.step_count = 0
        self.food_location_log = {f: [f.location.copy()] for f in self.food}
        self.ant_location_log = {a: [a.location.copy()] for a in self.ants}
        self.ant_angle_log = {a: [a.current_angle] for a in self.ants}

        observations = {a: get_observations(a, self.food, self.ant_range_radius) for a in self.ants}
        infos = {a: {} for a in self.ants}
        self.state = observations

        return observations, infos

    def step(self, actions):
        """
        Process one step:
          - Update ant positions based on chosen actions.
          - Provide shaping rewards if ants move closer to food.
          - Give a bonus reward when food is picked up.
          - Terminate episode after a fixed number of steps.
        """
        self.step_count += 1
        env_truncation = self.step_count >= self.num_steps
        if not actions:
            return {}, {}, {}

        terminations = {ant: False for ant in self.ants}
        rewards = {a: 0 for a in self.ants}

        ant_previous_location = {a: a.location.copy() for a in self.ants}

        # Process actions for each ant.
        for a in self.ants:
            self.ant_location_log[a].append(a.location.copy())
            self.ant_angle_log[a].append(a.current_angle)
            action = np.argmax(actions[a])
            if action == 0:
                a.update_angle(np.pi / 4)
            elif action == 1:
                a.update_angle(-np.pi / 4)
            elif action == 2:
                a.update_location(1)
                if not (-self.size <= a.location['x'] <= self.size and -self.size <= a.location['y'] <= self.size):
                    a.update_location(-1)

            # take out backwards movement
            # elif action == 3:
            #     a.update_location(-1)
            #     if not (-self.size <= a.location['x'] <= self.size and -self.size <= a.location['y'] <= self.size):
            #         a.update_location(1)

        # Shaping reward: encourage movement toward the nearest available food.
        for a in self.ants:
            available_food = [f for f in self.food if not f.picked_up]
            if available_food:
                prev_dists = [distance(ant_previous_location[a], f.location) for f in available_food]
                curr_dists = [distance(a.location, f.location) for f in available_food]
                prev_min = min(prev_dists)
                curr_min = min(curr_dists)
                # Reward the ant for reducing its distance to food.
                if curr_min < prev_min:
                    rewards[a] += (prev_min - curr_min)

        # Check if any ant picks up food (if within distance < 1) and assign bonus reward.
        for f in self.food:
            if not f.picked_up:
                for a in self.ants:
                    if distance(a.location, f.location) < 1:
                        f.picked_up = True
                        rewards[a] += 100
                self.food_location_log[f].append(f.location.copy())

        observations = {a: get_observations(a, self.food, self.ant_range_radius) for a in self.ants}
        self.state = observations

        if env_truncation:
            if self.render_mode == "video":
                self.make_video()
            self.ants = None

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations

    def make_video(self):
        plt.ioff()
        if self.render_mode == "video":
            frames_dir = "frames"
            os.makedirs(frames_dir, exist_ok=True)
            # Assuming all ants have the same number of logged steps.
            num_frames = len(next(iter(self.ant_location_log.values())))
            for i in range(num_frames):
                fig, ax = plt.subplots()
                for a in self.ants:
                    circle = plt.Circle((self.ant_location_log[a][i]['x'], self.ant_location_log[a][i]['y']),
                                        self.ant_range_radius, color='blue', fill=False, linestyle='--')
                    ax.add_artist(circle)
                    for section in range(8):
                        angle = section * np.pi/4 + self.ant_angle_log[a][i]
                        x_end = self.ant_location_log[a][i]['x'] + self.ant_range_radius * np.cos(angle)
                        y_end = self.ant_location_log[a][i]['y'] + self.ant_range_radius * np.sin(angle)
                        ax.plot([self.ant_location_log[a][i]['x'], x_end],
                                [self.ant_location_log[a][i]['y'], y_end],
                                color='orange', linestyle='--')
                    ax.scatter(self.ant_location_log[a][i]['x'], self.ant_location_log[a][i]['y'])
                    angle_rad = self.ant_angle_log[a][i]
                    ax.quiver(
                        self.ant_location_log[a][i]['x'],
                        self.ant_location_log[a][i]['y'],
                        np.cos(angle_rad),
                        np.sin(angle_rad),
                        scale=20,
                        color='red'
                    )
                for f in self.food:
                    if i < len(self.food_location_log[f]):
                        ax.scatter(self.food_location_log[f][i]['x'], self.food_location_log[f][i]['y'], color='blue')
                ax.set_xlim(-self.size, self.size)
                ax.set_ylim(-self.size, self.size)
                plt.savefig(f"{frames_dir}/frame_{i:05d}.png")
                plt.close(fig)
            frame_files = sorted(
                os.listdir(frames_dir),
                key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
            )
            frame_paths = [os.path.join(frames_dir, file) for file in frame_files]
            frame = cv2.imread(frame_paths[0])
            height, width, _ = frame.shape
            video_path = "foraging_vid.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                video.write(frame)
            video.release()
            for frame_path in frame_paths:
                os.remove(frame_path)
            os.rmdir(frames_dir)
            return video_path
        else:
            return None
