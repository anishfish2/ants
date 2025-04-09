import functools
import gymnasium
from gymnasium.spaces import Discrete, Box  # Added Box for continuous observations
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
import ant
import matplotlib.pyplot as plt
import numpy as np 
import cv2
import os
import re

# Environment parameters
num_steps = 100
size = 50
num_ants = 1

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "navigation_env_v2"}

    def __init__(self, render_mode=None, num_ants=None, num_food=None, size=None, num_steps=None, range_radius=None):
        """
        Initialize the environment with a list of ants and a rendering mode.
        """
        self.possible_ants = [ant() for _ in range(num_ants)]
        self.ant_name_mapping = dict(
            zip(self.possible_ants, list(range(len(self.possible_ants))))
        )
        self.render_mode = render_mode
        if self.render_mode == 'human':
            plt.ion() 
            self.figure, self.ax = plt.subplots()
            self.scatter = self.ax.scatter([], [], s=100)
            self.quiver = self.ax.quiver([], [], [], [], scale=50, color='red')
            self.ax.set_xlim(-size, size)
            self.ax.set_ylim(-size, size)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, ant):
        """
        Define the observation space as a Box:
          - x and y positions between -size and size
          - current_angle between -pi and pi
        """
        low = np.array([-size, -size, -np.pi], dtype=np.float32)
        high = np.array([size, size, np.pi], dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, ant):
        """
        Define a discrete action space with 4 actions:
          0: turn left (increase angle by pi/4)
          1: turn right (decrease angle by pi/4)
          2: move forward
          3: move backward
        """
        return Discrete(4)

    def render(self):
        """
        Render the current state of the environment.
        In human mode, display a plot showing ant positions and orientations.
        """
        if self.render_mode is None:
            gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return
        elif self.render_mode == "human":
            x = [ant.location['x'] for ant in self.ants]
            y = [ant.location['y'] for ant in self.ants]
            # Since current_angle is already in radians, we use it directly.
            u = [np.cos(ant.current_angle) for ant in self.ants]
            v = [np.sin(ant.current_angle) for ant in self.ants]
            self.scatter.set_offsets(np.c_[x, y])
            self.quiver.set_offsets(np.c_[x, y])
            self.quiver.set_UVC(u, v)
            plt.draw()
            plt.pause(0.1)
        elif self.render_mode == "video":
            pass

    def close(self):
        """
        Close any open rendering windows.
        """
        if self.render_mode == "human":
            plt.ioff()
            plt.close(self.figure)

    def reset(self, seed=None, options=None):
        """
        Reset the environment:
          - Create new ant instances.
          - Initialize location and angle logs.
          - Reset the move counter.
          - Return initial observations and infos.
        """
        self.ants = [ant() for _ in range(num_ants)]
        self.ant_location_log = {ant: [] for ant in self.ants}
        self.ant_angle_log = {ant: [] for ant in self.ants}
        self.num_moves = 0
        observations = {
            self.ants[i]: (self.ants[i].location['x'], self.ants[i].location['y'], self.ants[i].current_angle)
            for i in range(len(self.ants))
        }
        infos = {ant: {} for ant in self.ants}
        self.state = observations
        return observations, infos

    def step(self, actions):
        """
        Process one time-step in the environment given actions for each ant.
        Returns:
          - observations: updated state of each ant.
          - rewards: +1 if the ant moves closer to the target (size, size), else 0.
          - terminations: always False for now.
        """
        if not actions:
            self.ants = []
            return {}, {}, {}, {}

        terminations = {ant: False for ant in self.ants}
        self.num_moves += 1
        # End the episode if we've reached the max steps or the ant is within 1 unit of the target.
        env_truncation = self.num_moves >= num_steps or np.linalg.norm(
            np.array((size, size)) - np.array((self.ants[0].location['x'], self.ants[0].location['y']))
        ) < 1

        ant_previous_location = {ant: ant.location.copy() for ant in self.ants}
        for ant in self.ants:
            self.ant_location_log[ant].append(ant.location.copy())
            self.ant_angle_log[ant].append(ant.current_angle)
            # Use np.argmax so that the network's output maps to one of 4 actions.
            action = np.argmax(actions[ant])
            if action == 0:
                ant.update_angle(np.pi / 4)
            elif action == 1:
                ant.update_angle(-np.pi / 4)
            elif action == 2:
                ant.update_location(1)
                # Ensure the ant remains within bounds.
                if not (-size <= ant.location['x'] <= size and -size <= ant.location['y'] <= size):
                    ant.update_location(-1)
            elif action == 3:
                ant.update_location(-1)
                if not (-size <= ant.location['x'] <= size and -size <= ant.location['y'] <= size):
                    ant.update_location(1)

        observations = {
            self.ants[i]: (self.ants[i].location['x'], self.ants[i].location['y'], self.ants[i].current_angle)
            for i in range(len(self.ants))
        }
        rewards = {
            ant: 1 if np.linalg.norm(
                    np.array((size, size)) - np.array((ant.location['x'], ant.location['y']))
                 ) < np.linalg.norm(
                    np.array((size, size)) - np.array((ant_previous_location[ant]['x'], ant_previous_location[ant]['y']))
                 ) else 0
            for ant in self.ants
        }
        self.state = observations

        if env_truncation:
            if self.render_mode == "video":
                self.make_video()
            self.ants = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations

    def make_video(self):
        if self.render_mode == "video":
            frames_dir = "frames"
            os.makedirs(frames_dir, exist_ok=True)

            for i, _ in enumerate(self.ant_location_log[self.ants[0]]):
                fig, ax = plt.subplots()
                for ant in self.ants:
                    ax.scatter(self.ant_location_log[ant][i]['x'], self.ant_location_log[ant][i]['y'])
                    angle_rad = self.ant_angle_log[ant][i]
                    ax.quiver(
                        self.ant_location_log[ant][i]['x'],
                        self.ant_location_log[ant][i]['y'],
                        np.cos(angle_rad),
                        np.sin(angle_rad),
                        scale=20,
                        color='red'
                    )
                ax.set_xlim(-size, size)
                ax.set_ylim(-size, size)
                plt.savefig(f"{frames_dir}/frame_{i:05d}.png")
                plt.close(fig)

            frame_files = sorted(
                os.listdir(frames_dir),
                key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
            )
            frame_paths = [os.path.join(frames_dir, file) for file in frame_files]
            frame = cv2.imread(frame_paths[0])
            height, width, _ = frame.shape

            video_path = "output.mp4"
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