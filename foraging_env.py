import functools

import gymnasium
from gymnasium.spaces import Discrete

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
render_mode = "human"
size = 50
num_ants = 1
num_food = 1
range_radius = 10


def distance(ant_pos, food_pos):
            """Calculate the Euclidean distance between the ant and food."""
            return math.sqrt((food_pos['x'] - ant_pos['x']) ** 2 + (food_pos['y'] - ant_pos['y']) ** 2)

def get_observations(ant, food_list, range_limit):
    """Return the closest food distances in 8 directional sections around the ant within a given range."""
    observations = [float('inf')] * 8  
    section_angle = 360 / 8  

    for food in food_list:
        dist = distance(ant.location, food.location)
        
        # Only consider food within the specified range
        if dist <= range_limit and not food.picked_up:
            angle_to_food = math.degrees(math.atan2(food.location['y'] - ant.location['y'], food.location['x'] - ant.location['x'])) % 360
            
            adjusted_angle = (angle_to_food - ant.current_angle) % 360
            
            section_index = int(adjusted_angle // section_angle)  
            
            if dist < observations[section_index]:
                observations[section_index] = dist

    # Replace 'infinity' with None for sections with no food
    observations = [obs for obs in observations]

    return observations

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None, num_ants=1, num_food=1, size=50, num_steps=100, range_radius=10):
        """
        The init method takes in environment arguments and should define the following attributes:
        - ants
        - render_mode
        - food
        - size
        - num_steps
        - ant_range_radius
        """
        self.num_food = num_food
        self.ants = [ant() for _ in range(num_ants)]
        self.food = [food() for _ in range(num_food)]
        self.size = size
        self.step_count = 0
        self.num_steps = num_steps
        self.ant_range_radius = range_radius
        
        self.ant_name_mapping = dict(
            zip(self.ants, list(range(len(self.ants))))
        )
        self.render_mode = render_mode
        if self.render_mode == 'human':
            plt.ion() 
            self.figure, self.ax = plt.subplots()
            self.scatter = self.ax.scatter([], [], s=100)
            self.food_scatter = self.ax.scatter([], [], s=100, color='green')
            self.quiver = self.ax.quiver([], [], [], [], scale=50, color='red')  
            self.ax.set_xlim(-self.size, self.size)
            self.ax.set_ylim(-self.size, self.size)


    @functools.lru_cache(maxsize=None)
    def observation_space(self, ant):
        return Discrete(8)

    @functools.lru_cache(maxsize=None)
    def action_space(self, ant):
        return Discrete(8)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return
        elif self.render_mode == "human":
            x = [ant.location['x'] for ant in self.ants]
            y = [ant.location['y'] for ant in self.ants]
            u = [np.cos(np.deg2rad(ant.current_angle)) for ant in self.ants] 
            v = [np.sin(np.deg2rad(ant.current_angle)) for ant in self.ants]
            z_1 = [food.location['x'] for food in self.food]
            z_2 = [food.location['y'] for food in self.food]
            self.scatter.set_offsets(np.c_[x, y])
            self.quiver.set_offsets(np.c_[x, y])
            self.quiver.set_UVC(u, v)  
            self.food_scatter.set_offsets(np.c_[z_1, z_2])
            plt.draw()
            plt.pause(0.1)
        elif self.render_mode == "video":
            pass
   
    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        if self.render_mode == "human":
            plt.ioff()  
            plt.close(self.figure)

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `ants` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each ant
        """
        self.ants = [ant() for _ in range(num_ants)]
        self.food = [] 
        self.food_location_log = {food: [food.location.copy()] for food in self.food}
        self.ant_location_log = {ant: [ant.location] for ant in self.ants}
        self.ant_angle_log = {ant: [ant.current_angle] for ant in self.ants}
        self.step_count = 0

        observations = {
            self.ants[i]: get_observations(self.ants[i], self.food, self.ant_range_radius) for i in range(len(self.ants))
        } 
        infos = {ant: {} for ant in self.ants}
        self.state = observations

        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each ant and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {ant_1: item_1, ant_2: item_2}
        """

        self.step_count += 1

        if self.step_count >= self.num_steps:
            self.ants = None
            return {}, {}, {}
        
        if not actions:
            self.ants = []
            return {}, {}, {}

        terminations = {ant: False for ant in self.ants}

        # Handle the actions of each ant
        ant_previous_location = {ant: ant.location.copy() for ant in self.ants}
        for ant in self.ants:
            self.ant_location_log[ant].append(ant.location.copy())
            self.ant_angle_log[ant].append(ant.current_angle) 
            action = np.argmax(actions[ant])
            if action == 0:
                ant.update_angle(np.pi / 4)
            elif action == 1:
                ant.update_angle(-np.pi / 4)
            elif action == 2:
                ant.update_location(1)
                if ant.location['x'] <= size and ant.location['y'] <= size and ant.location['x'] >= -size and ant.location['y'] >= -size:
                    continue
                else:
                    ant.update_location(-1)
            elif action == 3:
                ant.update_location(-1)
                if ant.location['x'] <= size and ant.location['y'] <= size and ant.location['x'] >= -size and ant.location['y'] >= -size:
                    continue
                else:
                    ant.update_location(1)

        observations = {
            self.ants[i]: get_observations(self.ants[i], self.food, self.ant_range_radius) for i in range(len(self.ants))
        }
        self.state = observations

        rewards = {ant: 0 for ant in self.ants}

        # Check if the ant has picked up food
        for food in self.food:
            if not food.picked_up:
                for ant in self.ants:
                    if np.linalg.norm(np.array((food.location['x'], food.location['y'])) - np.array((ant.location['x'], ant.location['y']))) < 1:
                        food.picked_up = True
                        rewards[ant] = 100
                self.food_location_log[food].append(food.location.copy())

        
        # self.state = observations
        env_truncation = self.num_moves >= num_steps

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
            for i, frame in enumerate(self.ant_location_log[self.ants[0]]):
                fig, ax = plt.subplots()
                
                for ant in self.ants:
                    # Draw the circle for each ant's range
                    circle = plt.Circle((self.ant_location_log[ant][i]['x'], self.ant_location_log[ant][i]['y']),
                                        range_radius, color='blue', fill=False, linestyle='--')
                    ax.add_artist(circle)

                    # Draw the section lines for each ant
                    for section in range(8):
                        angle = section * np.pi/4 + self.ant_angle_log[ant][i]  # Adjust based on ant's current angle
                        x_end = self.ant_location_log[ant][i]['x'] + range_radius * np.cos(angle)
                        y_end = self.ant_location_log[ant][i]['y'] + range_radius * np.sin(angle)
                        ax.plot([self.ant_location_log[ant][i]['x'], x_end],
                                [self.ant_location_log[ant][i]['y'], y_end],
                                color='orange', linestyle='--')
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
                for food in self.food:
                    if i < len(self.food_location_log[food]):
                        ax.scatter(self.food_location_log[food][i]['x'], self.food_location_log[food][i]['y'], color='blue')
                ax.set_xlim(-size, size)
                ax.set_ylim(-size, size)
                plt.savefig(f"{frames_dir}/frame_{i:05d}.png")  # Zero-pad frame number
                plt.close(fig)

            # Sort frame files numerically
            frame_files = sorted(
                os.listdir(frames_dir),
                key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
            )
            frame_paths = [os.path.join(frames_dir, file) for file in frame_files]

            # Read the first frame to get video dimensions
            frame = cv2.imread(frame_paths[0])
            height, width, _ = frame.shape

            video_path = "foraging_vid.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                video.write(frame)

            video.release()

            # Remove the frames directory
            for frame_path in frame_paths:
                os.remove(frame_path)
            os.rmdir(frames_dir)

            # Return the video path
            return video_path
        else:
            return None