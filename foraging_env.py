import functools

import gymnasium
from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from agent import ant
import matplotlib.pyplot as plt
import numpy as np 
import cv2
import os
import re
import math

num_steps = 100
render_mode = "human"
size = 50
num_agents = 1
num_food = 1
range_radius = 10


def distance(agent_pos, food_pos):
            """Calculate the Euclidean distance between the agent and food."""
            return math.sqrt((food_pos['x'] - agent_pos['x']) ** 2 + (food_pos['y'] - agent_pos['y']) ** 2)

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

class food():
    def __init__(self):
        self.location = {'x': np.random.randint(-size, size), 'y': np.random.randint(-size, size)}
        self.picked_up = False

    def __init__(self, location):
        self.location = location
        self.picked_up = False
    
class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode
        """

        self.possible_agents = [ant() for _ in range(num_agents)]
        self.food = [ food({'x': -34, 'y': 22}),
                    food({'x': 10, 'y': -45}),
                    food({'x': 7, 'y': 31}),
                    food({'x': -50, 'y': 12}),
                    food({'x': 25, 'y': -5}),
                    food({'x': -12, 'y': 48}),
                    food({'x': 39, 'y': -17}),
                    food({'x': -26, 'y': -30}),
                    food({'x': 15, 'y': 15}),
                    food({'x': 44, 'y': 5}),
                    ]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode
        if self.render_mode == 'human':
            plt.ion() 
            self.figure, self.ax = plt.subplots()
            self.scatter = self.ax.scatter([], [], s=100)
            self.food_scatter = self.ax.scatter([], [], s=100, color='green')
            self.quiver = self.ax.quiver([], [], [], [], scale=50, color='red')  
            self.ax.set_xlim(-size, size)
            self.ax.set_ylim(-size, size)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Discrete(8)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
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
            x = [agent.location['x'] for agent in self.agents]
            y = [agent.location['y'] for agent in self.agents]
            u = [np.cos(np.deg2rad(agent.current_angle)) for agent in self.agents] 
            v = [np.sin(np.deg2rad(agent.current_angle)) for agent in self.agents]
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
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = [ant() for _ in range(num_agents)]
        self.food = [ food({'x': -34, 'y': 22}),
                    food({'x': 10, 'y': -45}),
                    food({'x': 7, 'y': 31}),
                    food({'x': -50, 'y': 12}),
                    food({'x': 25, 'y': -5}),
                    food({'x': -12, 'y': 48}),
                    food({'x': 39, 'y': -17}),
                    food({'x': -26, 'y': -30}),
                    food({'x': 15, 'y': 15}),
                    food({'x': 44, 'y': 5}),
                    ] 
        self.food_location_log = {food: [food.location.copy()] for food in self.food}
        self.agent_location_log = {agent: [agent.location] for agent in self.agents}
        self.agent_angle_log = {agent: [agent.current_angle] for agent in self.agents}

        self.num_moves = 0
        observations = {
            self.agents[i]: get_observations(self.agents[i], self.food, range_radius) for i in range(len(self.agents))
        } 
        infos = {agent: {} for agent in self.agents}
        self.state = observations

        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        terminations = {agent: False for agent in self.agents}

        self.num_moves += 1

        # Handle the actions of each agent
        agent_previous_location = {agent: agent.location.copy() for agent in self.agents}
        for agent in self.agents:
            self.agent_location_log[agent].append(agent.location.copy())
            self.agent_angle_log[agent].append(agent.current_angle) 
            action = np.argmax(actions[agent])
            if action == 0:
                agent.update_angle(np.pi / 4)
            elif action == 1:
                agent.update_angle(-np.pi / 4)
            elif action == 2:
                agent.update_location(1)
                if agent.location['x'] <= size and agent.location['y'] <= size and agent.location['x'] >= -size and agent.location['y'] >= -size:
                    continue
                else:
                    agent.update_location(-1)
            elif action == 3:
                agent.update_location(-1)
                if agent.location['x'] <= size and agent.location['y'] <= size and agent.location['x'] >= -size and agent.location['y'] >= -size:
                    continue
                else:
                    agent.update_location(1)

        observations = {
            self.agents[i]: get_observations(self.agents[i], self.food, range_radius) for i in range(len(self.agents))
        }
        self.state = observations

        rewards = {agent: 0 for agent in self.agents}

        # Check if the agent has picked up food
        for food in self.food:
            if not food.picked_up:
                for agent in self.agents:
                    if np.linalg.norm(np.array((food.location['x'], food.location['y'])) - np.array((agent.location['x'], agent.location['y']))) < 1:
                        food.picked_up = True
                        rewards[agent] = 100
                self.food_location_log[food].append(food.location.copy())

        
        # self.state = observations
        env_truncation = self.num_moves >= num_steps

        if env_truncation:
            if self.render_mode == "video":
                self.make_video()
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations
    
    def make_video(self):
        for food in self.food:
            print(f"Food location: {self.food_location_log[food]}")
            print("Food picked up: ", food.picked_up)
        if self.render_mode == "video":
            frames_dir = "frames"
            os.makedirs(frames_dir, exist_ok=True)
            for i, frame in enumerate(self.agent_location_log[self.agents[0]]):
                fig, ax = plt.subplots()
                
                for agent in self.agents:
                    # Draw the circle for each agent's range
                    circle = plt.Circle((self.agent_location_log[agent][i]['x'], self.agent_location_log[agent][i]['y']),
                                        range_radius, color='blue', fill=False, linestyle='--')
                    ax.add_artist(circle)

                    # Draw the section lines for each agent
                    for section in range(8):
                        angle = section * np.pi/4 + self.agent_angle_log[agent][i]  # Adjust based on agent's current angle
                        x_end = self.agent_location_log[agent][i]['x'] + range_radius * np.cos(angle)
                        y_end = self.agent_location_log[agent][i]['y'] + range_radius * np.sin(angle)
                        ax.plot([self.agent_location_log[agent][i]['x'], x_end],
                                [self.agent_location_log[agent][i]['y'], y_end],
                                color='orange', linestyle='--')
                    ax.scatter(self.agent_location_log[agent][i]['x'], self.agent_location_log[agent][i]['y'])
                    angle_rad = self.agent_angle_log[agent][i]
                    ax.quiver(
                        self.agent_location_log[agent][i]['x'],
                        self.agent_location_log[agent][i]['y'],
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