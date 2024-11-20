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


num_steps = 100
render_mode = "human"
size = 50
num_agents = 1

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode
        """

        self.possible_agents = [ant() for _ in range(num_agents)]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
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
            self.scatter.set_offsets(np.c_[x, y])
            self.quiver.set_offsets(np.c_[x, y])  
            self.quiver.set_UVC(u, v)  
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

        self.agent_location_log = {agent: [] for agent in self.agents}
        self.agent_angle_log = {agent: [] for agent in self.agents}

        self.num_moves = 0
        observations = {
            self.agents[i]: (self.agents[i].location['x'], self.agents[i].location['y'], self.agents[i].current_angle) for i in range(len(self.agents))
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
        env_truncation = self.num_moves >= num_steps or np.linalg.norm(np.array((size,size)) - np.array((self.agents[0].location['x'], self.agents[0].location['y']))) < 1

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
            self.agents[i]: (self.agents[i].location['x'], self.agents[i].location['y'], self.agents[i].current_angle) for i in range(len(self.agents))
        }

        rewards = {agent: 1 if np.linalg.norm(np.array((size,size)) - np.array((agent.location['x'], agent.location['y']))) < np.linalg.norm(np.array((size,size)) - np.array((agent_previous_location[agent]['x'], agent_previous_location[agent]['y']))) else 0 for agent in self.agents}

        self.state = observations

        if env_truncation:
            if self.render_mode == "video":
                self.make_video()
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations
    
    def make_video(self):
        if self.render_mode == "video":
            frames_dir = "frames"
            os.makedirs(frames_dir, exist_ok=True)

            for i, frame in enumerate(self.agent_location_log[self.agents[0]]):
                fig, ax = plt.subplots()
                for agent in self.agents:
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

            video_path = "output.mp4"
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