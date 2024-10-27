import functools

import gymnasium
from gymnasium.spaces import Discrete, Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from agent import ant
import matplotlib.pyplot as plt
import numpy as np 
import cv2
import os


num_steps = 25
render_mode = "video"
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

        self.render_mode = render_mode
        if self.render_mode == 'human':
            plt.ion()  # Turn on interactive mode
            self.figure, self.ax = plt.subplots()
            self.scatter = self.ax.scatter([], [], s=100)  # Initialize empty scatter plot
            self.quiver = self.ax.quiver([], [], [], [], scale=50, color='red')  # Initialize empty quiver for direction
            self.ax.set_xlim(-size, size)
            self.ax.set_ylim(-size, size)

    # Observation space should be defined here.
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box()

    # Action space should be defined here.
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)

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
            u = [np.cos(np.deg2rad(agent.current_angle)) for agent in self.agents]  # Calculate the x component of the direction
            v = [np.sin(np.deg2rad(agent.current_angle)) for agent in self.agents]  # Calculate the y component of the direction
            self.scatter.set_offsets(np.c_[x, y])  # Update the positions of agents
            self.quiver.set_offsets(np.c_[x, y])  # Update the base of the arrows
            self.quiver.set_UVC(u, v)  # Update the direction and length of the arrows
            plt.draw()
            plt.pause(0.1)
        elif self.render_mode == "video":
            # Implement video rendering if needed
            pass

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        
        if self.render_mode == "human":
            plt.ioff()  # Turn off interactive mode
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

        self.agent_location_log = {agent: [agent.location] for agent in self.agents}
        self.agent_angle_log = {agent: [agent.current_angle] for agent in self.agents}

        self.num_moves = 0
        observations = {
            self.agents[i]: ([np.linalg.norm(np.array((size, size)) - np.array((self.agents[i].location['x'], self.agents[i].location['y']))), self.agents[i].current_angle, self.agents[i].location['x'], self.agents[i].location['y']]) for i in range(len(self.agents))
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
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}



        terminations = {agent: False for agent in self.agents}
        rewards = {agent: 0 for agent in self.agents}  # Initialize rewards

        self.num_moves += 1
        env_truncation = self.num_moves >= num_steps

        for agent in self.agents:
            old_distance = np.linalg.norm(np.array((size, size)) - np.array((agent.location['x'], agent.location['y'])))

            # Process the actions similar to before
            if actions[agent][0] < .5:
                if agent.current_angle >= 2 * np.pi:
                    agent.current_angle = 0
                agent.update_angle(np.pi / 4)
            if actions[agent][0] >= .5:
                if agent.current_angle <= 0:
                    agent.current_angle = 2 * np.pi
                agent.update_angle(-np.pi / 4)
            if actions[agent][1] < .5:
                if agent.location['x'] <= size and agent.location['y'] <= size:
                    agent.update_location(1)
            if actions[agent][1] >= .5:
                if agent.location['x'] >= -size and agent.location['y'] >= -size:
                    agent.update_location(-1)

            new_distance = np.linalg.norm(np.array((size, size)) - np.array((agent.location['x'], agent.location['y'])))
            # Reward +1 if the agent moved closer to the goal
            if new_distance < old_distance:
                rewards[agent] = 1
            else:
                rewards[agent] = -2  # You can change this to a negative value to penalize moving away
            self.agent_location_log[agent].append(agent.location)
            self.agent_angle_log[agent].append(agent.current_angle)
            if np.linalg.norm(np.array((size, size)) - np.array((agent.location['x'], agent.location['y']))) < 1:
                terminations[agent] = True
                env_truncation = True

        # current observation is just the other player's most recent action
        observations = {
            self.agents[i]: ([np.linalg.norm(np.array((size, size)) - np.array((self.agents[i].location['x'], self.agents[i].location['y']))), self.agents[i].current_angle, self.agents[i].location['x'], self.agents[i].location['y']]) for i in range(len(self.agents))
        }
        self.state = observations

        if env_truncation:
            if self.render_mode == "video":
                self.make_video()
            self.agents = []
            
        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations

    def make_video(self):
        print(self.agent_location_log)
        if self.render_mode == "video":
            frames_dir = "frames"
            os.makedirs(frames_dir, exist_ok=True)

            # Save each frame as an image file
            for i, frame in enumerate(self.agent_location_log[self.agents[0]]):
                fig, ax = plt.subplots()
                for agent in self.agents:
                    # Scatter for agent position
                    ax.scatter(self.agent_location_log[agent][i]['x'], self.agent_location_log[agent][i]['y'])
                    # Adding arrow to represent the direction
                    angle_rad = np.deg2rad(self.agent_angle_log[agent][i])
                    ax.quiver(self.agent_location_log[agent][i]['x'], self.agent_location_log[agent][i]['y'], np.cos(angle_rad), np.sin(angle_rad), scale=20, color='red')
                    
                ax.set_xlim(-size, size)
                ax.set_ylim(-size, size)
                plt.savefig(f"{frames_dir}/frame_{i}.png")
                plt.close(fig)

            # Use OpenCV to stitch the frames into a video
            frame_files = sorted(os.listdir(frames_dir))
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

            # Remove the frames directory
            for frame_path in frame_paths:
                os.remove(frame_path)
            os.rmdir(frames_dir)

            # Return the video path
            return video_path