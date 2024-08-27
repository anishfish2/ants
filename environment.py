import functools

import gymnasium
from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from agent import ant
import matplotlib.pyplot as plt
import numpy as np 


num_steps = 100
render_mode = "human"
size = 50
num_agents = 2

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode
        """

        self.possible_agents = [ant() for _ in range(num_agents)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Discrete(4)

    # Action space should be defined here.
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(8)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        elif self.render_mode == "human":
            for agent in self.possible_agents:
                plt.scatter(agent.location['x'], agent.location['y'])
            plt.xlim(-size, size)
            plt.ylim(-size, size)
            plt.pause(0.1)
            plt.close()
   
    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {agent: None for agent in self.agents}
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

        self.num_moves += 1
        env_truncation = self.num_moves >= num_steps

        for agent in self.agents:
            if actions[agent] == 0:
                agent.update_angle(np.pi / 4)
            elif actions[agent] == 1:
                agent.update_angle(-np.pi / 4)
            elif actions[agent] == 2:
                agent.update_location(1)
            elif actions[agent] == 3:
                agent.update_location(-1)


        # current observation is just the other player's most recent action
        observations = {
            self.agents[i]: (self.agents[i].location, self.agents[i].current_angle) for i in range(len(self.agents))
        }
        rewards = {self.agents[i]: 50 - np.linalg.norm(np.array((size, size)) - np.array((self.agents[i].location['x'], self.agents[i].location['y']))) for i in range(len(self.agents))}
        self.state = observations

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations
