import math
import random
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class Predator:
    def __init__(self, x, y):
        self.location = {'x': x, 'y': y}
        self.current_angle = 0.0
        self.role = 'predator'
        
    def update_angle(self, delta):
        self.current_angle += delta
        
    def update_location(self, step_size):
        self.location['x'] += step_size * math.cos(self.current_angle)
        self.location['y'] += step_size * math.sin(self.current_angle)

class Prey:
    def __init__(self, x, y):
        self.location = {'x': x, 'y': y}
        self.current_angle = 0.0
        self.role = 'prey'
        
    def update_angle(self, delta):
        self.current_angle += delta
        
    def update_location(self, step_size):
        self.location['x'] += step_size * math.cos(self.current_angle)
        self.location['y'] += step_size * math.sin(self.current_angle)

class parallel_env(ParallelEnv):
    """
    A minimal, learnable predator–prey environment.
    
    - Both agents always move forward by a fixed step.
    - Action space (for each agent):
         0 = no turn, 1 = turn left, 2 = turn right.
    - Reward shaping:
         * Predator is rewarded for reducing the distance to the prey and gets a bonus upon capture.
         * Prey is rewarded for increasing the distance and penalized if caught.
    - Episodes terminate when the predator is within the catch radius of the prey
      or when a maximum number of steps is reached.
    """
    metadata = {"render_modes": ["human"], "name": "learnable_predator_prey_env_v1"}
    
    def __init__(self, render_mode=None, size=50, num_steps=200,
                 catch_radius=2, step_size=1.0, turn_angle=math.pi/12):
        self.size = size
        self.num_steps = num_steps
        self.catch_radius = catch_radius
        self.step_size = step_size
        self.turn_angle = turn_angle  # Approximately 15 degrees.
        self.step_count = 0
        self.prey_speed_factor = 1.2
        # Initialize agents; their positions will be randomized in reset().
        self.predator = Predator(x=-self.size/2, y=0)
        self.prey = Prey(x=self.size/2, y=0)
        self.agents = [self.predator, self.prey]
        
        self.render_mode = render_mode
        if self.render_mode == "human":
            plt.ion()
            self.figure, self.ax = plt.subplots()
            self.ax.set_xlim(-self.size, self.size)
            self.ax.set_ylim(-self.size, self.size)
            # Create scatter objects for predator and prey.
            self.predator_scatter = self.ax.scatter([], [], s=100, color='red', label='Predator')
            self.prey_scatter = self.ax.scatter([], [], s=100, color='green', label='Prey')
            # Create quiver objects to show headings.
            self.predator_quiver = self.ax.quiver([], [], [], [], scale=20, color='red')
            self.prey_quiver = self.ax.quiver([], [], [], [], scale=20, color='green')
            self.ax.legend()
            
        self.state = self._get_all_observations()
    
    def observation_space(self, agent):
        # Observation: [agent_x, agent_y, opponent_x, opponent_y, normalized_angle]
        low = np.array([-1, -1, -1, -1, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)
    
    def action_space(self, agent):
        # Three discrete actions: 0 = no turn, 1 = turn left, 2 = turn right.
        return Discrete(3)
    
    def _get_observation(self, agent):
        opponent = self.prey if agent.role == 'predator' else self.predator
        ax = agent.location['x'] / self.size
        ay = agent.location['y'] / self.size
        ox = opponent.location['x'] / self.size
        oy = opponent.location['y'] / self.size
        norm_angle = (agent.current_angle % (2 * math.pi)) / (2 * math.pi)
        return np.array([ax, ay, ox, oy, norm_angle], dtype=np.float32)
    
    def _get_all_observations(self):
        return {agent: self._get_observation(agent) for agent in self.agents}
    
    def reset(self, seed=None, options=None):
        offset = self.size * 0.1
        self.predator.location = {
            'x': -self.size/2 + random.uniform(-offset, offset),
            'y': random.uniform(-offset, offset)
        }
        self.predator.current_angle = random.uniform(-0.1, 0.1)
        
        self.prey.location = {
            'x': self.size/2 + random.uniform(-offset, offset),
            'y': random.uniform(-offset, offset)
        }
        self.prey.current_angle = random.uniform(0, 2 * math.pi)

        
        self.step_count = 0
        self.agents = [self.predator, self.prey]
        self.state = self._get_all_observations()
        infos = {agent: {} for agent in self.agents}
        return self.state, infos
    
    def step(self, actions):
        self.step_count += 1

        # Record previous distance for reward shaping.
        prev_distance = math.hypot(
            self.predator.location['x'] - self.prey.location['x'],
            self.predator.location['y'] - self.prey.location['y']
        )
        
        # Process actions for each agent.
        for agent in self.agents:
            a = np.argmax(actions[agent])
            if a == 1:
                agent.update_angle(self.turn_angle)
            elif a == 2:
                agent.update_angle(-self.turn_angle)
            # Use different step sizes: prey is faster.
            if agent.role == 'prey':
                agent.update_location(self.step_size * self.prey_speed_factor)
            else:
                agent.update_location(self.step_size)
            
            # Clamp positions.
            agent.location['x'] = max(-self.size, min(self.size, agent.location['x']))
            agent.location['y'] = max(-self.size, min(self.size, agent.location['y']))
        
        # Compute new distance after movement.
        new_distance = math.hypot(
            self.predator.location['x'] - self.prey.location['x'],
            self.predator.location['y'] - self.prey.location['y']
        )
        
        # Reward shaping.
        predator_reward = (prev_distance - new_distance) * 10
        if new_distance < self.catch_radius:
            predator_reward += 100
        prey_reward = (new_distance - prev_distance) * 10
        if new_distance < self.catch_radius:
            prey_reward -= 100
        else:
            # Reduced survival bonus for staying alive.
            prey_reward += 0.1

        wall_threshold = self.size * 0.2  # 20% of the arena size.
        max_wall_penalty = 150  # Maximum penalty.
        for agent in self.agents:
            dist_left = agent.location['x'] + self.size
            dist_right = self.size - agent.location['x']
            dist_bottom = agent.location['y'] + self.size
            dist_top = self.size - agent.location['y']
            min_distance_to_wall = min(dist_left, dist_right, dist_bottom, dist_top)
            if min_distance_to_wall < wall_threshold:
                penalty = -max_wall_penalty * (1 - (min_distance_to_wall / wall_threshold))
                if agent.role == 'predator':
                    predator_reward += penalty
                else:
                    prey_reward += penalty
                # Optionally terminate if an agent touches the wall exactly.
                if min_distance_to_wall == 0:
                    terminations = {agent: True for agent in self.agents}
                    self.agents = []
                    break
        
        rewards = {self.predator: predator_reward, self.prey: prey_reward}
        terminations = {agent: False for agent in self.agents}
        if new_distance < self.catch_radius or self.step_count >= self.num_steps:
            terminations = {agent: True for agent in self.agents}
            self.agents = []  # End episode.
        
        self.state = self._get_all_observations()

        if self.render_mode == "human":
            self.render()

        return self.state, rewards, terminations


    
    def render(self):
        if self.render_mode == "human":
            # Get current positions.
            x_pred = [self.predator.location['x']]
            y_pred = [self.predator.location['y']]
            x_prey = [self.prey.location['x']]
            y_prey = [self.prey.location['y']]
            
            # Update scatter plot positions.
            self.predator_scatter.set_offsets(np.c_[x_pred, y_pred])
            self.prey_scatter.set_offsets(np.c_[x_prey, y_prey])
            
            # Compute heading vectors.
            u_pred = [np.cos(self.predator.current_angle)]
            v_pred = [np.sin(self.predator.current_angle)]
            u_prey = [np.cos(self.prey.current_angle)]
            v_prey = [np.sin(self.prey.current_angle)]
            
            self.predator_quiver.set_offsets(np.c_[x_pred, y_pred])
            self.predator_quiver.set_UVC(u_pred, v_pred)
            self.prey_quiver.set_offsets(np.c_[x_prey, y_prey])
            self.prey_quiver.set_UVC(u_prey, v_prey)
            plt.draw()
            plt.pause(0.05)
    
    def close(self):
        if self.render_mode == "human":
            plt.ioff()
            plt.close(self.figure)
