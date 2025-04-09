import math
import random
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from food import food

class Agent:
    def __init__(self, x, y, agent_id):
        self.location = {'x': x, 'y': y}
        self.current_angle = 0.0
        self.agent_id = agent_id
        self.food_count = 0
        
    def update_angle(self, delta):
        self.current_angle += delta
        
    def update_location(self, step_size):
        self.location['x'] += step_size * math.cos(self.current_angle)
        self.location['y'] += step_size * math.sin(self.current_angle)

class parallel_env(ParallelEnv):
    """
    A competition environment where two agents compete for food.
    
    - Both agents move forward by a fixed step.
    - Action space (for each agent):
         0 = no turn, 1 = turn left, 2 = turn right.
    - Reward shaping:
         * Agents are rewarded for collecting food.
         * The agent with more food can eat the other agent.
         * The agent that eats the other gets a bonus.
    - Episodes terminate when one agent eats the other
      or when a maximum number of steps is reached.
    """
    metadata = {"render_modes": ["human", "video"], "name": "competition_env_v1"}
    
    def __init__(self, render_mode=None, size=50, num_steps=200,
                 catch_radius=2, step_size=1.0, turn_angle=math.pi/12,
                 num_food=20):
        self.size = size
        self.num_steps = num_steps
        self.catch_radius = catch_radius
        self.step_size = step_size
        self.turn_angle = turn_angle  # Approximately 15 degrees.
        self.num_food = num_food
        self.step_count = 0
        
        # Initialize agents; their positions will be randomized in reset().
        self.agent1 = Agent(x=-self.size/2, y=0, agent_id="agent1")
        self.agent2 = Agent(x=self.size/2, y=0, agent_id="agent2")
        self.agents = [self.agent1, self.agent2]
        
        # Initialize food
        self.food_items = []
        
        # For video rendering
        self.frames = []
        
        self.render_mode = render_mode
        if self.render_mode == "human":
            plt.ion()
            self.figure, self.ax = plt.subplots()
            self.ax.set_xlim(-self.size, self.size)
            self.ax.set_ylim(-self.size, self.size)
            # Create scatter objects for agents
            self.agent1_scatter = self.ax.scatter([], [], s=100, color='blue', label='Agent 1')
            self.agent2_scatter = self.ax.scatter([], [], s=100, color='red', label='Agent 2')
            # Create quiver objects to show headings
            self.agent1_quiver = self.ax.quiver([], [], [], [], scale=20, color='blue')
            self.agent2_quiver = self.ax.quiver([], [], [], [], scale=20, color='red')
            # Create scatter for food
            self.food_scatter = self.ax.scatter([], [], s=50, color='green', label='Food')
            self.ax.legend()
            
        self.state = self._get_all_observations()
    
    def observation_space(self, agent):
        # Observation: [agent_x, agent_y, opponent_x, opponent_y, relative_angle, agent_food_count, opponent_food_count]
        # relative_angle is normalized to [-1, 1].
        low = np.array([-1, -1, -1, -1, -1, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 10, 10], dtype=np.float32)
        return Box(low=low, high=high, dtype=np.float32)

    
    def action_space(self, agent):
        # Three discrete actions: 0 = no turn, 1 = turn left, 2 = turn right.
        return Discrete(3)
    
    def _get_observation(self, agent):
        opponent = self.agent2 if agent.agent_id == "agent1" else self.agent1
        # Normalize positions by the size of the environment.
        ax = agent.location['x'] / self.size
        ay = agent.location['y'] / self.size
        ox = opponent.location['x'] / self.size
        oy = opponent.location['y'] / self.size
        
        # Calculate the angle from the agent to its opponent.
        angle_to_opponent = math.atan2(oy - ay, ox - ax)
        # Compute the relative angle between the direction to the opponent and the agent's current heading.
        relative_angle = angle_to_opponent - agent.current_angle
        # Wrap the angle to be within [-pi, pi].
        relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
        # Normalize to [-1, 1].
        relative_angle_norm = relative_angle / math.pi

        # Add food counts to the observation
        agent_food = agent.food_count
        opponent_food = opponent.food_count

        return np.array([ax, ay, ox, oy, relative_angle_norm, agent_food, opponent_food], dtype=np.float32)

    
    def _get_all_observations(self):
        return {agent: self._get_observation(agent) for agent in self.agents}
    
    def reset(self, seed=None, options=None):
        offset = self.size * 0.1
        self.agent1.location = {
            'x': -self.size/2 + random.uniform(-offset, offset),
            'y': random.uniform(-offset, offset)
        }
        self.agent1.current_angle = random.uniform(-0.1, 0.1)
        self.agent1.food_count = 0
        
        self.agent2.location = {
            'x': self.size/2 + random.uniform(-offset, offset),
            'y': random.uniform(-offset, offset)
        }
        self.agent2.current_angle = random.uniform(0, 2 * math.pi)
        self.agent2.food_count = 0
        
        # Reset food
        self.food_items = []
        for _ in range(self.num_food):
            self.food_items.append(food(environment_size=self.size))
        
        self.step_count = 0
        self.agents = [self.agent1, self.agent2]
        self.state = self._get_all_observations()
        infos = {agent: {} for agent in self.agents}
        return self.state, infos
    
    def step(self, actions):
        self.step_count += 1

        # Process actions for each agent.
        for agent in self.agents:
            a = np.argmax(actions[agent])
            if a == 1:
                agent.update_angle(self.turn_angle)
            elif a == 2:
                agent.update_angle(-self.turn_angle)
            agent.update_location(self.step_size)
            
            # Clamp positions.
            agent.location['x'] = max(-self.size, min(self.size, agent.location['x']))
            agent.location['y'] = max(-self.size, min(self.size, agent.location['y']))

        # Check for food collection
        for agent in self.agents:
            for food_item in self.food_items[:]:  # Use a copy of the list to avoid modification during iteration
                if not food_item.picked_up:
                    distance = math.hypot(
                        agent.location['x'] - food_item.location['x'],
                        agent.location['y'] - food_item.location['y']
                    )
                    if distance < self.catch_radius:
                        food_item.picked_up = True
                        agent.food_count += 1
                        self.food_items.remove(food_item)

        # Check for agent interaction (eating)
        distance_between_agents = math.hypot(
            self.agent1.location['x'] - self.agent2.location['x'],
            self.agent1.location['y'] - self.agent2.location['y']
        )
        
        # Initialize rewards
        agent1_reward = 0
        agent2_reward = 0
        
        # If agents are close enough, the one with more food can eat the other
        if distance_between_agents < self.catch_radius:
            if self.agent1.food_count > self.agent2.food_count:
                # Agent 1 eats Agent 2
                agent1_reward += 100 + self.agent2.food_count  # Bonus for eating + food transfer
                agent2_reward -= 100  # Penalty for being eaten
                self.agents.remove(self.agent2)
            elif self.agent2.food_count > self.agent1.food_count:
                # Agent 2 eats Agent 1
                agent2_reward += 100 + self.agent1.food_count  # Bonus for eating + food transfer
                agent1_reward -= 100  # Penalty for being eaten
                self.agents.remove(self.agent1)
            else:
                # Equal food counts, no eating occurs
                pass
        
        # Add small rewards for having more food than the opponent
        if self.agent1.food_count > self.agent2.food_count:
            agent1_reward += 0.1
        elif self.agent2.food_count > self.agent1.food_count:
            agent2_reward += 0.1
        
        # Penalty for hitting walls
        for agent in self.agents:
            if abs(agent.location['x']) == self.size or abs(agent.location['y']) == self.size:
                if agent.agent_id == "agent1":
                    agent1_reward -= 10
                else:
                    agent2_reward -= 10

        rewards = {self.agent1: agent1_reward, self.agent2: agent2_reward}
        terminations = {agent: False for agent in self.agents}
        
        # End episode if one agent eats the other or max steps reached
        if len(self.agents) < 2 or self.step_count >= self.num_steps:
            terminations = {agent: True for agent in self.agents}
            self.agents = []  # End episode

        self.state = self._get_all_observations()

        if self.render_mode == "human":
            self.render()
        elif self.render_mode == "video":
            self._render_frame()

        return self.state, rewards, terminations
    
    def render(self):
        if self.render_mode is None:
            return
        
        if self.render_mode == "human":
            # Update agent positions
            self.agent1_scatter.set_offsets(np.c_[[self.agent1.location['x']], [self.agent1.location['y']]])
            self.agent2_scatter.set_offsets(np.c_[[self.agent2.location['x']], [self.agent2.location['y']]])
            
            # Update agent headings
            self.agent1_quiver.set_offsets(np.c_[[self.agent1.location['x']], [self.agent1.location['y']]])
            self.agent1_quiver.set_UVC([np.cos(self.agent1.current_angle)], [np.sin(self.agent1.current_angle)])
            
            self.agent2_quiver.set_offsets(np.c_[[self.agent2.location['x']], [self.agent2.location['y']]])
            self.agent2_quiver.set_UVC([np.cos(self.agent2.current_angle)], [np.sin(self.agent2.current_angle)])
            
            # Update food positions
            food_x = [f.location['x'] for f in self.food_items]
            food_y = [f.location['y'] for f in self.food_items]
            self.food_scatter.set_offsets(np.c_[food_x, food_y])
            
            # Update labels with food counts
            self.agent1_scatter.set_label(f'Agent 1 (Food: {self.agent1.food_count})')
            self.agent2_scatter.set_label(f'Agent 2 (Food: {self.agent2.food_count})')
            self.ax.legend()
            
            # Draw the plot
            plt.draw()
            plt.pause(0.01)
    
    def _render_frame(self):
        if self.render_mode == "video":
            fig, ax = plt.subplots()
            ax.set_xlim(-self.size, self.size)
            ax.set_ylim(-self.size, self.size)
            
            # Plot agents
            ax.scatter(self.agent1.location['x'], self.agent1.location['y'], s=100, color='blue', 
                      label=f'Agent 1 (Food: {self.agent1.food_count})')
            ax.scatter(self.agent2.location['x'], self.agent2.location['y'], s=100, color='red', 
                      label=f'Agent 2 (Food: {self.agent2.food_count})')
            
            # Plot agent headings
            ax.quiver(self.agent1.location['x'], self.agent1.location['y'], 
                     np.cos(self.agent1.current_angle), np.sin(self.agent1.current_angle),
                     scale=20, color='blue')
            ax.quiver(self.agent2.location['x'], self.agent2.location['y'], 
                     np.cos(self.agent2.current_angle), np.sin(self.agent2.current_angle),
                     scale=20, color='red')
            
            # Plot food
            food_x = [f.location['x'] for f in self.food_items]
            food_y = [f.location['y'] for f in self.food_items]
            ax.scatter(food_x, food_y, s=50, color='green', label='Food')
            
            ax.legend()
            ax.set_title(f'Step {self.step_count}')
            
            # Convert plot to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            self.frames.append(img)
            plt.close(fig)
    
    def make_video(self):
        if self.render_mode == "video" and self.frames:
            import cv2
            height, width, _ = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = "competition_simulation.mp4"
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
            
            for frame in self.frames:
                out.write(frame)
            
            out.release()
            self.frames = []
            return video_path
        return None
    
    def close(self):
        if self.render_mode == "human":
            plt.ioff()
            plt.close(self.figure) 