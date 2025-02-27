import math
import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
from matplotlib.patches import Wedge

# Import the food object.
# (Assumes you have a file food.py that defines a class 'food'
# with attributes: location (a dict with 'x' and 'y') and picked_up (a boolean).)
from food import food

def distance(pos_a, pos_b):
    """Euclidean distance between two positions."""
    return math.sqrt((pos_b['x'] - pos_a['x'])**2 + (pos_b['y'] - pos_a['y'])**2)

def get_food_observations(agent, food_list, sensor_range):
    """
    For the given agent (prey), returns an 8-element list.
    Each element corresponds to a 45° sector around the agent.
    The sensor reading is normalized so that 1 indicates very close food 
    and 0 indicates no food detected within sensor_range.
    """
    observations = [sensor_range] * 8
    sector_angle = 360 / 8  # 45 degrees per sector.
    for f in food_list:
        if f.picked_up:
            continue
        d = distance(agent.location, f.location)
        if d <= sensor_range:
            angle_to_food = math.degrees(math.atan2(
                f.location['y'] - agent.location['y'],
                f.location['x'] - agent.location['x']
            )) % 360
            sector_index = min(7, int(angle_to_food // sector_angle))
            if d < observations[sector_index]:
                observations[sector_index] = d
    # Normalize: a reading of 1 means food is right there (d=0) and 0 means no food detected.
    # normalized_obs = [1.0 - (d / sensor_range) for d in observations]
    return observations

# Define the Predator and Prey classes.
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
    Combined environment:
      - The predator learns to chase the prey.
      - The prey learns to avoid the predator and forage for food.
    """
    metadata = {"render_modes": ["human"], "name": "combined_predator_prey_foraging_env_v1"}
    
    def __init__(self, render_mode=None, size=10, num_steps=100, catch_radius=2, 
                 step_size=1, turn_angle=math.pi/16, prey_speed_factor=1.2, 
                 num_food=45, food_sensor_range=10):
        self.size = size
        self.num_steps = num_steps
        self.catch_radius = catch_radius
        self.step_size = step_size
        self.turn_angle = turn_angle
        self.prey_speed_factor = prey_speed_factor
        self.num_food = num_food
        self.food_sensor_range = food_sensor_range 
        self.step_count = 0
        
        # Initialize predator and prey.
        self.predator = Predator(x=-self.size/2, y=0)
        self.prey = Prey(x=self.size/2, y=0)
        self.agents = [self.predator, self.prey]
        
        # Initialize food objects.
        self.food = [food(self.size) for _ in range(self.num_food)]
        
        self.render_mode = render_mode
        if self.render_mode == "human":
            plt.ion()
            self.figure, self.ax = plt.subplots()
            self.ax.set_xlim(-self.size, self.size)
            self.ax.set_ylim(-self.size, self.size)
            self.predator_scatter = self.ax.scatter([], [], s=100, color='red', label='Predator')
            self.prey_scatter = self.ax.scatter([], [], s=100, color='green', label='Prey')
            self.food_scatter = self.ax.scatter([], [], s=100, color='blue', label='Food')
            self.predator_quiver = self.ax.quiver([], [], [], [], scale=20, color='red')
            self.prey_quiver = self.ax.quiver([], [], [], [], scale=20, color='green')
            self.ax.legend()
            
        self.state = self._get_all_observations()
    
    def observation_space(self, agent):
        if agent.role == 'predator':
            low = np.array([-1, -1, -1, -1, -1], dtype=np.float32)
            high = np.array([1, 1, 1, 1, 1], dtype=np.float32)
            return Box(low=low, high=high, dtype=np.float32)
        elif agent.role == 'prey':
            low = np.concatenate((np.array([-1, -1, -1, -1, -1], dtype=np.float32), 
                                  np.zeros(8, dtype=np.float32)))
            high = np.concatenate((np.array([1, 1, 1, 1, 1], dtype=np.float32), 
                                   np.ones(8, dtype=np.float32)))
            return Box(low=low, high=high, dtype=np.float32)
    
    def action_space(self, agent):
        return Discrete(3)
    
    def _get_observation(self, agent):
        if agent.role == 'predator':
            ax = agent.location['x'] / self.size
            ay = agent.location['y'] / self.size
            px = self.prey.location['x'] / self.size
            py = self.prey.location['y'] / self.size
            angle_to_prey = math.atan2(py - ay, px - ax)
            relative_angle = (angle_to_prey - agent.current_angle + math.pi) % (2 * math.pi) - math.pi
            relative_angle_norm = relative_angle / math.pi
            return np.array([ax, ay, px, py, relative_angle_norm], dtype=np.float32)
        elif agent.role == 'prey':
            ax = agent.location['x'] / self.size
            ay = agent.location['y'] / self.size
            pred_x = self.predator.location['x'] / self.size
            pred_y = self.predator.location['y'] / self.size
            angle_to_predator = math.atan2(pred_y - ay, pred_x - ax)
            relative_angle = (angle_to_predator - agent.current_angle + math.pi) % (2 * math.pi) - math.pi
            relative_angle_norm = relative_angle / math.pi
            predator_obs = np.array([ax, ay, pred_x, pred_y, relative_angle_norm], dtype=np.float32)
            food_obs = np.array(get_food_observations(agent, self.food, self.food_sensor_range), dtype=np.float32)
            return np.concatenate((predator_obs, food_obs))
    
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
        self.food = [food(self.size) for _ in range(self.num_food)]
        self.food_picked_count = 0  # <-- Initialize food pickup counter.
        self.step_count = 0
        self.agents = [self.predator, self.prey]
        self.state = self._get_all_observations()
        infos = {agent: {} for agent in self.agents}
        return self.state, infos
        
    def step(self, actions):
        self.step_count += 1
        prev_distance = distance(self.predator.location, self.prey.location)
        prev_prey_location = self.prey.location.copy()
        
        for agent in self.agents:
            a = np.argmax(actions[agent])
            if a == 0:
                agent.update_angle(self.turn_angle)
            elif a == 1:
                agent.update_angle(-self.turn_angle)
            if agent.role == 'prey':
                agent.update_location(self.step_size * self.prey_speed_factor)
            else:
                agent.update_location(self.step_size)
            agent.location['x'] = max(-self.size, min(self.size, agent.location['x']))
            agent.location['y'] = max(-self.size, min(self.size, agent.location['y']))
        
        prey_reward = 0
        predator_reward = 0
        
        new_distance = distance(self.predator.location, self.prey.location)
        for agent in self.agents:
            if abs(agent.location['x']) == self.size or abs(agent.location['y']) == self.size:
                if agent.role == 'predator':
                    predator_reward = -25
                else:
                    prey_reward = -25
            else:
                if agent.role == 'predator':
                    predator_reward = (prev_distance - new_distance) * 100
                # else:
                    # prey_reward = (new_distance - prev_distance) * 25
                    
               
        
        if new_distance < self.catch_radius:
            prey_reward -= 1000
            predator_reward += 1000
        
        # Reward for reducing distance to food.
        # available_food = [f for f in self.food if not f.picked_up]
        # if available_food:
        #     prev_dists = [distance(prev_prey_location, f.location) for f in available_food]
        #     curr_dists = [distance(self.prey.location, f.location) for f in available_food]
        #     prev_min = min(prev_dists)
        #     curr_min = min(curr_dists)
        #     if curr_min < prev_min:
        #         prey_reward += (prev_min - curr_min)
        
        # Reward for heading toward the closest food.
        # if available_food:
        #     closest_food = min(available_food, key=lambda f: distance(self.prey.location, f.location))
        #     vec_to_food = np.array([
        #         closest_food.location['x'] - self.prey.location['x'],
        #         closest_food.location['y'] - self.prey.location['y']
        #     ], dtype=np.float64)  # Ensure float type.
        #     dist_to_food = np.linalg.norm(vec_to_food)
        #     if dist_to_food > 0:
        #         vec_to_food /= dist_to_food  # Normalize.
        #     heading_vec = np.array([math.cos(self.prey.current_angle), math.sin(self.prey.current_angle)])
        #     alignment = np.dot(vec_to_food, heading_vec)
        #     if alignment > 0:
        #         prey_reward += 50 * alignment

        # Reward for picking up food.
        for f in self.food:
            if not f.picked_up and distance(self.prey.location, f.location) < 1:
                f.picked_up = True
                prey_reward += 200
                self.food_picked_count += 1  # <-- Increment food pickup counter.
        # Remove picked-up food.
        self.food = [f for f in self.food if not f.picked_up]
        
        
        
        # Store current rewards for display in render.
        self.current_predator_reward = predator_reward
        self.current_prey_reward = prey_reward

        rewards = {self.predator: predator_reward, self.prey: prey_reward}
        terminations = {agent: False for agent in self.agents}
        if new_distance < self.catch_radius or self.step_count >= self.num_steps:
            terminations = {agent: True for agent in self.agents}
            self.agents = []
        
        self.state = self._get_all_observations()
        if self.render_mode == "human":
            self.render()
        return self.state, rewards, terminations
    
    def render(self):
        if self.render_mode == "human":
            # Update predator and prey positions.
            pred_x = self.predator.location['x']
            pred_y = self.predator.location['y']
            prey_x = self.prey.location['x']
            prey_y = self.prey.location['y']
            self.predator_scatter.set_offsets(np.c_[[pred_x], [pred_y]])
            self.prey_scatter.set_offsets(np.c_[[prey_x], [prey_y]])
            # Update food positions.
            food_positions = np.array([[f.location['x'], f.location['y']] for f in self.food if not f.picked_up])
            if len(food_positions) > 0:
                self.food_scatter.set_offsets(food_positions)
            else:
                self.food_scatter.set_offsets(np.empty((0, 2)))
            # Update heading quivers.
            self.predator_quiver.set_offsets(np.c_[[pred_x], [pred_y]])
            self.predator_quiver.set_UVC([np.cos(self.predator.current_angle)], [np.sin(self.predator.current_angle)])
            self.prey_quiver.set_offsets(np.c_[[prey_x], [prey_y]])
            self.prey_quiver.set_UVC([np.cos(self.prey.current_angle)], [np.sin(self.prey.current_angle)])
            
            # --- Clear previous drawings ---
            for p in self.ax.patches[:]:
                p.remove()
            for line in self.ax.lines[:]:
                line.remove()
            for txt in self.ax.texts[:]:
                txt.remove()
                
            # Draw prey sensor range circle.
            sensor_circle = plt.Circle((prey_x, prey_y), self.food_sensor_range, color='green', fill=False, linestyle='--', linewidth=1)
            self.ax.add_patch(sensor_circle)
            
            # Draw sensor sectors.
            if self.prey in self.state:
                sensor_values = self.state[self.prey][5:13]
            else:
                sensor_values = [0.0] * 8
            num_sectors = 8
            sector_angle = 2 * math.pi / num_sectors
            for i in range(num_sectors):
                start_angle_deg = i * (360 / num_sectors)
                end_angle_deg = start_angle_deg + (360 / num_sectors)
                if sensor_values[i] < self.food_sensor_range:  # food detected in that sector.
                    wedge = Wedge(center=(prey_x, prey_y), r=self.food_sensor_range,
                                theta1=start_angle_deg, theta2=end_angle_deg,
                                color='yellow', alpha=0.3)
                    self.ax.add_patch(wedge)
                angle = i * sector_angle
                x_end = prey_x + self.food_sensor_range * math.cos(angle)
                y_end = prey_y + self.food_sensor_range * math.sin(angle)
                self.ax.plot([prey_x, x_end], [prey_y, y_end],
                            color='green', linestyle='--', linewidth=1)
            for i, sensor in enumerate(sensor_values):
                mid_angle = (i + 0.5) * sector_angle
                text_radius = self.food_sensor_range * 0.7
                text_x = prey_x + text_radius * math.cos(mid_angle)
                text_y = prey_y + text_radius * math.sin(mid_angle)
                self.ax.text(text_x, text_y, f"{sensor:.2f}", color="black", fontsize=8,
                            ha="center", va="center")
            
            # Draw line between predator and prey and display distance.
            dist = distance(self.predator.location, self.prey.location)
            self.ax.plot([pred_x, prey_x], [pred_y, prey_y], color='purple', linestyle='-', linewidth=2)
            mid_x = (pred_x + prey_x) / 2
            mid_y = (pred_y + prey_y) / 2
            self.ax.text(mid_x, mid_y, f"{dist:.2f}", color="purple", fontsize=10,
                        ha="center", va="center")
            
            # --- NEW: Display reward and food pickup info ---
            info_text = (
                f"Predator Reward: {self.current_predator_reward:.2f}\n"
                f"Prey Reward: {self.current_prey_reward:.2f}\n"
                f"Food Picked: {self.food_picked_count}"
            )
            # Display in the top-left corner of the plot.
            self.ax.text(0.02, 0.95, info_text, transform=self.ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.8))
            
            plt.draw()
            plt.pause(0.01)

    
    def close(self):
        if self.render_mode == "human":
            plt.ioff()
            plt.close(self.figure)
