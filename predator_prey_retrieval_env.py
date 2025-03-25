import math
import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
from matplotlib.patches import Wedge

# New imports for video rendering.
import os
import re
import cv2

# Import the food object and the new agent classes.
from food import food
from ant import Predator, Prey

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
    normalized_obs = [1.0 - (d / sensor_range) for d in observations]
    return normalized_obs

class parallel_env(ParallelEnv):
    """
    Environment where a predator chases a prey.
    Both predator and prey extend the base ant class.
    Only the prey carries cargo (one food item at a time) and must return
    to a home base to drop off its cargo before collecting more.
    """
    metadata = {"render_modes": ["human", "video"], "name": "predator_prey_foraging_env_v1"}
    
    def __init__(self, render_mode=None, size=10, num_steps=100, catch_radius=2, 
                 step_size=1, turn_angle=math.pi/16, prey_speed_factor=1.5, 
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
        
        # Instantiate predator and prey as subclasses of ant.
        self.predator = Predator()
        self.predator.location = {'x': -self.size/2, 'y': 0}
        self.predator.current_angle = 0.0
        
        self.prey = Prey()
        self.prey.location = {'x': self.size/2, 'y': 0}
        self.prey.current_angle = 0.0
        
        self.agents = [self.predator, self.prey]
        
        # Initialize food objects.
        self.food = [food(self.size) for _ in range(self.num_food)]
        
        # Define a fixed home base for the prey.
        self.home_base = {'x': 0, 'y': 0}
        
        self.render_mode = render_mode
        if self.render_mode == "human":
            plt.ion()
            self.figure, self.ax = plt.subplots()
            self.ax.set_xlim(-self.size, self.size)
            self.ax.set_ylim(-self.size, self.size)
            self.predator_scatter = self.ax.scatter([], [], s=100, color='red', label='Predator')
            self.prey_scatter = self.ax.scatter([], [], s=100, color='green', label='Prey')
            self.food_scatter = self.ax.scatter([], [], s=100, color='blue', label='Food')
            self.home_base_scatter = self.ax.scatter([self.home_base['x']], [self.home_base['y']],
                                                     s=150, marker='*', color='orange', label='Home Base')
            self.predator_quiver = self.ax.quiver([], [], [], [], scale=20, color='red')
            self.prey_quiver = self.ax.quiver([], [], [], [], scale=20, color='green')
            self.ax.legend()
            
        # Initialize logs for video rendering.
        self.predator_log = []
        self.prey_log = []
        self.predator_angle_log = []
        self.prey_angle_log = []
        self.food_log = []
        # New logs for food picked count and rewards.
        self.food_picked_count_log = []
        self.predator_reward_log = []
        self.prey_reward_log = []
        
        self.state = self._get_all_observations()
    
    def observation_space(self, agent):
        if agent.role == 'predator':
            low = np.array([-1, -1, -1, -1, -1], dtype=np.float32)
            high = np.array([1, 1, 1, 1, 1], dtype=np.float32)
            return Box(low=low, high=high, dtype=np.float32)
        elif agent.role == 'prey':
            # The prey's observation consists of its state plus 8 food sensor values.
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
            angle_to_pred = math.atan2(pred_y - ay, pred_x - ax)
            relative_angle = (angle_to_pred - agent.current_angle + math.pi) % (2 * math.pi) - math.pi
            relative_angle_norm = relative_angle / math.pi
            prey_state = np.array([ax, ay, pred_x, pred_y, relative_angle_norm], dtype=np.float32)
            food_obs = np.array(get_food_observations(agent, self.food, self.food_sensor_range), dtype=np.float32)
            return np.concatenate((prey_state, food_obs))
    
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
        self.prey.cargo = 0  # Reset prey cargo.
        
        self.food = [food(self.size) for _ in range(self.num_food)]
        self.food_picked_count = 0
        self.step_count = 0
        self.agents = [self.predator, self.prey]
        self.state = self._get_all_observations()
        
        # Initialize logs for video frames.
        self.predator_log = [self.predator.location.copy()]
        self.prey_log = [self.prey.location.copy()]
        self.predator_angle_log = [self.predator.current_angle]
        self.prey_angle_log = [self.prey.current_angle]
        self.food_log = [[f.location.copy() for f in self.food]]
        self.food_picked_count_log = [self.food_picked_count]
        self.predator_reward_log = [0]
        self.prey_reward_log = [0]
        
        infos = {agent: {} for agent in self.agents}
        return self.state, infos
        
    def step(self, actions):
        self.step_count += 1
        prev_distance = distance(self.predator.location, self.prey.location)
        
        # Update positions and angles.
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
        
        new_distance = distance(self.predator.location, self.prey.location)
        
        # Predator reward: proximity, alignment, and a small time penalty.
        max_possible_distance = math.sqrt((2 * self.size)**2 + (2 * self.size)**2)
        proximity_reward = 15 * (1 - (new_distance / max_possible_distance))
        vec_to_prey = np.array([
            self.prey.location['x'] - self.predator.location['x'],
            self.prey.location['y'] - self.predator.location['y']
        ])
        norm = np.linalg.norm(vec_to_prey)
        vec_to_prey_norm = vec_to_prey / norm if norm > 0 else vec_to_prey
        pred_heading = np.array([math.cos(self.predator.current_angle), math.sin(self.predator.current_angle)])
        alignment = np.dot(pred_heading, vec_to_prey_norm)
        alignment_reward = alignment * 50
        time_penalty = -1
        predator_reward = proximity_reward + alignment_reward + time_penalty
        
        if abs(self.predator.location['x']) == self.size or abs(self.predator.location['y']) == self.size:
            predator_reward = -25
        prey_reward = 0
        if abs(self.prey.location['x']) == self.size or abs(self.prey.location['y']) == self.size:
            prey_reward = -25
        
        if new_distance < self.catch_radius:
            predator_reward += 1000
            prey_reward -= 1000
        
        # Food Pickup (only for prey).
        if self.prey.cargo < self.prey.max_cargo:
            for f in self.food:
                if not f.picked_up and distance(self.prey.location, f.location) < 1:
                    f.picked_up = True
                    self.prey.cargo += 1
                    prey_reward += 100
                    self.food_picked_count += 1
                    break
        # Remove picked-up food.
        self.food = [f for f in self.food if not f.picked_up]
        
        # Incentivize moving toward the home base if carrying food.
        if self.prey.cargo > 0:
            dist_to_home = distance(self.prey.location, self.home_base)
            max_dist = self.size * math.sqrt(2)
            incentive_reward = 30 * (1 - (dist_to_home / max_dist))
            prey_reward += incentive_reward
            #prey_reward -= 0.5
        
        # Drop-off at the home base.
        if self.prey.cargo > 0 and distance(self.prey.location, self.home_base) < .5:
            self.prey.cargo = 0
            prey_reward += 1000
        
        self.current_predator_reward = predator_reward
        self.current_prey_reward = prey_reward

        rewards = {self.predator: predator_reward, self.prey: prey_reward}
        terminations = {agent: False for agent in self.agents}
        
        # Log the current positions, angles, food positions, and new info.
        self.predator_log.append(self.predator.location.copy())
        self.prey_log.append(self.prey.location.copy())
        self.predator_angle_log.append(self.predator.current_angle)
        self.prey_angle_log.append(self.prey.current_angle)
        self.food_log.append([f.location.copy() for f in self.food])
        self.food_picked_count_log.append(self.food_picked_count)
        self.predator_reward_log.append(self.current_predator_reward)
        self.prey_reward_log.append(self.current_prey_reward)
        
        if new_distance < self.catch_radius or self.step_count >= self.num_steps:
            terminations = {agent: True for agent in self.agents}
            if self.render_mode == "video":
                self.make_video()
            self.agents = []
        
        self.state = self._get_all_observations()
        if self.render_mode == "human":
            self.render()
        return self.state, rewards, terminations

    def render(self):
        if self.render_mode == "human":
            pred_x = self.predator.location['x']
            pred_y = self.predator.location['y']
            prey_x = self.prey.location['x']
            prey_y = self.prey.location['y']
            self.predator_scatter.set_offsets(np.c_[[pred_x], [pred_y]])
            self.prey_scatter.set_offsets(np.c_[[prey_x], [prey_y]])
            food_positions = np.array([[f.location['x'], f.location['y']] for f in self.food if not f.picked_up])
            if len(food_positions) > 0:
                self.food_scatter.set_offsets(food_positions)
            else:
                self.food_scatter.set_offsets(np.empty((0, 2)))
            self.predator_quiver.set_offsets(np.c_[[pred_x], [pred_y]])
            self.predator_quiver.set_UVC([np.cos(self.predator.current_angle)], [np.sin(self.predator.current_angle)])
            self.prey_quiver.set_offsets(np.c_[[prey_x], [prey_y]])
            self.prey_quiver.set_UVC([np.cos(self.prey.current_angle)], [np.sin(self.prey.current_angle)])
            
            # Clear additional drawings.
            for p in self.ax.patches[:]:
                p.remove()
            for line in self.ax.lines[:]:
                line.remove()
            for txt in self.ax.texts[:]:
                txt.remove()
                
            sensor_circle = plt.Circle((prey_x, prey_y), self.food_sensor_range, color='green', fill=False, linestyle='--', linewidth=1)
            self.ax.add_patch(sensor_circle)
            
            if self.prey in self.state:
                sensor_values = self.state[self.prey][5:13]
            else:
                sensor_values = [0.0] * 8
            num_sectors = 8
            sector_angle = 2 * math.pi / num_sectors
            for i in range(num_sectors):
                start_angle_deg = i * (360 / num_sectors)
                end_angle_deg = start_angle_deg + (360 / num_sectors)
                if sensor_values[i] < self.food_sensor_range:
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
            
            self.ax.plot([pred_x, prey_x], [pred_y, prey_y], color='purple', linestyle='-', linewidth=2)
            mid_x = (pred_x + prey_x) / 2
            mid_y = (pred_y + prey_y) / 2
            self.ax.text(mid_x, mid_y, f"{distance(self.predator.location, self.prey.location):.2f}",
                         color="purple", fontsize=10, ha="center", va="center")
            
            info_text = (
                f"Predator Reward: {self.current_predator_reward:.2f}\n"
                f"Prey Reward: {self.current_prey_reward:.2f}\n"
                f"Food Picked: {self.food_picked_count}\n"
                f"Cargo: {self.prey.cargo}/{self.prey.max_cargo}"
            )
            self.ax.text(0.02, 0.95, info_text, transform=self.ax.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(facecolor='white', alpha=0.8))
            
            plt.draw()
            plt.pause(0.01)
    
    def make_video(self):
        """
        Creates a video from the logged frames including food count and rewards.
        """
        plt.ioff()
        if self.render_mode == "video":
            frames_dir = "frames"
            os.makedirs(frames_dir, exist_ok=True)
            num_frames = len(self.predator_log)
            for i in range(num_frames):
                fig, ax = plt.subplots()
                # Draw predator.
                pred_loc = self.predator_log[i]
                pred_angle = self.predator_angle_log[i]
                ax.scatter(pred_loc['x'], pred_loc['y'], color='red', s=100, label='Predator')
                ax.quiver(pred_loc['x'], pred_loc['y'], np.cos(pred_angle), np.sin(pred_angle),
                          scale=20, color='red')
                # Draw prey.
                prey_loc = self.prey_log[i]
                prey_angle = self.prey_angle_log[i]
                ax.scatter(prey_loc['x'], prey_loc['y'], color='green', s=100, label='Prey')
                ax.quiver(prey_loc['x'], prey_loc['y'], np.cos(prey_angle), np.sin(prey_angle),
                          scale=20, color='green')
                # Draw food positions logged for this frame.
                for food_pos in self.food_log[i]:
                    ax.scatter(food_pos['x'], food_pos['y'], color='blue', s=100)
                # Draw home base.
                ax.scatter(self.home_base['x'], self.home_base['y'], color='orange', s=150, marker='*', label='Home Base')
                # Draw connecting line.
                ax.plot([pred_loc['x'], prey_loc['x']], [pred_loc['y'], prey_loc['y']], color='purple', linestyle='-', linewidth=2)
                
                # Add info text with rewards and food picked.
                info_text = (
                    f"Predator Reward: {self.predator_reward_log[i]:.2f}\n"
                    f"Prey Reward: {self.prey_reward_log[i]:.2f}\n"
                    f"Food Picked: {self.food_picked_count_log[i]}"
                )
                ax.text(0.02, 0.95, info_text, transform=ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.8))
                
                ax.set_xlim(-self.size, self.size)
                ax.set_ylim(-self.size, self.size)
                ax.legend()
                plt.savefig(f"{frames_dir}/frame_{i:05d}.png")
                plt.close(fig)
            frame_files = sorted(
                os.listdir(frames_dir),
                key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
            )
            frame_paths = [os.path.join(frames_dir, file) for file in frame_files]
            frame = cv2.imread(frame_paths[0])
            height, width, _ = frame.shape
            video_path = "predator_prey_foraging_vid.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                video.write(frame)
            video.release()
            # Clean up the frames.
            for frame_path in frame_paths:
                os.remove(frame_path)
            os.rmdir(frames_dir)
            return video_path
        else:
            return None

    def close(self):
        if self.render_mode == "human":
            plt.ioff()
            plt.close(self.figure)
