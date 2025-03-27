import random
import numpy as np

class Agent:
    def __init__(self, role, energy=100, position=None):
        self.role = role  # 'predator' or 'prey'
        self.energy = energy
        self.position = position if position else [random.randint(0, 50), random.randint(0, 50)]
    
    def move(self):
        # Move randomly
        self.position = [self.position[0] + random.randint(-1, 1), self.position[1] + random.randint(-1, 1)]
        self.energy -= 1  # Moving costs energy

    def eat(self, food_value):
        # Eating food gives the agent energy
        self.energy += food_value

    def get_position(self):
        return tuple(self.position)
    
    def lose_energy(self, amount):
        self.energy -= amount

class PredatorPreyEnv:
    def __init__(self, size=50, num_steps=100, num_food=10):
        self.size = size
        self.num_steps = num_steps
        self.num_food = num_food
        self.agents = [Agent(role='predator'), Agent(role='prey')]
        self.food_positions = [(random.randint(0, size), random.randint(0, size)) for _ in range(num_food)]
        self.steps_taken = 0
    
    def reset(self):
        self.agents = [Agent(role='predator'), Agent(role='prey')]
        self.food_positions = [(random.randint(0, self.size), random.randint(0, self.size)) for _ in range(self.num_food)]
        self.steps_taken = 0
        return self.get_state()

    def step(self, actions):
        predator, prey = self.agents
        self.steps_taken += 1
        
        # Move agents based on actions
        predator.move()
        prey.move()
        
        # Check if predator catches prey
        if predator.get_position() == prey.get_position():
            if predator.energy > prey.energy:
                predator.eat(50)  # Predator eats prey and gains energy
                prey.energy = 0  # Prey loses all energy
        
        # Check if agents eat food
        if predator.get_position() in self.food_positions:
            predator.eat(10)  # Predator gains energy from food
            self.food_positions.remove(predator.get_position())
        
        if prey.get_position() in self.food_positions:
            prey.eat(10)  # Prey gains energy from food
            self.food_positions.remove(prey.get_position())

        # Reward for predator (more energy, better fitness)
        predator_reward = predator.energy
        
        # Reward for prey (more energy, better fitness)
        prey_reward = prey.energy
        
        done = False
        if self.steps_taken >= self.num_steps or predator.energy <= 0 or prey.energy <= 0:
            done = True  # Episode ends when an agent runs out of energy or steps end
        
        rewards = {'predator': predator_reward, 'prey': prey_reward}
        
        return self.get_state(), rewards, done

    def get_state(self):
        # Return the positions of the predator and prey and remaining food positions as state
        return {
            'predator_position': self.agents[0].get_position(),
            'prey_position': self.agents[1].get_position(),
            'food_positions': self.food_positions,
        }

    def render(self):
        # This function can visualize the environment (e.g., using matplotlib)
        pass
