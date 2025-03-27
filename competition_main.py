import os
import neat
import predator_prey_energy_env
import numpy as np
import random
import importlib
import matplotlib.pyplot as plt

def eval_genomes(genomes, config, args):
    env = predator_prey_energy_env.PredatorPreyEnv(size=args.size, num_steps=args.num_steps, num_food=args.num_food)
    
    for genome_id, genome in genomes:
        observations = env.reset()
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        while True:
            actions = {}
            predator, prey = env.agents
            # Predator and prey action based on their positions
            actions['predator'] = np.argmax(net.activate([predator.position[0], predator.position[1]]))  # Action based on predator's position
            actions['prey'] = np.argmax(net.activate([prey.position[0], prey.position[1]]))  # Action based on prey's position
            
            observations, rewards, done = env.step(actions)
            genome.fitness += rewards['predator']  # The predator's energy is used for fitness

            if done:
                break

        print(f"Genome {genome_id} fitness:", genome.fitness)

def run(config_file, args):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)
    
    # Setup for plotting
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(lambda genomes, config: eval_genomes(genomes, config, args), args.episodes)
    print("Best genome's fitness:", winner.fitness)

    # Visualize the winner
    try:
        render_mode = "human"
        env = predator_prey_energy_env.PredatorPreyEnv(size=args.size, num_steps=args.num_steps, num_food=args.num_food)
        observations = env.reset()
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        
        while True:
            actions = {}
            predator, prey = env.agents
            actions['predator'] = np.argmax(net.activate([predator.position[0], predator.position[1]]))  # Action based on predator's position
            actions['prey'] = np.argmax(net.activate([prey.position[0], prey.position[1]]))  # Action based on prey's position

            observations, rewards, done = env.step(actions)
            if done:
                break
        env.render()
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the predator-prey environment with NEAT')
    parser.add_argument('--config', '--c', default='competition.txt', type=str, help='Path to the config file')
    parser.add_argument('--render', '--r', default='human', type=str, help='Render mode (video or human)')
    parser.add_argument('--episodes', '--ep', default=100, type=int, help='Number of episodes to run')
    parser.add_argument('--num_food', '--nf', default=10, type=int, help='Number of food items in the environment')
    parser.add_argument('--size', '--s', default=50, type=int, help='Size of the environment')
    parser.add_argument('--num_steps', '--ns', default=100, type=int, help='Number of steps to run the environment for')

    args = parser.parse_args()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, f'configs/{args.config}')
    run(config_path, args)
