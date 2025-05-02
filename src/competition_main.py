import os
import neat
import numpy as np
import random
import matplotlib.pyplot as plt
from competition_env import parallel_env
import pickle

def eval_genomes(genomes, config):
    """
    Evaluate a set of genomes against each other in the competition environment.
    Each genome is evaluated against all other genomes in the population.
    """
    # Create the environment
    env = parallel_env(render_mode=None, size=50, num_steps=200, num_food=20)
    
    # Initialize fitness for all genomes
    for genome_id, genome in genomes:
        genome.fitness = 0.0
    
    # Evaluate each genome against all others
    for i, (genome1_id, genome1) in enumerate(genomes):
        for j, (genome2_id, genome2) in enumerate(genomes[i+1:], i+1):
            # Create neural networks for both agents
            net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
            net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
            
            # Reset environment
            observations, _ = env.reset()
            
            # Run episode
            done = False
            while not done:
                # Get actions from both networks
                action1 = net1.activate(observations[env.agent1])
                action2 = net2.activate(observations[env.agent2])
                
                # Step environment
                observations, rewards, terminations = env.step({
                    env.agent1: action1,
                    env.agent2: action2
                })
                
                # Update fitness
                genome1.fitness += rewards[env.agent1]
                genome2.fitness += rewards[env.agent2]
                
                # Check if episode is done
                if any(terminations.values()):
                    done = True

def run(config_path, num_generations=50):
    """
    Run the NEAT algorithm to evolve agents for the competition environment.
    """
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    # Create population
    pop = neat.Population(config)
    
    # Add reporters
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    # Run evolution
    winner = pop.run(eval_genomes, num_generations)
    
    # Save the winner
    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)
    
    # Plot statistics
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([genome.fitness for genome in stats.most_fit_genomes], 'b-', label="Most Fit")
    plt.plot(stats.get_fitness_mean(), 'r-', label="Average")
    plt.title("Fitness History")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(stats.get_species_sizes(), 'g-')
    plt.title("Species Size History")
    plt.xlabel("Generation")
    plt.ylabel("Number of Species")
    
    plt.tight_layout()
    plt.savefig('evolution_stats.png')
    plt.close()
    
    return winner

def visualize_winner(config_path, winner_path):
    """
    Visualize the best agent from training.
    """
    # Load configuration and winner
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    with open(winner_path, 'rb') as f:
        winner = pickle.load(f)
    
    # Create environment with rendering
    env = parallel_env(render_mode="human", size=50, num_steps=200, num_food=20)
    
    # Create neural network for winner
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Create a random opponent network
    opponent = neat.DefaultGenome(1)
    opponent.configure_new(config.genome_config)
    opponent_net = neat.nn.FeedForwardNetwork.create(opponent, config)
    
    # Run episode
    observations, _ = env.reset()
    done = False
    while not done:
        # Get actions from both networks
        action1 = net.activate(observations[env.agent1])
        action2 = opponent_net.activate(observations[env.agent2])
        
        # Step environment
        observations, rewards, terminations = env.step({
            env.agent1: action1,
            env.agent2: action2
        })
        
        # Check if episode is done
        if any(terminations.values()):
            done = True
    
    env.close()

if __name__ == "__main__":
    # Get the directory containing this script
    local_dir = os.path.dirname(__file__)
    
    # Path to configuration file
    config_path = os.path.join(local_dir, 'config.txt')
    
    # Run evolution
    winner = run(config_path)
    
    # Visualize the winner
    visualize_winner(config_path, 'winner.pkl')
