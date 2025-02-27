import os
import argparse
import random
import neat
import predator_prey_foraging_env
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# Custom Reporter for Plotting
# --------------------------
class CoEvolutionPlotReporter(neat.reporting.BaseReporter):
    def __init__(self):
        self.generations = []
        self.predator_best_fitness = []
        self.prey_best_fitness = []
        self.predator_avg_fitness = []
        self.prey_avg_fitness = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Best Fitness Over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")

    def post_evaluate(self, generation, predator_best, prey_best, predator_avg, prey_avg):
        self.generations.append(generation)
        self.predator_best_fitness.append(predator_best)
        self.prey_best_fitness.append(prey_best)
        self.predator_avg_fitness.append(predator_avg)
        self.prey_avg_fitness.append(prey_avg)
        self.ax.clear()
        self.ax.plot(
            self.generations,
            self.predator_best_fitness,
            label='Predator Best Fitness',
            marker='o',
            linestyle='-'
        )
        self.ax.plot(
            self.generations,
            self.prey_best_fitness,
            label='Prey Best Fitness',
            marker='o',
            linestyle='-'
        )
        self.ax.plot(
            self.generations,
            self.predator_avg_fitness,
            label='Predator Avg Fitness',
            marker='x',
            linestyle='--'
        )
        self.ax.plot(
            self.generations,
            self.prey_avg_fitness,
            label='Prey Avg Fitness',
            marker='x',
            linestyle='--'
        )
        self.ax.set_title("Fitness Over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        self.ax.legend()
        plt.draw()
        plt.pause(0.001)

# --------------------------
# Evaluation Function
# --------------------------
def eval_coevolution(predator_genomes, predator_config, prey_genomes, prey_config, args, gen):
    """
    Evaluate each predator genome against the best prey genome over several episodes,
    and evaluate each prey genome against the best predator genome over several episodes.
    This gives both populations a fitness signal:
      - Predators are rewarded for reducing the distance (and capturing) the prey.
      - Prey are rewarded for increasing the distance (and avoiding capture) and for foraging.
    """
    episodes_per_genome = 5

    predator_ids = list(predator_genomes.keys())
    prey_ids = list(prey_genomes.keys())

    # --- Evaluate predators ---
    # Choose the best prey genome (if available, otherwise pick randomly).
    if any(g.fitness is not None for g in prey_genomes.values()):
        best_prey_id, _ = max(
            prey_genomes.items(), key=lambda item: item[1].fitness if item[1].fitness is not None else -float('inf')
        )
    else:
        best_prey_id = random.choice(prey_ids)

    for pid in predator_ids:
        total_predator_reward = 0.0
        for ep in range(episodes_per_genome):
            # Occasionally render in human mode.
            if ep == episodes_per_genome - 1 and gen % 25 == 0 and gen != 0 and pid == predator_ids[-1]:
                env = predator_prey_foraging_env.parallel_env(
                    render_mode='human',
                    size=args.size,
                    num_steps=args.num_steps
                )
            else:
                env = predator_prey_foraging_env.parallel_env(
                    render_mode=args.render,
                    size=args.size,
                    num_steps=args.num_steps
                )
            state, infos = env.reset()
            predator_net = neat.nn.FeedForwardNetwork.create(predator_genomes[pid], predator_config)
            prey_net = neat.nn.FeedForwardNetwork.create(prey_genomes[best_prey_id], prey_config)
            episode_predator_reward = 0.0
            while env.agents:
                actions = {}
                for agent in env.agents:
                    obs = state[agent]
                    if agent.role == 'predator':
                        actions[agent] = predator_net.activate(obs)
                    else:
                        actions[agent] = prey_net.activate(obs)
                state, rewards, terminations = env.step(actions)
                # Accumulate predator rewards.
                for agent, reward in rewards.items():
                    if agent.role == 'predator':
                        episode_predator_reward += reward
            total_predator_reward += episode_predator_reward
            env.close()
        predator_genomes[pid].fitness = total_predator_reward / episodes_per_genome

    # --- Evaluate prey ---
    # Choose the best predator genome (if available, otherwise pick randomly).
    if any(g.fitness is not None for g in predator_genomes.values()):
        best_predator_id, _ = max(
            predator_genomes.items(), key=lambda item: item[1].fitness if item[1].fitness is not None else -float('inf')
        )
    else:
        best_predator_id = random.choice(predator_ids)

    for pid in prey_ids:
        total_prey_reward = 0.0
        for ep in range(episodes_per_genome):
            env = predator_prey_foraging_env.parallel_env(
                render_mode=args.render,
                size=args.size,
                num_steps=args.num_steps
            )
            state, infos = env.reset()
            predator_net = neat.nn.FeedForwardNetwork.create(predator_genomes[best_predator_id], predator_config)
            prey_net = neat.nn.FeedForwardNetwork.create(prey_genomes[pid], prey_config)
            episode_prey_reward = 0.0
            while env.agents:
                actions = {}
                for agent in env.agents:
                    obs = state[agent]
                    if agent.role == 'predator':
                        actions[agent] = predator_net.activate(obs)
                    else:
                        actions[agent] = prey_net.activate(obs)
                state, rewards, terminations = env.step(actions)
                # Accumulate prey rewards.
                for agent, reward in rewards.items():
                    if agent.role == 'prey':
                        episode_prey_reward += reward
            total_prey_reward += episode_prey_reward
            env.close()
        prey_genomes[pid].fitness = total_prey_reward / episodes_per_genome

# --------------------------
# Final Run Function (Human Rendering)
# --------------------------
def run_best(predator_genome, predator_config, prey_genome, prey_config, args):
    """
    Run one episode using the best predator and prey genomes in human render mode.
    """
    env = predator_prey_foraging_env.parallel_env(
        render_mode='human',
        size=args.size,
        num_steps=args.num_steps
    )
    state, infos = env.reset()
    predator_net = neat.nn.FeedForwardNetwork.create(predator_genome, predator_config)
    prey_net = neat.nn.FeedForwardNetwork.create(prey_genome, prey_config)
    total_predator_reward = 0.0
    total_prey_reward = 0.0

    while env.agents:
        actions = {}
        for agent in env.agents:
            obs = state[agent]
            if agent.role == 'predator':
                actions[agent] = predator_net.activate(obs)
            else:
                actions[agent] = prey_net.activate(obs)
        state, rewards, terminations = env.step(actions)
        for agent, reward in rewards.items():
            if agent.role == 'predator':
                total_predator_reward += reward
            else:
                total_prey_reward += reward
        env.render()
    env.close()
    print("Final human run - Total Predator Reward:", total_predator_reward)
    print("Final human run - Total Prey Reward:", total_prey_reward)

# --------------------------
# Final Run Function (Video Generation)
# --------------------------
def run_best_video(predator_genome, predator_config, prey_genome, prey_config, args):
    """
    Run one episode using the best predator and prey genomes in video render mode.
    At termination, the environment will generate a video.
    """
    env = predator_prey_foraging_env.parallel_env(
        render_mode='video',
        size=args.size,
        num_steps=args.num_steps
    )
    state, infos = env.reset()
    predator_net = neat.nn.FeedForwardNetwork.create(predator_genome, predator_config)
    prey_net = neat.nn.FeedForwardNetwork.create(prey_genome, prey_config)
    total_predator_reward = 0.0
    total_prey_reward = 0.0

    while env.agents:
        actions = {}
        for agent in env.agents:
            obs = state[agent]
            if agent.role == 'predator':
                actions[agent] = predator_net.activate(obs)
            else:
                actions[agent] = prey_net.activate(obs)
        state, rewards, terminations = env.step(actions)
        for agent, reward in rewards.items():
            if agent.role == 'predator':
                total_predator_reward += reward
            else:
                total_prey_reward += reward
    env.close()
    print("Final video run - Total Predator Reward:", total_predator_reward)
    print("Final video run - Total Prey Reward:", total_prey_reward)

# --------------------------
# Main Evolution Loop
# --------------------------
def run(args):
    local_dir = os.path.dirname(__file__)
    predator_config_path = os.path.join(local_dir, f'configs/{args.config_predator}')
    prey_config_path = os.path.join(local_dir, f'configs/{args.config_prey}')
    
    predator_config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        predator_config_path
    )
    prey_config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        prey_config_path
    )
    
    predator_pop = neat.Population(predator_config)
    prey_pop = neat.Population(prey_config)
    
    predator_pop.add_reporter(neat.StdOutReporter(True))
    prey_pop.add_reporter(neat.StdOutReporter(True))
    predator_stats = neat.StatisticsReporter()
    prey_stats = neat.StatisticsReporter()
    predator_pop.add_reporter(predator_stats)
    prey_pop.add_reporter(prey_stats)
    
    coevo_plot_reporter = CoEvolutionPlotReporter()
    
    generations = args.episodes
    for gen in range(generations):
        print(f"\n*** Generation {gen} ***")
        
        # Reset fitness for all genomes.
        for g in predator_pop.population.values():
            g.fitness = 0
        for g in prey_pop.population.values():
            g.fitness = 0
        
        # Evaluate genomes over multiple episodes.
        eval_coevolution(
            predator_pop.population, predator_config,
            prey_pop.population, prey_config, args, gen
        )
        
        # Capture best fitness BEFORE reproduction.
        best_predator = max(predator_pop.population.values(), key=lambda g: g.fitness if g.fitness is not None else 0)
        best_prey = max(prey_pop.population.values(), key=lambda g: g.fitness if g.fitness is not None else 0)
        print("Best predator fitness this generation:", best_predator.fitness)
        
        predator_avg = np.mean([g.fitness for g in predator_pop.population.values()])
        prey_avg = np.mean([g.fitness for g in prey_pop.population.values()])
        
        # Update the custom reporter with best and average fitness values.
        coevo_plot_reporter.post_evaluate(
            gen,
            best_predator.fitness if best_predator.fitness is not None else 0,
            best_prey.fitness if best_prey.fitness is not None else 0,
            predator_avg,
            prey_avg
        )
        
        predator_pop.reporters.post_evaluate(
            predator_config, predator_pop.population, predator_pop.species, best_predator
        )
        prey_pop.reporters.post_evaluate(
            prey_config, prey_pop.population, prey_pop.species, best_prey
        )
        
        new_predator_population = predator_pop.reproduction.reproduce(
            predator_config, predator_pop.species, predator_config.pop_size, gen
        )
        new_prey_population = prey_pop.reproduction.reproduce(
            prey_config, prey_pop.species, prey_config.pop_size, gen
        )
        
        predator_pop.species.speciate(predator_config, new_predator_population, gen)
        prey_pop.species.speciate(prey_config, new_prey_population, gen)
        
        predator_pop.population = new_predator_population
        prey_pop.population = new_prey_population
        
        predator_pop.generation += 1
        prey_pop.generation += 1
        
    best_predator = max(predator_pop.population.values(), key=lambda g: g.fitness if g.fitness is not None else 0)
    best_prey = max(prey_pop.population.values(), key=lambda g: g.fitness if g.fitness is not None else 0)
    print("Best Predator Fitness:", best_predator.fitness)
    print("Best Prey Fitness:", best_prey.fitness)
    coevo_plot_reporter.fig.savefig("final_rewards_graph.png")
    print("Final rewards graph saved as 'final_rewards_graph.png'")
    
    # Optionally, run final demonstration episodes:
    # run_best(best_predator, predator_config, best_prey, prey_config, args)
    # run_best_video(best_predator, predator_config, best_prey, prey_config, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run combined predator-prey-foraging coevolution with separate NEAT populations and final demonstrations"
    )
    parser.add_argument('--config_predator', default='predator_config.txt', type=str,
                        help='Path to the predator config file')
    parser.add_argument('--config_prey', default='prey_predator_retrieval_config.txt', type=str,
                        help='Path to the prey config file')
    parser.add_argument('--render', default='cpu', type=str,
                        help='Render mode during evolution (video or cpu)')
    parser.add_argument('--episodes', default=100, type=int,
                        help='Number of generations/episodes')
    parser.add_argument('--size', default=50, type=int,
                        help='Size of the environment')
    parser.add_argument('--num_steps', default=500, type=int,
                        help='Number of steps per episode')
    args = parser.parse_args()
    run(args)
