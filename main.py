import environment
import navigation_env
import foraging_env
import neat
import os
import argparse
import importlib
import matplotlib.pyplot as plt

def eval_genomes(genomes, config, args):
    try:
        if args.env:
            env_module = importlib.import_module(args.env)
        else:
            print("Importing default foraging_env")
            env_module = foraging_env
        print("Successfully imported", args.env)
    except ModuleNotFoundError:
        print("Module", args.env, "not found. Using the default environment.")
        env_module = foraging_env

    render_mode = args.render if args.render else "cpu"
    env = env_module.parallel_env(render_mode=render_mode, 
                                  num_ants=args.num_agents, 
                                  num_food=args.num_food, 
                                  size=args.size, 
                                  num_steps=args.num_steps, 
                                  range_radius=args.range_radius)
    for genome_id, genome in genomes:
        observations, infos = env.reset()
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # TDNN instead of feed forward NN
        # net = TDNN.create(genome, config)
        while env.ants:
            actions = {ant: net.activate(env.state[ant]) for ant in env.ants}
            observations, rewards, terminations = env.step(actions)
            genome.fitness += sum(rewards.values())
        # Record the food-delivered count for this episode into the genome.
        genome.delivered_count = getattr(env, 'food_delivered_count', 0)
    env.close()

def view_winner(winner, config, args):
    try:
        if args.env:
            env_module = importlib.import_module(args.env)
        else:
            print("Importing default foraging_env")
            env_module = foraging_env
        print("Successfully imported", args.env)
    except ModuleNotFoundError:
        print("Module", args.env, "not found. Using the default environment.")
        env_module = foraging_env

    render_mode = "video"
    env = env_module.parallel_env(render_mode=render_mode, 
                                  num_ants=args.num_agents, 
                                  num_food=args.num_food, 
                                  size=args.size, 
                                  num_steps=args.num_steps, 
                                  range_radius=args.range_radius)
    observations, infos = env.reset()
    winner.fitness = 0
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    while env.ants:
        actions = {ant: net.activate(env.state[ant]) for ant in env.ants}
        observations, rewards, terminations = env.step(actions)
        winner.fitness += sum(rewards.values())
    print("Winner fitness:", winner.fitness)
    env.close()

def run(config_file, args):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(FitnessPlotReporter())
    p.add_reporter(FoodDeliveredPlotReporter())
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(lambda genomes, config: eval_genomes(genomes, config, args), args.episodes)
    view_winner(winner, config, args)

class FitnessPlotReporter(neat.reporting.BaseReporter):
    def __init__(self):
        self.generations = []
        self.best_fitness = []
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Best Fitness Over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Best Fitness")

    def post_evaluate(self, config, population, species, best_genome):
        gen = len(self.generations) + 1
        self.generations.append(gen)
        self.best_fitness.append(best_genome.fitness)
        
        self.ax.clear()
        self.ax.plot(self.generations, self.best_fitness, marker='o', linestyle='-')
        self.ax.set_title("Best Fitness Over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Best Fitness")
        plt.draw()
        plt.pause(0.001)

class FoodDeliveredPlotReporter(neat.reporting.BaseReporter):
    def __init__(self):
        self.generations = []
        self.max_delivered = []
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Max Food Delivered per Episode")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Food Delivered")

    def post_evaluate(self, config, population, species, best_genome):
        gen = len(self.generations) + 1
        self.generations.append(gen)
        # We stored the delivered count in the genome during evaluation.
        delivered = getattr(best_genome, 'delivered_count', 0)
        self.max_delivered.append(delivered)
        
        self.ax.clear()
        self.ax.plot(self.generations, self.max_delivered, marker='o', linestyle='-')
        self.ax.set_title("Max Food Delivered per Episode")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Food Delivered")
        plt.draw()
        plt.pause(0.001)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the foraging environment with NEAT')
    parser.add_argument('--config', '--c', default='default_config.txt', type=str, help='Path to the config file')
    parser.add_argument('--render', '--r', default='human', type=str, help='Render mode (video or human)')
    parser.add_argument('--env', '--e', default='foraging', type=str, help='Environment to run (foraging or navigation)')
    parser.add_argument('--episodes', '--ep', default=100, type=int, help='Number of episodes to run')
    parser.add_argument('--num_agents', '--na', default=1, type=int, help='Number of agents in the environment')
    parser.add_argument('--num_food', '--nf', default=1, type=int, help='Number of food items in the environment')
    parser.add_argument('--size', '--s', default=50, type=int, help='Size of the environment')
    parser.add_argument('--range_radius', '--rr', default=10, type=int, help='Range radius of the environment')
    parser.add_argument('--num_steps', '--ns', default=100, type=int, help='Number of steps to run the environment for')

    args = parser.parse_args()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, f'configs/{args.config}')
    run(config_path, args)
