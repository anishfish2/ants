import environment
import navigation_env
import foraging_env
import neat
import os
import argparse
import importlib

def eval_genomes(genomes, config, args):
    try:
        # Try to import the module dynamically
        env_module = importlib.import_module(args.env)
        print(f"Successfully imported {args.env}")
    except ModuleNotFoundError:
        print(f"Module {args.env} not found. Using the default environment.")
        env_module = foraging_env  # Fallback to default module

    # Pass the render_mode and any other relevant arguments
    render_mode = args.render if args.render else "cpu"
    env = env_module.parallel_env(render_mode=render_mode, num_ants=args.num_agents, num_food=args.nf, size=args.s, num_steps=args.ns, range_radius=args.rr)

    for genome_id, genome in genomes:
        observations, infos = env.reset()
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        while env.ants:
            actions = {ant: net.activate(env.state[ant]) for ant in env.ants}

            observations, rewards, terminations = env.step(actions)

            genome.fitness += sum(rewards.values())
    env.close()

def view_winner(winner, config):
    render_mode = args.render if args.render else "video"
    env = foraging_env.parallel_env(render_mode=render_mode)

    observations, infos = env.reset()
    winner.fitness = 0
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    done = False
    while env.ants:
        actions = {ant: net.activate(env.state[ant]) for ant in env.ants}
        observations, rewards, terminations = env.step(actions)
        winner.fitness += sum(rewards.values())
    print(f"Winner fitness: {winner.fitness}")

    env.close()

def run(config_file, args):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(lambda genomes, config: eval_genomes(genomes, config, args), args.episodes)

    view_winner(winner, config, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the foraging environment with NEAT')
    parser.add_argument('--config', '--c', type=str, help='Path to the config file')
    parser.add_argument('--render', '--r', type=str, help='Render mode (video or human)')
    parser.add_argument('--env', '--e', type=str, help='Environment to run (foraging or navigation)')
    parser.add_argument('--episodes', '--ep', type=int, help='Number of episodes to run')
    parser.add_argument('--num_agents', '--na', type=int, help='Number of agents in the environment')
    parser.add_argument('--num food', '--nf', type=int, help='Number of food items in the environment')
    parser.add_argument('--size', '--s', type=int, help='Size of the environment')
    parser.add_argument('--range_radius', '--rr', type=int, help='Range radius of the environment')
    parser.add_argument('--num_steps', '--ns', type=int, help='Number of steps to run the environment for')

    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, f'configs/{args.config}')

    run(config_path, args)
