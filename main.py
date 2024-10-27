import environment
import navigation_env
import foraging_env
import neat
import os


def eval_genomes(genomes, config):
    env = foraging_env.parallel_env(render_mode="cpu")

    for genome_id, genome in genomes:
        observations, infos = env.reset()
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        while env.agents:
            actions = {agent: net.activate(env.state[agent]) for agent in env.agents}

            observations, rewards, terminations = env.step(actions)

            genome.fitness += sum(rewards.values())
    env.close()

def view_winner(winner, config):
    env = foraging_env.parallel_env(render_mode="video")

    observations, infos = env.reset()
    winner.fitness = 0
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    done = False
    while env.agents:
        actions = {agent: net.activate(env.state[agent]) for agent in env.agents}
        observations, rewards, terminations = env.step(actions)
        winner.fitness += sum(rewards.values())
    print(f"Winner fitness: {winner.fitness}")
    # print('\Winner genome:\n{!s}'.format(winner))

    env.close()

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 100)

    # print('\nBest genome:\n{!s}'.format(winner))

    view_winner(winner, config)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'configs/foraging_config.txt')
    run(config_path)
