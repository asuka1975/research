import os
import json
import sys

import bson

import neat
import visualize
from devneat.config import CustomStaticConfig

import devneat
from devneat.nn import RecurrentNetwork, CRecurrentNetwork
from devneat.task import create_task

def scale(v, mn, mx):
    return (v - mn) / (mx - mn)

error_num = 0
all_num = 0

def eval_genomes(genomes, config):
    global error_num, all_num
    task_config = config.task_config
    observing = config.generation_step % config.observe_tick == 0
    observe_index = config.generation_step // config.observe_tick
    for idx, (genome_id, genome) in enumerate(genomes):
        if observing:
            os.makedirs(f"observe{observe_index}/genome{idx}", exist_ok=True)
        task_states = {}

        print(genome_id)
        net = CRecurrentNetwork.create(genome, config)
        net2 = RecurrentNetwork.create(genome, config)
        genome.fitness = 0
        break_flag = False
        for i in task_config["schedule"]:
            task = create_task(task_config["tasks"][i])
            if observing:
                task_states[f"task{i}"] = []
                task_states[f"task{i}"].append(task.state())

            while not task.finish():
                inputs = net.activate(task.get_output())
                inputs2 = net2.activate(task.get_output())
                task.update(inputs)

                all_num += 1
                if abs(inputs[0] - inputs2[0]) > 0.1:
                    error_num += 1

                if observing:
                    task_states[f"task{i}"].append(task.state())
            if break_flag:
                break
            genome.fitness += task.fitness()
        
        if observing:
            with open(f"observe{observe_index}/genome{idx}/other.json", "w") as f:
                f.write(json.dumps({
                    "fitness" : genome.fitness,
                    "genome_id" : genome_id
                }))
            with open(f"observe{observe_index}/genome{idx}/cartpole.bson", "wb") as f:
                f.write(bson.dumps(task_states))
    config.fitness_history["maxes"].append(scale(max(genome.fitness for _, genome in genomes), config.fitness_history["fitness_min"], config.fitness_history["fitness_max"]))
    config.fitness_history["means"].append(scale(sum(genome.fitness for _, genome in genomes) / len(genomes), config.fitness_history["fitness_min"], config.fitness_history["fitness_max"]))
    config.fitness_history["mins"].append(scale(min(genome.fitness for _, genome in genomes), config.fitness_history["fitness_min"], config.fitness_history["fitness_max"]))
    config.generation_step += 1


def run(config, task, out_dir):
    with open(task, "r") as f:
        task_config = json.load(f)
    # Load configuration.
    config = CustomStaticConfig(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         task_config, config)
    config.generation_step = 0
    config.fitness_history = {
        "maxes" : [],
        "means" : [],
        "mins" : []
    }
    mn, mx = 0, 0
    for i in config.task_config["schedule"]:
        task = create_task(config.task_config["tasks"][i])
        mn += task.min_fitness()
        mx += task.max_fitness()
    max_inputs = max(create_task(config.task_config["tasks"][i]).num_inputs() for i in config.task_config["schedule"])
    max_outputs = max(create_task(config.task_config["tasks"][i]).num_outputs() for i in config.task_config["schedule"])
    config.genome_config.num_inputs = max_inputs
    config.genome_config.num_outputs = max_outputs
    config.fitness_history["fitness_max"] = mx
    config.fitness_history["fitness_min"] = mn

    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(config.observe_tick, 10000))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, config.num_generations)
    with open("fitness.json", "w") as f:
        f.write(json.dumps(config.fitness_history))
    global error_num, all_num
    print("***", error_num / all_num)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = CRecurrentNetwork.create(winner, config)

    task_states = {}
    break_flag = False
    for i in config.task_config["schedule"]:
        print("**task", i)
        task = create_task(config.task_config["tasks"][i])
        task_states[f"task{i}"] = []
        task_states[f"task{i}"].append(task.state())
        while not task.finish():
            inputs = winner_net.activate(task.get_output())
            task.update(inputs)
            task_states[f"task{i}"].append(task.state())
        if break_flag:
            break
    
    with open("cartpole.bson", "wb") as f:
        f.write(bson.dumps(task_states))

    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    visualize.draw_net(config, winner, False, node_names=node_names, filename="solver.gv")
    visualize.plot_stats(stats, ylog=False, view=False)
    visualize.plot_species(stats, view=False)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    options = dict(arg[1:].split(":") for arg in sys.argv if arg[0] == "-")
    local_dir = os.path.dirname(__file__)
    run(options["config"], options["task"], options["out"])