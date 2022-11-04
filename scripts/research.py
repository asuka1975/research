import os
import json
import copy
import datetime
from re import M
import sys

import bson

import neat
import visualize
from devneat.genome import CustomGenome
from devneat.config import CustomConfig

import devneat
from devneat.nn.cdevelopmental import DevelopmentalNetwork
from devneat.nn.crecurrent import RecurrentNetwork
from devneat.task import create_task

def scale(v, mn, mx):
    return (v - mn) / (mx - mn)

def eval_genomes(genomes, config):
    task_config = config.task_config
    observing = config.generation_step % config.observe_tick == 0
    observe_index = config.generation_step // config.observe_tick
    for idx, (genome_id, genome) in enumerate(genomes):
        if observing:
            os.makedirs(f"observe{observe_index}/genome{idx}", exist_ok=True)
        task_states = {}
        network_states = {}

        print(genome_id)
        net1 = RecurrentNetwork.create(genome.creator, type("Hoge", (object, ), {"genome_config": config.genome_config.creator}))
        net2 = RecurrentNetwork.create(genome.deleter, type("Hoge", (object, ), {"genome_config": config.genome_config.deleter}))
        net3 = DevelopmentalNetwork.create(genome.solver, {
            "creator" : net1,
            "deleter" : net2,
            "hebb" : genome.hebb,
            "num_neighbors" : config.genome_config.num_neighbors,
            "num_develop_steps" : config.genome_config.num_develop_steps,
            "enable_devrule_per_neurocomponents" : config.enable_devrule_per_neurocomponents
        }, type("Hoge", (object, ), {"genome_config": config.genome_config.solver}))
        genome.fitness = 0
        break_flag = False
        for i in task_config["schedule"]:
            task = create_task(task_config["tasks"][i])
            if observing:
                task_states[f"task{i}"] = []
                task_states[f"task{i}"].append(task.state())
                network_states[f"task{i}"] = []
                network_states[f"task{i}"].append({
                    "nodes" : [{ "bias" : node[2], "energy" : node[5], "x" : node[4][0], "y" : node[4][1] } for node in net3.nodes],
                    "conns" : [{ "in" : conn[0][0], "out" : conn[0][1], "weight" : conn[2], "x" : conn[1][0], "y" : conn[1][1] } for conn in net3.conns],
                    "num_nodes" : len(net3.nodes),
                    "num_conns" : len(net3.conns),
                    "num_inputs" : config.genome_config.solver.num_inputs,
                    "num_outputs" : config.genome_config.solver.num_outputs
                })

            while not task.finish():
                if len(net3.conns) > 150 or len(net3.nodes) > 150:
                    genome.fitness += -100
                    break_flag = True
                    break
                inputs = net3.activate(task.get_output())
                task.update(inputs)

                if observing:
                    task_states[f"task{i}"].append(task.state())
                    network_states[f"task{i}"].append({
                        "nodes" : [{ "bias" : node[2], "energy" : node[5], "x" : node[4][0], "y" : node[4][1] } for node in net3.nodes],
                        "conns" : [{ "in" : conn[0][0], "out" : conn[0][1], "weight" : conn[2], "x" : conn[1][0], "y" : conn[1][1] } for conn in net3.conns],
                        "num_nodes" : len(net3.nodes),
                        "num_conns" : len(net3.conns),
                        "num_inputs" : config.genome_config.solver.num_inputs,
                        "num_outputs" : config.genome_config.solver.num_outputs
                    })
            if break_flag:
                break
            genome.fitness += task.fitness()
        
        if observing:
            with open(f"observe{observe_index}/genome{idx}/other.json", "w") as f:
                f.write(json.dumps({
                    "fitness" : genome.fitness,
                    "genome_id" : genome_id
                }))
            with open(f"observe{observe_index}/genome{idx}/network.bson", "wb") as f:
                f.write(bson.dumps(network_states))
            with open(f"observe{observe_index}/genome{idx}/cartpole.bson", "wb") as f:
                f.write(bson.dumps(task_states))
                
        genome.creator.fitness = genome.fitness
        genome.deleter.fitness = genome.fitness
        genome.solver.fitness = genome.fitness
    config.fitness_history["maxes"].append(scale(max(genome.fitness for _, genome in genomes), config.fitness_history["fitness_min"], config.fitness_history["fitness_max"]))
    config.fitness_history["means"].append(scale(sum(genome.fitness for _, genome in genomes) / len(genomes), config.fitness_history["fitness_min"], config.fitness_history["fitness_max"]))
    config.fitness_history["mins"].append(scale(min(genome.fitness for _, genome in genomes), config.fitness_history["fitness_min"], config.fitness_history["fitness_max"]))
    config.generation_step += 1


def run(config, task, out_dir):
    with open(task, "r") as f:
        task_config = json.load(f)
    # Load configuration.
    config = CustomConfig(CustomGenome, neat.DefaultReproduction,
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
    print(config.genome_config.solver.num_inputs)
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
    p.add_reporter(neat.Checkpointer(config.observe_tick, None))

    try:
        # Run for up to 300 generations.
        winner = p.run(eval_genomes, config.num_generations)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net1 = RecurrentNetwork.create(winner.creator, type("Hoge", (object, ), {"genome_config": config.genome_config.creator}))
        winner_net2 = RecurrentNetwork.create(winner.deleter, type("Hoge", (object, ), {"genome_config": config.genome_config.deleter}))
        net3 = DevelopmentalNetwork.create(winner.solver, {
            "creator" : winner_net1,
            "deleter" : winner_net2,
            "hebb" : winner.hebb,
            "num_neighbors" : config.genome_config.num_neighbors,
            "num_develop_steps" : config.genome_config.num_develop_steps,
            "enable_devrule_per_neurocomponents" : config.enable_devrule_per_neurocomponents
        }, type("Hoge", (object, ), {"genome_config": config.genome_config.solver}))

        task_states = {}
        network_states = {}
        break_flag = False
        for i in config.task_config["schedule"]:
            print("**task", i)
            task = create_task(config.task_config["tasks"][i])
            task_states[f"task{i}"] = []
            task_states[f"task{i}"].append(task.state())
            network_states[f"task{i}"] = []
            network_states[f"task{i}"].append({
                "nodes" : [{ "bias" : node[2], "energy" : node[5], "x" : node[4][0], "y" : node[4][1] } for node in net3.nodes],
                "conns" : [{ "in" : conn[0][0], "out" : conn[0][1], "weight" : conn[2], "x" : conn[1][0], "y" : conn[1][1] } for conn in net3.conns],
                "num_nodes" : len(net3.nodes),
                "num_conns" : len(net3.conns),
                "num_inputs" : config.genome_config.solver.num_inputs,
                "num_outputs" : config.genome_config.solver.num_outputs
            })
            while not task.finish():
                if len(net3.conns) > 400 or len(net3.nodes) > 400:
                    break_flag = True
                    break
                inputs = net3.activate(task.get_output())
                task.update(inputs)
                task_states[f"task{i}"].append(task.state())
                network_states[f"task{i}"].append({
                    "nodes" : [{ "bias" : node[2], "energy" : node[5], "x" : node[4][0], "y" : node[4][1] } for node in net3.nodes],
                    "conns" : [{ "in" : conn[0][0], "out" : conn[0][1], "weight" : conn[2], "x" : conn[1][0], "y" : conn[1][1] } for conn in net3.conns],
                    "num_nodes" : len(net3.nodes),
                    "num_conns" : len(net3.conns),
                    "num_inputs" : config.genome_config.solver.num_inputs,
                    "num_outputs" : config.genome_config.solver.num_outputs
                })
            if break_flag:
                break
        
        with open("cartpole.bson", "wb") as f:
            f.write(bson.dumps(task_states))
        with open("network.bson", "wb") as f:
            f.write(bson.dumps(network_states))

        node_names = task.label()
        visualize.draw_net(type("Hoge", (object, ), {"genome_config": config.genome_config.creator}), winner.creator, False, node_names=node_names, filename="creator.gv")
        visualize.draw_net(type("Hoge", (object, ), {"genome_config": config.genome_config.deleter}), winner.deleter, False, node_names=node_names, filename="deleter.gv")
        visualize.draw_net(type("Hoge", (object, ), {"genome_config": config.genome_config.solver}), winner.solver, False, node_names=node_names, filename="solver.gv")
    except neat.CompleteExtinctionException:
        pass
    with open("fitness.json", "w") as f:
        f.write(json.dumps(config.fitness_history))

    visualize.plot_stats(stats, ylog=False, view=False)
    visualize.plot_species(stats, view=False)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    options = dict(arg[1:].split(":") for arg in sys.argv if arg[0] == "-")
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-devnetwork')
    run(options["config"], options["task"], options["out"])
