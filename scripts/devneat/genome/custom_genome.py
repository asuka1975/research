import math
from random import random

from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.genome import DefaultGenomeConfig
from neat.config import ConfigParameter

from .creator_genome import CreatorGenome
from .deleter_genome import DeleterGenome
from .solver_genome import SolverGenome

from ..genes import DevnetNodeGene, DevnetConnectionGene

def randrange(mn, mx):
    return random() * (mx - mn) + mn

class CustomGenomeConfig:
    def __init__(self, params):
        self._params = [ConfigParameter('hebb_value_min', float),
                        ConfigParameter('hebb_value_max', float),
                        ConfigParameter('hebb_blx_alpha', float),
                        ConfigParameter('hebb_mutate_rate', float),
                        ConfigParameter('hebb_distance_rate', float),
                        ConfigParameter('num_neighbors', int),
                        ConfigParameter('num_develop_steps', int)]
        for p in self._params:
            setattr(self, p.name, p.interpret(params["CustomGenome"]))
        params["CreatorGenome"]["num_inputs"] = 4 * self.num_neighbors + 3
        params["DeleterGenome"]["num_inputs"] = 4 * 2 + 3

        self.creator = DefaultGenomeConfig(params["CreatorGenome"])
        self.deleter = DefaultGenomeConfig(params["DeleterGenome"])
        self.solver = DefaultGenomeConfig(params["SolverGenome"])


class CustomGenome:
    @classmethod
    def parse_config(cls, param_dict):
        param_dict["CreatorGenome"]['node_gene_type'] = DefaultNodeGene
        param_dict["CreatorGenome"]['connection_gene_type'] = DefaultConnectionGene
        param_dict["DeleterGenome"]['node_gene_type'] = DefaultNodeGene
        param_dict["DeleterGenome"]['connection_gene_type'] = DefaultConnectionGene
        param_dict["SolverGenome"]['node_gene_type'] = DevnetNodeGene
        param_dict["SolverGenome"]['connection_gene_type'] = DevnetConnectionGene
        return CustomGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        self.key = key

        self.creator = CreatorGenome(key)
        self.deleter = DeleterGenome(key)
        self.solver = SolverGenome(key)
        self.hebb = [0] * 5

        self.fitness = None

    def configure_new(self, config):
        self.creator.configure_new(config.creator)
        self.deleter.configure_new(config.deleter)
        self.solver.configure_new(config.solver)
        self.hebb = [randrange(config.hebb_value_min, config.hebb_value_max) for _ in range(5)]

    def configure_crossover(self, genome1, genome2, config):
        self.creator.configure_crossover(genome1.creator, genome2.creator, config.creator)
        self.deleter.configure_crossover(genome1.deleter, genome2.deleter, config.deleter)
        self.solver.configure_crossover(genome1.solver, genome2.solver, config.solver)
        self.hebb = [
            randrange(min(a, b) - abs(a - b) * config.hebb_blx_alpha, max(a, b) + abs(a - b) * config.hebb_blx_alpha) for a, b in zip(genome1.hebb, genome2.hebb)
        ]

    def mutate(self, config):
        self.creator.mutate(config.creator)
        self.deleter.mutate(config.deleter)
        self.solver.mutate(config.solver)
        self.hebb = [randrange(config.hebb_value_min, config.hebb_value_max) if random() < config.hebb_mutate_rate else v for v in self.hebb]

    def distance(self, other, config):
        return self.creator.distance(other.creator, config.creator) \
            + self.deleter.distance(other.deleter, config.deleter) \
                 + self.solver.distance(other.solver, config.solver) \
                    + config.hebb_distance_rate * math.sqrt(sum((a - b) ** 2 for a, b in zip(self.hebb, other.hebb)))

    def size(self):
        n1, c1 = self.creator.size()
        n2, c2 = self.deleter.size()
        n3, c3 = self.solver.size()
        return n1 + n2 + n3, c1 + c2 + c3