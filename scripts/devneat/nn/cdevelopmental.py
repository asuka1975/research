from .cffi import DevelopmentalNetwork as CDevelopmentalNetwork
from neat.six_util import itervalues, iteritems

class DevelopmentalNetwork(object):
    def __init__(self, num_inputs, num_outputs, dev_elements, conns, nodes):
        self.network = CDevelopmentalNetwork(num_inputs, num_outputs, conns, nodes, dev_elements, dev_elements["creator"].network, dev_elements["deleter"].network)

    def reset(self):
        self.reset()
        
    def activate(self, inputs):
        return self.network.activate(inputs)

    @property
    def conns(self):
        return self.network.conns

    @property
    def nodes(self):
        return self.network.nodes
        
    @staticmethod
    def create(genome, dev_elements, config):
        """ Receives a genome and returns its phenotype (a DevelopmentalNetwork). """
        genome_config = config.genome_config

        node_keys = {}
        i = 0
        for key in genome_config.input_keys:
            node_keys[key] = i
            i += 1
        for key in genome_config.output_keys:
            node_keys[key] = i
            i += 1

        conns = []
        # Gather inputs and expressed connections.
        for cg in itervalues(genome.connections):
            if not cg.enabled:
                continue

            i_, o_ = cg.key

            if i_ not in node_keys:
                node_keys[i_] = i
                i += 1
            if o_ not in node_keys:
                node_keys[o_] = i
                i += 1
            conns.append([cg.cx, cg.cy, node_keys[i_], node_keys[o_], cg.weight])
            
        node_keys_inverse = {
            v : k for k, v in node_keys.items()
        }
        
        nodes = [
            [float(i), 0.0, 0] for i in range(len(genome_config.input_keys))
        ]
        nodes += [
            [genome.nodes[node_keys_inverse[value]].nx, genome.nodes[node_keys_inverse[value]].ny, 
            genome.nodes[node_keys_inverse[value]].bias]
                for value in range(len(genome_config.input_keys), len(genome_config.input_keys) + len(genome_config.output_keys))
        ]
        nodes += [
            [genome.nodes[node_keys_inverse[value]].nx, genome.nodes[node_keys_inverse[value]].ny, 
            genome.nodes[node_keys_inverse[value]].bias]
                for value in range(len(genome_config.input_keys) + len(genome_config.output_keys), len(node_keys))
        ]
        return DevelopmentalNetwork(len(genome_config.input_keys), len(genome_config.output_keys), dev_elements, conns, nodes)