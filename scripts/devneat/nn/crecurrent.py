from .cffi import RecurrentNetwork as CRecurrentNetwork
from neat.graphs import required_for_output
from neat.six_util import itervalues, iteritems

class RecurrentNetwork(object):
    def __init__(self, num_inputs, num_outputs, nodes, conns, activations_def):
        self.network = CRecurrentNetwork(num_inputs, num_outputs, nodes, conns, activations_def)

    def reset(self):
        self.network.reset()

    def activate(self, inputs):
        return self.network.activate(inputs)

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a RecurrentNetwork). """
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

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
            if o_ not in required and i_ not in required:
                continue

            if i_ not in node_keys:
                node_keys[i_] = i
                i += 1
            if o_ not in node_keys:
                node_keys[o_] = i
                i += 1
            conns.append((node_keys[i_], node_keys[o_], cg.weight))
            
        node_keys_inverse = {
            v : k for k, v in node_keys.items()
        }
        
        functions = list(genome_config.activation_defs.functions.keys())
        nodes = [
            (0, 0) for _ in range(len(genome_config.input_keys))
        ]
        nodes += [
            (genome.nodes[node_keys_inverse[value]].bias, functions.index(genome.nodes[node_keys_inverse[value]].activation)) 
                for value in range(len(genome_config.input_keys), len(genome_config.input_keys) + len(genome_config.output_keys))
        ]
        nodes += [
            (genome.nodes[node_keys_inverse[value]].bias, functions.index(genome.nodes[node_keys_inverse[value]].activation)) 
                for value in range(len(genome_config.input_keys) + len(genome_config.output_keys), len(node_keys))
        ]

        return RecurrentNetwork(len(genome_config.input_keys), len(genome_config.output_keys), nodes, conns, functions)

