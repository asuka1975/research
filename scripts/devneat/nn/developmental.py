from neat.graphs import required_for_output
from neat.six_util import itervalues, iteritems
import math
from functools import reduce

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def distance(p1, p2):
    dx = (p1[0] - p2[0])
    dy = (p1[1] - p2[1])
    return dx * dx + dy * dy

def neighbor_nodes(nodes, index, num_neighbors):
    indices = [i for i in range(len(nodes)) if i != index]
    indices.sort(key=lambda i: distance(nodes[i][4], nodes[index][4]))
    r = [nodes[i] for i in indices[:num_neighbors]]
    dummy = [[
        nodes[0][0],
        nodes[0][1],
        0, 0, (0, 0), 0.5
    ] for _ in range(num_neighbors - len(r))]
    return r + dummy

def average_conns(indice, conns):
    conns = [conn for conn in conns if conn[0][1] == indice]
    conn_sum = reduce(lambda a, b: [None, (a[1][0] + b[1][0], a[1][1] + b[1][1]), a[2] + b[2]], conns, [None, (0, 0), 0])
    if len(conns) > 0:
        return (None, (conn_sum[1][0] / len(conns), conn_sum[1][1] / len(conns)), conn_sum[2] / len(conns))
    else:
        return conn_sum

def nearby_node(nodes, p):
    indices = list(range(len(nodes)))
    return min(indices, key=lambda i: distance(nodes[i][4], p))
    
def creator(net, inputs):
    nodes, conn_avg = inputs
    args = [*[s for node in nodes for s in [node[4][0], node[4][1], node[2], node[5]]], conn_avg[1][0], conn_avg[1][1], conn_avg[2]]
    output = net.activate(args)
    return (
        (output[0] > 0, (output[1], output[2]), output[3]),
        (output[4] > 0, (output[5], output[6]), output[7])
    )

def deleter(net, inputs):
    input_node, output_node, pos, weight = inputs

    args = [input_node[4][0], input_node[4][1], input_node[2], input_node[5], 
            output_node[4][0], output_node[4][1], output_node[2], output_node[5],
            pos[0], pos[1], weight]
    output = net.activate(args)
    return output[0] > 0

class DevelopmentalNetwork(object):
    def __init__(self, num_inputs, num_outputs, dev_elements, conns, nodes):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.conns = conns
        self.nodes = nodes

        self.values = [[0.0 for _ in self.nodes], [0.0 for _ in self.nodes]]
        self.active = 0

        self.creator = dev_elements["creator"]
        self.deleter = dev_elements["deleter"]
        self.hebb = dev_elements["hebb"]
        self.num_neighbors = dev_elements["num_neighbors"]
        self.num_develop_steps = dev_elements["num_develop_steps"]

        self.steps = 1

    def reset(self):
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        
    def activate(self, inputs):
        if self.num_inputs != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in enumerate(inputs):
            ivalues[i] = v
            ovalues[i] = v

        for states in self.conns:
            (i, o), _, w = states
            input = ivalues[i]
            output = ivalues[i] * w
            ovalues[o] += output
            # update weight
            states[-1] += self.hebb[0] * (self.hebb[1] * input * output + self.hebb[2] * input + self.hebb[3] * output + self.hebb[4])
            
        for i, states in enumerate(self.nodes):
            ovalues[i] = clamp(ovalues[i], -100, 100)
            activation, aggregation, bias, response, pos, energy = states
            ovalues[i] = activation(bias + ovalues[i])
            # update energy
            x = math.log(energy) - math.log(1 - energy) + ovalues[i]
            states[-1] = clamp(1 / (1 + math.exp(x)), 0.000001, 0.999999)

        if self.steps % self.num_develop_steps == 0:
            self.develop()

        self.steps += 1

        return ovalues[self.num_inputs:(self.num_inputs+self.num_outputs)]

    def develop(self):
        removes = [i for i in range(len(self.conns)) if deleter(self.deleter, [self.nodes[self.conns[i][0][0]], self.nodes[self.conns[i][0][1]], self.conns[i][1], self.conns[i][2]])]
        # 'Equal to
        # for i in range(len(self.conns)):
        #     conn = self.conns[i]
        #     ni = self.nodes[conn[0][0]]
        #     no = self.nodes[conn[0][1]]
        #     if deleter(self.deleter, [ni, no, conn[1], conn[2]]):
        #         removes.append(i)

        inserts = [creator(self.creator, [neighbor_nodes(self.nodes, i, self.num_neighbors), average_conns(i, self.conns)]) for i in range(len(self.nodes))]
        
        new_nodes = [[
            self.nodes[0][0],
            self.nodes[0][1],
            bias,
            1,
            pos,
            0.5
        ] for (f, pos, bias), _ in inserts if f]
        new_conns = [[
            (i, nearby_node(self.nodes, pos)), pos, weight
        ] for i, (_, (f, pos, weight)) in enumerate(inserts) if f]
        new_conns = [conn for conn in new_conns if sum(conn[0] == c[0] for c in self.conns) == 0]
        self.values[0] += [0.0] * len(new_nodes)
        self.values[1] += [0.0] * len(new_nodes)

        self.nodes += new_nodes
        self.conns += new_conns
        for i in reversed(removes):
            self.conns.pop(i)
        
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
            conns.append([(node_keys[i_], node_keys[o_]), (cg.cx, cg.cy), cg.weight])
            
        node_keys_inverse = {
            v : k for k, v in node_keys.items()
        }
        
        nodes = [
            [genome_config.activation_defs.get("tanh"),
            genome_config.aggregation_function_defs.get("sum"),
            0, 1, (float(i), 0.0), 0.5] for i in range(len(genome_config.input_keys))
        ]
        nodes += [
            [genome_config.activation_defs.get("tanh"),
            genome_config.aggregation_function_defs.get("sum"),
            genome.nodes[node_keys_inverse[value]].bias, 0,
            (genome.nodes[node_keys_inverse[value]].nx, genome.nodes[node_keys_inverse[value]].ny), 0.5]
                for value in range(len(genome_config.input_keys), len(genome_config.input_keys) + len(genome_config.output_keys))
        ]
        nodes += [
            [genome_config.activation_defs.get("tanh"),
            genome_config.aggregation_function_defs.get("sum"),
            genome.nodes[node_keys_inverse[value]].bias, 0,
            (genome.nodes[node_keys_inverse[value]].nx, genome.nodes[node_keys_inverse[value]].ny), 0.5]
                for value in range(len(genome_config.input_keys) + len(genome_config.output_keys), len(node_keys))
        ]
        return DevelopmentalNetwork(len(genome_config.input_keys), len(genome_config.output_keys), dev_elements, conns, nodes)