from neat.graphs import feed_forward_layers
from neat.six_util import itervalues
from collections import deque


class TDNN(object):
    def __init__(self, inputs, outputs, node_evals, time_delay=3):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.time_delay = time_delay

        # Create a deque for each input node to store the history of inputs.
        self.input_history = {key: deque([0.0] * time_delay, maxlen=time_delay) for key in inputs}
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        # Append new inputs to the input history deques
        for k, v in zip(self.input_nodes, inputs):
            self.input_history[k].append(v)

        # Flatten the input history for each input node
        flattened_inputs = []
        for k in self.input_nodes:
            flattened_inputs.extend(self.input_history[k])

        # Now assign flattened inputs (current and past) to the input nodes
        for idx, k in enumerate(self.input_nodes):
            self.values[k] = flattened_inputs[idx]

        # Evaluate the network
        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)

        return [self.values[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config, time_delay=3):
        """ Receives a genome and returns its phenotype (a TDNN). """

        # Gather expressed connections.
        connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                node_expr = []  # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        return TDNN(config.genome_config.input_keys, config.genome_config.output_keys, node_evals, time_delay)
