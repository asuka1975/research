from bokeh.plotting import figure
from bokeh.palettes import d3

from ..archive.observe import Observe

class NeuroComponents:
    def __new__(cls, observe: Observe):
        c = d3['Category10'][3]
        solver = observe.solver()
        num_conns = [step["num_conns"] for v in solver.values() for step in v]
        num_nodes = [step["num_nodes"] for v in solver.values() for step in v]
        self = figure(title="neurocomponents", plot_width=500, plot_height=400)

        self.xaxis.axis_label = 'step'
        self.yaxis.axis_label = 'fitness'
        self.line(range(len(num_conns)), num_conns, color=c[0], legend_label='synapse')
        self.line(range(len(num_nodes)), num_nodes, color=c[1], legend_label='neuron')
        return self