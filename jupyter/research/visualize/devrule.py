import networkx as nx
from bokeh.plotting import figure, from_networkx

from ..archive.genome import Genome

class DevRule:
    def __new__(cls, genome: Genome, type: str):
        network = dict(creator=genome.creator(), deleter=genome.deleter())[type]
        G = nx.DiGraph()
        G.add_weighted_edges_from([(c["in"], c["out"], c["weight"]) for c in network["conns"] if c["enable"]])
        self = figure(title=type, width=500, height=500)
        n = from_networkx(G, nx.spring_layout, scale=1, center=(0,0))
        self.renderers.append(n)

        return self