import sys
import math

import networkx as nx
from bokeh.plotting import figure, from_networkx
from bokeh.io import push_notebook
from bokeh.io.export import get_screenshot_as_png
from bokeh.models import Range1d, Circle, ColumnDataSource, Arrow, NormalHead, Scatter
from bokeh.palettes import RdBu11

from ..archive.observe import Observe

from .widgets import MediaPlayer

class SolverNetworkView:
    def __new__(cls, observe: Observe, xrange=None, yrange=None):
        solver = observe.solver()
        
        has_num_neurocomponents = "num_inputs" in solver["task0"][0]
        if has_num_neurocomponents:
            num_inputs = solver["task0"][0]["num_inputs"]
            num_outputs = solver["task0"][0]["num_outputs"]
        nodes = [[(node["x"], node["y"], node["energy"]) for node in n["nodes"]] if not has_num_neurocomponents else [(node["x"], node["y"], node["energy"]) for node in n["nodes"]][(num_inputs+num_outputs):] for task in solver.values() for n in task]
        indices = [(task, i) for task, networks in solver.items() for i in range(len(networks))]

        if len(nodes[0]) != 0:
            px0, py0, energy0 = zip(*nodes[0])
        else:
            px0, py0, energy0 = [], [], []
        sgn = lambda v: -1 if v < 0 else 1


        xscaled = max([x for step in nodes for x, y, e in step]) > 100
        yscaled = max([y for step in nodes for x, y, e in step]) > 100
        if xscaled: print("x scaled")
        if yscaled: print("y scaled")

        if xscaled:
            xscaler = lambda x: x if -1 <= x <= 1 else sgn(x) * math.log10(abs(x))
        else:
            xscaler = lambda x: x
        if yscaled:
            yscaler = lambda y: y if -1 <= y <= 1 else sgn(y) * math.log10(abs(y))
        else:
            yscaler = lambda y: y

        px0 = [xscaler(x) for x in px0]
        py0 = [yscaler(y) for y in py0]
        energy0 = [RdBu11[int(10 * e)] if not math.isnan(e) else "#000000" for e in energy0]

        self = figure(title="solver network", width=500, height=500, output_backend="webgl")
        if xrange is not None:
            self.x_range=Range1d(*xrange)
        if yrange is not None:
            self.y_range=Range1d(*yrange)
        node_data_source = ColumnDataSource(data=dict(px=list(px0), py=list(py0), fill_color=energy0))
        node = Circle(
            x="px", y="py", size=12,
            fill_color="fill_color", line_color="#cc6633", fill_alpha=0.5
        )
        self.add_glyph(node_data_source, node)

        if has_num_neurocomponents:
            input_nodes = [[(node["x"], node["y"], node["energy"]) for node in n["nodes"]][:num_inputs] for task in solver.values() for n in task]
            output_nodes = [[(node["x"], node["y"], node["energy"]) for node in n["nodes"]][num_inputs:(num_inputs+num_outputs)] for task in solver.values() for n in task]

            ipx0, ipy0, ienergy0 = zip(*input_nodes[0])
            ipx0 = [xscaler(x) for x in ipx0]
            ipy0 = [yscaler(y) for y in ipy0]
            ienergy0 = [RdBu11[int(10 * e)] if not math.isnan(e) else "#000000" for e in ienergy0]
            
            opx0, opy0, oenergy0 = zip(*output_nodes[0])
            opx0 = [xscaler(x) for x in opx0]
            opy0 = [yscaler(y) for y in opy0]
            oenergy0 = [RdBu11[int(10 * e)] if not math.isnan(e) else "#000000" for e in oenergy0]

            input_node_data_source = ColumnDataSource(data=dict(px=list(ipx0), py=list(ipy0), fill_color=ienergy0))
            output_node_data_source = ColumnDataSource(data=dict(px=list(opx0), py=list(opy0), fill_color=oenergy0))

            input_node = Scatter(x="px", y="py", fill_color="fill_color", line_color="#cc6633", fill_alpha=0.5, size=12, marker="square")
            output_node = Scatter(x="px", y="py", fill_color="fill_color", line_color="#cc6633", fill_alpha=0.5, size=12, marker="inverted_triangle")
            self.add_glyph(input_node_data_source, input_node)
            self.add_glyph(output_node_data_source, output_node)


        conns = [[([xscaler(n["nodes"][conn["in"]]["x"]), xscaler(n["nodes"][conn["out"]]["x"])], [yscaler(n["nodes"][conn["in"]]["y"]), yscaler(n["nodes"][conn["out"]]["y"])], conn["weight"]) for conn in n["conns"]] for task in solver.values() for n in task]
        cx0, cy0, ws0 = zip(*conns[0])
        nh = NormalHead(size=12, fill_color="color", line_color="color", fill_alpha=0.5, line_alpha=0.5)
        conn_data_source = ColumnDataSource(data={
            "x_start" : [cx0[i][0] for i in range(len(cx0))],
            "x_end" : [cx0[i][1] for i in range(len(cx0))],
            "y_start" : [cy0[i][0] for i in range(len(cy0))],
            "y_end" : [cy0[i][1] for i in range(len(cy0))],
            "color" : ["blue" if w < 0 else "red" for w in ws0],
            "weight" : [4 * abs(math.tanh(w)) for w in ws0]
        })
        self.add_layout(Arrow(end=nh, x_start="x_start", y_start="y_start", x_end="x_end", y_end="y_end", 
            line_color="color", line_alpha=0.5, line_width="weight",
            source=conn_data_source))

        def update_fig(change):
            self.title.text = f"solver network (step={change['new']})"
            if len(nodes[change["new"]]) != 0:
                px, py, energy = zip(*nodes[change["new"]])
            else:
                px, py, energy = [], [], []
            px = [xscaler(x) for x in px]
            py = [yscaler(y) for y in py]
            energy = [RdBu11[int(10 * e)] if not math.isnan(e) else "#000000" for e in energy]
            node_data_source.data = {
                "px" : px,
                "py" : py,
                "fill_color" : energy
            }
            if len(conns[change["new"]]) == 0:
                conn_data_source.data = {
                    "x_start" : [],
                    "x_end" : [],
                    "y_start" : [],
                    "y_end" : [],
                    "color" : [],
                    "weight" : []
                }
            else:
                cx, cy, ws = zip(*conns[change["new"]])
                conn_data_source.data = {
                    "x_start" : [cx[i][0] for i in range(len(cx))],
                    "x_end" : [cx[i][1] for i in range(len(cx))],
                    "y_start" : [cy[i][0] for i in range(len(cy))],
                    "y_end" : [cy[i][1] for i in range(len(cy))],
                    "color" : ["blue" if w < 0 else "red" for w in ws],
                    "weight" : [4 * abs(math.tanh(w)) for w in ws]
                }
            if has_num_neurocomponents:
                ipx, ipy, ienergy = zip(*input_nodes[change["new"]])
                ipx = [xscaler(x) for x in ipx]
                ipy = [yscaler(y) for y in ipy]
                ienergy = [RdBu11[int(10 * e)] if not math.isnan(e) else "#000000" for e in ienergy]
                
                opx, opy, oenergy = zip(*output_nodes[change["new"]])
                opx = [xscaler(x) for x in opx]
                opy = [yscaler(y) for y in opy]
                oenergy = [RdBu11[int(10 * e)] if not math.isnan(e) else "#000000" for e in oenergy]
                input_node_data_source.data = {
                    "px" : ipx,
                    "py" : ipy,
                    "fill_color" : ienergy
                }
                output_node_data_source.data = {
                    "px" : opx,
                    "py" : opy,
                    "fill_color" : oenergy
                }
            push_notebook()
            pass

        player_tooltips = MediaPlayer(len(indices), update_fig, lambda: get_screenshot_as_png(self))
        player_tooltips.show()

        return self