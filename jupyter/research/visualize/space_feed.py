import asyncio
import base64
import hashlib
from io import BytesIO
import json

from bokeh.plotting import figure
from bokeh.models import Range1d, ColumnDataSource, Scatter, FactorRange
from bokeh.layouts import layout, column
from bokeh.io import push_notebook
from bokeh.io.export import get_screenshot_as_png
import ipywidgets
from IPython.display import display, HTML
from PIL import Image

from ..archive.observe import Observe

from .widgets import MediaPlayer
    
class SpaceFeed:
    def __new__(cls, observe: Observe, task_index):
        field = figure(title="field (step=0)", width=500, height=500)
        with open(f"unuse_setting/{observe.trial.experiment}.json") as f:
            cfg = json.load(f)
        setting = cfg["tasks"]["tasks"][cfg["tasks"]["schedule"][task_index]]["setting"]

        field.x_range = Range1d(0, setting["width"])
        field.y_range = Range1d(0, setting["height"])

        task = observe.task(task_index)
        t0 = task[0]
        agent = field.scatter([t0["position"]["x"]], [t0["position"]["y"]], marker="square", size=10)
        feed = field.scatter([v[0] for v in t0["feeds"] if v[2] == 1], [v[1] for v in t0["feeds"] if v[2] == 1], color="#0000FF", legend_label="feed", size=10)
        poizon = field.scatter([v[0] for v in t0["feeds"] if v[2] == 2], [v[1] for v in t0["feeds"] if v[2] == 2], color="#FF0000", legend_label="poizon", size=10)
        l = sorted([v for v in t0["feeds"]], key=lambda v: (v[0] - t0["position"]["x"]) * (v[0] - t0["position"]["x"]) + (v[1] - t0["position"]["y"]) * (v[1] - t0["position"]["y"]))
        
        orbital_data = { "x": [t0["position"]["x"]], "y": [t0["position"]["y"]] }
        orbital = field.line(orbital_data["x"], orbital_data["y"], line_dash="4 4", color="red")

        approaching = field.line([t0["position"]["x"], l[0][0]], [t0["position"]["y"], l[0][1]], line_dash="4 4")

        inout = figure(width=270, height=250)
        inout.x_range = Range1d(0, 1)
        inout.y_range = Range1d(0, 1)
        inout.xgrid.visible = False
        inout.ygrid.visible = False
        inout.xaxis.visible = False
        inout.yaxis.visible = False
        inout.outline_line_width = 0
        switching = inout.scatter(x=[0.75, 0.75], y=[0.75, 0.25], size=10, fill_color=["white", "white"], line_color="black")
        inout.text([0.1, 0.1], [0.7125, 0.2125], text=["input_switched", "output_switched"])

        feed_plot = figure(width=270, height=250, x_range=FactorRange(factors=["feed", "poizon"]))
        feed_plot.y_range = Range1d(0, max(setting["num_feeds"], setting["num_poisons"]) + 1)
        feed_bar = feed_plot.vbar(x=["feed", "poizon"], top=[0, 0], width=0.9)

        def update_step(change):
            t = task[change["new"]]
            field.title.text = f"field (step={change['new']})"
            agent.data_source.data = dict(x=[t["position"]["x"]], y=[t["position"]["y"]])
            feed.data_source.data = dict(x=[v[0] for v in t["feeds"] if v[2] == 1], y=[v[1] for v in t["feeds"] if v[2] == 1])
            poizon.data_source.data = dict(x=[v[0] for v in t["feeds"] if v[2] == 2], y=[v[1] for v in t["feeds"] if v[2] == 2])
            switching.data_source.data["fill_color"] = ["black" if t["input_switch"] else "white", "black" if t["output_switch"] else "white"]
            feed_bar.data_source.data["top"] = [t["eaten_feeds"], t["eaten_poisons"]]
            l = sorted([v for v in t["feeds"]], key=lambda v: (v[0] - t["position"]["x"]) * (v[0] - t["position"]["x"]) + (v[1] - t["position"]["y"]) * (v[1] - t["position"]["y"]))
            if len(l) != 0:
                approaching.data_source.data["x"] = [t["position"]["x"], l[0][0]]
                approaching.data_source.data["y"] = [t["position"]["y"], l[0][1]]
            ln = change["new"] + 1
            orbital.data_source.data = {
                "x" : [task[i]["position"]["x"] for i in range(ln)],
                "y" : [task[i]["position"]["y"] for i in range(ln)]
            }
            push_notebook()


        self = layout([[field, column(inout, feed_plot)]])

        player_tooltips = MediaPlayer(len(task), update_step, lambda: get_screenshot_as_png(self))
        player_tooltips.show()
        
        return self