import asyncio
import base64
import hashlib
from io import BytesIO
import json
import math

from bokeh.plotting import figure
from bokeh.models import Range1d, ColumnDataSource, Scatter, FactorRange, Rect
from bokeh.layouts import layout, column
from bokeh.io import push_notebook
from bokeh.io.export import get_screenshot_as_png
import ipywidgets
from IPython.display import display, HTML
from PIL import Image

from ..archive.observe import Observe

from .widgets import MediaPlayer
    
class CartPole:
    def __new__(cls, observe: Observe, task_index):
        with open(f"unuse_setting/{observe.trial.experiment.split('-')[1]}.json") as f:
            cfg = json.load(f)
        setting = cfg["tasks"][cfg["schedule"][task_index]]["setting"]

        field = figure(title="field (step=0)", plot_width=500, plot_height=500)
        field.x_range = Range1d(-1, 1)
        field.y_range = Range1d(-1, 1)

        task = observe.task(task_index)
        t0 = task[0]
        pole = field.line([t0["cart_position"], t0["cart_position"] + setting["pole_length"] * math.sin(t0["pole_angle"])], [0, setting["pole_length"] * math.cos(t0["pole_angle"])], color="blue", line_width=10)
        glyph = Rect(x=t0["cart_position"], y=0, width=0.2, height=0.1, fill_color="#cab2d6")
        field.add_glyph(glyph)

        def update_step(change):
            t = task[change["new"]]
            field.title.text = f"field (step={change['new']})"
            
            pole.data_source.data = dict(
                x=[t["cart_position"], t["cart_position"] + setting["pole_length"] * math.sin(t["pole_angle"])], 
                y=[0, setting["pole_length"] * math.cos(t["pole_angle"])],
                color=["blue", "blue"]
            )
            glyph.x = t["cart_position"]
            if math.ceil(t["cart_position"]) % 2 == 0:
                field.x_range.start, field.x_range.end = (math.floor(t["cart_position"]), math.floor(t["cart_position"]) + 2)
            else:
                field.x_range.start, field.x_range.end = (math.ceil(t["cart_position"]) - 2, math.ceil(t["cart_position"]))

            push_notebook()
        self = field

        player_tooltips = MediaPlayer(len(task), update_step, lambda: get_screenshot_as_png(self))
        player_tooltips.show()
        
        return self

class BrokenPoleCartPole:
    def __new__(cls, observe: Observe, task_index):
        with open(f"unuse_setting/{observe.trial.experiment.split('-')[1]}.json") as f:
            cfg = json.load(f)
        setting = cfg["tasks"][cfg["schedule"][task_index]]["setting"]

        field = figure(title="field (step=0)", plot_width=500, plot_height=500)
        field.x_range = Range1d(-1, 1)
        field.y_range = Range1d(-1, 1)

        task = observe.task(task_index)
        t0 = task[0]
        pole = field.line([t0["cart_position"], t0["cart_position"] + setting["pole_length"] * math.sin(t0["pole_angle"])], [0, setting["pole_length"] * math.cos(t0["pole_angle"])], color="blue", line_width=10)
        glyph = Rect(x=t0["cart_position"], y=0, width=0.2, height=0.1, fill_color="#cab2d6")
        field.add_glyph(glyph)

        def update_step(change):
            t = task[change["new"]]
            field.title.text = f"field (step={change['new']})"
            if t["broken"]:
                pole.data_source.data = dict(
                    x=[t["cart_position"], t["cart_position"] + t["pole1_length"] * math.sin(t["pole1_angle"]), t["cart_position"] + t["pole1_length"] * math.sin(t["pole1_angle"]) + t["pole2_length"] * math.sin(t["pole2_angle"])],
                    y=[0, t["pole1_length"] * math.cos(t["pole1_angle"]), t["pole1_length"] * math.cos(t["pole1_angle"]) + t["pole2_length"] * math.cos(t["pole2_angle"])],
                    color=["red", "red", "red"]
                )
            else:
                pole.data_source.data = dict(
                    x=[t["cart_position"], t["cart_position"] + setting["pole_length"] * math.sin(t["pole_angle"])], 
                    y=[0, setting["pole_length"] * math.cos(t["pole_angle"])],
                    color=["blue", "blue"]
                )
            glyph.x = t["cart_position"]
            if math.ceil(t["cart_position"]) % 2 == 0:
                field.x_range.start, field.x_range.end = (math.floor(t["cart_position"]), math.floor(t["cart_position"]) + 2)
            else:
                field.x_range.start, field.x_range.end = (math.ceil(t["cart_position"]) - 2, math.ceil(t["cart_position"]))

            push_notebook()
        self = field

        player_tooltips = MediaPlayer(len(task), update_step, lambda: get_screenshot_as_png(self))
        player_tooltips.show()
        
        return self

class DoublePoleCartPole:
    def __new__(cls, observe: Observe, task_index):
        with open(f"unuse_setting/{observe.trial.experiment.split('-')[1]}.json") as f:
            cfg = json.load(f)
        setting = cfg["tasks"][cfg["schedule"][task_index]]["setting"]

        field = figure(title="field (step=0)", plot_width=500, plot_height=500)
        field.x_range = Range1d(-1, 1)
        field.y_range = Range1d(-1, 1)

        task = observe.task(task_index)
        t0 = task[0]
        pole = field.line([t0["cart_position"], t0["cart_position"] + setting["pole1_length"] * math.sin(t0["pole_angle"])], [0, setting["pole1_length"] * math.cos(t0["pole_angle"])], color="blue", line_width=10)
        glyph = Rect(x=t0["cart_position"], y=0, width=0.2, height=0.1, fill_color="#cab2d6")
        field.add_glyph(glyph)

        def update_step(change):
            t = task[change["new"]]
            field.title.text = f"field (step={change['new']})"
            if t["double"]:
                pole.data_source.data = dict(
                    x=[t["cart_position"] + t["pole1_length"] * math.sin(t["pole1_angle"]), t["cart_position"], t["cart_position"] + t["pole2_length"] * math.sin(t["pole2_angle"])],
                    y=[t["pole1_length"] * math.cos(t["pole1_angle"]), 0, t["pole2_length"] * math.cos(t["pole2_angle"])],
                    color=["red", "red", "red"]
                )
            else:
                pole.data_source.data = dict(
                    x=[t["cart_position"], t["cart_position"] + setting["pole1_length"] * math.sin(t["pole_angle"])], 
                    y=[0, setting["pole1_length"] * math.cos(t["pole_angle"])],
                    color=["blue", "blue"]
                )
            glyph.x = t["cart_position"]
            if math.ceil(t["cart_position"]) % 2 == 0:
                field.x_range.start, field.x_range.end = (math.floor(t["cart_position"]), math.floor(t["cart_position"]) + 2)
            else:
                field.x_range.start, field.x_range.end = (math.ceil(t["cart_position"]) - 2, math.ceil(t["cart_position"]))

            push_notebook()
        self = field

        player_tooltips = MediaPlayer(len(task), update_step, lambda: get_screenshot_as_png(self))
        player_tooltips.show()
        
        return self