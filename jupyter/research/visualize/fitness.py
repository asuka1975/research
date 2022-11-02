from bokeh.plotting import figure
from bokeh.palettes import d3
from bokeh.models import Range1d

from ..archive.trial import Trial

class Fitness:
    def __new__(cls, trial: Trial, ymin=None, ymax=None, extras={}):
        c = d3['Category10'][max(2 + len(extras), 3)]
        f = trial.fitness()
        self = figure(title="fitness", plot_width=500, plot_height=400)

        self.xaxis.axis_label = 'generation'
        self.yaxis.axis_label = 'fitness'
        fmax = f["fitness_max"]
        fmin = f["fitness_min"]
        self.y_range = Range1d(ymin if ymin is not None else fmin, ymax if ymax is not None else fmax)
        self.line(range(len(f["maxes"])), [v * (fmax - fmin) + fmin for v in f["maxes"]], color=c[0], legend_label='max')
        self.line(range(len(f["maxes"])), [v * (fmax - fmin) + fmin for v in f["means"]], color=c[1], legend_label='mean')
        
        for i, (label, fn) in enumerate(extras.items()):
            self.line(range(len(f["maxes"])), fn(f), color=c[i + 2], legend_label=label)
        return self