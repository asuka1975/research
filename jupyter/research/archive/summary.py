import glob
import json
import pandas as pd

from .trial import Trial

class Summary:
    def __init__(self):
        def get_summary(file):
            with open(file, "r") as f:
                return json.load(f)
        summaries = { summary.replace(".summary.json", "").replace("archive/", "") : get_summary(summary) for summary in glob.glob("archive/*.json") }
        self.summaries = {}
        for date, summary in summaries.items():
            for cfg, content in summary.items():
                self.summaries[(cfg, date)] = {}
                self.summaries[(cfg, date)]["trials"] = content["sorted"]["fitness"]
                self.summaries[(cfg, date)]["fitness_max"] = content["fitness_max"]
                self.summaries[(cfg, date)]["real_fitness_max"] = content["fitness_max"]
                self.summaries[(cfg, date)]["real_trials"] = content["sorted"]["real_fitness"]
        
    def table(self):
        df = pd.DataFrame(self.summaries).transpose()
        return df.sort_index(level=0)

    def trial(self, experiment, date, index, extract_entities = None):
        return Trial(date, experiment, index, extract_entities)


