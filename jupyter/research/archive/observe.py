import bson
import json

class Observe:
    def __init__(self, trial, index, genome):
        self.trial = trial
        self.index = index
        self.genome = genome

    def solver(self):
        with open(f"analysis/{self.trial.root}/{self.trial.directory}observe{self.index}/genome{self.genome}/network.bson", "rb") as f:
            return bson.loads(f.read())

    def task(self, index):
        with open(f"analysis/{self.trial.root}/{self.trial.directory}observe{self.index}/genome{self.genome}/cartpole.bson", "rb") as f:
            return bson.loads(f.read())[f"task{index}"]

    def other(self):
        with open(f"analysis/{self.trial.root}/{self.trial.directory}observe{self.index}/genome{self.genome}/other.json", "r") as f:
            return json.load(f)

