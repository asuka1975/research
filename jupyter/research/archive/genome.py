import json

class Genome:
    def __init__(self, trial, index):
        self.trial = trial
        self.index = index
        with open(f"analysis/{self.trial.root}/{self.trial.directory}genome{self.index}.json") as f:
            self.data = json.load(f)

    def creator(self):
        return self.data["creator"]

    def deleter(self):
        return self.data["deleter"]

    def initial_solver(self):
        return self.data["devnetwork"]
    
    def solver(self):
        return self.data["solver"]

    def hebb(self):
        return self.data["hebb"]
