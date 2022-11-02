import subprocess
import hashlib
import os
import shutil
import json
import glob
import pandas as pd

from . import genome
from . import observe

class Trial:
    def __init__(self, date, experiment, trial, extract_entities=None):
        self.archive = f"{date}.tar.xz"
        self.root = hashlib.sha256(date.encode()).hexdigest()
        self.directory = f"data/{experiment}/{trial}/"
        self.experiment = experiment
        self.extract_entities = extract_entities

    def __enter__(self):
        if not os.path.exists(f"analysis/{self.root}"):
            os.mkdir(f"analysis/{self.root}")
        if self.extract_entities is None:
            subprocess.run(["tar", "-I", "pixz", f"--directory=analysis/{self.root}/", "-xf", f"archive/{self.archive}", self.directory])
        else:
            subprocess.run(["tar", "-I", "pixz", f"--directory=analysis/{self.root}/", "-xf", f"archive/{self.archive}", *[f"{self.directory}{entity}" for entity in self.extract_entities]])
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(f"analysis/{self.root}/{self.directory}")
        if len(os.listdir(f"analysis/{self.root}/data/{self.experiment}")) == 0:
            shutil.rmtree(f"analysis/{self.root}/data/{self.experiment}")
        if len(os.listdir(f"analysis/{self.root}/data")) == 0:
            shutil.rmtree(f"analysis/{self.root}/data")
        if len(os.listdir(f"analysis/{self.root}")) == 0:
            shutil.rmtree(f"analysis/{self.root}")

    def fitness(self):
        with open(f"analysis/{self.root}/{self.directory}fitness.json", "r") as f:
            j = json.load(f)
            return j

    def genome(self, index):
        return genome.Genome(self, index)

    def observe(self, index, genome):
        return observe.Observe(self, index, genome)

    def table(self):
        def get_fitness(observe, genome):
            with open(f"analysis/{self.root}/{self.directory}observe{observe}/genome{genome}/other.json", "r") as f:
                return json.load(f)["fitness"]
        return pd.DataFrame({
            f"observe{i}" : {
                f"genome{j}" : get_fitness(i, j) for j in range(len(glob.glob(f"analysis/{self.root}/{self.directory}observe{i}/genome*")))
            } for i in range(len(glob.glob(f"analysis/{self.root}/{self.directory}observe*")))
        })

def clean():
    shutil.rmtree(f"analysis/*")