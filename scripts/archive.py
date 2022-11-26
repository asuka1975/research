import os
import json
import datetime
import subprocess
import glob
import shutil
import subprocess

from devneat.task import create_task

def do_command(command):
    with subprocess.Popen(command, stdout=subprocess.PIPE) as proc:
        while True:
            msg = proc.stdout.readline()[:-1]
            if msg == b'': break
            if os.path.isfile(msg): yield msg

data_dir = "data/"
experiments = ["data/" + e + "/" for e in os.listdir(data_dir)]

l2 = subprocess.run(["sh", "-c", "find data/ | grep -v /observe | grep -v /neat | grep -e '/[0123456789]$'"], capture_output=True, text=True).stdout.split()
l1 = subprocess.run(["sh", "-c", "find data/ | grep fitness.json | sed -e 's/\\/fitness.json//g'"], capture_output=True, text=True).stdout.split()

for v in l2:
    if v not in l1:
        shutil.rmtree(v)

fm = {}
for experiment in experiments:
    if experiment.split("/")[1] == "summary.json": continue
    trials_int = [t for t in os.listdir(experiment)]
    trials = [experiment + str(t) + "/" for t in trials_int]
    f = []
    real_f = []
    for trial in trials:
        with open(trial + "fitness.json", "r") as fp:
            j = json.load(fp)
            f.append(max(j["maxes"]))
        
        with open(f"unuse_setting/{experiment.split('/')[1].split('-')[0]}", "r") as fp:
            content = fp.readlines()
            observe_tick = [int(line.split("=")[1]) for line in content if "observe_tick" in line][0]
            
            num_generations = [int(line.split("=")[1]) for line in content if "num_generations" in line][0]
            observe_epochs = list(range(0, num_generations, observe_tick))
            real_f.append(max([j["maxes"][i] for i in observe_epochs if i < len(j["maxes"])]))
    i = list(range(len(f)))
    i.sort(key=lambda x: -f[x])
    k = list(range(len(real_f)))
    k.sort(key=lambda x: -real_f[x])
    fm[experiment.split("/")[1]] = { 
        "trials" : len(os.listdir(experiment)), 
        "fitness_max" : f[i[0]], 
        "real_fitness_max" : real_f[k[0]],
        "sorted" : { "fitness" : [trials_int[j] for j in i], "real_fitness" : [trials_int[j] for j in k] } 
    }

archive_name = str(datetime.datetime.now()).replace(":", "_")

with open("archive/" + archive_name + ".summary.json", "w") as f:
    json.dump(fm, f)

s = sum(os.path.isfile(path) for path in glob.glob("data/**/*", recursive=True))

command = ["tar", "-I", "pixz", "-cvf", "archive/{0}.tar.xz".format(archive_name), "data/"]
u = 0
print("\r{:.3f}%".format(u / s * 100), end="")
for _ in do_command(command):
    u += 1
    print("\r{0} {1:.3f}% {2}".format(s, u / s * 100, u), end="")
for d in os.listdir("data"):
    shutil.rmtree("data/" + d)
print("")
