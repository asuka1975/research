import os
import sys
import subprocess
import multiprocessing
import json

def current_research_index(setting_name, worker_index):
    path = "data/" + setting_name
    if not os.path.exists(path):
        return 0
    dirs = [d for d in os.listdir(path) if int(d.split("-")[0]) == worker_index]
    return len(dirs)


def worker(worker_index, num_epochs):
    for _ in range(num_epochs):
        with open("settings/workers.json", "r") as f:
            workers = json.load(f)
        for config, task in workers["settings"]:
            file = "settings/" + config
            setting_name = f"{config}-{task}"
            index = current_research_index(setting_name, worker_index)
            data_path = f"data/{setting_name}/{worker_index}-{index}"
            try:
                exe = None
                if "static" in setting_name:
                    exe = f"python3 scripts/static_research.py -config:{file} -task:settings/{task}.json -out:{data_path}"
                else:
                    exe = f"python3 scripts/research.py -config:{file} -task:settings/{task}.json -out:{data_path}"
                subprocess.run(exe, shell=True)
            except subprocess.CalledProcessError as e:
                with open(data_path + "/error", "w") as f:
                    f.write("raise error", e.returncode, e.stderr)
            except Exception as e:
                with open(data_path + "/error", "w") as f:
                    f.write("unknown error", e)


def main():
    num_epochs = int(sys.argv[1])
    num_workers = int(sys.argv[2])

    ps = [multiprocessing.Process(target=worker, args=(i, num_epochs)) for i in range(num_workers)]
    for p in ps:
        p.start()

    for p in ps:
        p.join()

    for setting in os.listdir("data"):
        for i, trial in enumerate(os.listdir("data/" + setting)):
            os.rename(f"data/{setting}/{trial}", f"data/{setting}/{i}")


if __name__ == "__main__":
    main()