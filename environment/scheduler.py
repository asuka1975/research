import subprocess
import time
import json
import os
import shutil
import glob
import multiprocessing
import logging
import logging.config

def monitor(schedule, queue):
    while True:
        monitor_dict = {}
        for config, task in schedule["settings"]:
            with open(f"/opt/app/unuse_setting/{config}", "r") as f:
                contents = f.readlines()
            observe_tick = [int(line.split("=")[1]) for line in contents if "observe_tick" in line][0]
            num_generations = [int(line.split("=")[1]) for line in contents if "num_generations" in line][0]
            observe_epochs = num_generations // observe_tick + 1
            monitor_dict[f"{config}-{task}"] = {
                "progress" : int(len(glob.glob(f"/opt/app/data/{config}-{task}/*/observe*")) * 100 / (observe_epochs * schedule["parallel"] * schedule["epoch"]))
            }
            with open("/opt/app/schedule/monitor.json", "w") as f:
                f.write(json.dumps(monitor_dict))
        try:
            queue.get(timeout=1)
            break
        except:
            pass
        time.sleep(60)

def main():
    logging.config.fileConfig('/opt/app/config/logger.conf', disable_existing_loggers=False)
    logger = logging.getLogger("scheduler")
    logger.info("environment start")
    
    while True:
        with open("/opt/app/schedule/schedule.json", "r") as f:
            schedule = json.load(f)
        if len(schedule) == 0:
            time.sleep(600)
            continue

        s = schedule.pop(-1)
        with open("/opt/app/schedule/schedule.json", "w") as f:
            f.write(json.dumps(schedule, indent=4))
        for file in glob.glob("/opt/app/settings/*"):
            os.remove(file)
        for setting, task in s["settings"]:
            shutil.copyfile(f"/opt/app/unuse_setting/{setting}", f"/opt/app/settings/{setting}")
            shutil.copyfile(f"/opt/app/unuse_setting/{task}.json", f"/opt/app/settings/{task}.json")
        print("******************************")
        with open("/opt/app/settings/workers.json", "w") as f:
            f.write(json.dumps(s))

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=monitor, args=(s, queue))
        process.start()
        logger.info("[%s] are submitted", ", ".join(f"({setting}, {task})" for setting, task in s["settings"]))
        subprocess.run(["python3", "scripts/parallel_repeat.py", str(s["epoch"]), str(s["parallel"])], stdout=subprocess.DEVNULL)

        num_expected_trials = [s["epoch"] * s["parallel"] for _ in s["settings"]]
        num_real_trials = [len(glob.glob(f"/opt/app/data/{config}-{task}/*/fitness.json")) for config, task in s["settings"]]
        for i, (config, task) in enumerate(s["settings"]):
            if num_expected_trials[i] != num_real_trials[i]:
                logger.warning("%s; %d trials have dropped out", f"({config}, {task})", num_expected_trials[i] - num_real_trials[i])
        
        subprocess.run(["python3", "scripts/archive.py"], stdout=subprocess.DEVNULL)

        queue.put(False)
        process.join()

        time.sleep(1)
        
        with open("/opt/app/schedule/monitor.json", "w") as f:
            f.write("{}")
        

if __name__ == "__main__":
    main()
