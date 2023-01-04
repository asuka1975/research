import subprocess
import time
import json
import os
import shutil
import glob
import multiprocessing
import logging
import logging.config
import configparser

import redis

def monitor(schedule, rds, queue):
    rds.hmset("monitor", *[v for config, task in schedule["settings"] for v in [f"{config}-{task}", 0]])
    while True:
        monitor_dict = {}
        for config, task in schedule["settings"]:
            cfg = configparser.ConfigParser()
            cfg.read(f"/opt/app/unuse_setting/{config}")
            observe_tick = cfg["NEAT"].getint("observe_tick")
            num_generations = cfg["NEAT"].getint("num_generations")
            observe_epochs = num_generations // observe_tick + 1
            percent = int(len(glob.glob(f"/opt/app/data/{config}-{task}/*/observe*")) * 100 / (observe_epochs * schedule["parallel"] * schedule["epoch"]))
            monitor_dict[f"{config}-{task}"] = {
                "progress" : percent
            }
            with open("/opt/app/schedule/monitor.json", "w") as f:
                f.write(json.dumps(monitor_dict))
            rds.hset("monitor", f"{config}-{task}", percent)
        try:
            queue.get(timeout=1)
            break
        except:
            pass
        time.sleep(60)
    rds.delete("monitor")

def main():
    logging.config.fileConfig('/opt/app/config/logger.conf', disable_existing_loggers=False)
    logger = logging.getLogger("scheduler")
    logger.info("environment start")

    pool = redis.ConnectionPool(host="redis", port=6379, db=0)
    rd = redis.StrictRedis(connection_pool=pool)
    
    while True:
        if rd.llen("schedule") == 0:
            time.sleep(600)
            continue

        s = json.loads(rd.rpop("schedule").decode("utf-8"))
        for file in glob.glob("/opt/app/settings/*"):
            os.remove(file)
        for setting, task in s["settings"]:
            shutil.copyfile(f"/opt/app/unuse_setting/{setting}", f"/opt/app/settings/{setting}")
            shutil.copyfile(f"/opt/app/unuse_setting/{task}.json", f"/opt/app/settings/{task}.json")
        print("******************************")
        with open("/opt/app/settings/workers.json", "w") as f:
            f.write(json.dumps(s))

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=monitor, args=(s, redis.StrictRedis(connection_pool=pool), queue))
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
