import os
import sys
import subprocess


def current_research_index(setting_name):
    path = "data/" + setting_name
    if not os.path.exists(path):
        return 0
    dirs = os.listdir(path)
    return len(dirs)

def main():
    num_epochs = int(sys.argv[1])
    for _ in range(num_epochs):
        settings = os.listdir("settings")
        for setting in settings:
            file = "settings/" + setting
            setting_name = os.path.splitext(os.path.basename(setting))[0]
            index = current_research_index(setting_name)
            data_path = f"data/{setting_name}/{index}"
            try:
                if "static" in file:
                    exe = f"scripts/static_research -in:{file} -out:{data_path}"
                else:
                    exe = f"scripts/research -in:{file} -out:{data_path}"
                subprocess.run(exe, shell=True)
            except subprocess.CalledProcessError as e:
                with open(data_path + "/error", "w") as f:
                    f.write("raise error", e.returncode, e.stderr)
            except Exception as e:
                with open(data_path + "/error", "w") as f:
                    f.write("unknown error", e)

if __name__ == "__main__":
    main()