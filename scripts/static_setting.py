import json
import sys

setting_name = sys.argv[1]

with open(setting_name, "r") as f:
    j = json.load(f)

j["main_network"]["bias_mutate_rate"] = 0.01
j["main_network"]["activation_mutate_rate"] = 0.01
j["main_network"]["enable_mutate_rate"] = 0.01
j["main_network"]["weight_mutate_rate"] = 0.01
j["main_network"]["conn_add_prob"] = 0.01
j["main_network"]["conn_delete_prob"] = 0.01
j["main_network"]["node_add_prob"] = 0.01
j["main_network"]["node_delete_prob"] = 0.01

j["main_network"]["activation_functions"] = [
    "relu",
    "y=x",
    "tanh",
    "sin",
    "cos",
    "neg",
    "abs",
    "y=2x-1",
    "gauss",
    "signbit",
    "sign"
]

ls = setting_name.split("/")
new_setting_name = "/".join([ls[0], "static_" + ls[1]])
with open(new_setting_name, "w") as f:
    f.write(json.dumps(j, indent=4))
