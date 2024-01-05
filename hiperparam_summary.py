import os
import numpy as np
import json

# Directory path to search for files
directory = "parameter_comparison"

data = {}

def flatten_data(data):
    classifier = data["classifier"]
    params = '-'.join([f"{key}-{value}" for key, value in data["params"].items()])
    accuracy = data["accuracy"]
    return [f"{classifier}-{params}", accuracy]

for filename in os.listdir(directory):
    if filename.startswith("summary"):
        l = json.load(open(directory + "/" + filename))
        for obj in l:
            name, val = flatten_data(obj)
            if name not in data:
                data[name] = {}
                data[name]["values"] = []
            data[name]["values"].append(val)

for name, instance in data.items():
    vals = instance.pop("values")
    instance["min accuracy"] = round(np.min(vals), 2)
    instance["max accuracy"] = round(np.max(vals), 2)
    instance["mean accuracy"] = round(np.mean(vals), 2)
    instance["standard deviation"] = round(np.std(vals), 2)
    instance["instances"] = len(vals)


with open("summary.json", "w") as f:
    json.dump(data, f, indent=4)