import os

directory = "data/all"

seconds = 0
peract = {
    "chodzenie": 0,
    "chodzenie_reka": 0,
    "schodzenie": 0,
    "wchodzenie": 0,
    "siadanie": 0,
    "upadek": 0,
}

for dir, _, files in os.walk(directory):
    if dir == directory:
        continue

    sensor = f"{dir}/Accelerometer.csv"
    act = dir.partition('\\')[2].partition('-')[0]
    if act not in peract:
        if act.startswith('upadek'):
            act = "upadek"
        elif act.startswith('wchodzenie'):
            act = "wchodzenie"
        else:
            raise "wtf"

    with open(sensor) as f:
        lines = f.readlines()
        first_time, _, _ = lines[1].partition(',')
        last_time, _, _ = lines[-1].partition(',')
        diff = int(last_time) - int(first_time)
        seconds += diff * 1e-9
        peract[act] += (diff * 1e-9) / 60

print(seconds)
print(seconds / 60)
for key in peract:
    peract[key] = round(peract[key], 2)
print(peract)
# 2562.5075358000013
# 42.70845893000002
# {'chodzenie': 839.5031118, 'chodzenie_reka': 451.203857, 'schodzenie': 379.5250166, 'wchodzenie': 341.671932, 'siadanie': 264.80752650000005, 'upadek': 285.7960919}

import matplotlib.pyplot as plt

labels = peract.keys()
sizes = peract.values()

plt.pie(sizes, labels=labels, autopct='%1.1f%%')
# plt.title('Minuty na aktywność')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()