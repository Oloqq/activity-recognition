import pandas as pd
from pandas import DataFrame as DF
import os
from .samples import *
from .suite import Suite
from .utils import verbosity

def with_intervals(df: DF):
    secs = (df["time"] - df["time"].shift(1)) / 1e9
    df["dt_ms"] = secs * 1000


def split_for_continuity(df: DF):
    with_intervals(df)

    acceptable = df.dt_ms.median() + 0.5
    split_indices = [0] \
                  + [i for i, val in enumerate(df['dt_ms'] > acceptable) if val] \
                  + [len(df)]
    df.drop(columns=['dt_ms'], inplace=True)
    return [df.iloc[split_indices[i]:split_indices[i+1]] for i in range(len(split_indices)-1)]


def load_sensor(path, trim: tuple | None) -> list[DF]:
    df = pd.read_csv(path)
    df['time'] = df['time'] - df['time'][0]
    df.drop(columns=['seconds_elapsed'], inplace=True)

    if trim != None:
        (trim_start, trim_end) = trim
        df = df[df['time'] * 1e-9 > trim_start]
        df = df[df['time'] < df['time'].iloc[-1] - trim_end * 1e9]

    return split_for_continuity(df)


def get_trim(path):
    if not os.path.isfile(path):
        return None

    with open(path, "r") as f:
        match f.read().split("|"):
            case [start, end]:
                return (int(start), int(end))
            case _:
                raise Exception(f"{path}: trim.txt has invalid format")


def load_raw_sample(sample_dir, activity) -> list[Sample]:
    trim = get_trim(os.path.join(sample_dir, "trim.txt"))
    gyro = load_sensor(os.path.join(sample_dir, "Gyroscope.csv"), trim)
    accelerometer = load_sensor(os.path.join(
        sample_dir, "Accelerometer.csv"), trim)
    orient = load_sensor(os.path.join(sample_dir, "Orientation.csv"), trim)
    gravity = load_sensor(os.path.join(sample_dir, "Gravity.csv"), trim)

    # assert len(gyro) == len(accelerometer) == len(orient) == len(gravity) # tu wywala
    # if not (len(gyro) == len(accelerometer) == len(orient) == len(gravity)):
    #     print("ERROR: len(gyro) == len(accelerometer) == len(orient) == len(gravity)")
    #     print(sample_dir)
    #     print(len(gyro), len(accelerometer), len(orient), len(gravity))
    #     return []

    if verbosity() > 0:
        print(f"{sample_dir} -> {len(gyro)} samples")

    samples = []
    for i in range(len(gyro)):
        samples.append(Sample(sample_dir, i, activity, gyro[i], accelerometer[i], orient[i], gravity[i]))

    return samples


def infer_activity(dirname: str):
    activity = dirname.split("-")[0]
    match activity.split("_"):
        case ["chodzenie", "reka"]:
            return "chodz_reka"
    return dirname[0:5]


def load_raw_suite(suite_dir) -> Suite:
    samples = []
    for sample_dir in [f for f in os.listdir(suite_dir) if not os.path.isfile(f)]:
        samples.extend(
            load_raw_sample(
                os.path.join(suite_dir, sample_dir),
                infer_activity(sample_dir)
            ))
    return Suite(suite_dir, samples)
