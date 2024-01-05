from copy import deepcopy
from pandas import DataFrame as DF
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

id = 0
USE_SLERP = False
class Sample:
    def __init__(
        self,
        source: str,
        continuity_index: int,
        activity: str,
        gyro: DF,
        accelerometer: DF,
        orient: DF,
        gravity: DF,
    ):
        global id
        self.id = id
        id += 1

        self.source = source
        self.continuity_index = continuity_index
        self.activity = activity

        self.gyro = gyro
        self.accelerometer = accelerometer
        self.orient = orient
        self.gravity = gravity

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname} {self.id}: {self.activity}"

    def plot_sensor_data(self, sensor_data, title, domain = (0, float("inf"))):
        """Helper function to plot a specific sensor's data."""
        df = sensor_data

        filtered = df[df["time"] < domain[1] * 1e10]
        filtered = filtered[filtered["time"] > domain[0] * 1e10]
        for col in df.columns:
            # if col != "time":
            if col not in ["time", "roll", "pitch", "yaw"]:
                plt.plot(filtered["time"], filtered[col], label=col)

        plt.title(title)
        plt.legend()

    def getLength(self):
        return self.gyro["time"].iloc[-1] * 1e-9 - self.gyro["time"].iloc[0]* 1e-9

    def graph(self, domain=(0, float("inf"))):
        """Plots the data for all sensors in separate subplots."""
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))

        fig.suptitle(f"Data Source: {self.source}", fontsize=16, y=1.05)

        plt.subplot(4, 1, 1)
        self.plot_sensor_data(self.gyro, "Gyroscope Data", domain)

        plt.subplot(4, 1, 2)
        self.plot_sensor_data(self.accelerometer, "Accelerometer Data", domain)

        plt.subplot(4, 1, 3)
        self.plot_sensor_data(self.orient, "Orientation Data", domain)

        plt.subplot(4, 1, 4)
        self.plot_sensor_data(self.gravity, "Gravity Data", domain)

        plt.tight_layout()
        plt.show()

    def _plot_sensor_data_edges(self, subplot, sensor_data, title, margin_sec: int):
        """Helper function to plot a specific sensor's data."""
        df = sensor_data

        offset_nanosec = margin_sec * 1e9
        start_split = df[df["time"] > offset_nanosec].index[0]
        df_start = df.loc[: start_split - 1]

        maxtime = df["time"].iloc[-1]
        end_split = df[df["time"] > maxtime - offset_nanosec].index[0]
        df_end = df.loc[end_split:]

        plt.subplot(subplot)
        time_data = df_start["time"]
        for col in df_start.columns:
            if col != "time":
                plt.plot(time_data * 1e-9, df_start[col], label=col)

        plt.title(title)
        plt.legend()

        plt.subplot(subplot + 1)
        plt.title(title)
        # plt.legend() results in: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
        time_data = df_end["time"]
        for col in df_end.columns:
            if col != "time":
                plt.plot((time_data - maxtime) * 1e-9, df_end[col], label=col)

    def graph_edges(self, margin):
        """Plots the data for all sensors in separate subplots."""
        fig, axs = plt.subplots(4, 2, figsize=(10, 20))

        fig.suptitle(f"Data Source: {self.source}", fontsize=16, y=1.05)

        self._plot_sensor_data_edges(421, self.gyro, "Gyroscope Data", margin)
        self._plot_sensor_data_edges(
            423, self.accelerometer, "Accelerometer Data", margin
        )
        self._plot_sensor_data_edges(425, self.orient, "Orientation Data", margin)
        self._plot_sensor_data_edges(427, self.gravity, "Gravity Data", margin)

        plt.tight_layout()
        plt.show(block=True)

    def showAccPlot(self, window=128):
        acc_128 = self.accelerometer[0:window]
        print(acc_128.shape)

        x = acc_128['x']
        y = acc_128['y']
        z = acc_128['z']
        index = np.arange(0, window, 1)

        fig = plt.gcf()
        fig.set_size_inches(15, 8)
        plt.title(f"Accelerometer - {self.activity} - {window} samples")

        plt.plot(index, x, label="x")
        plt.plot(index, y, label="y")
        plt.plot(index, z, label="z")

        plt.legend()
        plt.show()

    def save_trim(self, start, end):
        with open(os.path.join(self.source, "trim.txt"), "w") as f:
            f.write(f"{start}|{end}")

    def resample(self, frequency_hz: float):
        interval_ms = (1 / frequency_hz) * 1000
        self.gyro = resample_sensor(self.gyro, interval_ms)
        self.accelerometer = resample_sensor(self.accelerometer, interval_ms)
        if USE_SLERP:
            self.orient = resample_sensor_quaternion(self.orient, interval_ms)
        else:
            self.orient = resample_sensor(self.orient, interval_ms)
        self.gravity = resample_sensor(self.gravity, interval_ms)

    def synchronize(self):
        """Effect:
        All sensors are split into the same time windows
        e.g. if sample had gyro readings in seconds [1, 5]
        but accelerometer [3, 6], synchronized would have readings
        for both in time window [3, 5].
        Necessary for models taking multiple sensors as inputs
        """
        pass  # TODO

def resample_sensor(sensor: DF, interval_ms: float):
    df = deepcopy(sensor) # for immutability, if performance gets too bad, remove and pray nothing breaks
    mi = np.min(df['time'])
    ma = np.max(df['time'])
    # print(mi, ma)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    resampled_df = df.resample(f'{interval_ms}L').mean()
    resampled_df_interpolated = resampled_df.interpolate(method='linear')
    df_reset = resampled_df_interpolated.reset_index()
    df_reset['time'] = df_reset['time'].astype('int64')
    df_reset.rename(columns={'index': 'time'}, inplace=True)
    # return df_reset
    return df_reset[(df_reset['time'] >= mi) & (df_reset['time'] <= ma)]

def resample_sensor_quaternion(df: DF, interval_ms: float):
    if df.empty:
        return df

    times = pd.to_timedelta(df['time'] / 1e9, unit="seconds").dt.total_seconds()

    if df[['qx', 'qy', 'qz', 'qw']].isnull().values.any():
        raise ValueError("NaN values found in quaternion data.")

    quaternions = df[['qx', 'qy', 'qz', 'qw']].to_numpy()
    if quaternions.shape[1] != 4:
        raise ValueError(f"Quaternion data has invalid shape: {quaternions.shape}")

    df = resample_sensor(df[['time', 'roll', 'pitch', 'yaw']], interval_ms)
    rotations = R.from_quat(quaternions)
    slerp = Slerp(times, rotations)
    new_rotations = slerp(df.time / 1e9)
    new_quats = new_rotations.as_quat()
    quaternion_df = pd.DataFrame(new_quats, columns=['qx', 'qy', 'qz', 'qw'])
    final_df = pd.concat([df, quaternion_df], axis=1)

    return final_df