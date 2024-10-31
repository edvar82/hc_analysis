import ordpy
import numpy as np
import pandas as pd
from matplotlib import pylab as plt

plt.rcParams.update({"font.size": 12})

def get_hc_plan_quota(quota_file_name):
    data = []
    with open(quota_file_name, "r") as file:
        for line in file:
            if line.strip() and not line.startswith(">>>"):
                try:
                    values = line.split()
                    data.append((float(values[0]), float(values[1])))
                except ValueError:
                    continue

    ht, cjt = zip(*data)
    return ht, cjt

colors = {
    'android.sensor.linear_acceleration#max': 'deepskyblue',
    'android.sensor.linear_acceleration#mean': 'steelblue',
    'speed#min': 'lightcoral',
    'speed#max': 'indianred',
    'speed#mean': 'royalblue',
    'android.sensor.linear_acceleration#min': 'cornflowerblue',
    'android.sensor.magnetic_field#max': 'purple',
    'android.sensor.magnetic_field#min': 'mediumpurple',
    'android.sensor.gyroscope#mean': 'darkgrey',
    'android.sensor.gyroscope#min': 'silver',
    'android.sensor.accelerometer#std': 'limegreen',
    'android.sensor.gyroscope#max': 'dimgray',
    'android.sensor.magnetic_field#mean': 'orchid',
    'android.sensor.light#max': 'gold',
    'android.sensor.light#min': 'khaki',
    'sound#mean': 'chocolate',
    'sound#min': 'saddlebrown',
    'android.sensor.light#mean': 'yellow',
    'android.sensor.step_counter#max': 'darkorange',
    'android.sensor.magnetic_field_uncalibrated#min': 'blueviolet',
    'android.sensor.gravity#min': 'slateblue',
    'android.sensor.gyroscope_uncalibrated#min': 'lightgrey',
    'android.sensor.magnetic_field_uncalibrated#mean': 'plum',
    'sound#max': 'darkred',
    'android.sensor.magnetic_field_uncalibrated#max': 'fuchsia',
    'android.sensor.step_counter#min': 'tomato',
    'android.sensor.gyroscope_uncalibrated#mean': 'gainsboro',
    'android.sensor.step_counter#mean': 'orangered',
    'activityrecognition#1': 'darkblue',
    'android.sensor.gravity#mean': 'mediumslateblue',
    'android.sensor.linear_acceleration#std': 'dodgerblue',
    'android.sensor.gravity#max': 'slategrey',
    'android.sensor.pressure#max': 'darkkhaki',
    'android.sensor.pressure#min': 'olive',
    'android.sensor.gyroscope_uncalibrated#max': 'darkslategray',
    'android.sensor.rotation_vector#max': 'teal',
    'android.sensor.game_rotation_vector#mean': 'cadetblue',
    'android.sensor.game_rotation_vector#max': 'turquoise',
    'android.sensor.orientation#std': 'lightseagreen',
    'android.sensor.proximity#min': 'seagreen',
    'android.sensor.pressure#mean': 'greenyellow',
    'android.sensor.game_rotation_vector#min': 'springgreen',
    'android.sensor.rotation_vector#mean': 'mediumseagreen',
    'android.sensor.rotation_vector#min': 'lightgreen',
    'android.sensor.gyroscope#std': 'mediumaquamarine',
    'android.sensor.proximity#mean': 'aquamarine',
    'android.sensor.game_rotation_vector#std': 'mediumturquoise',
    'android.sensor.accelerometer#min': 'darkcyan',
    'android.sensor.gravity#std': 'deepskyblue',
    'android.sensor.gyroscope_uncalibrated#std': 'aqua',
    'android.sensor.orientation#mean': 'paleturquoise',
    'android.sensor.orientation#max': 'powderblue',
    'android.sensor.orientation#min': 'aliceblue',
    'android.sensor.proximity#max': 'lavender',
    'android.sensor.accelerometer#max': 'lightsteelblue',
    'sound#std': 'lightslategray',
    'android.sensor.rotation_vector#std': 'whitesmoke',
    'android.sensor.accelerometer#mean': 'azure',
    'android.sensor.light#std': 'lightcyan'
}

def plot_plan(series, series_name, x_label, y_label, fig_name, quota_files):
    fig, ax = plt.subplots(figsize=[8, 6])
    labels = {name: 0 for name in series_name}
    markers = {name: 'o' for name in series_name}

    for measures, name in zip(series, series_name):
        ax.scatter(*measures, label=name, s=40, marker=markers[name], color=colors[name])
        labels[name] += 1

    for quota_file in quota_files:
        ht, cjt = get_hc_plan_quota(quota_file)
        plt.plot(ht, cjt, color="black", linewidth=0.8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    plt.savefig(f"{fig_name}.pdf", format="pdf", dpi=300)

def main():
    df = pd.read_csv("dataset_halfSecondWindow.csv")

    columns = [
        'android.sensor.linear_acceleration#max', 'android.sensor.linear_acceleration#mean', 
        'speed#min', 'speed#max', 'speed#mean', 'android.sensor.linear_acceleration#min', 
        'android.sensor.magnetic_field#max', 'android.sensor.magnetic_field#min', 
        'android.sensor.gyroscope#mean', 'android.sensor.gyroscope#min', 
        'android.sensor.accelerometer#std', 'android.sensor.gyroscope#max', 
        'android.sensor.magnetic_field#mean', 'android.sensor.light#max', 
        'android.sensor.light#min', 'sound#mean', 'sound#min', 
        'android.sensor.light#mean', 'android.sensor.step_counter#max', 
        'android.sensor.magnetic_field_uncalibrated#min', 'android.sensor.gravity#min', 
        'android.sensor.gyroscope_uncalibrated#min', 'android.sensor.magnetic_field_uncalibrated#mean', 
        'sound#max', 'android.sensor.magnetic_field_uncalibrated#max', 
        'android.sensor.step_counter#min', 'android.sensor.gyroscope_uncalibrated#mean', 
        'android.sensor.step_counter#mean', 'activityrecognition#1', 
        'android.sensor.gravity#mean', 'android.sensor.linear_acceleration#std', 
        'android.sensor.gravity#max', 'android.sensor.pressure#max', 
        'android.sensor.pressure#min', 'android.sensor.gyroscope_uncalibrated#max', 
        'android.sensor.rotation_vector#max', 'android.sensor.game_rotation_vector#mean', 
        'android.sensor.game_rotation_vector#max', 'android.sensor.orientation#std', 
        'android.sensor.proximity#min', 'android.sensor.pressure#mean', 
        'android.sensor.game_rotation_vector#min', 'android.sensor.rotation_vector#mean', 
        'android.sensor.rotation_vector#min', 'android.sensor.gyroscope#std', 
        'android.sensor.proximity#mean', 'android.sensor.game_rotation_vector#std', 
        'android.sensor.accelerometer#min', 'android.sensor.gravity#std', 
        'android.sensor.gyroscope_uncalibrated#std', 'android.sensor.orientation#mean', 
        'android.sensor.orientation#max', 'android.sensor.orientation#min', 
        'android.sensor.proximity#max', 'android.sensor.accelerometer#max', 
        'sound#std', 'android.sensor.rotation_vector#std', 'android.sensor.accelerometer#mean', 
        'android.sensor.light#std'
    ]

    df_clean = df[columns].dropna()
    time_series = [df_clean[col].values for col in columns]

    samples = columns

    dx = 4
    HC = [ordpy.complexity_entropy(series, dx=dx) for series in time_series if len(series) >= dx]

    quota_files = ["limits/trozos-N24.q1", "limits/continua-N24.q1"]

    plot_plan(
        series=HC,
        series_name=samples,
        x_label="Permutation entropy, $H$",
        y_label="Statistical complexity, $C$",
        fig_name="Entropy-Complexity-Plane",
        quota_files=quota_files
    )

    plt.show()


if __name__ == "__main__":
    main()
