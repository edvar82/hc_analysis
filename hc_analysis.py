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
    "Acelerômetro (std)": "limegreen",
    "Aceleração Linear (max)": "deepskyblue",
    "Velocidade (mean)": "royalblue",
    "Giroscópio (mean)": "darkgrey",
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
    df = pd.read_csv("dataset_5secondWindow.csv")

    columns = [
        'android.sensor.accelerometer#std',
        'android.sensor.linear_acceleration#max',
        'speed#mean',
        'android.sensor.gyroscope#mean'
    ]

    df_clean = df[columns].dropna()

    time_series = [df_clean[col].values for col in columns]

    samples = ["Acelerômetro (std)", "Aceleração Linear (max)", "Velocidade (mean)", "Giroscópio (mean)"]

    HC = [ordpy.complexity_entropy(series, dx=4) for series in time_series]

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
