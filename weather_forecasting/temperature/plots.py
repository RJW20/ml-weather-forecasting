import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from weather_forecasting.temperature.load_data import (
    clean_data,
    feature_engineer,
)


def subplots() -> None:

    raw_data = pd.read_csv("weather_data/2017_2023.csv", index_col="Date Time")
    raw_data = clean_data(raw_data, -9999.0)
    raw_data = raw_data[5::6]
    plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)', 'rain (mm)']
    plot_features = raw_data[plot_cols]
    plot_features.index = pd.to_datetime(
        raw_data.index,
        format='%d.%m.%Y %H:%M:%S',
    )
    _ = plot_features.plot(subplots=True)
    plt.show()


def violin() -> None:

    raw_data = pd.read_csv(
        "weather_data/2017_2023.csv",
        index_col="Date Time",
    )
    raw_data = clean_data(raw_data, -9999.0)
    feature_engineer(raw_data)

    num_train_samples = int(0.5 * len(raw_data.index))
    mean = raw_data[:num_train_samples].mean(axis=0)
    raw_data -= mean
    std = raw_data[:num_train_samples].std(axis=0)
    raw_data /= std

    labels = raw_data.keys()
    raw_data = raw_data.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=raw_data)
    ax.set_xticklabels(labels, rotation=90)
    plt.show()


if __name__ == "__main__":
    #subplots()
    violin()
