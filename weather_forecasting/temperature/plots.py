import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from weather_forecasting.temperature.load_data import clean_data


def temperature() -> None:

    raw_data = pd.read_csv("weather_data/2017_2023.csv", index_col="Date Time")
    raw_data = clean_data(raw_data, -9999.0)
    temperature = raw_data["T (degC)"]
    #plt.plot(range(len(temperature)), temperature)
    plt.plot(range(1440), temperature[:1440])
    plt.show()


def pressure() -> None:

    raw_data = pd.read_csv("weather_data/2017_2023.csv", index_col="Date Time")
    raw_data = clean_data(raw_data, -9999.0)
    pressure = raw_data["p (mbar)"]
    plt.plot(range(len(pressure)), pressure)
    plt.show()


def rain() -> None:

    raw_data = pd.read_csv("weather_data/2017_2023.csv", index_col="Date Time")
    raw_data = clean_data(raw_data, -9999.0)
    rain = raw_data["rain (mm)"]
    plt.plot(range(len(rain)), rain)
    plt.show()


def violin() -> None:

    raw_data = pd.read_csv(
        "weather_data/2017_2023.csv",
        index_col="Date Time",
    )
    labels = raw_data.keys()
    raw_data = clean_data(raw_data, -9999.0)

    num_train_samples = int(0.5 * len(raw_data.index))
    mean = raw_data[:num_train_samples].mean(axis=0)
    raw_data -= mean
    std = raw_data[:num_train_samples].std(axis=0)
    raw_data /= std

    raw_data = raw_data.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=raw_data)
    ax.set_xticklabels(labels, rotation=90)
    plt.show()


if __name__ == "__main__":
    
    rain()
