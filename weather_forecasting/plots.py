import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from weather_forecasting.temperature.load_data import (
    clean_data,
    feature_engineer,
    normalize,
)


def subplots(
    plot_cols: list[str] = ["p (mbar)", "T (degC)", "rain (mm)"],
) -> None:
    
    raw_data = pd.read_csv("weather_data/2017_2023.csv", index_col="Date Time")
    raw_data = clean_data(raw_data, -9999.0)
    feature_engineer(raw_data)
    raw_data = raw_data[5::6]
    raw_data = raw_data[plot_cols]
    raw_data.index = pd.to_datetime(raw_data.index, format="%d.%m.%Y %H:%M:%S")
    raw_data.plot(subplots=True)
    plt.show()


def violin() -> None:

    raw_data = pd.read_csv(
        "weather_data/2017_2023.csv",
        index_col="Date Time",
    )
    raw_data = clean_data(raw_data, -9999.0)
    feature_engineer(raw_data)
    normalize(raw_data, int(0.5 * len(raw_data.index)))

    labels = raw_data.keys()
    raw_data = raw_data.melt(var_name="Column", value_name="Normalized")
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x="Column", y="Normalized", data=raw_data)
    ax.set_xticklabels(labels, rotation=90)
    plt.show()


if __name__ == "__main__":
    subplots()
    # violin()
