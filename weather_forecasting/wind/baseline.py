import numpy as np
import pandas as pd
import tensorflow as tf

from weather_forecasting.data_preprocessing import clean_data, feature_engineer
from weather_forecasting.wind.load_data import load_data
from weather_forecasting.wind.settings import settings


def evaluate_baseline(
    dataset: tf.data.Dataset,
    target_mean: float,
    target_std: float,
) -> float:
    """Return the MAE achieved by using the baseline method on the given
    dataset."""

    total_abs_error = 0
    samples_seen = 0
    for samples, targets in dataset:
        # Wind x and y are in columns 12 and 13
        predictions = samples[:, -1, 11:13] * target_std + target_mean
        total_abs_error += np.sum(np.abs(predictions - targets))
        samples_seen += samples.shape[0]

    return total_abs_error / samples_seen


def baseline_predictor() -> None:
    """Simple baseline for prediction that predicts that the wind vector in 10
    minutes is the exact same as it is currently.

    Prints the MAE on the validation and test datasets.
    """

    train_dataset, val_dataset, test_dataset = load_data(
        **settings,
    )

    raw_data = pd.read_csv(settings['data_location'], index_col="Date Time")
    raw_data = clean_data(raw_data, -9999.0)
    feature_engineer(raw_data)
    wind = raw_data[['wx (m/s)', 'wy (m/s)']]
    num_train_samples = int(settings['train_prop'] * len(wind))
    mean = wind[:num_train_samples].mean()
    std = wind[:num_train_samples].std()

    print(f"Validation MAE: {evaluate_baseline(val_dataset, mean, std):.8f}")
    print(f"Test MAE: {evaluate_baseline(test_dataset, mean, std):.8f}")


if __name__ == "__main__":
    baseline_predictor()
