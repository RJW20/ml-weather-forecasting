import numpy as np
import pandas as pd
import tensorflow as tf

from weather_forecasting.rainfall.load_data import load_data
from weather_forecasting.rainfall.settings import settings


def evaluate_baseline(
    dataset: tf.data.Dataset,
    target_delay: int,
    target_mean: float,
    target_std: float,
) -> float:
    """Return the MAE achieved by using the baseline method on the given
    dataset."""

    total_abs_error = 0
    samples_seen = 0
    for samples, targets in dataset:
        # Rain is 11th column
        predictions = sum(
            samples[:, i, 10] * target_std + target_mean
            for i in range(-target_delay,0)
        )
        #predictions = np.zeros(shape=targets.shape)
        total_abs_error += np.sum(np.abs(predictions - targets))
        samples_seen += samples.shape[0]

    return total_abs_error / samples_seen


def baseline_predictor() -> None:
    """Simple baseline for prediction that predicts that the rainfall in the
    next timeframe is the exact same as in the previous timeframe.

    Prints the MAE on the validation and test datasets.
    """

    train_dataset, val_dataset, test_dataset = load_data(
        **settings,
    )

    rainfall = pd.read_csv(settings['data_location'])["rain (mm)"]
    num_train_samples = int(settings['train_prop'] * len(rainfall))
    mean = rainfall[:num_train_samples].mean()
    std = rainfall[:num_train_samples].std()

    target_delay = settings['target_delay']
    print(
        "Validation MAE: "
        f"{evaluate_baseline(val_dataset, target_delay, mean, std):.8f}"
    )
    print(
        "Test MAE: "
        f"{evaluate_baseline(test_dataset, target_delay, mean, std):.8f}"
    )


if __name__ == "__main__":
    baseline_predictor()
