import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.typing import NDArray

from weather_forecasting.evaluate_model import evaluate_model
from weather_forecasting.rainfall.load_data import load_data
from weather_forecasting.rainfall.settings import settings


class Baseline:
    """Rainfall baseline that uses the sum of rainfall in the last target_delay
    timesteps in each window as its prediction."""

    def __init__(
        self,
        target_mean: float,
        target_std: float,
        target_delay: float,
    ) -> None:
        self.target_mean: float = target_mean
        self.target_std: float = target_std
        self.target_delay: float = target_delay

    def predict(
        self,
        dataset: tf.data.Dataset,
        **kwargs,
    ) -> NDArray[np.float32]:
        """Return a NumPy array of predictions for each sample in each batch in
        the given Dataset."""

        return np.concatenate([
            np.sum(
                samples[:, -self.target_delay:, 11] * \
                   self.target_std + self.target_mean,
                axis=1,
            )
            for samples, targets in dataset
        ])

    def evaluate(self, dataset: tf.data.Dataset, **kwargs) -> dict[str, float]:
        """Return a dictionary containing the MAE achieved by using the baseline
        method for prediction on the given dataset."""

        total_abs_error = 0
        samples_seen = 0
        for samples, targets in dataset:
            # Rainfall is 12th column
            predictions = np.sum(
                samples[:, -self.target_delay:, 11] * \
                    self.target_std + self.target_mean,
                axis=1,
            )
            #predictions = np.zeros(shape=targets.shape)   # For 2nd baseline
            total_abs_error += np.sum(np.abs(predictions - targets))
            samples_seen += samples.shape[0]

        return {'mae': total_abs_error / samples_seen}


def baseline_predictor() -> None:
    """Simple baseline for prediction that predicts that the cumulative rainfall
    in the next target_delay timesteps is the exact same as in the previous
    target_delay timesteps.

    Displays a sample of the predictions against their targets for the test
    dataset, and saves the plot to figures/rainfall/baseline_evaluation.png
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
    model = Baseline(mean, std, target_delay)

    print(f"Validation MAE: {model.evaluate(val_dataset)['mae']:.8f}")

    evaluate_model(model, test_dataset, "rain (mm)")
    plt.savefig(
        "figures/rainfall/baseline_evaluation.png",
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    baseline_predictor()
