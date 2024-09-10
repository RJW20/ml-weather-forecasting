import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.typing import NDArray

from weather_forecasting.evaluate_model import evaluate_model
from weather_forecasting.temperature.load_data import load_data
from weather_forecasting.temperature.settings import settings


class Baseline:
    """Temperature baseline that uses the last timestep in each window as its
    prediction."""

    def __init__(self, target_mean: float, target_std: float) -> None:
        self.target_mean: float = target_mean
        self.target_std: float = target_std

    def predict(
        self,
        dataset: tf.data.Dataset,
        **kwargs,
    ) -> NDArray[np.float32]:
        """Return a NumPy array of predictions for each sample in each batch in
        the given Dataset."""

        return np.concatenate([
            samples[:,-1,1] * self.target_std + self.target_mean    #T in col 2
            for samples, targets in dataset
        ])

    def evaluate(self, dataset: tf.data.Dataset, **kwargs) -> dict[str, float]:
        """Return a dictionary containing the MAE achieved by using the baseline
        method for prediction on the given dataset."""

        total_abs_error = 0
        samples_seen = 0
        for samples, targets in dataset:
            # Temperature is 2nd column
            predictions = samples[:,-1,1] * self.target_std + self.target_mean
            total_abs_error += np.sum(np.abs(predictions - targets))
            samples_seen += samples.shape[0]

        return {'mae': total_abs_error / samples_seen}


def baseline_predictor() -> None:
    """Simple baseline for prediction that predicts that the temperature in 24
    hours is the exact same as it is currently.

    Displays a sample of the predictions against their targets for the test
    dataset, and saves the plot to figures/temperature/baseline_evaluation.png
    Prints the MAE on the validation and test datasets.
    """

    train_dataset, val_dataset, test_dataset = load_data(
        **settings,
    )

    temperature = pd.read_csv(settings['data_location'])["T (degC)"]
    num_train_samples = int(settings['train_prop'] * len(temperature))
    mean = temperature[:num_train_samples].mean()
    std = temperature[:num_train_samples].std()
    model = Baseline(mean, std)

    print(f"Validation MAE: {model.evaluate(val_dataset)['mae']:.8f}")

    evaluate_model(model, test_dataset, "T (degC)")
    plt.savefig("figures/temperature/dense_evaluation.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    baseline_predictor()
