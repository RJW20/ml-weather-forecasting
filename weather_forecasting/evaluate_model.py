import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


def test_predictions(
    model: keras.Model,
    test_dataset: tf.data.Dataset,
    target_label: str,
) -> None:
    """Compute predictions for the test dataset and generate a plt.figure
    displaying a sample of 25 (consecutive) hours of predicted values and their
    targets."""

    targets = np.concatenate([targets for samples, targets in test_dataset])
    predictions = model.predict(test_dataset, verbose=0)
    start = random.randrange(0, len(targets) - 150)
    plt.figure(figsize=(15, 4.8))
    plt.plot(
        range(25),
        targets[start:start + 150:6],
        marker="o",
        label="Targets",
    )
    plt.plot(
        range(25),
        predictions[start:start + 150:6],
        marker="x",
        label="Predictions",
    )
    plt.legend()
    plt.ylabel(target_label)


def evaluate_model(
    model: keras.Model,
    test_dataset: tf.data.Dataset,
    target_label: str,
) -> None:
    """Evaluate the given model on the given testing dataset.
    
    Prints the MAE on the test dataset.
    Displays a sample of the predictions against their targets for the test
    dataset.
    """

    print(
        "Test MAE: "
        f"{model.evaluate(
            test_dataset,
            verbose=0,
            return_dict=True,
        )['mae']:.8f}"
    )
    test_predictions(model, test_dataset, target_label)
    plt.show()
