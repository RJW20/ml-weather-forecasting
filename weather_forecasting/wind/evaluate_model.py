import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


def test_predictions(model: keras.Model, test_dataset: tf.data.Dataset) -> None:
    """Compute predictions for the test dataset and generate a plt.figure
    displaying the last 25 predicted vectors and their targets."""

    targets = np.concatenate(list(targets for samples, targets in test_dataset))
    targets_speed = np.linalg.norm(targets, axis=1, ord=2, keepdims=True)
    targets_directions = targets / targets_speed
    predictions = model.predict(test_dataset, verbose=0)
    predictions_speed = np.linalg.norm(
        predictions,
        axis=1,
        ord=2,
        keepdims=True,
    )
    predictions_directions = predictions / predictions_speed

    plt.figure(figsize=(18, 9))
    plt.subplot(2, 1, 1)
    plt.quiver(
        [_ for _ in range(25)],
        [2] * 25,
        targets_directions[-25:, 0],
        targets_directions[-25:,1],
        pivot="mid",
        label="Targets",
        color='C0',
        scale=1.2,
        scale_units='x',
        width=0.13,
        headwidth=3,
        headlength=2.5,
        headaxislength=2.5,
        units='x',
    )
    plt.quiver(
        [_ for _ in range(25)],
        [1] * 25,
        predictions_directions[-25:, 0],
        predictions_directions[-25:,1],
        pivot="mid",
        label="Predictions",
        color='C1',
        scale=1.2,
        scale_units='x',
        width=0.13,
        headwidth=3,
        headlength=2.5,
        headaxislength=2.5,
        units='x',
    )
    plt.ylim(0, 3)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("wind direction")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(range(25), targets_speed[-25:], marker="o", label="Targets")
    plt.plot(
        range(25),
        predictions_speed[-25:],
        marker="x",
        label="Predictions",
    )
    plt.ylabel("wind speed (m/s)")
    plt.legend()



def evaluate_model(model: keras.Model, test_dataset: tf.data.Dataset) -> None:
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
    test_predictions(model, test_dataset)
    plt.show()
