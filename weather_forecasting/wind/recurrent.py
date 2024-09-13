import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from weather_forecasting.train_model import train_model
from weather_forecasting.wind.evaluate_model import evaluate_model
from weather_forecasting.wind.load_data import load_data
from weather_forecasting.wind.settings import settings


def simple_recurrent() -> None:
    """Implements a simple recurrent neural network for wind prediction.
    
    Saves the final model to models/wind/recurrent.keras
    Displays the loss curves for validation and training MAE, and saves
    the plot to figures/wind/recurrent_training.png.
    Displays a sample of the predictions against their targets for the test
    dataset, and saves the plot to figures/wind/recurrent_evaluation.png.
    Prints the MAE on the test dataset.
    """

    train_dataset, val_dataset, test_dataset = load_data(
        **settings,
    )

    num_features = train_dataset.element_spec[0].shape[2]

    inputs = keras.Input(shape=(settings['window_size'], num_features))
    x = layers.LSTM(32)(inputs)
    outputs = layers.Dense(2)(x)
    model = keras.Model(inputs, outputs)

    save_location = "models/wind/recurrent.keras"
    train_model(model, train_dataset, val_dataset, save_location)
    plt.savefig("figures/wind/recurrent_training.png", bbox_inches="tight")
    model = keras.models.load_model(save_location)
    evaluate_model(model, test_dataset)
    plt.savefig("figures/wind/recurrent_evaluation.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    simple_recurrent()
