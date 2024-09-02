from tensorflow import keras
from tensorflow.keras import layers

from weather_forecasting.evaluate_model import evaluate_model
from weather_forecasting.rainfall.load_data import load_data
from weather_forecasting.rainfall.settings import settings
from weather_forecasting.train_model import train_model


def simple_dense() -> None:
    """Implements a simple dense neural network for rainfall prediction.
    
    Saves the final model to models/rainfall/dense.keras
    Prints the MAE on the test dataset.
    Displays the loss curves for validation and training MAE.
    """

    train_dataset, val_dataset, test_dataset = load_data(
        **settings,
    )

    num_features = train_dataset.element_spec[0].shape[2]

    inputs = keras.Input(shape=(settings['window_size'], num_features))
    x = layers.Flatten()(inputs)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    save_location = "models/rainfall/dense.keras"
    train_model(model, train_dataset, val_dataset, save_location)
    model = keras.models.load_model(save_location)
    evaluate_model(model, test_dataset, "rain (mm)")


if __name__ == "__main__":
    simple_dense()
