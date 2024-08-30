import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from weather_forecasting.temperature.load_data import load_data
from weather_forecasting.temperature.settings import settings


def recurrent_dropout() -> None:
    """Implements a recurrent neural network with dropout for temperature
    prediction.
    
    Saves the final model to models/temperature/recurrent_dropout.keras
    Prints the MAE on the validation datasets.
    Displays the loss curves for validation and training.
    """

    train_dataset, val_dataset, test_dataset = load_data(
        **settings,
    )

    num_features = train_dataset.element_spec[0].shape[2]

    inputs = keras.Input(shape=(settings['window_size'], num_features))
    x = layers.LSTM(32, recurrent_dropout=0.25)(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "models/temperature/recurrent_dropout.keras",
            save_best_only=True,
        )
    ]
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    history = model.fit(
        train_dataset,
        epochs=25,
        validation_data=val_dataset,
        callbacks=callbacks,
    )

    model = keras.models.load_model("models/temperature/recurrent_dropout.keras")
    print(f"Test MAE: {model.evaluate(test_dataset, verbose=0)[1]:.8f}")

    mae = history.history["mae"]
    val_mae = history.history["val_mae"]
    epochs = range(1, len(mae) + 1)
    plt.figure()
    plt.plot(epochs, mae, "bo", label="Training MAE")
    plt.plot(epochs, val_mae, "b", label="Validation MAE")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    recurrent_dropout()
