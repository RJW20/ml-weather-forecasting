import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from weather_forecasting.temperature.load_data import load_data
from weather_forecasting.temperature.settings import settings


def simple_dense() -> None:
    """Implements a simple dense neural network for temperature prediction.
    
    Saves the final model to models/dense.keras
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

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "models/dense.keras",
            save_best_only=True,
        )
    ]
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        callbacks=callbacks,
    )

    model = keras.models.load_model("models/dense.keras")
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
    simple_dense()
