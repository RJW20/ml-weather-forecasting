import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

MAX_EPOCHS = 25
PATIENCE = 5


def loss_curves(history: keras.callbacks.History) -> None:
    """Generate a plt.figure of the loss curves for validation and training
    MAE contained in the given history."""

    mae = history.history["mae"]
    val_mae = history.history["val_mae"]
    epochs = range(1, len(mae) + 1)
    plt.figure()
    plt.plot(epochs, mae, "bo", label="Training MAE")
    plt.plot(epochs, val_mae, "b", label="Validation MAE")
    plt.legend()


def train_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    save_location: str,
) -> None:
    """Train the given model on the given training dataset
    
    Saves the best version of the model to the given save location (should end
    in .keras).
    Displays the loss curves for validation and training MAE.
    """

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            save_location,
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            mode='min',
        )
    ]
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    history = model.fit(
        train_dataset,
        epochs=MAX_EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
    )

    loss_curves(history)
