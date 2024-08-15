import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.typing import NDArray
from tensorflow import keras


def generate_datasets(
    data: NDArray[np.float32],
    targets: NDArray[np.float32],
    sampling_rate: int,
    window_size: int,
    batch_size: int,
    target_delay: int,
    num_train_samples: int,
    num_val_samples: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Return 3 tf.data.Datasets that iterate over the given data and targets.
     
    The samples in the Datasets are batches of windows of given window_size and
    batch_size, sampling at rate sampling_rate.
    The targets in the Datasets are batches of values that are target_delay *
    sampling_rate ahead of the samples.
    """

    delay = sampling_rate * (window_size + target_delay - 1)

    train_dataset = keras.utils.timeseries_dataset_from_array(
        data[:-delay],
        targets[delay:],
        sampling_rate=sampling_rate,
        sequence_length=window_size,
        batch_size=batch_size,
        shuffle=True,
        start_index=0,
        end_index=num_train_samples,
    )
    val_dataset = keras.utils.timeseries_dataset_from_array(
        data[:-delay],
        targets[delay:],
        sampling_rate=sampling_rate,
        sequence_length=window_size,
        batch_size=batch_size,
        shuffle=True,
        start_index=num_train_samples,
        end_index=num_train_samples + num_val_samples,
    )
    test_dataset = keras.utils.timeseries_dataset_from_array(
        data[:-delay],
        targets[delay:],
        sampling_rate=sampling_rate,
        sequence_length=window_size,
        batch_size=batch_size,
        shuffle=True,
        start_index=num_train_samples+num_val_samples,
    )

    return train_dataset, val_dataset, test_dataset


def ensure_shape(
    dataset: tf.data.Dataset,
    window_size: int,
    num_features: int
) -> tf.data.Dataset:
    """Set the shape of samples in the batches of the dataset manually."""

    return dataset.map(
        lambda x, y: (tf.ensure_shape(x, (None, window_size, num_features)), y)
    )


def load_data(
    data_location: str,
    *,
    train_prop: float = 0.5,
    val_prop: float = 0.25,
    sampling_rate: int = 6,
    window_size: int = 10,
    batch_size: int = 64,
    target_delay: int = 24,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load and prepare the weather data saved at the given data_location.

    Returns 3 tf.data.Datasets, each containing the training, validation and
    testing data respectively.
    Parameters:
    - data_location: the filename (csv) for the saved weather data to open.
    - train_prop: the proportion of the data to include in the training Dataset.
    - val_prop: the proportion of the data to include in the validation Dataset.
    - sampling_rate: the period between successive individual timesteps within
    windows.
    - window_size: the number of timesteps to include per window in the
    Datasets.
    - batch_size: the number of windows to return per call to next on each of
    the 3 Datasets.
    - target_delay: sampling_rate * target_delay is the number of timesteps into
    the future the target is from the data used to predict it.
    Raises a FileNotFoundError if the file does not exist.
    """

    raw_data = pd.read_csv(data_location, index_col="Date Time").drop(
        columns=["rain (mm)"],
    )
    targets = raw_data['T (degC)'].to_numpy(dtype=np.float32)

    num_train_samples = int(train_prop * len(raw_data.index))
    num_val_samples = int(val_prop * len(raw_data.index))

    mean = raw_data[:num_train_samples].mean(axis=0)
    raw_data -= mean
    std = raw_data[:num_train_samples].std(axis=0)
    raw_data /= std

    data = raw_data.to_numpy(dtype=np.float32)

    train_dataset, val_dataset, test_dataset = generate_datasets(
        data,
        targets,
        sampling_rate=sampling_rate,
        window_size=window_size,
        batch_size=batch_size,
        target_delay=target_delay,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples
    )

    train_dataset = ensure_shape(train_dataset, window_size, data.shape[-1])
    val_dataset = ensure_shape(val_dataset, window_size, data.shape[-1])
    test_dataset = ensure_shape(test_dataset, window_size, data.shape[-1])

    return train_dataset, val_dataset, test_dataset
