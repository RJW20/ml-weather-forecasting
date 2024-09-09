import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.typing import NDArray
from tensorflow import keras


def clean_data(data: pd.DataFrame, nan_value: float) -> pd.DataFrame:
    """Return the cleaned given DataFrame.
    
    Linearly interpolates all data at points with the given nan_value.
    """

    return data.replace(nan_value, np.nan).interpolate()


def feature_engineer(data: pd.DataFrame) -> None:
    """Carry out feature engineering on the given DataFrame.
    
    Turns the wind velocity and wind direction columns into wind x and y speeds.
    Adds daily and yearly sine and cosine time-signal columns.
    """

    wv = data.pop('wv (m/s)')
    wd_rad = data.pop('wd (deg)')*np.pi / 180
    data['wx (m/s)'] = wv*np.cos(wd_rad)
    data['wy (m/s)'] = wv*np.sin(wd_rad)

    timestamp_seconds = pd.to_datetime(
        data.index,
        format='%d.%m.%Y %H:%M:%S',
    ).map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day
    data['day sin'] = np.sin(timestamp_seconds * (2 * np.pi / day))
    data['day cos'] = np.cos(timestamp_seconds * (2 * np.pi / day))
    data['year sin'] = np.sin(timestamp_seconds * (2 * np.pi / year))
    data['year cos'] = np.cos(timestamp_seconds * (2 * np.pi / year))


def normalize(
    data: pd.DataFrame,
    num_train_samples: int,
) -> tuple[float, float]:
    """Normalize the given DataFrame, using the mean and standard deviation
    calculated from the rows to be used for training.
    
    Returns the mean and standard deviation used.
    """

    mean = data[:num_train_samples].mean(axis=0)
    data -= mean
    std = data[:num_train_samples].std(axis=0)
    data /= std

    return mean, std


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
    The targets in the Datasets are batches of values that are sampling_rate *
    target_delay ahead of the samples.
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
        shuffle=False,
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
