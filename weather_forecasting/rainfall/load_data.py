import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.typing import NDArray

from weather_forecasting.data_preprocessing import (
    clean_data,
    ensure_shape,
    feature_engineer,
    generate_datasets,
    normalize,
)


def create_targets(rainfall: pd.Series, timesteps: int) -> NDArray[np.float32]:
    """Return an (np.float32) ndarray containing the cumulative rainfall within
    the previous (inclusive) given timesteps for each entry in the given
    rainfall data."""

    return rainfall.rolling(timesteps).sum().to_numpy(dtype=np.float32)


def load_data(
    data_location: str,
    *,
    train_prop: float,
    val_prop: float,
    sampling_rate: int,
    window_size: int,
    batch_size: int,
    target_delay: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load and prepare the weather data saved at the given data_location.

    Returns 3 tf.data.Datasets, each containing the training, validation and
    testing data respectively. The targets within each Dataset are the
    cumulative rainfall within the (indirectly) specified number of timesteps
    into the future.
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
    the future the target is created from cumulatively.
    Raises a FileNotFoundError if no file is found at the given data_location.
    """

    raw_data = pd.read_csv(data_location, index_col="Date Time")
    raw_data = clean_data(raw_data, -9999.0)
    feature_engineer(raw_data)
    targets = create_targets(raw_data['ran (mm)'], sampling_rate * target_delay)

    num_train_samples = int(train_prop * len(raw_data.index))
    num_val_samples = int(val_prop * len(raw_data.index))

    normalize(raw_data, num_train_samples)

    data = raw_data.to_numpy(dtype=np.float32)

    train_dataset, val_dataset, test_dataset = generate_datasets(
        data,
        targets,
        sampling_rate=sampling_rate,
        window_size=window_size,
        batch_size=batch_size,
        target_delay=target_delay,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
    )

    train_dataset = ensure_shape(train_dataset, window_size, data.shape[-1])
    val_dataset = ensure_shape(val_dataset, window_size, data.shape[-1])
    test_dataset = ensure_shape(test_dataset, window_size, data.shape[-1])

    return train_dataset, val_dataset, test_dataset
