# ML Weather Forecasting
An exploration of different supervised machine learning models for predicting future weather.

## Data

### Retrieval
- The data used is measured at WS Saaleaue and available at https://www.bgc-jena.mpg.de/wetter/weather_data.html. It contains measurements for various meteorological variables every 10 minutes dating back to midway through 2002.
- The data is downloaded using the function `data_download(years)`, where the years specified is the number of years into the past to download.
- Before saving any data, some columns are dropped such as rainfall duration.
- The data is then saved .csv file for easy opening and processing.

### Preprocessing
- The data has -9999.0 entries as placeholders for no data measured -  to resolve this we replace them with linearly interpolated values.
- We carry out feature engineering on the data - wind direction and speed are turned into a wind vector and the Date Time index is used to create daily and yearly time signals.
- Target data is extracted/copied depending on what we are predicting.
- The data is then normalized into the standard normal distribution.
- We make use of TensorFlow's in-built `timeseries_dataset_from_array` function to create Dataset objects that work effectively with keras' model training API.

## Training and Testing Models
We package the training and testing procedures into two functions for easy reusability. 

### `train_model`
- Compiles the model with rmsprop optimizer, mean squared error loss, and mean absolute error (MAE) as a metric.
- Trains the model on the given training dataset.
- Applies 2 keras callbacks, namely ModelCheckpoint and EarlyStopping, to save the best version of the model and to prevent wasted runtime.
- Displays the MAE curves for the training and validation datasets during the training.

### `evaluate_model`
- Runs the model on the given testing dataset and outputs the MAE.
- Displays a plot containing a sample window of the model's predictions against the targets for the testing dataset.

## Temperature Prediction
We will attempt to predict the temperature in 24 hours time. We will sample the data hourly, so the target data will be the temperature 24 readings ahead. We will also use a window of 120 timesteps (so data spanning 120 hours in this case). We will use a training, validation, testing data split of 0.7, 0.2, 0.1. The loss used in any model training is the mean squared error, but we will track the mean absolute error (MAE) on the validation and testing datasets.

### Baseline
The common-sense baseline that we should look to beat is to predict that the temperature in 24 hours time will be exactly the same as it is now. This results in a validation MAE of 3.03 degrees Celsius, a test MAE of 2.85 degrees Celsius, and the following (sample of) predictions vs targets:

![Baseline predictions](/figures/temperature/baseline_evaluation.png)

### Dense
The simplest and cheapest machine learning model we can try is a small densely connected one. 

#### Model

```
inputs = keras.Input(shape=(settings['window_size'], num_features))
x = layers.Flatten()(inputs)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```

#### Training
The best validation MAE of 2.89 degrees Celsius occurs in the 3rd epoch:

![Dense network loss curves](/figures/temperature/dense_training.png)

#### Evaluation
The best version of the model achieves a test MAE of 2.63 degrees Celsius, and the following (sample of) predictions vs targets:

![Dense network predictions](/figures/temperature/dense_evaluation.png)

### Simple Recurrent
The simplest model that takes advantage of the fact that our data is a timeseries is one containing a recurrent layer.

#### Model

```
inputs = keras.Input(shape=(settings['window_size'], num_features))
x = layers.LSTM(32)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```

#### Training
The best validation MAE of 2.70 degrees Celsius occurs in the 1st epoch:

![Recurrent network loss curves](/figures/temperature/recurrent_training.png)

#### Evaluation
The best version of the model achieves a test MAE of 2.53 degrees Celsius, and the following (sample of) predictions vs targets:

![Recurrent network predictions](/figures/temperature/recurrent_evaluation.png)

### Recurrent with Dropout
Since the simple recurrent model performed well and is clearly overfitting on the training dataset, we can use a similar model but with dropout in an attempt to combat it.

#### Model

```
inputs = keras.Input(shape=(settings['window_size'], num_features))
x = layers.LSTM(48, recurrent_dropout=0.25)(inputs)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```

#### Training
The best validation MAE of 2.65 degrees Celsius occurs in the 4th epoch:

![Recurrent dropout network loss curves](/figures/temperature/recurrent_dropout_training.png)

#### Evaluation
The best version of the model achieves a test MAE of 2.49 degrees Celsius, and the following (sample of) predictions vs targets:

![Recurrent dropout network predictions](/figures/temperature/recurrent_dropout_evaluation.png)

### Stacked Recurrent Layers
Since the recurrent model with dropout is no longer obviously overfitting, we can try to extend the size of our model to improve performance by stacking LTSM layers.

#### Model

```
inputs = keras.Input(shape=(settings['window_size'], num_features))
x = layers.LSTM(48, recurrent_dropout=0.5, return_sequences=True)(inputs)
x = layers.LSTM(48, recurrent_dropout=0.5)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```

#### Training
The best validation MAE of 2.69 degrees Celsius occurs in the 2nd epoch:

![Stacked recurrent dropout network loss curves](/figures/temperature/stacked_recurrent_training.png)

#### Evaluation
The best version of the model achieves a test MAE of 2.53 degrees Celsius, and the following (sample of) predictions vs targets:

![Stacked recurrent dropout network predictions](/figures/temperature/stacked_recurrent_evaluation.png)

## Wind Prediction
We will attempt to predict the wind vector at the next measurement (in 10 minutes time). We will use a window of 60 timesteps (so data spanning 10 hours in this case). We will use a training, validation, testing data split of 0.7, 0.2, 0.1. The loss used in any model training is the mean squared error, but we will track the mean absolute error (MAE) on the validation and testing datasets.

### Baseline
The common-sense baseline that we should look to beat is to predict that the wind vector in 10 minutes time will be exactly the same as it is now. This results in a validation MAE of 0.864 m/s, a test MAE of 0.841 m/s, and the following (sample of) predictions vs targets:

![Baseline predictions](/figures/wind/baseline_evaluation.png)

### Dense
The simplest and cheapest machine learning model we can try is a small densely connected one.

#### Model

```
inputs = keras.Input(shape=(settings['window_size'], num_features))
x = layers.Flatten()(inputs)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(2)(x)
model = keras.Model(inputs, outputs)
```

#### Training
The best validation MAE of 0.416 m/s occurs in the 16th epoch:

![Dense network loss curves](/figures/wind/dense_training.png)

#### Evaluation
The best version of the model achieves a test MAE of 0.397 m/s, and the following (sample of) predictions vs targets:

![Dense network predictions](/figures/wind/dense_evaluation.png)

### Simple Recurrent
The simplest model that takes advantage of the fact that our data is a timeseries is one containing a recurrent layer.

#### Model

```
inputs = keras.Input(shape=(settings['window_size'], num_features))
x = layers.LSTM(32)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```

#### Training
The best validation MAE of 0.399 m/s occurs in the 11th epoch:

![Recurrent network loss curves](/figures/wind/recurrent_training.png)

#### Evaluation
The best version of the model achieves a test MAE of 0.386 m/s, and the following (sample of) predictions vs targets:

![Recurrent network predictions](/figures/wind/recurrent_evaluation.png)

### Stacked Recurrent Layers
Since the recurrent model is not obviously overfitting, we will increase the size and complexity of our model in attempt to improve performance. We can do this by stacking LTSM layers.

#### Model

```
inputs = keras.Input(shape=(settings['window_size'], num_features))
x = layers.LSTM(64, recurrent_dropout=0.5, return_sequences=True)(inputs)
x = layers.LSTM(64, recurrent_dropout=0.5)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```

#### Training
The best validation MAE of 0.401 m/s occurs in the 9th epoch:

![Stacked recurrent dropout network loss curves](/figures/wind/stacked_recurrent_training.png)

#### Evaluation
The best version of the model achieves a test MAE of 0.388 m/s, and the following (sample of) predictions vs targets:

![Stacked recurrent dropout network predictions](/figures/wind/stacked_recurrent_evaluation.png)

## Rain Prediction
Look to predict the amount of rainfall within the next hour.
