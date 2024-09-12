# ML Weather Forecasting
An exploration of different supervised machine learning models for predicting future weather.

## Data

### Retrieval
- The data used is measured at WS Saaleaue and available at https://www.bgc-jena.mpg.de/wetter/weather_data.html. Contains various measurements for various meteorological variables every 10 minutes dating back to midway through 2002.
- Downloaded using data_download(years).
- Drop some columns such as rainfall duration.
- Saved in .csv file.

### Preprocessing
- Remove any incorrect values, the data has -9999.0 entries, linearly interpolate
- Carry out feature engineering on the data - turn wind direction and speed into a wind vector and use the Date Time index to create daily and yearly time signals.
- Extract/copy data out as target data depending on what we are predicting.
- Normalize the data into the standard normal distribution - violin plot.
- Make use of TensorFlow's in-built timeseries_dataset_from_array to create Dataset objects that work effectively for keras Models.

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

![Basline predictions](/figures/temperature/baseline_evaluation.png)

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
x = layers.LSTM(16)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```

#### Training
The best validation MAE of 2.76 degrees Celsius occurs in the 6th epoch:

![Recurrent network loss curves](/figures/temperature/recurrent_training.png)

#### Evaluation
The best version of the model achieves a test MAE of 2.51 degrees Celsius, and the following (sample of) predictions vs targets:

![Recurrent network predictions](/figures/temperature/recurrent_evaluation.png)

### Recurrent with Dropout
Since the simple recurrent model performed well and is clearly overfitting on the training dataset, we can use a similar model but with dropout in an attempt to combat it.

#### Model

```
inputs = keras.Input(shape=(settings['window_size'], num_features))
x = layers.LSTM(32, recurrent_dropout=0.25)(inputs)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```

#### Training
The best validation MAE of 2.66 degrees Celsius occurs in the 13th epoch:

![Recurrent dropout network loss curves](/figures/temperature/recurrent_dropout_training.png)

#### Evaluation
The best version of the model achieves a test MAE of 2.48 degrees Celsius, and the following (sample of) predictions vs targets:

![Recurrent dropout network predictions](/figures/temperature/recurrent_dropout_evaluation.png)

### Stacked Recurrent Layers

## Wind Prediction
Look to predict the wind vector at the next measurement (10 minutes time).

## Rain Prediction
Look to predict the amount of rainfall within the next hour.
