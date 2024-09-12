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
The common-sense baseline that we should look to beat is to predict that the temperature in 24 hours time will be exactly the same as it is now. This results in a validation MAE of 3.03 degrees Celsius and a test MAE of 2.85 degrees Celsius.

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

#### Model Training
- Running the train_model function with this model and the training and validation datasets, we get the following loss curves:

![Dense network loss curves](/figures/temperature/dense_training.png)

- The best validation MAE of 2.82 degrees Celsius occurs in the first epoch.

#### Model Evaluation
- The best version of the model achieves a test MAE of 2.63 degrees Celsius.
- A sample of 25 consecutive hours of testing data targets and predictions is shown below:

![Dense network predictions](/figures/temperature/dense_evaluation.png)

### Simple Recurrent

### Recurrent with Dropout

### Stacked Recurrent Layers

## Wind Prediction
Look to predict the wind vector at the next measurement (10 minutes time).

## Rain Prediction
Look to predict the amount of rainfall within the next hour.
