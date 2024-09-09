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

## Temperature Prediction
Look to predict the temperature in 24 hours time.

## Wind Prediction
Look to predict the wind vector at the next measurement (10 minutes time).

## Rain Prediction
Look to predict the amount of rainfall within the next hour.
