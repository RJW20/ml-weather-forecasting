from datetime import datetime
from io import BytesIO
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd


def check_valid_years(years) -> range:
    """Raise a ValueError if the given years isn't strictly positive or is too
    large (if the current_year - years < 2003).
    
    Returns a list containing the previous {years} years (excluding the current
    year).
    """

    if years <= 0:
        raise ValueError(f"Requested {years=} must be strictly positive.")
    
    end_year = datetime.today().year - 1
    start_year = end_year - (years - 1)
    
    if start_year < 2003:
        raise ValueError(
            f"Requested {years=} is too large, it must be less than or equal to"
            f" {end_year - 2002}."
        )
    
    return list(range(start_year, end_year + 1))


def retrieve_weather_data_for(filename: str) -> pd.DataFrame:
    """Return a pd.DataFrame containing the weather data for the given filename
    found at 'https://www.bgc-jena.mpg.de/wetter/weather_data.html'.
    
    Raises a FileNotFoundError if the requested filename cannot be found, or if
    the csv file cannot be found within the zip file.
    """

    url = f"https://www.bgc-jena.mpg.de/wetter/{filename}.zip"

    with urlopen(url) as data:
        try:
            with ZipFile(BytesIO(data.read())) as zfile:
                return pd.read_csv(
                    zfile.open(f"{filename}.csv"),
                    index_col="Date Time",
                )
        except BadZipFile:
            raise FileNotFoundError(f"Cannot find the requested file {url}.")
        except KeyError:
            raise FileNotFoundError(
                f"The file {filename}.csv does not exist within {url}."
            )


def download_weather_data(years=5) -> None:
    """Download the weather data for the number of given years (ending at the
    end of the previous year) for the data measured at WS Saaleaue available at
    'https://www.bgc-jena.mpg.de/wetter/weather_data.html'.
    
    Raises a ValueError if the given years isn't strictly positive or is too
    large (data is only available back to the start of 2003).
    Saves the data to 'weather_data/{start_year}_{end_year}.csv'.
    """

    years_to_download = check_valid_years(years)
    filenames = [
        f"mpi_saale_{year}{part}" for year in years_to_download
        for part in ("a", "b")
    ]
    pd.concat(
        retrieve_weather_data_for(filename) for filename in filenames
    ).to_csv(
        f"weather_data/{years_to_download[0]}_{years_to_download[-1]}.csv",
        mode="w",
    )


if __name__ == "__main__":

    download_weather_data(years=7)
