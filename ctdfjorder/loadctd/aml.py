import polars as pl
from datetime import datetime
from os import path
from ctdfjorder.constants.constants import *
from ctdfjorder.exceptions.exceptions import CTDCorruptError


def load_file_aml(aml_file_path):
    """
    Loads and processes an AML CTD file.

    Parameters
    ----------
    aml_file_path : str
        The file path to the AML CTD file.

    Returns
    -------
    pl.DataFrame
        The processed AML CTD data.

    Raises
    ------
    CTDCorruptError
        If the AML file is unable to be opened or processed.
    MissingTimestampError
        If the timestamp information is missing from the file.
    """
    filename = path.basename(aml_file_path)

    try:
        with open(aml_file_path, 'r') as file:
            header = {}
            measurement_metadata = {}
            data = []

            # Parse header
            for line in file:
                if line.strip() == '[MeasurementMetadata]':
                    break
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    header[key] = value

            # Parse measurement metadata
            for line in file:
                if line.strip() == '[MeasurementData]':
                    break
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    measurement_metadata[key] = value.split(',')

            # Parse data
            for line in file:
                data.append(line.strip().split(','))

    except Exception as e:
        raise CTDCorruptError(filename=filename) from e

    # Create DataFrame
    df = pl.DataFrame(data, schema=measurement_metadata['Columns'], orient='row')

    # Convert types and rename columns
    df = df.with_columns([
        pl.col('Date').str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col('Time').str.strptime(pl.Time, "%H:%M:%S.%f"),
        pl.col('Conductivity').cast(pl.Float64).alias(CONDUCTIVITY.label),
        pl.col('Temperature').cast(pl.Float64).alias(TEMPERATURE.label),
        pl.col('Pressure').cast(pl.Float64).alias(PRESSURE.label),
        pl.col('Sound Velocity').cast(pl.Float64).alias(SPEED_OF_SOUND.label),
        pl.col('Depth').cast(pl.Float64).alias(DEPTH.label)
    ])

    # Combine Date and Time into a single Timestamp column
    df = df.with_columns([
        (pl.col('Date').cast(pl.Datetime) + pl.col('Time').cast(pl.Duration)).alias(TIMESTAMP.label)
    ]).drop(['Date', 'Time'])

    # Convert conductivity from mS/cm to µS/cm
    df = df.with_columns([
        (pl.col(CONDUCTIVITY.label) * 1000).alias(CONDUCTIVITY.label)
    ])

    # Add sea pressure column (assuming atmospheric pressure of 10.1325 dbar)
    df = df.with_columns([
        (pl.col(PRESSURE.label) - 10.1325).alias(SEA_PRESSURE.label)
    ])

    # Add metadata columns
    start_time = datetime.strptime(header['Time'], "%d/%m/%Y %H:%M:%S")
    df = df.with_columns([
        pl.lit(filename).alias(FILENAME.label),
        pl.lit(0).alias(PROFILE_ID.label),
        pl.lit(float(header['Latitude'])).alias(LATITUDE.label),
        pl.lit(float(header['Longitude'])).alias(LONGITUDE.label),
    ])

    return df


def is_aml_file(filename):
    """
    Checks if the given filename follows the AML naming convention.

    Parameters
    ----------
    filename : str
        The filename to check.

    Returns
    -------
    bool
        True if the filename follows the AML naming convention, False otherwise.
    """
    try:
        # Check if the filename starts with a date in the format YYYY-MM-DD
        datetime.strptime(filename[:10], "%Y-%m-%d")
        return True
    except ValueError:
        return False
