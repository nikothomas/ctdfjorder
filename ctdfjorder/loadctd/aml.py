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
                line = line.strip().rstrip(',')
                if line == '[MeasurementMetadata]':
                    break
                if '=' in line:
                    key, value = line.split('=', 1)
                    header[key.strip()] = value.strip()

            # Parse measurement metadata
            for line in file:
                line = line.strip().rstrip(',')
                if line == '[MeasurementData]':
                    break
                if '=' in line:
                    key, value = line.split('=', 1)
                    measurement_metadata[key.strip()] = [v.strip() for v in value.split(',')]

            # Parse data
            for line in file:
                line = line.strip().rstrip(',')
                data.append([v.strip() for v in line.split(',')])

    except Exception as e:
        raise CTDCorruptError(filename=filename) from e

    # Define the columns to keep
    desired_columns = ['Date', 'Time', 'Conductivity', 'Temperature', 'Pressure', 'Sound Velocity', 'Depth']

    # Check if the measurement metadata columns include the desired columns
    columns = measurement_metadata.get('Columns', [])
    valid_columns = [col for col in columns if col in desired_columns]
    valid_indices = [i for i, col in enumerate(columns) if col in valid_columns]

    # Filter out undesired columns from data
    filtered_data = [[row[i] for i in valid_indices] for row in data]

    # Create DataFrame with valid columns only
    df = pl.DataFrame(filtered_data, schema=valid_columns, orient='row')

    # Detect date format and parse accordingly
    if df['Date'].str.contains(r'^\d{1,2}/\d{1,2}/\d{4}$').all():
        df = df.with_columns(
            pl.col('Date').str.strptime(pl.Date, "%m/%d/%Y")
        )
    else:
        df = df.with_columns(
            pl.col('Date').str.strptime(pl.Date, "%Y-%m-%d")
        )

    # Check if 'Time' column is in 'mm:ss.s' format
    if df['Time'].str.contains(r'^\d{1,2}:\d{2}\.\d$').all():
        # Convert 'mm:ss.s' to 'hh:mm:ss.s'
        df = df.with_columns(
            (pl.lit("00:") + pl.col('Time')).alias('Time')
        )

    # Convert time and other columns
    df = df.with_columns([
        pl.col('Time').str.strptime(pl.Time, "%H:%M:%S.%f"),
        pl.col('Conductivity').alias(CONDUCTIVITY.label).cast(pl.Float64),
        pl.col('Temperature').alias(TEMPERATURE.label).cast(pl.Float64),
        pl.col('Pressure').alias(PRESSURE.label).cast(pl.Float64),
        pl.col('Sound Velocity').alias(SPEED_OF_SOUND.label).cast(pl.Float64),
        pl.col('Depth').alias(DEPTH.label).cast(pl.Float64)
    ]).drop(['Conductivity', 'Temperature', 'Pressure', 'Sound Velocity', 'Depth'])

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
        (pl.col(PRESSURE.label).alias(SEA_PRESSURE.label))
    ])
    df = df.with_columns((pl.col(SEA_PRESSURE.label) + 10.1325).alias(PRESSURE.label))

    # Handle missing latitude and longitude
    latitude = pl.lit(float(header['Latitude'])) if 'Latitude' in header else pl.lit(None)
    longitude = pl.lit(float(header['Longitude'])) if 'Longitude' in header else pl.lit(None)

    # Add metadata columns
    start_time = datetime.strptime(header['Time'], "%d/%m/%Y %H:%M:%S")
    df = df.with_columns([
        pl.lit(filename).alias(FILENAME.label),
        pl.lit(0).alias(PROFILE_ID.label),
        latitude.alias(LATITUDE.label),
        longitude.alias(LONGITUDE.label),
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
