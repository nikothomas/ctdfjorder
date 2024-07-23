from ctdfjorder.constants import *
from ctdfjorder.CTDExceptions.CTDExceptions import CTDError

from datetime import datetime, timedelta
import numpy as np
import polars as pl

from os import path


def load_file_castaway(castaway_file_path):
    filename = path.basename(castaway_file_path)
    with open(castaway_file_path) as file:
        profile = pl.read_csv(file, comment_prefix="%", null_values="#N/A")
    if profile.is_empty():
        raise CTDError(message=ERROR_NO_SAMPLES, filename=filename)
    if CASTAWAY_DATETIME_LABEL in profile.columns:
        profile = profile.with_columns(
            pl.col(CASTAWAY_DATETIME_LABEL)
            .str.to_datetime(
                format=TIME_FORMAT,
                time_zone=TIME_ZONE,
                time_unit=TIME_UNIT,
            )
            .cast(pl.Datetime)
            .alias(CASTAWAY_DATETIME_LABEL)
        )
        start_time = profile.select(pl.col(CASTAWAY_DATETIME_LABEL).first()).item()
    else:
        start_time = extract_utc_cast_time(castaway_file_path)
    if type(start_time) == type(None):
        raise CTDError(filename=filename, message=ERROR_CASTAWAY_START_TIME)
    timestamps = [
        start_time + timedelta(milliseconds=200 * i) for i in range(profile.height)
    ]
    profile = profile.with_columns(
        pl.Series(timestamps)
        .dt.convert_time_zone(TIME_ZONE)
        .dt.cast_time_unit(TIME_UNIT)
        .alias(TIMESTAMP_LABEL)
    )
    for header, maps_to in csvLabels_to_labelInternal.items():
        if header in profile.columns:
            profile = profile.rename({header: maps_to})
    profile = profile.drop(CASTAWAY_FILE_ID_LABEL, strict=False).with_columns(
        (pl.col(SEA_PRESSURE_LABEL) + 10.1325).alias(PRESSURE_LABEL),
        pl.lit(0).alias(PROFILE_ID_LABEL),
        pl.lit(filename).alias(FILENAME_LABEL),
    )
    if LATITUDE_LABEL not in profile.collect_schema().names():
        lat, long = extract_lat_long_castaway(castaway_file_path)
        profile = profile.with_columns(
            pl.lit(lat).alias(LATITUDE_LABEL), pl.lit(long).alias(LONGITUDE_LABEL)
        )
    data = profile
    return data


def extract_utc_cast_time(ctd_file_path):
    """
    Function to extract the UTC cast time from a castaway file and convert it to ISO 8601 format.

    Parameters
    ----------
    ctd_file_path : str
        The file path of the castaway file to extract the time from.

    Returns
    -------
    str
        Cast time (UTC) of the castaway file in ISO 8601 format.
    """
    # Initialize variable to store UTC cast time
    cast_time_utc = None
    # Open the file and read line by line
    with open(ctd_file_path, "r") as file:
        for line in file:
            if line.startswith("% Cast time (UTC)"):
                # Extract the UTC cast time after the comma
                parts = line.strip().split(",")
                if len(parts) > 1:
                    cast_time_str = parts[1].strip()  # Take only the datetime part
                    # Convert the datetime string to ISO format if possible
                    cast_time_utc = datetime.strptime(
                        cast_time_str, "%Y-%m-%d %H:%M:%S"
                    )
                    break  # Stop reading once the timestamp is found

    return cast_time_utc


def extract_lat_long_castaway(ctd_file_path):
    """
    Function extract start lat/long from castaway file.

    Parameters
    ----------
    ctd_file_path : str
        Filepath to castaway ctd file.

    Returns
    -------
    tuple
        (latitude, longitude)
    """
    latitude = None
    longitude = None
    # Open the file and read line by line
    with open(ctd_file_path, "r") as file:
        for line in file:
            if line.startswith("% Start latitude"):
                # Assume the format is '% description, latitude'
                parts = line.strip().split(",")
                if len(parts) > 1:
                    latitude = parts[1].strip()
                    if latitude == "":
                        latitude = np.nan
            if line.startswith("% Start longitude"):
                # Assume the format is '% description, longitude'
                parts = line.strip().split(",")
                if len(parts) > 1:
                    longitude = parts[1].strip()
                    if longitude == "":
                        longitude = np.nan
    return latitude, longitude
