from ctdfjorder.constants.constants import *
from ctdfjorder.exceptions.exceptions import CTDError

from datetime import datetime, timedelta
import polars as pl

from os import path

# Column label mapping from castaway to internal
csvLabels_to_labelInternal: dict[str, str] = {
    "Pressure (Decibar)": SEA_PRESSURE_LABEL,
    "Depth (Meter)": DEPTH_LABEL,
    "Temperature (Celsius)": TEMPERATURE_LABEL,
    "Conductivity (MicroSiemens per Centimeter)": CONDUCTIVITY_LABEL,
    "Specific conductance (MicroSiemens per Centimeter)": SPECIFIC_CONDUCTIVITY_LABEL,
    "Salinity (Practical Salinity Scale)": SALINITY_LABEL,
    "Sound velocity (Meters per Second)": SPEED_OF_SOUND_LABEL,
    "Density (Kilograms per Cubic Meter)": DENSITY_LABEL,
}


def load_file_castaway(castaway_file_path):
    """
    Loads and processes a Castaway CTD file.

    Parameters
    ----------
    castaway_file_path : str
        The file path to the Castaway CTD file.

    Returns
    -------
    pl.DataFrame
        The processed Castaway CTD data.

    Raises
    ------
    CTDError
        If the Castaway CTD profile is empty or if no samples are found in the file, or if the start time is missing.
    """
    filename = path.basename(castaway_file_path)
    with open(castaway_file_path) as file:
        profile = pl.read_csv(file, comment_prefix="%", null_values=["#N/A", "null"])
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
            pl.lit(lat).cast(pl.Float64).alias(LATITUDE_LABEL),
            pl.lit(long).cast(pl.Float64).alias(LONGITUDE_LABEL),
        )
    data = profile
    return data


def extract_utc_cast_time(ctd_file_path):
    """
    Extracts the UTC cast time from a Castaway file and converts it to ISO 8601 format.

    Parameters
    ----------
    ctd_file_path : str
        The file path of the Castaway file to extract the time from.

    Returns
    -------
    datetime.datetime | None
        The cast time (UTC) of the Castaway file in ISO 8601 format, or None if not found.
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
    Extracts the start latitude and longitude from a Castaway file.

    Parameters
    ----------
    ctd_file_path : str
        The file path to the Castaway CTD file.

    Returns
    -------
    tuple
        A tuple containing the latitude and longitude as strings, or None if not found.
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
                        latitude = None
            if line.startswith("% Start longitude"):
                # Assume the format is '% description, longitude'
                parts = line.strip().split(",")
                if len(parts) > 1:
                    longitude = parts[1].strip()
                    if longitude == "":
                        longitude = None
    return latitude, longitude
