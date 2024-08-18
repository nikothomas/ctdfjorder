import polars as pl
from ctdfjorder.pyrsktools import RSK, Geo, Region
from ctdfjorder.constants.constants import *
from ctdfjorder.exceptions.exceptions import CTDError
import numpy as np
from os import path
from typing import Any

# Column label mapping from rsk to internal
rskLabels_to_labelInternal: dict[str, str] = {
    "temperature_00": TEMPERATURE.label,
    "chlorophyll_00": CHLOROPHYLL.label,
    "seapressure_00": SEA_PRESSURE.label,
    "depth_00": DEPTH.label,
    "salinity_00": SALINITY.label,
    "speedofsound_00": SPEED_OF_SOUND.label,
    "specificconductivity_00": SPECIFIC_CONDUCTIVITY.label,
    "conductivity_00": CONDUCTIVITY.label,
    "pressure_00": PRESSURE.label,
}


def load_file_rsk(rbr_file_path: str = None) -> pl.DataFrame:
    """
    Loads and processes an RSK file, extracting profiles and adding geospatial information.

    Parameters
    ----------
    rbr_file_path : str, optional
        The file path to the RSK file.

    Returns
    -------
    pl.DataFrame
        The processed RSK file data.

    Raises
    ------
    CTDError
        If the RSK profile is empty or if no samples are found in the file.
    """

    def process_rsk(
        rsk_profile: pl.DataFrame,
        geo: [Geo, Any, Any] = None,
        filename: str = None,
    ) -> pl.DataFrame | None:
        """
        Processes an RSK profile dataframe, adding geospatial information.

        Parameters
        ----------
        rsk_profile : pl.DataFrame
            The RSK profile data to process.
        geo : Geo, optional
            Geospatial information generator.
        filename : str, optional
            The filename of the RSK profile.

        Returns
        -------
        pl.DataFrame | None
            The processed RSK profile dataframe with latitude and longitude columns added, or None if the profile is empty.

        Raises
        ------
        CTDError
            If the RSK profile is empty.
        """
        rsk_profile = rsk_profile.with_columns(
            pl.col(TIMESTAMP.label)
            .cast(pl.String)
            .str.to_datetime(
                format="%Y-%m-%d %H:%M:%S%.f",
                time_zone=TIME_ZONE,
                time_unit=TIME_UNIT,
            )
            .cast(pl.Datetime(time_unit=TIME_UNIT))
            .dt.convert_time_zone(time_zone=TIME_ZONE)
        )
        data = rsk_profile
        try:
            profile_geodata = next(geo)
            return data.with_columns(
                pl.lit(profile_geodata.latitude).alias(LATITUDE.label),
                pl.lit(profile_geodata.longitude).alias(LONGITUDE.label),
            )

        # No geodata found in rsk file
        except StopIteration:
            return data.with_columns(
                pl.lit(None).alias(LATITUDE.label), pl.lit(None).alias(LONGITUDE.label)
            )

    data = None
    filename = path.basename(rbr_file_path)
    rsk = RSK(rbr_file_path)
    num_profiles = 0
    # Processing for rsk files with cast regions, loops through casts and indicates separate casts with profile num
    rsk_casts_down = rsk.casts(Region.CAST_DOWN)
    for i, endpoints in enumerate(rsk_casts_down):
        rsk_numpy_array = np.array(
            rsk.npsamples(endpoints.start_time, endpoints.end_time)
        )
        for x, timestamp in enumerate(rsk_numpy_array[TIMESTAMP.label]):
            rsk_numpy_array[TIMESTAMP.label][x] = timestamp.strftime(TIME_FORMAT)
        profile_to_process = (
            pl.DataFrame(rsk_numpy_array)
            .rename(rskLabels_to_labelInternal)
            .drop_nulls()
            .with_columns(
                pl.lit(filename).alias(FILENAME.label),
                pl.lit(num_profiles).alias(PROFILE_ID.label),
            )
        )
        geodata = rsk.geodata(endpoints.start_time, endpoints.end_time)
        processed_profile = process_rsk(
            rsk_profile=profile_to_process, geo=geodata, filename=filename
        )
        if data is None:
            data = processed_profile
            num_profiles += 1
        elif processed_profile is not None:
            data = pl.concat([processed_profile, data], how=CONCAT_HOW)
            num_profiles += 1

    # Processing for rsk files without cast regions
    if type(data) is type(None) or data.is_empty():
        num_profiles = 0
        rsk_numpy_array = np.array(rsk.npsamples())
        for x, timestamp in enumerate(rsk_numpy_array[TIMESTAMP.label]):
            rsk_numpy_array[TIMESTAMP.label][x] = timestamp.strftime(TIME_FORMAT)
        profile = (
            pl.DataFrame(rsk_numpy_array)
            .rename(rskLabels_to_labelInternal)
            .with_columns(
                pl.lit(filename).alias(FILENAME.label),
                pl.lit(num_profiles).alias(PROFILE_ID.label),
            )
        )
        geodata = rsk.geodata()
        processed_profile = process_rsk(
            rsk_profile=profile, geo=geodata, filename=filename
        )
        if data is None:
            data = processed_profile
            num_profiles += 1
        elif processed_profile is not None:
            data = pl.concat([processed_profile, data], how=CONCAT_HOW)
            num_profiles += 1

    return data
