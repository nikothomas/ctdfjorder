import polars as pl
from ctdfjorder.pyrsktools import RSK, Geo, Region
from ctdfjorder.Mastersheet import Mastersheet
from ctdfjorder.constants import *
from ctdfjorder.CTDExceptions.CTDExceptions import (
    CTDError,
    raise_warning_native_location,
)
import numpy as np
from os import path
from typing import Any


def process_rsk(
    rsk_profile: pl.DataFrame,
    geo: [Geo, Any, Any] = None,
    filename: str = None,
    mastersheet: Mastersheet = None,
) -> pl.DataFrame | None:
    rsk_profile = rsk_profile.with_columns(
        pl.col(TIMESTAMP_LABEL)
        .cast(pl.String)
        .str.to_datetime(
            format=TIME_FORMAT,
            time_zone=TIME_ZONE,
            time_unit=TIME_UNIT,
        )
        .cast(pl.Datetime(time_unit=TIME_UNIT))
        .dt.convert_time_zone(time_zone=TIME_ZONE)
    )
    rsk_profile = rsk_profile.with_columns(pl.lit(value=0).alias(name=PROFILE_ID_LABEL))
    if rsk_profile.is_empty():
        raise CTDError(filename=filename, message=ERROR_NO_SAMPLES)
    data = rsk_profile.with_columns(
        pl.lit(filename + FILENAME_CM_ENDING).alias(FILENAME_LABEL)
    )
    try:
        profile_geodata = next(geo)
        return data.with_columns(
            pl.lit(profile_geodata.latitude).alias(LATITUDE_LABEL),
            pl.lit(profile_geodata.longitude).alias(LONGITUDE_LABEL),
        )
    except StopIteration:
        raise_warning_native_location(
            filename=filename,
            message=DEBUG_FILE_LACKS_LOCATION,
        )
        lat, long, _, _ = mastersheet.find_match(data)
        return data.with_columns(
            pl.lit(lat).alias(LATITUDE_LABEL),
            pl.lit(long).alias(LONGITUDE_LABEL),
            pl.lit(filename + FILENAME_CM_ENDING).alias(FILENAME_LABEL),
        )


def load_file_rsk(
    rbr_file_path: str = None, mastersheet: Mastersheet = None
) -> pl.DataFrame:
    data = None
    num_profiles = 0
    filename = path.basename(rbr_file_path)
    rsk = RSK(rbr_file_path)

    # Processing for rsk files with cast regions, loops through casts and indicates separate casts with profile num
    rsk_casts_down = rsk.casts(Region.CAST_DOWN)
    for i, endpoints in enumerate(rsk_casts_down):
        rsk_numpy_array = np.array(
            rsk.npsamples(endpoints.start_time, endpoints.end_time)
        )
        for x, timestamp in enumerate(rsk_numpy_array[TIMESTAMP_LABEL]):
            rsk_numpy_array[TIMESTAMP_LABEL][x] = timestamp.strftime(TIME_FORMAT)
        profile_to_process = (
            pl.DataFrame(rsk_numpy_array)
            .rename(rskLabels_to_labelInternal)
            .drop_nulls()
            .with_columns(pl.lit(filename).alias(FILENAME_LABEL))
        )
        geodata = rsk.geodata(endpoints.start_time, endpoints.end_time)
        processed_profile = process_rsk(
            rsk_profile=profile_to_process,
            geo=geodata,
            filename=filename,
            mastersheet=mastersheet,
        )
        if type(data) is type(None):
            data = processed_profile
            num_profiles += 1
        elif processed_profile is not None:
            data = pl.concat([processed_profile, data], how=CONCAT_HOW)
            num_profiles += 1

    # Processing for rsk files without cast regions
    if type(data) is type(None) or data.is_empty():
        rsk_numpy_array = np.array(rsk.npsamples())
        for x, timestamp in enumerate(rsk_numpy_array[TIMESTAMP_LABEL]):
            rsk_numpy_array[TIMESTAMP_LABEL][x] = timestamp.strftime(TIME_FORMAT)
        profile = (
            pl.DataFrame(rsk_numpy_array)
            .rename(rskLabels_to_labelInternal)
            .drop_nulls()
            .with_columns(pl.lit(filename).alias(FILENAME_LABEL))
        )
        geodata = rsk.geodata()
        processed_profile = process_rsk(
            rsk_profile=profile, geo=geodata, filename=filename, mastersheet=mastersheet
        )
        if data is None:
            data = processed_profile
            num_profiles += 1
        if processed_profile is not None:
            data = pl.concat([processed_profile, data], how=CONCAT_HOW)
            num_profiles += 1
        else:
            CTDError(message=ERROR_NO_SAMPLES, filename=filename)

    return data
