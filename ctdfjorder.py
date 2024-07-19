# -*- coding: utf-8 -*-
import itertools
import openpyxl
import tensorflow as tf
import pandas as pd
from keras.api.models import Model
from keras.api.layers import Input, Dense, TimeDistributed, GRU
from keras.api.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras import mixed_precision
import os
import sys
from datetime import datetime
from datetime import timedelta
from sqlite3 import OperationalError
from os import path
import numpy as np
import polars as pl
from pyrsktools import RSK, Geo
from pyrsktools import Region
import gsw
import matplotlib.pyplot as plt
import statsmodels.api
import logging
from typing import Generator
from typing import Any
from typing import Literal
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")

mixed_precision.set_global_policy("float64")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.CRITICAL)
logger = logging.getLogger("ctdfjorder")
logger.propagate = 0


class CTD:
    """
    CTD
    ---

    Class representing a CTD object for processing and analyzing CTD data.

    Attributes
    ----------
    master_sheet_path : str
        Path to the mastersheet.

    Examples
    --------
    Removing non positive samples from data:

    >>> from ctdfjorder import CTD
    >>> ctd_data = CTD('CC1531002_20181225_114931.csv')
    >>> ctd_data.remove_non_positive_samples()
    >>> output = ctd_data.get_df()
    >>> print(output.head(3))
    shape: (3, 13)
    ┌──────────────┬──────────┬─────────────┬──────────────┬───┬────────────┬───────────────────────────────┬────────────┬────────────┐
    │ sea_pressure ┆ depth    ┆ temperature ┆ conductivity ┆ … ┆ profile_id ┆ filename                      ┆ latitude   ┆ longitude  │
    │ ---          ┆ ---      ┆ ---         ┆ ---          ┆   ┆ ---        ┆ ---                           ┆ ---        ┆ ---        │
    │ f64          ┆ f64      ┆ f64         ┆ f64          ┆   ┆ i32        ┆ str                           ┆ f64        ┆ f64        │
    ╞══════════════╪══════════╪═════════════╪══════════════╪═══╪════════════╪═══════════════════════════════╪════════════╪════════════╡
    │ 0.15         ┆ 0.148676 ┆ 0.32895     ┆ 28413.735648 ┆ … ┆ 0          ┆ CC1531002_20181225_114931.csv ┆ -64.668455 ┆ -62.641775 │
    │ 0.45         ┆ 0.446022 ┆ 0.316492    ┆ 28392.966662 ┆ … ┆ 0          ┆ CC1531002_20181225_114931.csv ┆ -64.668455 ┆ -62.641775 │
    │ 0.75         ┆ 0.743371 ┆ 0.310613    ┆ 28386.78011  ┆ … ┆ 0          ┆ CC1531002_20181225_114931.csv ┆ -64.668455 ┆ -62.641775 │
    └──────────────┴──────────┴─────────────┴──────────────┴───┴────────────┴───────────────────────────────┴────────────┴────────────┘

    Notes
    -----
    - A mastersheet is not necessary, but if your files are missing location data then many functions will not work

    """

    # Column labels for internal use
    _TIMESTAMP_LABEL: str = "timestamp"
    _FILENAME_LABEL: str = "filename"
    _CHLOROPHYLL_LABEL: str = "chlorophyll"
    _TEMPERATURE_LABEL: str = "temperature"
    _SEA_PRESSURE_LABEL: str = "sea_pressure"
    _DEPTH_LABEL: str = "depth"
    _SALINITY_LABEL: str = "salinity"
    _SPEED_OF_SOUND_LABEL: str = "speed_of_sound"
    _SPECIFIC_CONDUCTIVITY_LABEL: str = "specific_conductivity"
    _CONDUCTIVITY_LABEL: str = "conductivity"
    _PRESSURE_LABEL: str = "pressure"
    _SALINITY_ABS_LABEL: str = "salinity_abs"
    _SURFACE_SALINITY_LABEL: str = "surface_salinity"
    _SURFACE_TEMPERATURE_LABEL: str = "surface_temperature"
    _SURFACE_DENSITY_LABEL: str = "surface_density"
    _MELTWATER_FRACTION_LABEL: str = "meltwater_fraction"
    _DENSITY_LABEL: str = "density"
    _POTENTIAL_DENSITY_LABEL: str = "potential_density"
    _LATITUDE_LABEL: str = "latitude"
    _LONGITUDE_LABEL: str = "longitude"
    _UNIQUE_ID_LABEL: str = "unique_id"
    _PROFILE_ID_LABEL: str = "profile_id"
    _BV_LABEL: str = "brunt_vaisala_frequency_squared"
    _P_MID_LABEL: str = "p_mid"
    _SECCHI_DEPTH_LABEL: str = "secchi_depth"

    # Column label mapping from rsk to internal
    _rskLabels_to_labelInternal: dict[str, str] = {
        "temperature_00": _TEMPERATURE_LABEL,
        "chlorophyll_00": _CHLOROPHYLL_LABEL,
        "seapressure_00": _SEA_PRESSURE_LABEL,
        "depth_00": _DEPTH_LABEL,
        "salinity_00": _SALINITY_LABEL,
        "speedofsound_00": _SPEED_OF_SOUND_LABEL,
        "specificconductivity_00": _SPECIFIC_CONDUCTIVITY_LABEL,
        "conductivity_00": _CONDUCTIVITY_LABEL,
        "pressure_00": _PRESSURE_LABEL,
    }

    # Column label mapping from castaway to internal
    _csvLabels_to_labelInternal: dict[str, str] = {
        "Pressure (Decibar)": _SEA_PRESSURE_LABEL,
        "Depth (Meter)": _DEPTH_LABEL,
        "Temperature (Celsius)": _TEMPERATURE_LABEL,
        "Conductivity (MicroSiemens per Centimeter)": _CONDUCTIVITY_LABEL,
        "Specific conductance (MicroSiemens per Centimeter)": _SPECIFIC_CONDUCTIVITY_LABEL,
        "Salinity (Practical Salinity Scale)": _SALINITY_LABEL,
        "Sound velocity (Meters per Second)": _SPEED_OF_SOUND_LABEL,
        "Density (Kilograms per Cubic Meter)": _DENSITY_LABEL,
    }
    # Column labels of master sheet
    _MASTER_SHEET_TIME_LOCAL_LABEL = "time_local"
    _MASTER_SHEET_DATE_LOCAL_LABEL = "date_local"
    _MASTER_SHEET_DATETIME_LABEL = "datetime"
    _MASTER_SHEET_SECCHI_DEPTH_LABEL = "secchi depth"

    # Time string constants
    _TIME_ZONE: str = "UTC"
    _TIME_UNIT: Literal["ns", "us", "ms"] = "ns"
    _TIME_FORMAT: str = "%Y-%m-%d %H:%M:%S.%f"

    # Error messages
    _ERROR_NO_SAMPLES: str = "No samples in file"
    _ERROR_NO_LOCATION: str = "No location could be found"
    _ERROR_DENSITY_CALCULATION: str = "Could not calculate density on this dataset"
    _ERROR_SALINITY_ABS: str = "Could not calculate salinity absolute on this dataset"
    _ERROR_NO_MASTER_SHEET: str = (
        "No mastersheet provided, could not update the file's missing location data"
    )
    _ERROR_RSK_CORRUPT: str = "Ruskin file is corrupted and could not be read"
    _ERROR_LOCATION_DATA_INVALID: str = (
        "Location data invalid, probably due to malformed master sheet data"
    )
    _ERROR_NO_TIMESTAMP_IN_FILE: str = "No timestamp in file, could not get location"
    _ERROR_NO_TIMESTAMP_IN_MASTER_SHEET: str = (
        "No timestamp in master sheet, could not get location"
    )
    _ERROR_MLD_DEPTH_RANGE: str = "Insufficient depth range to calculate MLD"

    # Warning messages
    _WARNING_DROPPED_PROFILE: str = "No samples in profile number "

    # Info messages
    _INFO_CTD_OBJECT_INITITALIZED: str = "New CTD object initialized from file"
    _INFO_CTD_SURFACE_MEASUREMENT: str = (
        "First measurment lies below {end} dbar, cannot compute surface measurements"
    )
    # Debug messages
    _DEBUG_FILE_LACKS_LOCATION: str = "File lacks native location data"

    # Filename constants
    _FILENAME_GPS_ENDING: str = "_gps"
    _FILENAME_CM_ENDING: str = "cm"
    _RSK_FILE_MARKER: str = ".rsk"
    _CASTAWAY_FILE_MARKER: str = ".csv"

    # Castaway column labels
    _CASTAWAY_DATETIME_LABEL: str = "datetime_utc"
    _CASTAWAY_FILE_ID_LABEL: str = "file_id"

    # Concatenation parameters
    _CONCAT_HOW: Literal["diagonal_relaxed"] = "diagonal_relaxed"

    # Sea pressure to pressure difference
    _SEA_PRESSURE_TO_PRESSURE_DIFF: float = 10.1325

    # Initialization Constants
    _data: pl.DataFrame = pl.DataFrame()
    _cached_master_sheet: pl.DataFrame = pl.DataFrame()
    _filename: str = None
    _filepath: str = None
    _cwd: str = None
    master_sheet_path: str = None
    _add_unique_id: bool = False
    _num_profiles: int = 0
    _mld_col_labels: list[str] = []

    def __init__(
        self,
        ctd_file_path: str,
        cached_master_sheet: pl.DataFrame = None,
        master_sheet_path=None,
        add_unique_id=False,
    ):
        """
        Initialize a new CTD object.

        Parameters
        ----------
        ctd_file_path : str
            The file path to the RSK or Castaway file.
        cached_master_sheet : pl.Dataframe, default pl.DataFrame()
            Polars dataframe representation of a master sheet.
        master_sheet_path : str, default None
            Path to master sheet.
        add_unique_id : bool, default False
            If true adds unique id and secchi depth from master sheet.

        Examples
        --------
        Castaway CTD profile with valid data

        >>> from ctdfjorder import CTD
        >>> from ctdfjorder import CTDError
        >>> ctd_data = CTD('CC1531002_20181225_114931.csv')
        >>> output = ctd_data.get_df()
        >>> print(output.head(3))
        shape: (3, 13)
        ┌──────────────┬──────────┬─────────────┬──────────────┬───┬────────────┬───────────────────────────────┬────────────┬────────────┐
        │ sea_pressure ┆ depth    ┆ temperature ┆ conductivity ┆ … ┆ profile_id ┆ filename                      ┆ latitude   ┆ longitude  │
        │ ---          ┆ ---      ┆ ---         ┆ ---          ┆   ┆ ---        ┆ ---                           ┆ ---        ┆ ---        │
        │ f64          ┆ f64      ┆ f64         ┆ f64          ┆   ┆ i32        ┆ str                           ┆ f64        ┆ f64        │
        ╞══════════════╪══════════╪═════════════╪══════════════╪═══╪════════════╪═══════════════════════════════╪════════════╪════════════╡
        │ 0.15         ┆ 0.148676 ┆ 0.32895     ┆ 28413.735648 ┆ … ┆ 0          ┆ CC1531002_20181225_114931.csv ┆ -64.668455 ┆ -62.641775 │
        │ 0.45         ┆ 0.446022 ┆ 0.316492    ┆ 28392.966662 ┆ … ┆ 0          ┆ CC1531002_20181225_114931.csv ┆ -64.668455 ┆ -62.641775 │
        │ 0.75         ┆ 0.743371 ┆ 0.310613    ┆ 28386.78011  ┆ … ┆ 0          ┆ CC1531002_20181225_114931.csv ┆ -64.668455 ┆ -62.641775 │
        └──────────────┴──────────┴─────────────┴──────────────┴───┴────────────┴───────────────────────────────┴────────────┴────────────┘

        Castaway CTD profile with no data

        >>> from ctdfjorder import CTD
        >>> ctd_data = CTD('CC1627007_20191220_195931.csv') # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ctdfjorder.CTDError: CC1627007_20191220_195931.csv - No samples in file

        Raises
        ------
        CTDError
            For ctdfjorder related errors

        """
        self._filename = path.basename(ctd_file_path)
        if type(self._cached_master_sheet) is type(None):
            self._cached_master_sheet = pl.DataFrame()
        else:
            self._cached_master_sheet = cached_master_sheet
        self.master_sheet_path = master_sheet_path
        self._cwd = CTD.Utility.get_cwd()

        def _process_rsk_profile(
            lf: pl.DataFrame, geo: Generator[Geo, Any, None]
        ) -> pl.DataFrame:
            lf = lf.with_columns(
                pl.lit(self._filename + self._FILENAME_CM_ENDING).alias(
                    self._FILENAME_LABEL
                )
            )
            try:
                profile_geodata = next(geo)
                return lf.with_columns(
                    pl.lit(profile_geodata.latitude).alias(self._LATITUDE_LABEL),
                    pl.lit(profile_geodata.longitude).alias(self._LONGITUDE_LABEL),
                )
            except StopIteration:
                CTDLogger(
                    filename=self._filename,
                    message=self._DEBUG_FILE_LACKS_LOCATION,
                    level="debug",
                )
                _lat, _long, _, _ = self._process_master_sheet(lf)
                return lf.with_columns(
                    pl.lit(_lat).alias(self._LATITUDE_LABEL),
                    pl.lit(_long).alias(self._LONGITUDE_LABEL),
                    pl.lit(self._filename + self._FILENAME_CM_ENDING).alias(
                        self._FILENAME_LABEL
                    ),
                )

        def _process_profile(rsk_profile: pl.DataFrame, geodata) -> pl.DataFrame | None:
            rsk_profile = rsk_profile.with_columns(
                pl.col(self._TIMESTAMP_LABEL)
                .cast(pl.String)
                .str.to_datetime(
                    format=self._TIME_FORMAT,
                    time_zone=self._TIME_ZONE,
                    time_unit=self._TIME_UNIT,
                )
                .cast(pl.Datetime(time_unit=self._TIME_UNIT))
                .dt.replace_time_zone(self._TIME_ZONE)
            )
            rsk_profile = rsk_profile.with_columns(pl.lit(0).alias(self._PROFILE_ID_LABEL))
            if not rsk_profile.is_empty():
                return _process_rsk_profile(rsk_profile, geodata)
            return None

        def _load_rsk_file():
            _rsk = RSK(ctd_file_path)
            rsk_casts_down = _rsk.casts(Region.CAST_DOWN)
            for i, endpoints in enumerate(rsk_casts_down):
                rsk_numpy_array = np.array(
                    _rsk.npsamples(endpoints.start_time, endpoints.end_time)
                )
                for x, timestamp in enumerate(rsk_numpy_array[self._TIMESTAMP_LABEL]):
                    rsk_numpy_array[self._TIMESTAMP_LABEL][x] = timestamp.strftime(
                        self._TIME_FORMAT
                    )
                profile_to_process = (
                    pl.DataFrame(rsk_numpy_array)
                    .rename(self._rskLabels_to_labelInternal)
                    .drop_nulls()
                )
                geodata = _rsk.geodata(endpoints.start_time, endpoints.end_time)
                processed_profile = _process_profile(profile_to_process, geodata)
                if processed_profile is not None:
                    self._data = pl.concat(
                        [processed_profile, self._data], how=self._CONCAT_HOW
                    )
                    self._num_profiles += 1
                else:
                    CTDLogger(
                        filename=self._filename,
                        message=self._WARNING_DROPPED_PROFILE + str(self._num_profiles),
                        level="warning",
                    )
            if self._data.is_empty():
                rsk_numpy_array = np.array(_rsk.npsamples())
                for x, timestamp in enumerate(rsk_numpy_array[self._TIMESTAMP_LABEL]):
                    rsk_numpy_array[self._TIMESTAMP_LABEL][x] = timestamp.strftime(
                        self._TIME_FORMAT
                    )
                profile = (
                    pl.DataFrame(rsk_numpy_array)
                    .rename(self._rskLabels_to_labelInternal)
                    .drop_nulls()
                )
                geodata = _rsk.geodata()
                processed_profile = _process_profile(profile, geodata)
                if processed_profile is not None:
                    self._data = pl.concat(
                        [processed_profile, self._data], how=self._CONCAT_HOW
                    )
                    self._num_profiles += 1
                else:
                    CTDError(message=self._ERROR_NO_SAMPLES, filename=self._filename)

        def _load_castaway_file():
            with open(ctd_file_path) as file:
                profile = pl.read_csv(file, comment_prefix="%")
            if profile.is_empty():
                raise CTDError(message=self._ERROR_NO_SAMPLES, filename=self._filename)
            if self._CASTAWAY_DATETIME_LABEL in profile.columns:
                profile = profile.with_columns(
                    pl.col(self._CASTAWAY_DATETIME_LABEL)
                    .str.to_datetime(
                        format=self._TIME_FORMAT,
                        time_zone=self._TIME_ZONE,
                        time_unit=self._TIME_UNIT,
                    )
                    .cast(pl.Datetime)
                    .alias(self._CASTAWAY_DATETIME_LABEL)
                )
                start_time = profile.select(
                    pl.col(self._CASTAWAY_DATETIME_LABEL).first()
                ).item()
            else:
                start_time = CTD.Utility.extract_utc_cast_time(ctd_file_path)
            timestamps = [
                start_time + timedelta(milliseconds=200 * i)
                for i in range(profile.height)
            ]
            profile = profile.with_columns(
                pl.Series(timestamps)
                .dt.replace_time_zone(self._TIME_ZONE)
                .dt.cast_time_unit(self._TIME_UNIT)
                .alias(self._TIMESTAMP_LABEL)
            )
            for header, maps_to in self._csvLabels_to_labelInternal.items():
                if header in profile.columns:
                    profile = profile.rename({header: maps_to})
            profile = profile.drop(
                self._CASTAWAY_FILE_ID_LABEL, strict=False
            ).with_columns(
                (pl.col(self._SEA_PRESSURE_LABEL) + 10.1325).alias(
                    self._PRESSURE_LABEL
                ),
                pl.lit(0).alias(self._PROFILE_ID_LABEL),
                pl.lit(self._filename).alias(self._FILENAME_LABEL),
            )
            if self._LATITUDE_LABEL not in profile.collect_schema().names():
                lat, long = self.Utility.extract_lat_long_castaway(ctd_file_path)
                profile = profile.with_columns(
                    pl.lit(lat).alias(self._LATITUDE_LABEL),
                    pl.lit(long).alias(self._LONGITUDE_LABEL),
                )
            self._data = profile
            self._num_profiles += 1

        if self._RSK_FILE_MARKER in ctd_file_path:
            try:
                _load_rsk_file()
            except OperationalError:
                raise CTDError(filename=self._filename, message=self._ERROR_RSK_CORRUPT)
        elif self._CASTAWAY_FILE_MARKER in ctd_file_path:
            _load_castaway_file()
        else:
            raise CTDError(filename=self._filename, message=self._ERROR_NO_SAMPLES)

        if self._data.is_empty():
            raise CTDError(filename=self._filename, message=self._ERROR_NO_SAMPLES)

        try:
            self._data = self._data.with_columns(
                pl.col(self._LATITUDE_LABEL).cast(pl.Float64),
                pl.col(self._LONGITUDE_LABEL).cast(pl.Float64),
            )
        except pl.exceptions.InvalidOperationError:
            raise CTDError(
                message=self._ERROR_LOCATION_DATA_INVALID, filename=self._filename
            )

        if add_unique_id:
            self._data = self._data.with_columns(
                pl.lit(None, dtype=pl.String).alias(self._UNIQUE_ID_LABEL),
                pl.lit(None, dtype=pl.String).alias(self._SECCHI_DEPTH_LABEL),
            )
            for profile_id in (
                self._data.select(self._PROFILE_ID_LABEL)
                .unique(keep="first")
                .to_series()
                .to_list()
            ):
                profile = self._data.filter(
                    pl.col(self._PROFILE_ID_LABEL) == profile_id
                )
                _, _, unique_id, secchi_depth = self._process_master_sheet(
                    profile, for_id=True
                )
                profile = profile.with_columns(
                    pl.lit(unique_id, dtype=pl.String).alias(self._UNIQUE_ID_LABEL),
                    pl.lit(secchi_depth, dtype=pl.String).alias(
                        self._SECCHI_DEPTH_LABEL
                    ),
                )
                self._data = self._data.filter(
                    pl.col(self._PROFILE_ID_LABEL) != profile_id
                ).vstack(profile)

        CTDLogger(
            filename=self._filename,
            message=self._INFO_CTD_OBJECT_INITITALIZED,
            level="info",
        )

    def _find_master_sheet_file(self) -> None:
        """
        Function to find and the master sheet path. Uses the first xlsx file in the current working directory.

        """
        cwd = CTD.Utility.get_cwd()
        xlsx_files = [file for file in os.listdir(cwd) if file.endswith(".xlsx")]
        if len(xlsx_files) > 0:
            self.master_sheet_path = os.path.abspath(xlsx_files[0])

    def get_df(self, pandas=False) -> pl.DataFrame | Any:
        """
        Returns the dataframe of the CTD object for integration with custom pipelines.

        Parameters
        ----------
        pandas : bool, default False
            If True returns a pandas df, if False returns a polars DataFrame. Defaults to False.

        Examples
        --------
        Accessing CTD data as a polars dataframe

        >>> from ctdfjorder import CTD
        >>> ctd_data = CTD('CC1531002_20181225_114931.csv')
        >>> ctd_data.remove_non_positive_samples()
        >>> output = ctd_data.get_df()
        >>> print(output.head(3))
        shape: (3, 13)
        ┌──────────────┬──────────┬─────────────┬──────────────┬───┬────────────┬───────────────────────────────┬────────────┬────────────┐
        │ sea_pressure ┆ depth    ┆ temperature ┆ conductivity ┆ … ┆ profile_id ┆ filename                      ┆ latitude   ┆ longitude  │
        │ ---          ┆ ---      ┆ ---         ┆ ---          ┆   ┆ ---        ┆ ---                           ┆ ---        ┆ ---        │
        │ f64          ┆ f64      ┆ f64         ┆ f64          ┆   ┆ i32        ┆ str                           ┆ f64        ┆ f64        │
        ╞══════════════╪══════════╪═════════════╪══════════════╪═══╪════════════╪═══════════════════════════════╪════════════╪════════════╡
        │ 0.15         ┆ 0.148676 ┆ 0.32895     ┆ 28413.735648 ┆ … ┆ 0          ┆ CC1531002_20181225_114931.csv ┆ -64.668455 ┆ -62.641775 │
        │ 0.45         ┆ 0.446022 ┆ 0.316492    ┆ 28392.966662 ┆ … ┆ 0          ┆ CC1531002_20181225_114931.csv ┆ -64.668455 ┆ -62.641775 │
        │ 0.75         ┆ 0.743371 ┆ 0.310613    ┆ 28386.78011  ┆ … ┆ 0          ┆ CC1531002_20181225_114931.csv ┆ -64.668455 ┆ -62.641775 │
        └──────────────┴──────────┴─────────────┴──────────────┴───┴────────────┴───────────────────────────────┴────────────┴────────────┘

        Accessing CTD data as a pandas dataframe

        >>> from ctdfjorder import CTD
        >>> ctd_data = CTD('CC1531002_20181225_114931.csv')
        >>> ctd_data.remove_non_positive_samples()
        >>> output = ctd_data.get_df(pandas=True)
        >>> print(output.head(3))
           sea_pressure     depth  temperature  conductivity  specific_conductivity  ...  pressure  profile_id                       filename   latitude  longitude
        0          0.15  0.148676      0.32895  28413.735648           56089.447456  ...   10.2825           0  CC1531002_20181225_114931.csv -64.668455 -62.641775
        1          0.45  0.446022     0.316492  28392.966662           56076.028991  ...   10.5825           0  CC1531002_20181225_114931.csv -64.668455 -62.641775
        2          0.75  0.743371     0.310613   28386.78011           56076.832208  ...   10.8825           0  CC1531002_20181225_114931.csv -64.668455 -62.641775
        [3 rows x 13 columns]

        Returns
        -------
        pl.DataFrame | pd.DataFrame
            CTD data in pandas when pandas=True, polars when pandas=False.

        Notes
        -----
        There is no supported method to reinsert the dataframe back into the :class:`CTD` object. Any changes made on
        this dataframe will not be reflected in the :class:`CTD` objects internal data.

        """
        # Convert each DataFrame to a DataFrame and collect them in a list
        if pandas:
            return self._data.to_pandas(use_pyarrow_extension_array=True)
        else:
            return self._data

    def _is_profile_empty(self, func: str) -> bool:
        if self._data.is_empty():
            raise CTDError(
                filename=self._filename,
                message=f"No valid samples in file after running {func}",
            )
        return True

    def remove_upcasts(self) -> None:
        """
        Removes upcasts by dropping rows where pressure decreases from one sampling event to the next.

        """
        for profile_id in (
            self._data.select(self._PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(self._PROFILE_ID_LABEL) == profile_id)
            profile = profile.filter((pl.col(self._PRESSURE_LABEL).diff()) > 0.0)
            self._data = self._data.filter(pl.col(self._PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_profile_empty(CTD.remove_upcasts.__name__)

    def remove_non_positive_samples(self) -> None:
        """
        Removes rows with non-positive values for depth, pressure, practical salinity, absolute salinity, or density.

        """
        for profile_id in (
            self._data.select(self._PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(self._PROFILE_ID_LABEL) == profile_id)
            cols = list(
                {
                    self._DEPTH_LABEL,
                    self._PRESSURE_LABEL,
                    self._SALINITY_LABEL,
                    self._SALINITY_ABS_LABEL,
                    self._DENSITY_LABEL,
                }.intersection(profile.collect_schema().names())
            )
            for col in cols:
                profile = profile.filter(
                    pl.col(col) > 0.0, ~pl.col(col).is_null(), pl.col(col).is_not_nan()
                )
            self._data = self._data.filter(pl.col(self._PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_profile_empty(CTD.remove_non_positive_samples.__name__)

    def remove_invalid_salinity_values(self) -> None:
        """
        Removes rows with practical salinity values <= 10.

        """
        # if len(self._data) < 1:
        # raise CTDError(filename=self._filename, message=self._NO_SAMPLES_ERROR)
        for profile_id in (
            self._data.select(self._PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(self._PROFILE_ID_LABEL) == profile_id)
            profile = profile.filter(pl.col(self._SALINITY_LABEL) > 10)
            self._data = self._data.filter(pl.col(self._PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_profile_empty(CTD.remove_invalid_salinity_values.__name__)

    def clean(self, method) -> None:
        """
        Applies data cleaning methods to the specified feature using the selected method.
        Currently supports cleaning practical salinity using 'clean_salinity_ai' method.

        Parameters
        ----------
        method : str, default 'clean_salinity_ai'
            The cleaning method to apply.

        Raises
        ------
        CTDError
            When the cleaning method is invalid.

        """
        for profile_id in (
            self._data.select(self._PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(self._PROFILE_ID_LABEL) == profile_id)
            if method == "clean_salinity_ai":
                profile = self.clean_salinity_ai(profile, profile_id)
            else:
                raise CTDError(message="Method invalid for clean.", filename=self._filename)
            self._data = self._data.filter(pl.col(self._PROFILE_ID_LABEL) != profile_id)
            self._data = pl.concat([self._data, profile], how=self._CONCAT_HOW)
        self._is_profile_empty(CTD.clean.__name__)

    def add_absolute_salinity(self) -> None:
        """
        Calculates and adds absolute salinity to the CTD data using the TEOS-10 salinity conversion formula.

        Notes
        -----
        - gsw.conversions.SA_from_SP used to calculate absolute salinity using TEOS-10 standards.
        - More info on the gsw function can be found here https://www.teos-10.org/pubs/gsw/html/gsw_SA_from_SP.html

        """
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(self._SALINITY_ABS_LABEL)
        )
        for profile_id in (
            self._data.select(self._PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(self._PROFILE_ID_LABEL) == profile_id)
            s = profile.select(pl.col(self._SALINITY_LABEL)).to_numpy()
            p = profile.select(pl.col(self._SEA_PRESSURE_LABEL)).to_numpy()
            lat = profile.select(pl.col(self._LATITUDE_LABEL)).to_numpy()
            long = profile.select(pl.col(self._LONGITUDE_LABEL)).to_numpy()
            salinity_abs_list = gsw.conversions.SA_from_SP(s, p, lat, long)
            salinity_abs = pl.Series(
                np.array(salinity_abs_list).flatten(), dtype=pl.Float64, strict=False
            ).to_frame(self._SALINITY_ABS_LABEL)
            profile = profile.with_columns(salinity_abs)
            self._data = self._data.filter(pl.col(self._PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_profile_empty(CTD.add_absolute_salinity.__name__)

    def add_density(self):
        """
        Calculates and adds density to CTD data using the TEOS-10 formula.
        If absolute salinity is not present it is calculated first.

        Notes
        -----
        - gsw.density.rho_t_exact used to calculate absolute density using TEOS-10 standards.
        - More info on the gsw function can be found here https://www.teos-10.org/pubs/gsw/html/gsw_rho_t_exact.html

        """
        if self._SALINITY_ABS_LABEL not in self._data.columns:
            self.add_absolute_salinity()
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(self._DENSITY_LABEL)
        )
        for profile_id in (
            self._data.select(self._PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(self._PROFILE_ID_LABEL) == profile_id)
            sa = profile.select(pl.col(self._SALINITY_ABS_LABEL)).to_numpy()
            t = profile.select(pl.col(self._TEMPERATURE_LABEL)).to_numpy()
            p = profile.select(pl.col(self._SEA_PRESSURE_LABEL)).to_numpy()
            density = pl.Series(
                np.array(gsw.density.rho_t_exact(sa, t, p)).flatten(),
                dtype=pl.Float64,
                strict=False,
            ).to_frame(self._DENSITY_LABEL)
            profile = profile.with_columns(density)
            self._data = self._data.filter(pl.col(self._PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_profile_empty(CTD.add_density.__name__)

    def add_potential_density(self):
        """
        Calculates and adds potential density to the CTD data using the TEOS-10 formula.
        If absolute salinity is not present it is calculated first.

        Notes
        -----
        - gsw.sigma0 used to calculate absolute density using TEOS-10 standards.
        - More info on the gsw function can be found here https://www.teos-10.org/pubs/gsw/html/gsw_sigma0.html

        """
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(self._POTENTIAL_DENSITY_LABEL)
        )
        if self._SALINITY_ABS_LABEL not in self._data.columns:
            self.add_absolute_salinity()
        for profile_id in (
            self._data.select(self._PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(self._PROFILE_ID_LABEL) == profile_id)
            sa = profile.select(pl.col(self._SALINITY_ABS_LABEL)).to_numpy()
            t = profile.select(pl.col(self._TEMPERATURE_LABEL)).to_numpy()
            potential_density = pl.Series(
                np.array(gsw.sigma0(sa, t)).flatten()
            ).to_frame(self._POTENTIAL_DENSITY_LABEL)
            profile = profile.with_columns(pl.Series(potential_density))
            self._data = self._data.filter(pl.col(self._PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_profile_empty(CTD.add_potential_density.__name__)

    def add_surface_salinity_temp_meltwater(self, start=10.1325, end=12.1325):
        """
        Calculates the surface salinity, surface temperature and meltwater fraction of a CTD profile.
        Adds these values to the CTD data.

        Parameters
        ----------
        start : float, default 10.1325
            Upper bound of surface pressure.
        end : float, default 12.1325
            Lower bound of surface pressure.

        Notes
        -----
        - Surface temperature is the mean temperature from pressure start to end.
        - Surface salinity calculated as the salinity value at the lowest pressure.

        Meltwater fraction equation where S₀ is surface salinity :math:`(-0.021406 * S₀ + 0.740392) * 100`

        """
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(self._SURFACE_SALINITY_LABEL),
            pl.lit(None, dtype=pl.Float64).alias(self._SURFACE_TEMPERATURE_LABEL),
            pl.lit(None, dtype=pl.Float64).alias(self._MELTWATER_FRACTION_LABEL),
        )
        for profile_id in (
            self._data.select(self._PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(self._PROFILE_ID_LABEL) == profile_id)
            surface_data = profile.filter(
                pl.col(self._PRESSURE_LABEL) > start, pl.col(self._PRESSURE_LABEL) < end
            )
            if surface_data.is_empty():
                CTDLogger(
                    filename=self._filename,
                    message=self._INFO_CTD_SURFACE_MEASUREMENT,
                    level="info",
                )
                self._is_profile_empty(CTD.add_surface_salinity_temp_meltwater.__name__)
                continue
            surface_salinity = np.array(
                surface_data.select(pl.col(self._SALINITY_LABEL)).to_numpy()
            )
            surface_salinity = surface_salinity.item(0)
            surface_temperature = np.array(
                surface_data.select(pl.col(self._TEMPERATURE_LABEL).mean()).to_numpy()
            ).item(0)
            mwf = (-0.021406 * surface_salinity + 0.740392) * 100
            profile = profile.with_columns(
                pl.lit(surface_salinity).alias(self._SURFACE_SALINITY_LABEL),
                pl.lit(surface_temperature).alias(self._SURFACE_TEMPERATURE_LABEL),
                pl.lit(mwf).alias(self._MELTWATER_FRACTION_LABEL),
            )
            self._data = self._data.filter(pl.col(self._PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_profile_empty(CTD.add_surface_salinity_temp_meltwater.__name__)

    def add_mean_surface_density(self, start=10.1325, end=12.1325):
        """
        Calculates the mean surface density from the density values and adds it as a new column
        to the CTD data table.
        Requires absolute salinity and absolute density to be calculated first.

        Parameters
        ----------
        start : float, default 10.1325
            Upper bound of surface pressure.
        end : float, default 12.1325
            Lower bound of surface pressure.

        Notes
        -----
        - Mean surface density calculated as the mean of density from pressure start to end.

        """
        # Filtering data within the specified pressure range
        for profile_id in (
            self._data.select(self._PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(self._PROFILE_ID_LABEL) == profile_id)
            surface_data = profile.filter(
                pl.col(self._PRESSURE_LABEL) > start, pl.col(self._PRESSURE_LABEL) < end
            )
            surface_density = surface_data.select(
                pl.col(self._DENSITY_LABEL).mean()
            ).item()
            profile = profile.with_columns(
                pl.lit(surface_density).alias(self._SURFACE_DENSITY_LABEL)
            )
            self._data = self._data.filter(pl.col(self._PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_profile_empty(CTD.add_mean_surface_density.__name__)

    def add_mld(self, reference: int, method="potential_density_avg", delta=0.05):
        """
        Calculates and adds the mixed layer depth (MLD) using the density threshold method.

        Parameters
        ----------
        reference : int
            The reference depth for MLD calculation.
        method : str, default "potential_density_avg"
            The MLD calculation method options are "abs_density_avg" or "potential_density_avg".
             (default: "potential_density_avg").
        delta : float, default 0.05
            The change in density or potential density from the reference that would define the MLD.

        Notes
        -----
        MLD equation :math:`min(D + ∆ > Dᵣ)`

        - Dᵣ is the reference density, defined as the mean density up to the reference depth
        - D is all densities
        - ∆ is a constant

        """
        supported_methods = ["abs_density_avg", "potential_density_avg"]
        self._mld_col_labels.append(f"MLD {reference}")
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(self._mld_col_labels[-1])
        )
        for profile_id in (
            self._data.select(self._PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(self._PROFILE_ID_LABEL) == profile_id)
            unpack = None
            df_filtered = profile.filter(pl.col(CTD._DEPTH_LABEL) <= reference)
            if method == supported_methods[0]:
                reference_density = df_filtered.select(
                    pl.col(CTD._DENSITY_LABEL).mean()
                ).item()
                df_filtered = profile.filter(
                    pl.col(CTD._DENSITY_LABEL) >= reference_density + delta
                )
            elif method == supported_methods[1]:
                reference_density = df_filtered.select(
                    pl.col(CTD._POTENTIAL_DENSITY_LABEL).mean()
                ).item()
                df_filtered = profile.filter(
                    pl.col(CTD._POTENTIAL_DENSITY_LABEL) >= reference_density + delta
                )
            else:
                raise CTDError(
                    message=f'add_mld: Invalid method "{method}" not in {supported_methods}',
                    filename=self._filename,
                )
            mld = df_filtered.select(pl.col(CTD._DEPTH_LABEL).first()).item()
            profile = profile.with_columns(pl.lit(mld).alias(self._mld_col_labels[-1]))
            self._data = self._data.filter(pl.col(self._PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_profile_empty(CTD.add_mld.__name__)

    def add_bf_squared(self):
        """
        Calculates buoyancy frequency squared and adds to the CTD data.
        Requires potential density to be calculated first.

        Notes
        -----
        - gsw.sigma0 used to calculate absolute density using TEOS-10 standards.
        - More info on the gsw function can be found here https://www.teos-10.org/pubs/gsw/html/gsw_Nsquared.html

        """
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(self._BV_LABEL),
            pl.lit(None, dtype=pl.Float64).alias(self._P_MID_LABEL),
        )
        for profile_id in (
            self._data.select(self._PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(self._PROFILE_ID_LABEL) == profile_id)
            sa = profile.select(pl.col(self._SALINITY_ABS_LABEL)).to_numpy().flatten()
            t = profile.select(pl.col(self._TEMPERATURE_LABEL)).to_numpy().flatten()
            p = profile.select(pl.col(self._SEA_PRESSURE_LABEL)).to_numpy().flatten()
            lat = profile.select(pl.col(self._LATITUDE_LABEL)).to_numpy().flatten()
            ct = gsw.CT_from_t(sa, t, p).flatten()
            n_2, p_mid = gsw.Nsquared(SA=sa, CT=ct, p=p, lat=lat)
            buoyancy_frequency = (
                pl.Series(np.array(n_2).flatten())
                .extend_constant(None, n=1)
                .to_frame(self._BV_LABEL)
            )
            p_mid = (
                pl.Series(p_mid).extend_constant(None, n=1).to_frame(self._P_MID_LABEL)
            )
            profile = profile.with_columns(
                pl.Series(buoyancy_frequency), pl.Series(p_mid)
            )
            self._data = self._data.filter(pl.col(self._PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_profile_empty(CTD.add_surface_salinity_temp_meltwater.__name__)

    def plot(self, measurement, plot_type="scatter"):
        """
        Generates a plot of depth vs. specified measurement (salinity, density, temperature).

        Parameters
        ----------
        measurement : str
            Options are self._SALINITY_LABEL, self._DENSITY_LABEL, 'potential_density, or self._TEMPERATURE_LABEL.
        plot_type : str
            Options are 'scatter' or 'line'.

        Notes
        -----
        - Adds horizontal lines indicating the mixed layer depth (MLD) if present.
        - Allows for both scatter and line plot types.
        - Saves the plot as an image file.

        """
        plt.rcParams.update({"font.size": 16})
        plot_folder = os.path.join(self._cwd, "plots")
        os.makedirs(plot_folder, exist_ok=True)
        for profile_id in (
            self._data.select(self._PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(self._PROFILE_ID_LABEL) == profile_id)
            fig, ax1 = plt.subplots(figsize=(18, 18))
            ax1.invert_yaxis()
            ax1.set_ylim([profile.select(pl.col(self._DEPTH_LABEL)).max().item(), 0])
            color_map = {
                self._SALINITY_LABEL: "tab:blue",
                self._DENSITY_LABEL: "tab:red",
                self._POTENTIAL_DENSITY_LABEL: "tab:red",
                self._TEMPERATURE_LABEL: "tab:blue",
            }
            label_map = {
                self._SALINITY_LABEL: "Practical Salinity (PSU)",
                self._DENSITY_LABEL: "Density (kg/m^3)",
                self._POTENTIAL_DENSITY_LABEL: "Potential Density (kg/m^3)",
                self._TEMPERATURE_LABEL: "Temperature (°C)",
            }
            if plot_type == "line":
                lowess = statsmodels.api.nonparametric.lowess
                y, x = zip(
                    *lowess(
                        profile.select(pl.col(f"{measurement}")).to_numpy(),
                        profile.select(pl.col(self._DEPTH_LABEL)).to_numpy(),
                        frac=0.1,
                    )
                )
            else:
                x, y = (
                    profile.select(pl.col(f"{measurement}")).to_numpy(),
                    profile.select(pl.col(self._DEPTH_LABEL)).to_numpy(),
                )
            (
                ax1.plot(
                    x, y, color=color_map[measurement], label=label_map[measurement]
                )
                if plot_type == "line"
                else ax1.scatter(
                    x, y, color=color_map[measurement], label=label_map[measurement]
                )
            )
            ax1.set_xlabel(label_map[measurement], color=color_map[measurement])
            ax1.set_ylabel("Depth (m)")
            ax1.tick_params(axis="x", labelcolor=color_map[measurement])
            mld_col = None
            mld = profile.select(pl.col(r"^.*MLD.*$").first()).item()
            if mld is not None:
                # Plot MLD line
                ax1.axhline(
                    y=mld, color="green", linestyle="--", linewidth=2, label=f"{mld}"
                )
                ax1.text(
                    0.95,
                    mld,
                    f"{mld}",
                    va="center",
                    ha="right",
                    backgroundcolor="white",
                    color="green",
                    transform=ax1.get_yaxis_transform(),
                )
            plt.title(
                f"{self._filename} \n Profile {profile_id} \n Depth vs. {label_map[measurement]}\n MLD {mld}"
            )
            ax1.grid(True)
            ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
            plot_path = os.path.join(
                plot_folder,
                f"{self._filename}_{profile_id}_depth_{measurement}_{plot_type}_plot.png",
            )
            plt.savefig(plot_path)
            plt.close(fig)

    def _process_master_sheet(
        self, cached_master_sheet: pl.DataFrame = pl.DataFrame(), for_id=False
    ) -> Tuple[Any, Any, str, float | None]:
        """
        Extracts the date and time components from the filename and compares them with the data
        in the master sheet. Calculates the absolute differences between the dates and times to
        find the closest match. Returns the estimated latitude, longitude, and updated filename
        based on the closest match.

        Parameters
        ----------
        cached_master_sheet : pl.DataFrame, default pl.DataFrame()
            CTD profile

        for_id : bool, default False
            Flag for logging purposes, indicates if we processed for location or id.

        Returns
        -------
        tuple
            A tuple containing the estimated latitude, longitude, unique id, and secchi depth.

        Raises
        ------
        CTDError
            When there is no timestamp data in the master sheet and/or CTD file.

        """
        if (
            type(self._cached_master_sheet) is type(None)
            or self._cached_master_sheet.is_empty()
        ):
            if self.master_sheet_path:
                self._cached_master_sheet = CTD.Utility.load_master_sheet(
                    self.master_sheet_path
                )
            else:
                raise CTDError(
                    message=self._ERROR_NO_MASTER_SHEET, filename=self._filename
                )
        if self._TIMESTAMP_LABEL not in cached_master_sheet.collect_schema().names():
            raise CTDError(
                message=self._ERROR_NO_TIMESTAMP_IN_FILE, filename=self._filename
            )
        if "datetime" not in self._cached_master_sheet.collect_schema().names():
            raise CTDError(
                message=self._ERROR_NO_TIMESTAMP_IN_MASTER_SHEET,
                filename=self._filename,
            )
        timestamp_highest = cached_master_sheet.select(
            pl.last(self._TIMESTAMP_LABEL).dt.datetime()
        ).item()
        closest_row_overall = self._cached_master_sheet.select(
            pl.all().sort_by((pl.col("datetime") - timestamp_highest).abs())
        )
        latitude = closest_row_overall.select(
            pl.col(self._LATITUDE_LABEL).first()
        ).item()
        longitude = closest_row_overall.select(
            pl.col(self._LONGITUDE_LABEL).first()
        ).item()
        distance = (
            closest_row_overall.select(pl.col("datetime").first()).item()
            - timestamp_highest
        )
        unique_id = closest_row_overall.select(pl.col("UNIQUE ID CODE ").first()).item()
        secchi_depth = None
        if for_id:
            secchi_depth = closest_row_overall.select(
                pl.col(self._MASTER_SHEET_SECCHI_DEPTH_LABEL).first()
            ).item()
        # Extract days, hours, and minutes from the time difference
        days = abs(distance.days)
        hours, remainder = divmod(distance.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        if for_id:
            message = (
                f"Guessed Location : Matched to unique ID '{unique_id}' "
                f"with distance {days} days and time difference {hours}:{minutes}"
            )
        else:
            message = (
                f"Guessed Unique ID : Matched to unique ID '{unique_id}' with "
                f"distance {days} days and time difference {hours}:{minutes}"
            )
        if abs(days) > 2:
            CTDLogger(filename=self._filename, message=message, level="warning")
        else:
            CTDLogger(filename=self._filename, message=message, level="info")
        return latitude, longitude, unique_id, secchi_depth

    def clean_salinity_ai(self, profile: pl.DataFrame, profile_id: int) -> pl.DataFrame:
        """
        Cleans salinity using a GRU ML model.

        Parameters
        ----------
        profile : pl.DataFrame
            Single profile of CTD data.
        profile_id : int
            Profile number.

        Returns
        -------
        pl.DataFrame
            CTD data with clean salinity values.

        Notes
        -----
        - This process will bin the data every 0.5 dbar of pressure.
        - Uses a 16 width GRU layer with a loss function to penalize decreasing salinity values w.r.t. pressure

        """

        def loss_function(y_true, y_pred):
            """
            MAE loss with additional term to penalize salinity predictions that increase with pressure.

            Parameters
            ----------
            y_true
                True salinity tensor.
            y_pred
                Predicted salinity tensor.
            Returns
            -------
            float
                Loss value.

            """
            # Assuming salinity is at index 0
            salinity_true = y_true[:, :, 0]
            salinity_pred = y_pred[:, :, 0]

            # Calculate differences between consecutive values
            delta_sal_true = salinity_true[:, 1:] - salinity_true[:, :-1]
            delta_sal_pred = salinity_pred[:, 1:] - salinity_pred[:, :-1]

            # Penalize predictions where salinity decreases while pressure increases
            penalties = tf.where(delta_sal_pred < 0, -tf.minimum(delta_sal_pred, 0), 0)

            # Calculate mean absolute error
            mae = tf.reduce_mean(tf.abs(y_true - y_pred))

            # Add penalties
            return mae + 6.0 * tf.reduce_mean(
                penalties
            )  # Adjust weighting of penalty as needed

        def build_gru(input_shape):
            """
            GRU architecture.

            """
            inputs = Input(shape=(input_shape[0], input_shape[1]), name="inputs")
            gru1 = GRU(16, activation="tanh", return_sequences=True)(inputs)
            output = TimeDistributed(
                Dense(input_shape[1], activation="linear"), name="output"
            )(gru1)

            gru = Model(inputs, output)
            optimizer = Adam(learning_rate=0.01)
            gru.compile(
                optimizer=optimizer, loss=loss_function
            )  # Specify your loss function here
            return gru

        def plot_original_data(salinity, depths, filename):
            plt.figure(figsize=(10, 6))
            plt.scatter(salinity, -depths, alpha=0.6)
            plt.xlabel("Salinity (PSU)")
            plt.ylabel("Depth (m)")
            plt.title(f"Original Salinity vs. Depth - {filename}")
            plt.grid(True)
            plt.savefig(os.path.join("plots", f"{filename}_original.png"))
            xlim = plt.xlim()
            ylim = plt.ylim()
            # plt.show()
            plt.close()
            return xlim, ylim

        def plot_predicted_data(salinity, depths, xlim, ylim, filename):
            plt.figure(figsize=(10, 6))
            plt.scatter(salinity, -depths, alpha=0.6, color="red")
            title = f"Predicted Salinity vs. Depth - {filename}"
            plt.title(title)
            plt.xlabel("Salinity (PSU)")
            plt.ylabel("Depth (m)")
            plt.grid(True)
            plt.xlim((xlim[0], xlim[1]))
            plt.ylim((ylim[0], ylim[1]))
            plt.savefig(os.path.join("plots", f"{filename}_predicted.png"))
            # plt.show()
            plt.close()

        def run_gru(data: pl.DataFrame, show_plots=False):
            """
            Runs the GRU.

            Parameters
            ----------
            data : DataFrame
                CTD dataframe
            show_plots : bool, default False
                True saves plots in working directory within the plots folder.
            Returns
            -------
            DataFrame
                CTD data with clean salinity values.

            Raises
            ------
            CTDError
                When there are not enough values to train on.

            """
            filtered_data = data.filter(pl.col(self._DEPTH_LABEL) > 1)
            filtered_data = filtered_data.with_columns(
                (pl.col(self._PRESSURE_LABEL) // 0.5 * 0.5).alias("pressure_bin")
            )
            # Define the desired columns and their aggregation functions
            column_agg_dict = {
                "temperature": pl.mean("temperature"),
                "chlorophyll": pl.mean("chlorophyll"),
                "sea_pressure": pl.mean("sea_pressure"),
                "depth": pl.mean("depth"),
                "salinity": pl.median("salinity"),
                "speed_of_sound": pl.mean("speed_of_sound"),
                "specific_conductivity": pl.mean("specific_conductivity"),
                "conductivity": pl.mean("conductivity"),
                "density": pl.mean("density"),
                "potential_density": pl.mean("potential_density"),
                "salinity_abs": pl.mean("salinity_abs"),
                "timestamp": pl.first("timestamp"),
                "longitude": pl.first("longitude"),
                "latitude": pl.first("latitude"),
                "unique_id": pl.first("unique_id"),
                "filename": pl.first("filename"),
                "profile_id": pl.first("profile_id"),
                "secchi_depth": pl.first("secchi_depth"),
            }
            available_columns = {
                col: agg_func
                for col, agg_func in column_agg_dict.items()
                if col in data.columns
            }
            data_binned = filtered_data.group_by(
                "pressure_bin", maintain_order=True
            ).agg(list(available_columns.values()))
            data_binned = data_binned.rename({"pressure_bin": self._PRESSURE_LABEL})
            scaler = MinMaxScaler(feature_range=(-1, 1))
            if data.limit(4).height < 2:
                raise CTDError(
                    message="Not enough values to run the GRU on this data",
                    filename=self._filename,
                )
            logger.debug(f"{self._filename} - About to scale salinity")
            salinity = np.array(
                data_binned.select(pl.col(self._SALINITY_LABEL)).to_numpy()
            )
            scaled_sequence = scaler.fit_transform(salinity)
            logger.debug(f"{self._filename} - Salinity scaled")
            scaled_seq = np.expand_dims(scaled_sequence, axis=0)
            min_pres = data_binned.select(pl.min(self._DEPTH_LABEL)).item()
            max_pres = data_binned.select(pl.max(self._DEPTH_LABEL)).item()
            pres_range = max_pres - min_pres
            epochs = int(pres_range * 16)
            autoencoder = build_gru(scaled_seq.shape[1:])
            logger.debug(f"{self._filename} - GRU model built")
            autoencoder.fit(
                scaled_seq, scaled_seq, epochs=epochs, verbose=0, batch_size=4
            )
            X_pred = autoencoder.predict(scaled_seq, verbose=None)
            predicted_seq = np.array(scaler.inverse_transform(X_pred[0])).flatten()
            if show_plots:
                xlim, ylim = plot_original_data(
                    data.select(self._SALINITY_LABEL).to_numpy(),
                    data.select(self._DEPTH_LABEL).to_numpy(),
                    self._filename + str(profile_id),
                )
                plot_predicted_data(
                    salinity=predicted_seq,
                    depths=data_binned.select(self._DEPTH_LABEL).to_numpy(),
                    filename=self._filename + str(profile_id),
                    xlim=xlim,
                    ylim=ylim,
                )
            data_binned = data_binned.with_columns(
                pl.Series(predicted_seq, dtype=pl.Float64).alias(self._SALINITY_LABEL)
            )
            return data_binned

        return run_gru(profile)

    def save_to_csv(self, output_file: str):
        """
        Renames the columns of the CTD data table based on a predefined mapping and saves the
        data to the specified CSV file.

        Parameters
        ----------
        output_file : str
            The output CSV file path.

        Notes
        -----
        - Will overwrite exisiting files of the same name

        """
        self.Utility.save_to_csv(self._data, output_file=output_file)

    class Utility:
        """
        Utility class for CTD data processing.

        """

        @staticmethod
        def save_to_csv(data: pl.DataFrame | pl.DataFrame, output_file: str):
            """
            Renames the columns of the CTD data table based on a predefined mapping and saves the
            data to the specified CSV file.

            Parameters
            ----------
            data : pl.DataFrame
                The output CSV file path.
            output_file : str
                The output CSV file path.
            """

            def relabel_ctd_data(label: str):
                data_label_mapping = {
                    "timestamp": "timestamp",
                    "temperature": "Temperature_(°C)",
                    "pressure": "Pressure_(dbar)",
                    "chlorophyll": "Chlorophyll_a_(µg/l)",
                    "sea_pressure": "Sea Pressure_(dbar)",
                    "depth": "Depth_(m)",
                    "salinity": "Salinity_(PSU)",
                    "speed_of_sound": "Speed of Sound_(m/s)",
                    "specific_conductivity": "Specific Conductivity_(µS/cm)",
                    "conductivity": "Conductivity_(mS/cm)",
                    "density": "Density_(kg/m^3)",
                    "potential_density": "Potential_Density_(kg/m^3)",
                    "salinity_abs": "Absolute Salinity_(g/kg)",
                    "stratification": "Stratification_(J/m^2)",
                    "mean_surface_density": "Mean_Surface_Density_(kg/m^3)",
                    "surface_salinity": "Surface_Salinity_(PSU)",
                    "surface_temperature": "Surface_Temperature_(°C)",
                    "meltwater_fraction": "Meltwater_Fraction_(%)",
                    "longitude": "longitude",
                    "latitude": "latitude",
                    "filename": "filename",
                    "Profile_ID": "Profile_ID",
                    "Unique_ID": "Unique_ID",
                    "buoyancy_frequency": "Brunt_Vaisala_Frequency_Squared",
                    "p_mid": "Mid_Pressure_Used_For_BV_Calc",
                }
                if label in data_label_mapping.keys():
                    return data_label_mapping[label]
                else:
                    return label

            data = data.rename(relabel_ctd_data)
            if type(data) is pl.DataFrame:
                data.write_csv(output_file)
            elif type(data) is pl.DataFrame:
                data.write_csv(output_file)

        @staticmethod
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
                            cast_time_str = parts[
                                1
                            ].strip()  # Take only the datetime part
                            # Convert the datetime string to ISO format if possible
                            cast_time_utc = datetime.strptime(
                                cast_time_str, "%Y-%m-%d %H:%M:%S"
                            )
                            break  # Stop reading once the timestamp is found

            return cast_time_utc

        @staticmethod
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

        @staticmethod
        def load_master_sheet(
            master_sheet_path: str, secchi_depth: bool = False
        ) -> pl.DataFrame:
            _masterSheetLabels_to_dtypeInternal: dict[str, type(pl.String)] = {
                "time_local": pl.String,
                "date_local": pl.String,
                "time (UTC)": pl.String,
                "date (UTC)": pl.String,
                "secchi depth": pl.String,
            }
            df = pl.read_excel(
                master_sheet_path,
                infer_schema_length=None,
                schema_overrides=_masterSheetLabels_to_dtypeInternal,
            )
            df = df.drop_nulls("time_local")
            df = df.filter(~pl.col("time_local").eq("-999"))
            df = df.filter(~pl.col("time_local").eq("NA"))
            df = df.filter(~pl.col("date_local").eq("NA"))
            if secchi_depth:
                df = df.with_columns(
                    pl.col("secchi depth").cast(pl.Float64, strict=False)
                )
            df = df.with_columns(
                pl.col("date_local").str.strptime(
                    format="%Y-%m-%d %H:%M:%S%.3f", dtype=pl.Date, strict=False
                )
            )
            df = df.drop_nulls("date_local")
            df = df.with_columns(
                pl.col("time_local").str.strptime(
                    format="%Y-%m-%d %H:%M:%S%.3f", dtype=pl.Time, strict=False
                )
            )
            df = df.drop_nulls("time_local")
            df = df.with_columns(
                (
                    pl.col("date_local").dt.combine(pl.col("time_local").cast(pl.Time))
                ).alias("datetime")
            )
            return df

        @staticmethod
        def get_cwd():
            working_directory_path = None
            # determine if application is a script file or frozen exe
            if getattr(sys, "frozen", False):
                working_directory_path = os.path.dirname(sys.executable)
            elif __file__:
                working_directory_path = os.getcwd()
            else:
                working_directory_path = os.getcwd()
            return working_directory_path


class CTDError(Exception):
    """
    Exception raised for CTD related errors.

    Parameters
    ----------
    filename : str, default None
        Input dataset which caused the error.
    message : str
        Explanation of the error.
    """

    def __init__(self, message, filename=None):
        super().__init__(filename + " - " + message)


class CTDLogger:
    """
    Wrapper for logger.

    Parameters
    ----------
    filename : str
        Input dataset which caused the warning.
    message : str
        Explanation of the warning.
    """

    def __init__(self, message, filename=None, level="info"):
        if level == "info":
            logger.info(filename + " - " + message)
        if level == "debug":
            logger.debug(filename + " - " + message)
        if level == "warning":
            logger.warning(filename + " - " + message)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
