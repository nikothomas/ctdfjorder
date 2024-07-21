# -*- coding: utf-8 -*-
from ctdfjorder.constants import *
from ctdfjorder.ctdplot import plot_original_data
from ctdfjorder.ctdplot import plot_predicted_data
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys
from datetime import datetime
from datetime import timedelta
from sqlite3 import OperationalError
from os import path
import numpy as np
import polars as pl
from ctdfjorder.pyrsktools import RSK, Geo
from ctdfjorder.pyrsktools import Region
import gsw
from typing import Generator
from typing import Any
from typing import Tuple
import logging
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger("ctdfjorder")
logger.propagate = 0


class CTD:
    """
    Class representing a CTD object for processing and analyzing CTD data. A mastersheet is not necessary,
    but if your files are missing location data then many functions will not work

    Attributes
    ----------
    master_sheet_path : str
        Path to the mastersheet.

    Examples
    ---------
    Removing non-positive samples from data:

    >>> from ctdfjorder.ctdfjorder import CTD
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

    """

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
    _plot: bool = False

    def __init__(
        self,
        ctd_file_path: str,
        cached_master_sheet: pl.DataFrame = None,
        master_sheet_path=None,
        add_unique_id=False,
        plot=False,
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
        plots : bool, default False
            If true saves plots to 'plots' folder in working directory.

        Examples
        --------
        Castaway CTD profile with valid data

        >>> from ctdfjorder.ctdfjorder import CTD
        >>> from ctdfjorder.ctdfjorder import CTDError
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

        >>> from ctdfjorder.ctdfjorder import CTD
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
        self._plot = plot

        def _process_rsk_profile(
            lf: pl.DataFrame, geo: Generator[Geo, Any, None]
        ) -> pl.DataFrame:
            lf = lf.with_columns(
                pl.lit(self._filename + FILENAME_CM_ENDING).alias(FILENAME_LABEL)
            )
            try:
                profile_geodata = next(geo)
                return lf.with_columns(
                    pl.lit(profile_geodata.latitude).alias(LATITUDE_LABEL),
                    pl.lit(profile_geodata.longitude).alias(LONGITUDE_LABEL),
                )
            except StopIteration:
                CTDLogger(
                    filename=self._filename,
                    message=DEBUG_FILE_LACKS_LOCATION,
                    level="debug",
                )
                _lat, _long, _, _ = self._process_master_sheet(lf)
                return lf.with_columns(
                    pl.lit(_lat).alias(LATITUDE_LABEL),
                    pl.lit(_long).alias(LONGITUDE_LABEL),
                    pl.lit(self._filename + FILENAME_CM_ENDING).alias(FILENAME_LABEL),
                )

        def _process_profile(rsk_profile: pl.DataFrame, geodata) -> pl.DataFrame | None:
            rsk_profile = rsk_profile.with_columns(
                pl.col(TIMESTAMP_LABEL)
                .cast(pl.String)
                .str.to_datetime(
                    format=TIME_FORMAT,
                    time_zone=TIME_ZONE,
                    time_unit=TIME_UNIT,
                )
                .cast(pl.Datetime(time_unit=TIME_UNIT))
                .dt.convert_time_zone(TIME_ZONE)
            )
            rsk_profile = rsk_profile.with_columns(pl.lit(0).alias(PROFILE_ID_LABEL))
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
                for x, timestamp in enumerate(rsk_numpy_array[TIMESTAMP_LABEL]):
                    rsk_numpy_array[TIMESTAMP_LABEL][x] = timestamp.strftime(
                        TIME_FORMAT
                    )
                profile_to_process = (
                    pl.DataFrame(rsk_numpy_array)
                    .rename(rskLabels_to_labelInternal)
                    .drop_nulls()
                )
                geodata = _rsk.geodata(endpoints.start_time, endpoints.end_time)
                processed_profile = _process_profile(profile_to_process, geodata)
                if processed_profile is not None:
                    self._data = pl.concat(
                        [processed_profile, self._data], how=CONCAT_HOW
                    )
                    self._num_profiles += 1
                else:
                    CTDWarning(
                        filename=self._filename,
                        message=WARNING_DROPPED_PROFILE + str(self._num_profiles),
                    )
            if self._data.is_empty():
                rsk_numpy_array = np.array(_rsk.npsamples())
                for x, timestamp in enumerate(rsk_numpy_array[TIMESTAMP_LABEL]):
                    rsk_numpy_array[TIMESTAMP_LABEL][x] = timestamp.strftime(
                        TIME_FORMAT
                    )
                profile = (
                    pl.DataFrame(rsk_numpy_array)
                    .rename(rskLabels_to_labelInternal)
                    .drop_nulls()
                )
                geodata = _rsk.geodata()
                processed_profile = _process_profile(profile, geodata)
                if processed_profile is not None:
                    self._data = pl.concat(
                        [processed_profile, self._data], how=CONCAT_HOW
                    )
                    self._num_profiles += 1
                else:
                    CTDError(message=ERROR_NO_SAMPLES, filename=self._filename)

        def _load_castaway_file():
            with open(ctd_file_path) as file:
                profile = pl.read_csv(file, comment_prefix="%")
            if profile.is_empty():
                raise CTDError(message=ERROR_NO_SAMPLES, filename=self._filename)
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
                start_time = profile.select(
                    pl.col(CASTAWAY_DATETIME_LABEL).first()
                ).item()
            else:
                start_time = CTD.Utility.extract_utc_cast_time(ctd_file_path)
            if type(start_time) == type(None):
                CTDError(filename=self._filename, message=ERROR_CASTAWAY_START_TIME)
            timestamps = [
                start_time + timedelta(milliseconds=200 * i)
                for i in range(profile.height)
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
                pl.lit(self._filename).alias(FILENAME_LABEL),
            )
            if LATITUDE_LABEL not in profile.collect_schema().names():
                lat, long = self.Utility.extract_lat_long_castaway(ctd_file_path)
                profile = profile.with_columns(
                    pl.lit(lat).alias(LATITUDE_LABEL),
                    pl.lit(long).alias(LONGITUDE_LABEL),
                )
            self._data = profile
            self._num_profiles += 1

        if RSK_FILE_MARKER in ctd_file_path:
            try:
                _load_rsk_file()
            except OperationalError:
                raise CTDError(filename=self._filename, message=ERROR_RSK_CORRUPT)
        elif CASTAWAY_FILE_MARKER in ctd_file_path:
            _load_castaway_file()
        else:
            raise CTDError(filename=self._filename, message=ERROR_NO_SAMPLES)

        if self._data.is_empty():
            raise CTDError(filename=self._filename, message=ERROR_NO_SAMPLES)

        try:
            self._data = self._data.with_columns(
                pl.col(LATITUDE_LABEL).cast(pl.Float64),
                pl.col(LONGITUDE_LABEL).cast(pl.Float64),
            )
        except pl.exceptions.InvalidOperationError:
            raise CTDError(message=ERROR_LOCATION_DATA_INVALID, filename=self._filename)

        if add_unique_id:
            self._data = self._data.with_columns(
                pl.lit(None, dtype=pl.String).alias(UNIQUE_ID_LABEL),
                pl.lit(None, dtype=pl.Float32).alias(SECCHI_DEPTH_LABEL),
            )
            for profile_id in (
                self._data.select(PROFILE_ID_LABEL)
                .unique(keep="first")
                .to_series()
                .to_list()
            ):
                profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
                _, _, unique_id, secchi_depth = self._process_master_sheet(
                    profile, for_id=True
                )
                profile = profile.with_columns(
                    pl.lit(unique_id, dtype=pl.String).alias(UNIQUE_ID_LABEL),
                    pl.lit(secchi_depth, dtype=pl.Float32).alias(SECCHI_DEPTH_LABEL),
                )
                self._data = self._data.filter(
                    pl.col(PROFILE_ID_LABEL) != profile_id
                ).vstack(profile)

        CTDLogger(
            filename=self._filename,
            message=DEBUG_CTD_OBJECT_INITITALIZED,
            level="debug",
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

        >>> from ctdfjorder.ctdfjorder import CTD
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

        >>> from ctdfjorder.ctdfjorder import CTD
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
            self._data.select(PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
            profile = profile.filter((pl.col(PRESSURE_LABEL).diff()) > 0.0)
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_profile_empty(CTD.remove_upcasts.__name__)

    def remove_non_positive_samples(self) -> None:
        """
        Removes rows with non-positive values for depth, pressure, practical salinity, absolute salinity, or density.

        """
        for profile_id in (
            self._data.select(PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
            cols = list(
                {
                    DEPTH_LABEL,
                    PRESSURE_LABEL,
                    SALINITY_LABEL,
                    SALINITY_ABS_LABEL,
                    DENSITY_LABEL,
                }.intersection(profile.collect_schema().names())
            )
            for col in cols:
                profile = profile.filter(
                    pl.col(col) > 0.0, ~pl.col(col).is_null(), pl.col(col).is_not_nan()
                )
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_profile_empty(CTD.remove_non_positive_samples.__name__)


    def remove_invalid_salinity_values(self) -> None:
        """
        Removes rows with practical salinity values <= 10.

        """
        for profile_id in (
            self._data.select(PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
            profile = profile.filter(pl.col(SALINITY_LABEL) > 10)
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
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
            self._data.select(PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
            if method == "clean_salinity_ai":
                profile = self.clean_salinity_ai(profile, profile_id)
            else:
                raise CTDError(
                    message="Method invalid for clean.", filename=self._filename
                )
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
            self._data = pl.concat([self._data, profile], how=CONCAT_HOW)
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
            pl.lit(None, dtype=pl.Float64).alias(SALINITY_ABS_LABEL)
        )
        for profile_id in (
            self._data.select(PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
            s = profile.select(pl.col(SALINITY_LABEL)).to_numpy()
            p = profile.select(pl.col(SEA_PRESSURE_LABEL)).to_numpy()
            lat = profile.select(pl.col(LATITUDE_LABEL)).to_numpy()
            long = profile.select(pl.col(LONGITUDE_LABEL)).to_numpy()
            salinity_abs_list = gsw.conversions.SA_from_SP(s, p, lat, long)
            salinity_abs = pl.Series(
                np.array(salinity_abs_list).flatten(), dtype=pl.Float64, strict=False
            ).to_frame(SALINITY_ABS_LABEL)
            profile = profile.with_columns(salinity_abs)
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
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
        if SALINITY_ABS_LABEL not in self._data.columns:
            self.add_absolute_salinity()
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(DENSITY_LABEL)
        )
        for profile_id in (
            self._data.select(PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
            sa = profile.select(pl.col(SALINITY_ABS_LABEL)).to_numpy()
            t = profile.select(pl.col(TEMPERATURE_LABEL)).to_numpy()
            p = profile.select(pl.col(SEA_PRESSURE_LABEL)).to_numpy()
            density = pl.Series(
                np.array(gsw.density.rho_t_exact(sa, t, p)).flatten(),
                dtype=pl.Float64,
                strict=False,
            ).to_frame(DENSITY_LABEL)
            profile = profile.with_columns(density)
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
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
            pl.lit(None, dtype=pl.Float64).alias(POTENTIAL_DENSITY_LABEL)
        )
        if SALINITY_ABS_LABEL not in self._data.columns:
            self.add_absolute_salinity()
        for profile_id in (
            self._data.select(PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
            sa = profile.select(pl.col(SALINITY_ABS_LABEL)).to_numpy()
            t = profile.select(pl.col(TEMPERATURE_LABEL)).to_numpy()
            potential_density = pl.Series(
                np.array(gsw.sigma0(sa, t)).flatten()
            ).to_frame(POTENTIAL_DENSITY_LABEL)
            profile = profile.with_columns(pl.Series(potential_density))
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
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
            pl.lit(None, dtype=pl.Float64).alias(SURFACE_SALINITY_LABEL),
            pl.lit(None, dtype=pl.Float64).alias(SURFACE_TEMPERATURE_LABEL),
            pl.lit(None, dtype=pl.Float64).alias(MELTWATER_FRACTION_LABEL),
        )
        for profile_id in (
            self._data.select(PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
            surface_data = profile.filter(
                pl.col(PRESSURE_LABEL) > start, pl.col(PRESSURE_LABEL) < end
            )
            if surface_data.is_empty():
                CTDLogger(
                    filename=self._filename,
                    message=INFO_CTD_SURFACE_MEASUREMENT,
                    level="info",
                )
                self._is_profile_empty(CTD.add_surface_salinity_temp_meltwater.__name__)
                continue
            surface_salinity = np.array(
                surface_data.select(pl.col(SALINITY_LABEL)).to_numpy()
            )
            surface_salinity = surface_salinity.item(0)
            surface_temperature = np.array(
                surface_data.select(pl.col(TEMPERATURE_LABEL).mean()).to_numpy()
            ).item(0)
            mwf = (-0.021406 * surface_salinity + 0.740392) * 100
            profile = profile.with_columns(
                pl.lit(surface_salinity).alias(SURFACE_SALINITY_LABEL),
                pl.lit(surface_temperature).alias(SURFACE_TEMPERATURE_LABEL),
                pl.lit(mwf).alias(MELTWATER_FRACTION_LABEL),
            )
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
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
            self._data.select(PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
            surface_data = profile.filter(
                pl.col(PRESSURE_LABEL) > start, pl.col(PRESSURE_LABEL) < end
            )
            surface_density = surface_data.select(pl.col(DENSITY_LABEL).mean()).item()
            profile = profile.with_columns(
                pl.lit(surface_density).alias(SURFACE_DENSITY_LABEL)
            )
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
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
            self._data.select(PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
            unpack = None
            mld = None
            df_filtered = profile.filter(pl.col(DEPTH_LABEL) <= reference)
            if method == supported_methods[0]:
                reference_density = df_filtered.select(
                    pl.col(DENSITY_LABEL).mean()
                ).item()
                df_filtered = profile.filter(
                    pl.col(DENSITY_LABEL) >= reference_density + delta
                )
            elif method == supported_methods[1]:
                reference_density = df_filtered.select(
                    pl.col(POTENTIAL_DENSITY_LABEL).mean()
                ).item()
                df_filtered = profile.filter(
                    pl.col(POTENTIAL_DENSITY_LABEL) >= reference_density + delta
                )
            else:
                raise CTDError(
                    message=f'add_mld: Invalid method "{method}" not in {supported_methods}',
                    filename=self._filename,
                )
            mld = df_filtered.select(pl.col(DEPTH_LABEL).first()).item()
            CTDLogger(filename=self._filename, message=f"MLD: {mld}", level="debug")
            profile = profile.with_columns(pl.lit(mld).alias(self._mld_col_labels[-1]))
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
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
            pl.lit(None, dtype=pl.Float64).alias(BV_LABEL),
            pl.lit(None, dtype=pl.Float64).alias(P_MID_LABEL),
        )
        for profile_id in (
            self._data.select(PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
            sa = profile.select(pl.col(SALINITY_ABS_LABEL)).to_numpy().flatten()
            t = profile.select(pl.col(TEMPERATURE_LABEL)).to_numpy().flatten()
            p = profile.select(pl.col(SEA_PRESSURE_LABEL)).to_numpy().flatten()
            lat = profile.select(pl.col(LATITUDE_LABEL)).to_numpy().flatten()
            ct = gsw.CT_from_t(sa, t, p).flatten()
            n_2, p_mid = gsw.Nsquared(SA=sa, CT=ct, p=p, lat=lat)
            buoyancy_frequency = (
                pl.Series(np.array(n_2).flatten())
                .extend_constant(None, n=1)
                .to_frame(BV_LABEL)
            )
            p_mid = pl.Series(p_mid).extend_constant(None, n=1).to_frame(P_MID_LABEL)
            profile = profile.with_columns(
                pl.Series(buoyancy_frequency), pl.Series(p_mid)
            )
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_profile_empty(CTD.add_surface_salinity_temp_meltwater.__name__)

    def _process_master_sheet(
        self, profile: pl.DataFrame = None, for_id=False
    ) -> Tuple[Any, Any, str, float | None]:
        """
        Extracts the date and time components from the filename and compares them with the data
        in the master sheet. Calculates the absolute differences between the dates and times to
        find the closest match. Returns the estimated latitude, longitude, and updated filename
        based on the closest match.

        Parameters
        ----------
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
                raise CTDError(message=ERROR_NO_MASTER_SHEET, filename=self._filename)
        if TIMESTAMP_LABEL not in profile.collect_schema().names():
            raise CTDError(message=ERROR_NO_TIMESTAMP_IN_FILE, filename=self._filename)
        if "datetime" not in self._cached_master_sheet.collect_schema().names():
            raise CTDError(
                message=ERROR_NO_TIMESTAMP_IN_MASTER_SHEET,
                filename=self._filename,
            )
        timestamp_highest = profile.select(
            pl.last(TIMESTAMP_LABEL)
            .dt.convert_time_zone(TIME_ZONE)
            .cast(pl.Datetime(time_unit=TIME_UNIT, time_zone=TIME_ZONE))
        ).item()
        closest_row_overall = self._cached_master_sheet.select(
            pl.all().sort_by(
                (pl.col(MASTER_SHEET_DATETIME_LABEL) - timestamp_highest).abs()
            )
        )
        latitude = closest_row_overall.select(pl.col(LATITUDE_LABEL).first()).item()
        longitude = closest_row_overall.select(pl.col(LONGITUDE_LABEL).first()).item()
        distance = (
            closest_row_overall.select(
                pl.col(MASTER_SHEET_DATETIME_LABEL).first()
            ).item()
            - timestamp_highest
        )
        unique_id = closest_row_overall.select(pl.col("UNIQUE ID CODE ").first()).item()
        secchi_depth = None
        if for_id:
            secchi_depth = closest_row_overall.select(
                pl.col(MASTER_SHEET_SECCHI_DEPTH_LABEL).cast(pl.Float32, strict=False).first()
            ).item()
        CTDLogger(
            message=f"Secchi Depth: {secchi_depth}",
            filename=self._filename,
            level="debug",
        )
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
            CTDWarning(filename=self._filename, message=message)
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

        class GRUModel(nn.Module):
            def __init__(self, input_shape):
                super(GRUModel, self).__init__()
                self.gru = nn.GRU(input_shape[1], 16, batch_first=True)
                self.output_layer = nn.Linear(16, input_shape[1])

            def forward(self, x):
                gru_out, _ = self.gru(x)
                output = self.output_layer(gru_out)
                return output

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
            torch.Tensor
                Loss value as a tensor.
            """
            # Assuming salinity is at index 0
            salinity_true = y_true[:, :, 0]
            salinity_pred = y_pred[:, :, 0]

            # Calculate differences between consecutive values
            delta_sal_pred = salinity_pred[:, 1:] - salinity_pred[:, :-1]

            # Penalize predictions where salinity decreases while pressure increases
            penalties = torch.where(
                delta_sal_pred < 0,
                -torch.min(
                    delta_sal_pred, torch.tensor(0.0, device=delta_sal_pred.device)
                ),
                torch.tensor(0.0, device=delta_sal_pred.device),
            )

            # Calculate mean absolute error
            mae = torch.mean(torch.abs(y_true - y_pred))

            # Add penalties
            total_loss = mae + 12.0 * torch.mean(
                penalties
            )  # Adjust weighting of penalty as needed
            return total_loss

        def build_gru(input_shape):
            model = GRUModel(input_shape)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            return model, optimizer


        def run_gru(data: pl.DataFrame, show_plots=self._plot):
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
            filtered_data = data.filter(pl.col(DEPTH_LABEL) > 1)
            filtered_data = filtered_data.with_columns(
                (pl.col(PRESSURE_LABEL) // 0.5 * 0.5).alias("pressure_bin")
            )
            # Define the desired columns and their aggregation functions
            column_agg_dict = {
                TEMPERATURE_LABEL: pl.mean(TEMPERATURE_LABEL),
                CHLOROPHYLL_LABEL: pl.mean(CHLOROPHYLL_LABEL),
                SEA_PRESSURE_LABEL: pl.mean(SEA_PRESSURE_LABEL),
                DEPTH_LABEL: pl.mean(DEPTH_LABEL),
                SALINITY_LABEL: pl.median(SALINITY_LABEL),
                SPEED_OF_SOUND_LABEL: pl.mean(SPEED_OF_SOUND_LABEL),
                SPECIFIC_CONDUCTIVITY_LABEL: pl.mean(SPECIFIC_CONDUCTIVITY_LABEL),
                CONDUCTIVITY_LABEL: pl.mean(CONDUCTIVITY_LABEL),
                DENSITY_LABEL: pl.mean(DENSITY_LABEL),
                POTENTIAL_DENSITY_LABEL: pl.mean(POTENTIAL_DENSITY_LABEL),
                SALINITY_ABS_LABEL: pl.mean(SALINITY_ABS_LABEL),
                TIMESTAMP_LABEL: pl.first(TIMESTAMP_LABEL),
                LONGITUDE_LABEL: pl.first(LONGITUDE_LABEL),
                LATITUDE_LABEL: pl.first(LATITUDE_LABEL),
                UNIQUE_ID_LABEL: pl.first(UNIQUE_ID_LABEL),
                FILENAME_LABEL: pl.first(FILENAME_LABEL),
                PROFILE_ID_LABEL: pl.first(PROFILE_ID_LABEL),
                SECCHI_DEPTH_LABEL: pl.first(SECCHI_DEPTH_LABEL),
            }
            available_columns = {
                col: agg_func
                for col, agg_func in column_agg_dict.items()
                if col in data.columns
            }
            data_binned = filtered_data.group_by(
                "pressure_bin", maintain_order=True
            ).agg(list(available_columns.values()))
            data_binned = data_binned.rename({"pressure_bin": PRESSURE_LABEL})
            scaler = MinMaxScaler(feature_range=(-1, 1))
            if data_binned.limit(4).height < 2:
                raise CTDError(
                    message=ERROR_GRU_INSUFFICIENT_DATA,
                    filename=self._filename,
                )
            salinity = np.array(data_binned.select(pl.col(SALINITY_LABEL)).to_numpy())
            scaler = MinMaxScaler()
            scaled_sequence = scaler.fit_transform(salinity)
            scaled_seq = np.expand_dims(scaled_sequence, axis=0)
            min_pres = data_binned.select(pl.min(DEPTH_LABEL)).item()
            max_pres = data_binned.select(pl.max(DEPTH_LABEL)).item()
            pres_range = max_pres - min_pres
            epochs = int(pres_range * 12)
            input_shape = scaled_seq.shape[1:]
            model, optimizer = build_gru(input_shape)
            criterion = loss_function
            tensor_data = torch.tensor(scaled_seq, dtype=torch.float32)
            dataset = TensorDataset(tensor_data, tensor_data)
            data_loader = DataLoader(dataset, batch_size=4, shuffle=False)
            for epoch in range(epochs):
                model.train()
                for x_batch, y_batch in data_loader:
                    optimizer.zero_grad()  # Zero the gradients
                    y_pred = model(x_batch)  # Forward pass
                    loss = loss_function(y_batch, y_pred)  # Compute the loss
                    loss.backward()  # Backward pass
                    optimizer.step()  # Update the weights
            model.eval()
            with torch.no_grad():
                X_pred = model(tensor_data).numpy()
            predicted_seq = np.array(scaler.inverse_transform(X_pred[0])).flatten()
            if show_plots:
                xlim, ylim = plot_original_data(
                    data.select(SALINITY_LABEL).to_numpy(),
                    data.select(DEPTH_LABEL).to_numpy(),
                    self._filename + str(profile_id),
                    plot_path=os.path.join(self._cwd, "plots", f"{self._filename}_original.png")
                )
                plot_predicted_data(
                    salinity=predicted_seq,
                    depths=data_binned.select(DEPTH_LABEL).to_numpy(),
                    filename=self._filename + str(profile_id),
                    xlim=xlim,
                    ylim=ylim,
                    plot_path=os.path.join(self._cwd, "plots", f"{self._filename}_predicted.png")
                )
            data_binned = data_binned.with_columns(
                pl.Series(predicted_seq, dtype=pl.Float64).alias(SALINITY_LABEL)
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
                    TIMESTAMP_LABEL: "timestamp",
                    TEMPERATURE_LABEL: "Temperature_(°C)",
                    PRESSURE_LABEL: "Pressure_(dbar)",
                    CHLOROPHYLL_LABEL: "Chlorophyll_a_(µg/l)",
                    SEA_PRESSURE_LABEL: "Sea Pressure_(dbar)",
                    DEPTH_LABEL: "Depth_(m)",
                    SALINITY_LABEL: "Salinity_(PSU)",
                    SPEED_OF_SOUND_LABEL: "Speed of Sound_(m/s)",
                    SPECIFIC_CONDUCTIVITY_LABEL: "Specific Conductivity_(µS/cm)",
                    CONDUCTIVITY_LABEL: "Conductivity_(mS/cm)",
                    DENSITY_LABEL: "Density_(kg/m^3)",
                    POTENTIAL_DENSITY_LABEL: "Potential_Density_(kg/m^3)",
                    SALINITY_ABS_LABEL: "Absolute Salinity_(g/kg)",
                    SURFACE_DENSITY_LABEL: "Mean_Surface_Density_(kg/m^3)",
                    SURFACE_SALINITY_LABEL: "Surface_Salinity_(PSU)",
                    SURFACE_TEMPERATURE_LABEL: "Surface_Temperature_(°C)",
                    MELTWATER_FRACTION_LABEL: "Meltwater_Fraction_(%)",
                    LONGITUDE_LABEL: "longitude",
                    LATITUDE_LABEL: "latitude",
                    FILENAME_LABEL: "filename",
                    PROFILE_ID_LABEL: "Profile_ID",
                    UNIQUE_ID_LABEL: "Unique_ID",
                    BV_LABEL: "Brunt_Vaisala_Frequency_Squared",
                    P_MID_LABEL: "Mid_Pressure_Used_For_BV_Calc",
                    SECCHI_DEPTH_LABEL: "Secchi_Depth_(m)",
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
                MASTER_SHEET_TIME_LOCAL_LABEL: pl.String,
                MASTER_SHEET_DATE_LOCAL_LABEL: pl.String,
                MASTER_SHEET_TIME_UTC_LABEL: pl.String,
                MASTER_SHEET_DATE_UTC_LABEL: pl.String,
                MASTER_SHEET_SECCHI_DEPTH_LABEL: pl.String,
            }
            df = pl.read_excel(
                master_sheet_path,
                infer_schema_length=None,
                schema_overrides=_masterSheetLabels_to_dtypeInternal,
            )
            df = df.drop_nulls(MASTER_SHEET_TIME_LOCAL_LABEL)
            df = df.filter(
                ~pl.col(MASTER_SHEET_TIME_LOCAL_LABEL).eq("-999"),
                ~pl.col(MASTER_SHEET_TIME_LOCAL_LABEL).eq("NA"),
                ~pl.col(MASTER_SHEET_DATE_LOCAL_LABEL).eq("NA"),
            )

            df = df.with_columns(
                pl.col(MASTER_SHEET_DATE_LOCAL_LABEL).str.strptime(
                    format="%Y-%m-%d %H:%M:%S", dtype=pl.Date, strict=False
                ),
                pl.col(MASTER_SHEET_TIME_LOCAL_LABEL).str.strptime(
                    format="%Y-%m-%d %H:%M:%S", dtype=pl.Time, strict=False
                ),
                pl.col(MASTER_SHEET_SECCHI_DEPTH_LABEL).cast(pl.Float64, strict=False),
            )
            df = df.drop_nulls(MASTER_SHEET_DATE_LOCAL_LABEL)
            df = df.drop_nulls(MASTER_SHEET_TIME_LOCAL_LABEL)
            df = df.with_columns(
                (
                    pl.col(MASTER_SHEET_DATE_LOCAL_LABEL).dt.combine(
                        pl.col(MASTER_SHEET_TIME_LOCAL_LABEL).cast(pl.Time)
                    )
                )
                .alias(MASTER_SHEET_DATETIME_LABEL)
                .cast(pl.Datetime)
                .dt.replace_time_zone(TIME_ZONE),
                pl.when(pl.col(MASTER_SHEET_SECCHI_DEPTH_LABEL) == -999)
                .then(None)
                .otherwise(pl.col(MASTER_SHEET_SECCHI_DEPTH_LABEL))
                .alias(MASTER_SHEET_SECCHI_DEPTH_LABEL),
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


def CTDWarning(message, filename=None):
    """
    CTD warning function.

    Parameters
    ----------
    filename : str, default None
        Input dataset which caused the error.
    message : str
        Explanation of the error.
    """
    warnings.warn(message=f"{filename} - {message}")


def CTDLogger(message, filename=None, level="info"):
    """
    Wrapper for logger.

    Parameters
    ----------
    filename : str
        Input dataset which caused the warning.
    message : str
        Explanation of the warning.
    """
    if level == "info":
        logger.info(filename + " - " + message)
    if level == "debug":
        logger.debug(filename + " - " + message)
    if level == "warning":
        logger.warning(filename + " - " + message)
