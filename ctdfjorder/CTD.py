from ctdfjorder.exceptions.exceptions import CTDError, Critical
from ctdfjorder.exceptions.exceptions import raise_warning_calculatuion
from ctdfjorder.metadata.master_sheet import MasterSheet
from ctdfjorder.constants.constants import *
from ctdfjorder.loadctd.rsk import load_file_rsk
from ctdfjorder.loadctd.castaway import load_file_castaway
from ctdfjorder.utils import utils
from ctdfjorder.ai import ai
import pandas as pd
from sqlite3 import OperationalError
from os import path
import numpy as np
import polars as pl
import gsw
from typing import Any
import logging
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger("ctdfjorder")
logger.propagate = 0


class CTD:
    """
    Object representing a single profile.

    Parameters
    ----------

    ctd_file_path : str
        The file path to the RSK or Castaway file.
    cached_master_sheet : masterSheet, default None
        CTDFjorder's internal representation of a master sheet.
    master_sheet_path : str, default None
        Path to a master sheet.
    plot : bool, default False
        If true saves plots to 'plots' folder in working directory.

    Examples
    --------
    Castaway CTD profile with valid data

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

    >>> ctd_data = CTD('CC1627007_20191220_195931.csv') # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    ctdfjorder.CTDError: CC1627007_20191220_195931.csv - No samples in file

    Raises
    ------
    CTDError
        For ctdfjorder related errors

    """

    # Initialization Constants
    _data: pl.DataFrame = pl.DataFrame()
    _cached_master_sheet: MasterSheet = None
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
        cached_master_sheet: MasterSheet = None,
        master_sheet_path=None,
        plot=False,
    ):
        # Define instance vars, load master sheet if path given and master sheet is not cached
        self._filename = path.basename(ctd_file_path)
        if type(self._cached_master_sheet) is type(None) and master_sheet_path:
            try:
                self._cached_master_sheet = MasterSheet(master_sheet_path)
            except Critical:
                pass
        else:
            self._cached_master_sheet = cached_master_sheet
        self.master_sheet_path = master_sheet_path
        self._cwd = utils.get_cwd()
        self._plot = plot

        # Processing RSK Files
        if RSK_FILE_MARKER in ctd_file_path:
            try:
                self._data = load_file_rsk(ctd_file_path)
            except OperationalError:
                raise CTDError(filename=self._filename, message=ERROR_RSK_CORRUPT)

        # Processing Castaway Files
        elif CASTAWAY_FILE_MARKER in ctd_file_path:
            self._data = load_file_castaway(ctd_file_path)
        else:
            raise CTDError(filename=self._filename, message=ERROR_CTD_FILENAME_ENDING)

        # Checking if data is empty
        if self._data.is_empty():
            raise CTDError(filename=self._filename, message=ERROR_NO_SAMPLES)

        # Adding year and month columns
        self._data = self._data.with_columns(
            pl.col(TIMESTAMP_LABEL)
            .dt.convert_time_zone(TIME_ZONE)
            .cast(pl.Datetime(time_unit=TIME_UNIT, time_zone=TIME_ZONE))
        )
        self._data = self._data.with_columns(
            pl.col(TIMESTAMP_LABEL).dt.year().alias(YEAR_LABEL),
            pl.col(TIMESTAMP_LABEL).dt.month().alias(MONTH_LABEL),
        )

        # If master sheet or cached master sheet is present, find the matching information and correct missing location
        if self._cached_master_sheet:
            self._data = self._data.with_columns(
                pl.lit(None, dtype=pl.String).alias(UNIQUE_ID_LABEL),
                pl.lit(None, dtype=pl.Float32).alias(SECCHI_DEPTH_LABEL),
                pl.lit(None, dtype=pl.String).alias(SITE_NAME_LABEL),
                pl.lit(None, dtype=pl.String).alias(SITE_ID_LABEL),
            )
            for profile_id in (
                self._data.select(PROFILE_ID_LABEL)
                .unique(keep="first")
                .to_series()
                .to_list()
            ):
                profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)

                # Find master sheet match
                master_sheet_match = self._cached_master_sheet.find_match(profile)
                # Add secchi depth and Unique ID
                profile = profile.with_columns(
                    pl.lit(master_sheet_match.unique_id, dtype=pl.String).alias(
                        UNIQUE_ID_LABEL
                    ),
                    pl.lit(master_sheet_match.secchi_depth)
                    .cast(pl.Float32)
                    .alias(SECCHI_DEPTH_LABEL),
                    pl.lit(master_sheet_match.site_name)
                    .cast(pl.String)
                    .alias(SITE_NAME_LABEL),
                    pl.lit(master_sheet_match.site_id)
                    .cast(pl.String)
                    .alias(SITE_ID_LABEL),
                )
                # Add location data if not present
                if (
                    LATITUDE_LABEL not in profile.columns
                    or (
                        profile.select(pl.col(LATITUDE_LABEL).has_nulls()).item()
                        or profile.select(pl.col(LATITUDE_LABEL).is_nan().any()).item()
                    )
                    or profile.select(pl.col(LATITUDE_LABEL)).is_empty()
                    or profile.select(pl.col(LATITUDE_LABEL)).limit(1).item() is None
                ):
                    self._data = self._data.with_columns(
                        pl.lit(None, dtype=pl.Float64).alias(LATITUDE_LABEL),
                        pl.lit(None, dtype=pl.Float64).alias(LONGITUDE_LABEL),
                    )
                    latitude = master_sheet_match.latitude
                    longitude = master_sheet_match.longitude
                    if (
                        latitude is None
                        or longitude is None
                        or np.isnan(latitude)
                        or np.isnan(longitude)
                    ):
                        latitude = None
                        longitude = None
                    new_fname = self._filename + "cm"
                    profile = profile.with_columns(
                        pl.lit(latitude).cast(pl.Float64).alias(LATITUDE_LABEL),
                        pl.lit(longitude).cast(pl.Float64).alias(LONGITUDE_LABEL),
                        pl.lit(new_fname, pl.String).alias("filename"),
                    )
                self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
                self._data = self._data.vstack(profile)

        # Try casting location to float
        try:
            self._data = self._data.with_columns(
                pl.col(LATITUDE_LABEL).cast(pl.Float64),
                pl.col(LONGITUDE_LABEL).cast(pl.Float64),
            )
        except pl.exceptions.InvalidOperationError:
            raise CTDError(message=ERROR_LOCATION_DATA_INVALID, filename=self._filename)

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

    def _is_empty(self, func: str) -> bool:
        if self._data.is_empty():
            raise CTDError(
                filename=self._filename,
                message=f"No valid samples in file after running {func}",
            )
        return True

    def remove_upcasts(self) -> None:
        r"""
        Removes upcasts by dropping rows where pressure decreases from one sampling event to the next.

        Notes
        -----
        An upcast in CTD (Conductivity, Temperature, Depth) profiles occurs when the sensor package is
        moved upward through the water column, causing a decrease in pressure readings. This method
        identifies and removes such events by ensuring pressure monotonically increases within each profile.

        The procedure is as follows:

        1. For each unique profile identified by `profile_id`, the method extracts the profile's data.
        2. It then computes the difference in pressure between consecutive samples within the profile.
        3. Rows where the pressure difference is not positive are removed, indicating a non-increasing
           pressure (i.e., an upcast or no movement).
        4. The cleaned profile is then reintegrated into the main dataset, replacing the original data.

        Let :math:`p_i` be the pressure at the :math:`i`-th sampling event. The condition for retaining a
        data point is given by:

        .. math::

            \Delta p_i = p_{i} - p_{i-1} > 0 \quad \text{for} \quad i = 1, 2, \ldots, N

        where :math:`N` is the total number of sampling events in the profile. Rows not satisfying this
        condition are considered upcasts and are removed.

        Examples
        --------
        >>> ctd_data = CTD('example.csv')
        >>> ctd_data.remove_upcasts()
        >>> # This will clean the dataset by removing upcasts, ensuring all profiles have monotonically
        >>> # increasing pressure readings.

        See Also
        --------
        Other related methods for data cleaning or preprocessing within the CTD class.

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
        self._is_empty(CTD.remove_upcasts.__name__)

    def filter_columns_by_range(
        self,
        filters: zip = None,
        columns: list[str] = None,
        upper_bounds: list[float | int] = None,
        lower_bounds: list[float | int] = None,
    ):
        """
        Filters columns of the dataset based on specified upper and lower bounds.

        This method allows filtering of a dataset by applying specified upper and lower bounds
        on the columns. It processes the data profile by profile and updates the dataset accordingly.

        Parameters
        ----------
        filters : zip, optional
            An iterable of tuples, where each tuple contains a column name, an upper bound, and a lower bound.
            If provided, this takes precedence over the individual `columns`, `upper_bounds`, and `lower_bounds` parameters.
        columns : list of str, optional
            A list of column names to be filtered. Must be provided along with `upper_bounds` and `lower_bounds`.
        upper_bounds : list of float or int, optional
            A list of upper bound values corresponding to each column in `columns`.
        lower_bounds : list of float or int, optional
            A list of lower bound values corresponding to each column in `columns`.

        Notes
        -----
        The method performs the following steps for each unique profile identified by `profile_id`:

        1. Extracts the data for the profile.
        2. If `columns` is provided, iterates over each column and applies the corresponding upper and lower bounds
           to filter the data.
        3. If `filters` is provided, iterates over each filter tuple and applies the specified bounds to the relevant column.
        4. Updates the dataset by removing the original profile data and reintegrating the filtered profile data.

        Examples
        --------
        >>> ctd_data = CTD('example.csv')
        >>> filters = zip(['temperature', 'salinity'], [20.0, 35.0], [10.0, 30.0])
        >>> ctd_data.filter_columns_by_range(filters=filters)
        >>> # This will filter the `temperature` column to be between 10.0 and 20.0, and `salinity` to be between 30.0 and 35.0.

        >>> columns = ['temperature', 'salinity']
        >>> upper_bounds = [20.0, 35.0]
        >>> lower_bounds = [10.0, 30.0]
        >>> ctd_data.filter_columns_by_range(columns=columns, upper_bounds=upper_bounds, lower_bounds=lower_bounds)
        >>> # This will filter the `temperature` column to be between 10.0 and 20.0, and `salinity` to be between 30.0 and 35.0.

        See Also
        --------
        remove_non_positive_samples : Method to remove rows with non-positive values for specific parameters.

        """
        for profile_id in (
            self._data.select(PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
            if type(columns) is not type(None):
                for x, column in enumerate(columns):
                    upper_bound = upper_bounds[x]
                    lower_bound = lower_bounds[x]
                    if type(upper_bound) is not type(None) and type(
                        lower_bound
                    ) is not type(None):
                        profile = profile.filter(
                            pl.col(column) <= upper_bound, pl.col(column) >= lower_bound
                        )
                    elif type(upper_bound) is not type(None):
                        profile = profile.filter(pl.col(column) <= upper_bound)
                    elif type(lower_bound) is not type(None):
                        profile = profile.filter(pl.col(column) >= lower_bound)

            elif type(filters) is not type(None):
                for filter in filters:
                    if filter[0] not in profile.columns:
                        continue
                    column = filter[0]
                    upper_bound = filter[1]
                    lower_bound = filter[2]
                    profile = profile.filter(
                        pl.col(column) <= upper_bound, pl.col(column) >= lower_bound
                    )
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_empty(CTD.filter_columns_by_range.__name__)

    def remove_non_positive_samples(self) -> None:
        r"""
        Removes rows with non-positive values for depth, pressure, practical salinity, absolute salinity, or density.

        Notes
        -----
        This method cleans the CTD (Conductivity, Temperature, Depth) dataset by removing any samples
        that have non-positive values for key parameters. Non-positive values in these parameters
        are generally invalid and indicate erroneous measurements.

        The procedure is as follows:

        1. For each unique profile identified by `profile_id`, the method extracts the profile's data.
        2. It then iterates over a predefined set of key parameters: depth, pressure, practical salinity,
           absolute salinity, and density.
        3. For each parameter present in the profile, it filters out rows where the parameter's value
           is non-positive, null, or NaN.
        4. The cleaned profile is then reintegrated into the main dataset, replacing the original data.

        Let :math:`( x_i )` represent the value of a parameter (depth, pressure, practical salinity, absolute salinity,
        or density) at the :math:`( i )`-th sampling event. The condition for retaining a data point is given by:

        .. math::

            x_i > 0 \quad \text{and} \quad x_i \neq \text{NaN} \quad \text{and} \quad x_i \neq \text{null}

        Rows not satisfying this condition for any of the parameters are removed.

        Examples
        --------
        >>> ctd_data = CTD('example.csv')
        >>> ctd_data.remove_non_positive_samples()
        >>> # This will clean the dataset by removing samples with non-positive, null, or NaN values
        >>> # for the specified key parameters.

        See Also
        --------
        Other related methods for data cleaning or preprocessing within the CTD class.

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
        self._is_empty(CTD.remove_non_positive_samples.__name__)

    def remove_invalid_salinity_values(self) -> None:
        r"""
        Removes rows with practical salinity values less than or equal to 10.

        Notes
        -----
        This method cleans the CTD (Conductivity, Temperature, Depth) dataset by removing any samples
        that have practical salinity values that are less than or equal to 10. Salinity values in this
        range are generally considered invalid for typical oceanographic studies and may indicate
        erroneous measurements or freshwater influence.

        The procedure is as follows:

        1. For each unique profile identified by `profile_id`, the method extracts the profile's data.
        2. It then filters out rows where the practical salinity value is less than or equal to 10.
        3. The cleaned profile is then reintegrated into the main dataset, replacing the original data.

        Let :math:`( S_i )` represent the practical salinity at the :math:`( i )`-th sampling event. The condition for
        retaining a data point is given by:

        .. math::

            S_i > 10

        Rows not satisfying this condition are considered invalid and are removed.

        Examples
        --------
        >>> ctd_data = CTD('example.csv')
        >>> ctd_data.remove_invalid_salinity_values()
        >>> # This will clean the dataset by removing samples with practical salinity values less than
        >>> # or equal to 10.

        See Also
        --------
        Other related methods for data cleaning or preprocessing within the CTD class.

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
        self._is_empty(CTD.remove_invalid_salinity_values.__name__)

    def clean(self, method) -> None:
        r"""
        Applies data cleaning methods to the specified feature using the selected method.
        Supports cleaning practical salinity using the 'clean_salinity_ai' method.

        Parameters
        ----------
        method : str, default 'clean_salinity_ai'
            The cleaning method to apply. Currently, only 'clean_salinity_ai' is supported,
            which uses a GRU-based machine learning model to clean the salinity values.

        Raises
        ------
        CTDError
            When the cleaning method is invalid.

        Notes
        -----
        The 'clean_salinity_ai' method uses a Gated Recurrent Unit (GRU) model to correct
        salinity measurements. This model is designed to smooth out unrealistic fluctuations
        in salinity with respect to pressure.

        The procedure for 'clean_salinity_ai' is as follows:

        1. For each unique profile identified by `profile_id`, extract the profile's data.
        2. Bin the data every 0.5 dbar of pressure.
        3. Use a GRU model with a loss function that penalizes non-monotonic salinity increases with decreasing pressure.
        4. Train the model on the binned data to predict clean salinity values.
        5. Replace the original salinity values in the profile with the predicted clean values.
        6. Reintegration of the cleaned profile into the main dataset.

        The loss function :math:`( L )` used in training is defined as:

        .. math::

            L = \text{MAE}(y_{\text{true}}, y_{\text{pred}}) + \lambda \cdot \text{mean}(P)

        where :math:`( P )` are penalties for predicted salinity increases with decreasing pressure,
        and :math:`( \lambda )` is a weighting factor.

        Examples
        --------
        >>> ctd_data = CTD('example.csv')
        >>> ctd_data.clean('clean_salinity_ai')
        >>> # This will clean the salinity data using the AI-based method, correcting any unrealistic
        >>> # values and ensuring smoother transitions with respect to pressure.

        See Also
        --------
        _AI.clean_salinity_ai : Method used to clean salinity with 'clean_salinity_ai' option.
        """
        for profile_id in (
            self._data.select(PROFILE_ID_LABEL)
            .unique(keep="first")
            .to_series()
            .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
            if method == "clean_salinity_ai":
                profile = ai.AI.clean_salinity_ai(profile, profile_id)
            else:
                raise CTDError(
                    message="Method invalid for clean.", filename=self._filename
                )
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
            self._data = pl.concat([self._data, profile], how=CONCAT_HOW, rechunk=True)
        self._is_empty(CTD.clean.__name__)

    def add_absolute_salinity(self) -> None:
        r"""
        Calculates and adds absolute salinity to the CTD data using the TEOS-10 salinity conversion formula.

        Notes
        -----
        This method computes the absolute salinity from practical salinity for each profile in the dataset
        using the TEOS-10 standard. Absolute salinity provides a more accurate representation of salinity
        by accounting for the variations in seawater composition.

        The procedure is as follows:

        1. Initialize a new column for absolute salinity in the dataset.
        2. For each unique profile identified by `profile_id`, extract the profile's data.
        3. Use the `gsw.conversions.SA_from_SP` function to compute absolute salinity from practical salinity.
        4. Update the profile with the computed absolute salinity values.
        5. Reintegration of the updated profile into the main dataset.

        The TEOS-10 formula for converting practical salinity :math:`( S_P )` to absolute salinity :math:`( S_A )` is used:

        .. math::

            S_A = f(S_P, p, \phi, \lambda)

        where :math:`( p )` is the sea pressure, :math:`( \phi )` is the latitude, and :math:`( \lambda )` is the longitude.

        The `gsw.conversions.SA_from_SP` function from the Gibbs SeaWater (GSW) Oceanographic Toolbox is utilized
        for this conversion. More information about this function can be found at the
        `TEOS-10 website <https://www.teos-10.org/pubs/gsw/html/gsw_SA_from_SP.html>`__.

        Examples
        --------
        >>> ctd_data = CTD('example.csv')
        >>> ctd_data.add_absolute_salinity()
        >>> # This will add a new column with absolute salinity values to the dataset, calculated using the
        >>> # TEOS-10 formula.

        See Also
        --------
        gsw.conversions.SA_from_SP : Function used for the conversion from practical salinity to absolute salinity.

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
        self._is_empty(CTD.add_absolute_salinity.__name__)

    def add_density(self):
        r"""
        Calculates and adds density to CTD data using the TEOS-10 formula.
        If absolute salinity is not present, it is calculated first.


        Notes
        -----
        This method computes the density of seawater from absolute salinity, in-situ temperature,
        and sea pressure using the TEOS-10 standard. The density is a critical parameter for
        understanding the physical properties of seawater and its buoyancy characteristics.

        The procedure is as follows:

        1. Check if absolute salinity is already present in the dataset. If not, calculate it using `add_absolute_salinity()`.
        2. Initialize a new column for density in the dataset.
        3. For each unique profile identified by `profile_id`, extract the profile's data.
        4. Use the `gsw.density.rho_t_exact` function to compute density from absolute salinity, temperature, and pressure.
        5. Update the profile with the computed density values.
        6. Reintegration of the updated profile into the main dataset.

        The TEOS-10 formula for calculating density :math:`( \rho )` is used:

        .. math::

            \rho = f(S_A, T, p)

        where :math:`( S_A )` is the absolute salinity, :math:`( T )` is the in-situ temperature, and :math:`( p )` is the sea pressure.

        The `gsw.density.rho_t_exact` function from the Gibbs SeaWater (GSW) Oceanographic Toolbox is utilized
        for this calculation. More information about this function can be found at the
        `TEOS-10 website <https://www.teos-10.org/pubs/gsw/html/gsw_rho_t_exact.html>`__.

        Examples
        --------
        >>> ctd_data = CTD('example.csv')
        >>> ctd_data.add_density()
        >>> # This will add a new column with density values to the dataset, calculated using the
        >>> # TEOS-10 formula.

        See Also
        --------
        gsw.density.rho_t_exact : Function used for the calculation of density.
        add_absolute_salinity : Method to add absolute salinity if it is not already present in the dataset.

        """
        if SALINITY_ABS_LABEL not in self._data.columns:
            self.add_absolute_salinity()
        self._data = self._data.with_columns(
            pl.lit(None).cast(pl.Float64).alias(DENSITY_LABEL)
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
                np.array(gsw.density.rho_t_exact(sa, t, p)).flatten(), dtype=pl.Float64, strict=True).to_frame(DENSITY_LABEL)
            profile = profile.with_columns(density)
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_empty(CTD.add_density.__name__)

    def add_potential_density(self):
        r"""
        Calculates and adds potential density to the CTD data using the TEOS-10 formula.
        If absolute salinity is not present, it is calculated first.

        Notes
        -----
        This method computes the potential density of seawater from absolute salinity and in-situ temperature
        using the TEOS-10 standard. Potential density is the density a parcel of seawater would have if
        it were adiabatically brought to the sea surface, which helps in understanding the stability and
        stratification of the water column.

        The procedure is as follows:

        1. Check if absolute salinity is already present in the dataset. If not, calculate it using `add_absolute_salinity()`.
        2. Initialize a new column for potential density in the dataset.
        3. For each unique profile identified by `profile_id`, extract the profile's data.
        4. Use the `gsw.sigma0` function to compute potential density from absolute salinity and temperature.
        5. Update the profile with the computed potential density values.
        6. Reintegration of the updated profile into the main dataset.

        The TEOS-10 formula for calculating potential density :math:`( \sigma_0 )` is used:

        .. math::

            \sigma_0 = f(S_A, T)

        where :math:`( S_A )` is the absolute salinity and :math:`( T )` is the in-situ temperature.

        The `gsw.sigma0` function from the Gibbs SeaWater (GSW) Oceanographic Toolbox is utilized
        for this calculation. More information about this function can be found at the
        `TEOS-10 website <https://www.teos-10.org/pubs/gsw/html/gsw_sigma0.html>`__.

        Examples
        --------
        >>> ctd_data = CTD('example.csv')
        >>> ctd_data.add_potential_density()
        >>> # This will add a new column with potential density values to the dataset, calculated using the
        >>> # TEOS-10 formula.

        See Also
        --------
        gsw.sigma0 : Function used for the calculation of potential density.
        add_absolute_salinity : Method to add absolute salinity if it is not already present in the dataset.

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
        self._is_empty(CTD.add_potential_density.__name__)

    def add_surface_salinity_temp_meltwater(self, start=10.1325, end=12.1325):
        r"""
        Calculates the surface salinity, surface temperature, and meltwater fraction of a CTD profile.
        Adds these values to the CTD data.

        Parameters
        ----------
        start : float, default 10.1325
            Upper bound of surface pressure.
        end : float, default 12.1325
            Lower bound of surface pressure.

        Notes
        -----
        This method adds three new columns to the dataset: surface salinity, surface temperature, and
        meltwater fraction. The values are calculated as follows:

        - Surface temperature is the mean temperature from pressure `start` to `end`.
        - Surface salinity is the salinity value at the lowest pressure within the range from `start` to `end`.
        - Meltwater fraction is calculated using the formula from `Pan et. al 2019 <https://doi.org/10.1371/journal.pone.0211107>`__:

        .. math::

            \text{Meltwater fraction} = (-0.021406 \cdot S_0 + 0.740392) \cdot 100

        where :math:`( S_0 )` is the surface salinity.

        The procedure is as follows:

        1. Initialize new columns for surface salinity, surface temperature, and meltwater fraction in the dataset.
        2. For each unique profile identified by `profile_id`, extract the profile's data.
        3. Filter the data to include only the samples within the specified pressure range (`start` to `end`).
        4. Calculate the mean surface temperature, surface salinity, and meltwater fraction based on the filtered data.
        5. Update the profile with the computed values.
        6. Reintegration of the updated profile into the main dataset.

        Raises
        ------
        CTDError
            When there are no measurements within the specified pressure range for a profile.

        Examples
        --------
        >>> ctd_data = CTD('example.csv')
        >>> ctd_data.add_surface_salinity_temp_meltwater(start=10.1325, end=12.1325)
        >>> # This will add new columns with surface salinity, surface temperature, and meltwater fraction
        >>> # values to the dataset, calculated using the specified pressure range.

        See Also
        --------
        Other related methods for data analysis or preprocessing within the CTD class.

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
                raise_warning_calculatuion(
                    filename=self._filename,
                    message=WARNING_CTD_SURFACE_MEASUREMENT,
                )
                self._is_empty(CTD.add_surface_salinity_temp_meltwater.__name__)
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
        self._is_empty(CTD.add_surface_salinity_temp_meltwater.__name__)

    def add_mean_surface_density(self, start=10.1325, end=12.1325):
        """
        Calculates the mean surface density from the density values and adds it as a new column
        to the CTD data table. Requires absolute salinity and absolute density to be calculated first.

        Parameters
        ----------
        start : float, default 10.1325
            Upper bound of surface pressure.
        end : float, default 12.1325
            Lower bound of surface pressure.

        Notes
        -----
        This method adds a new column for the mean surface density to the dataset. The values are calculated as follows:

        - Mean surface density is computed as the mean of density values within the specified pressure range (`start` to `end`).

        The procedure is as follows:

        1. For each unique profile identified by `profile_id`, extract the profile's data.
        2. Filter the data to include only the samples within the specified pressure range (`start` to `end`).
        3. Calculate the mean surface density based on the filtered data.
        4. Update the profile with the computed mean surface density value.
        5. Reintegration of the updated profile into the main dataset.

        Raises
        ------
        CTDError
            When there are no measurements within the specified pressure range for a profile.

        Examples
        --------
        >>> ctd_data = CTD('example.csv')
        >>> ctd_data.add_mean_surface_density(start=10.1325, end=12.1325)
        >>> # This will add a new column with mean surface density values to the dataset, calculated using the
        >>> # specified pressure range.

        See Also
        --------
        add_absolute_salinity : Method to add absolute salinity if it is not already present in the dataset.
        add_density : Method to add density if it is not already present in the dataset.

        """
        # Filtering data within the specified pressure range
        self._data = self._data.with_columns(pl.lit(None).cast(pl.Float64).alias(SURFACE_DENSITY_LABEL))
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
        self._is_empty(CTD.add_mean_surface_density.__name__)

    def add_mld(self, reference: int, method="potential_density_avg", delta=0.05):
        r"""
        Calculates and adds the mixed layer depth (MLD) using the density threshold method.

        Parameters
        ----------
        reference : int
            The reference depth for MLD calculation.
        method : str, default "potential_density_avg"
            The MLD calculation method. Options are "abs_density_avg" or "potential_density_avg".
        delta : float, default 0.05
            The change in density or potential density from the reference that would define the MLD.

        Notes
        -----
        The mixed layer depth (MLD) is calculated using the density threshold method, defined as the depth
        at which the density increases by a specified amount (delta) from the reference density. The reference
        density is calculated as the mean density up to the reference depth.

        The procedure is as follows:

        1. Initialize a new column for MLD in the dataset.
        2. For each unique profile identified by `profile_id`, extract the profile's data.
        3. Filter the data to include only the samples up to the reference depth.
        4. Calculate the reference density based on the chosen method:
           - "abs_density_avg": mean absolute density up to the reference depth.
           - "potential_density_avg": mean potential density up to the reference depth.
        5. Identify the MLD as the shallowest depth where the density exceeds the reference density by the specified delta.
        6. Update the profile with the calculated MLD value.
        7. Reintegration of the updated profile into the main dataset.

        The MLD equation is given by:

        .. math::

            \text{MLD} = \min(D + \Delta > D_r)

        where:

        * :math:`( D_r )` is the reference density, defined as the mean density up to the reference depth.
        * :math:`( D )` represents all densities in the profile.
        * :math:`( \Delta )` is the specified change in density (delta).

        Raises
        ------
        CTDError
            When the specified method is invalid.

        Examples
        --------
        >>> ctd_data = CTD('example.csv')
        >>> ctd_data.add_mld(reference=10, method="potential_density_avg", delta=0.05)
        >>> # This will add a new column with MLD values to the dataset, calculated using the specified method
        >>> # and parameters.

        See Also
        --------
        Other related methods for data analysis or preprocessing within the CTD class.

        """
        supported_methods = ["abs_density_avg", "potential_density_avg"]
        self._mld_col_labels.append(f"MLD {reference} (m)")
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
            profile = profile.with_columns(pl.lit(mld).alias(self._mld_col_labels[-1]))
            self._data = self._data.filter(pl.col(PROFILE_ID_LABEL) != profile_id)
            self._data = self._data.vstack(profile)
        self._is_empty(CTD.add_mld.__name__)

    def add_bf_squared(self):
        r"""
        Calculates buoyancy frequency squared and adds it to the CTD data.
        Requires potential density to be calculated first.

        This method computes the buoyancy frequency squared (also known as the Brunt-Väisälä frequency squared)
        for each profile in the dataset using the TEOS-10 standard. This parameter is essential for understanding
        the stability of the water column and its propensity to mix vertically.

        The procedure is as follows:

        1. Initialize new columns for buoyancy frequency squared and the mid-pressure values in the dataset.
        2. For each unique profile identified by `profile_id`, extract the profile's data.
        3. Use the `gsw.Nsquared` function to compute buoyancy frequency squared and mid-pressure values from
           absolute salinity, conservative temperature, pressure, and latitude.
        4. Update the profile with the computed buoyancy frequency squared and mid-pressure values.
        5. Reintegration of the updated profile into the main dataset.

        Notes
        -----
        The buoyancy frequency squared :math:`( N^2 )` is calculated using the formula:

        .. math::

            N_2 = g_2 \cdot \frac{\beta \cdot d(SA) - \alpha \cdot d(CT)}{\text{specvol_local} \cdot dP}

        Note. This routine uses rho from "gsw_specvol", which is the
          computationally efficient 75-term expression for specific volume in
          terms of SA, CT and p (Roquet et al., 2015).
        Note also that the pressure increment, dP, in the above formula is in
          Pa, so that it is 104 times the pressure increment dp in dbar.

        The `gsw.Nsquared` function from the Gibbs SeaWater (GSW) Oceanographic Toolbox is utilized
        for this calculation. More information about this function can be found at the
        `TEOS-10 website <https://www.teos-10.org/pubs/gsw/html/gsw_Nsquared.html>`__.

        Examples
        --------
        >>> ctd_data = CTD('example.csv')
        >>> ctd_data.add_bf_squared()
        >>> # This will add new columns with buoyancy frequency squared values and mid-pressure values to the dataset,
        >>> # calculated using the TEOS-10 formula.

        See Also
        --------
        gsw.Nsquared : Function used for the calculation of buoyancy frequency squared.

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
        self._is_empty(CTD.add_surface_salinity_temp_meltwater.__name__)

    def save_to_csv(self, output_file: str, null_value: str | None):
        """
        Renames the columns of the CTD data table based on a predefined mapping and saves the
        data to the specified CSV file.

        Parameters
        ----------
        output_file : str
            The output CSV file path.
        null_value : str or None
            The value to represent null cells in the saved file.

        Notes
        -----
        This method will rename the columns of the CTD dataset based on a predefined mapping. After renaming,
        the dataset is saved to the specified CSV file. If a file with the same name already exists at the
        specified path, it will be overwritten.

        The procedure is as follows:

        1. Rename the columns of the CTD data table using a predefined mapping.
        2. Save the modified data table to the specified CSV file path.

        The predefined column mapping ensures that the column names in the output CSV file adhere to a specific
        naming convention or format required for further analysis or sharing.

        Raises
        ------
        IOError
            If there is an error in writing to the specified file path.

        Examples
        --------
        >>> ctd_data = CTD('example.csv')
        >>> ctd_data.save_to_csv(output_file='path/to/output.csv')
        >>> # This will rename the columns of the CTD dataset and save it to 'path/to/output.csv'.
        >>> # Any existing file with the same name at that location will be overwritten.

        See Also
        --------
        utils.save_to_csv : Utility function used to save the data to a CSV file.

        """
        utils.save_to_csv(self._data, output_file=output_file, null_value=null_value)
