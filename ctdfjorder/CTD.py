from scipy.stats import stats
from ctdfjorder.exceptions.exceptions import (raise_warning_calculatuion,
                                              CTDCorruptError,
                                              InvalidCTDFilenameError,
                                              NoSamplesError)
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
from typing import Any, Union
import logging
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger("ctdfjorder")
logger.propagate = 0


class CTD:
    """
    Read your data and initialize a CTD object.

    Parameters
    ----------

    ctd_file_path : str
        The file path to the RSK or Castaway file.

    Raises
    ------
    RskCorruptError
        When an RBR '.rsk' file is unable to be opened.

    InvalidCTDFilenameError
        When the filename parameter does not have '.rsk' or '.csv'.

    Examples
    --------
    Castaway CTD profile with valid data
    .. code-block:: python

        ctd_data = CTD('CC1531002_20181225_114931.csv')
        output = ctd_data.get_df()
        print(output.head(3))

    """

    # Initialization Constants
    _data: pl.DataFrame = pl.DataFrame()
    _cached_master_sheet: MasterSheet = None
    _filename: str = None

    def __init__(
            self,
            ctd_file_path: str,
    ):
        self._filename = path.basename(ctd_file_path)
        # Processing RSK Files
        if RSK_FILE_MARKER in ctd_file_path:
            try:
                self._data = load_file_rsk(ctd_file_path)
            except OperationalError:
                raise CTDCorruptError(filename=self._filename)
            except pl.exceptions.ComputeError:
                raise CTDCorruptError(filename=self._filename)
        # Processing Castaway Files
        elif CASTAWAY_FILE_MARKER in ctd_file_path:
            try:
                self._data = load_file_castaway(ctd_file_path)
            except pl.exceptions.ComputeError:
                raise CTDCorruptError(filename=self._filename)
        else:
            raise InvalidCTDFilenameError(filename=self._filename)

    def expand_date(self, year: bool = True, month: bool = True, day: bool = True):
        """
        Expands the timestamp column into separate year, month, and day columns.

        Parameters
        ----------
        year : bool, default True
            If True, adds a year column.
        month : bool, default True
            If True, adds a month column.
        day : bool, default True
            If True, adds a day column.

        Raises
        ------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Notes
        -----
        This method modifies the CTD data in-place by adding new columns for year, month, and/or day,
        depending on the parameters provided. The original timestamp column is preserved.

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.expand_date(year=True, month=True, day=False)
            # This will add year and month columns to the dataset, but not a day column.

        """
        self.assert_data_not_empty(CTD.expand_date.__name__)
        self._data = self._data.with_columns(
            pl.col(TIMESTAMP.label)
            .dt.convert_time_zone(TIME_ZONE)
            .cast(pl.Datetime(time_unit=TIME_UNIT, time_zone=TIME_ZONE))
        )
        if year:
            self._data = self._data.with_columns(
                pl.col(TIMESTAMP.label).dt.year().alias(YEAR.label))
        if month:
            self._data = self._data.with_columns(
                pl.col(TIMESTAMP.label).dt.month().alias(MONTH.label))
        if day:
            self._data = self._data.with_columns(
                pl.col(TIMESTAMP.label).dt.ordinal_day().alias(DAY.label))

    def add_metadata(self, master_sheet_path: str, master_sheet_polars: pl.DataFrame = None):
        """
        Adds metadata to the CTD data from a master sheet.

        Parameters
        ----------
        master_sheet_path : str
            Path to the master sheet file.
        master_sheet_polars : pl.DataFrame, optional
            Pre-loaded master sheet as a Polars DataFrame.

        Raises
        ------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Notes
        -----
        This method adds metadata such as unique ID, Secchi depth, site name, and site ID to each profile
        in the CTD data. It also corrects missing location data if present in the master sheet.

        If a preloaded master sheet is not provided, the method will load the master sheet from the
        specified file path.

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.add_metadata('path/to/master_sheet.csv')
            # This will add metadata from the master sheet to the CTD dataset.

        """
        self.assert_data_not_empty(CTD.add_metadata.__name__)
        # If master sheet or cached master sheet is present, find the matching information and correct missing location
        if not master_sheet_polars:
            master_sheet_polars = MasterSheet(master_sheet_path=master_sheet_path)

        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.String).alias(UNIQUE_ID.label),
            pl.lit(None, dtype=pl.Float32).alias(SECCHI_DEPTH.label),
            pl.lit(None, dtype=pl.String).alias(SITE_NAME.label),
            pl.lit(None, dtype=pl.String).alias(SITE_ID.label),
        )
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            # Find master sheet match
            master_sheet_match = master_sheet_polars.find_match(profile)
            # Add secchi depth and Unique ID
            profile = profile.with_columns(
                pl.lit(master_sheet_match.unique_id, dtype=pl.String).alias(
                    UNIQUE_ID.label
                ),
                pl.lit(master_sheet_match.secchi_depth)
                .cast(pl.Float32)
                .alias(SECCHI_DEPTH.label),
                pl.lit(master_sheet_match.site_name)
                .cast(pl.String)
                .alias(SITE_NAME.label),
                pl.lit(master_sheet_match.site_id)
                .cast(pl.String)
                .alias(SITE_ID.label),
            )
            # Add location data if not present
            if (
                    LATITUDE.label not in profile.columns
                    or (
                    profile.select(pl.col(LATITUDE.label).has_nulls()).item()
                    or profile.select(pl.col(LATITUDE.label).is_nan().any()).item()
            )
                    or profile.select(pl.col(LATITUDE.label)).is_empty()
                    or profile.select(pl.col(LATITUDE.label)).limit(1).item() is None
            ):
                self._data = self._data.with_columns(
                    pl.lit(None, dtype=pl.Float64).alias(LATITUDE.label),
                    pl.lit(None, dtype=pl.Float64).alias(LONGITUDE.label),
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
                    pl.lit(latitude).cast(pl.Float64).alias(LATITUDE.label),
                    pl.lit(longitude).cast(pl.Float64).alias(LONGITUDE.label),
                    pl.lit(new_fname, pl.String).alias("filename"),
                )
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)

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
        .. code-block:: python

            from ctdfjorder import CTD
            ctd_data = CTD('CC1531002_20181225_114931.csv')
            ctd_data.remove_non_positive_samples()
            output = ctd_data.get_df()
            print(output.head(3))

        Accessing CTD data as a pandas dataframe
        .. code-block:: python

             from ctdfjorder import CTD
             ctd_data = CTD('CC1531002_20181225_114931.csv')
             ctd_data.remove_non_positive_samples()
             output = ctd_data.get_df(pandas=True)
             print(output.head(3))

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

    def assert_data_not_empty(self, func: str) -> bool:
        """
        Checks if the CTD data is not empty.

        Parameters
        ----------
        func : str
            Name of the calling function, used for error reporting.

        Returns
        -------
        bool
            True if the data is not empty.

        Raises
        ------
        NoSamplesError
            When the CTD object has no data.

        Notes
        -----
        This method is typically used internally by other methods to ensure
        that operations are not performed on empty datasets.

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.assert_data_not_empty('example_function')

        """
        if self._data.is_empty():
            raise NoSamplesError(
                filename=self._filename,
                func=func
            )
        return True

    def remove_upcasts(self) -> None:
        r"""
        Removes upcasts by dropping rows where pressure decreases from one sampling event to the next.

        Raises
        -------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Notes
        -----
        An upcast in CTD (Conductivity, Temperature, Depth) profiles occurs when the sensor package is
        moved upward through the water column, causing a decrease in pressure readings. This method
        identifies and removes such events by ensuring pressure monotonically increases within each profile.

        The procedure is as follows:

        1. It then computes the difference in pressure between consecutive rows within the profile.
        2. Rows where the pressure difference is not positive are removed, indicating a non-increasing
           pressure (i.e., an upcast or no movement).
        3. The cleaned profile is then reintegrated into the main dataset, replacing the original data.

        Let :math:`p_i` be the pressure at the :math:`i`-th sampling event. The condition for retaining a
        data point is given by:

        .. math::

            \Delta p_i = p_{i} - p_{i-1} > 0 \quad \text{for} \quad i = 1, 2, \ldots, N

        where :math:`N` is the total number of sampling events in the profile. Rows not satisfying this
        condition are considered upcasts and are removed.

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.remove_upcasts()
            # This will clean the dataset by removing upcasts, ensuring all profiles have monotonically
            # increasing pressure readings.

        See Also
        --------
        CTD : To initialize an object

        """
        self.assert_data_not_empty(CTD.add_metadata.__name__)
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            profile = profile.filter((pl.col(PRESSURE.label).diff()) > 0.0,
                                     ~(pl.col(PRESSURE.label).diff().is_nan()),
                                     ~(pl.col(PRESSURE.label).diff().is_null()),
                                     (pl.col(SEA_PRESSURE.label).diff()) > 0.0,
                                     ~(pl.col(SEA_PRESSURE.label).diff().is_nan()),
                                     ~(pl.col(SEA_PRESSURE.label).diff().is_null()), )
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)

    def filter_columns_by_range(
            self,
            column: str = None,
            lower_bound: int | float = None,
            upper_bound: int | float = None,
            filters: zip = None,
            strict: bool = True
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
        column : list of str, optional
            A list of column names to be filtered. Must be provided along with `upper_bounds` and `lower_bounds`.
        upper_bound : list of float or list of int, optional
            A list of upper bound values corresponding to each column in `columns`.
        lower_bound : list of float or list of int, optional
            A list of lower bound values corresponding to each column in `columns`.
        strict : bool, default True
            When True throws an error if column does not exist.
        Raises
        -------
        NoSamplesError
            When the function is called on a CTD object with no data.
        ValueError
            When the column is not present in the profile data.

        Notes
        -----
        The method performs the following steps for each unique profile identified by `profile_id`:

        1. If `filters` is provided, it iterates over each filter tuple and applies the specified bounds
           to the relevant column, filtering out data that does not meet the criteria.
        2. If `columns`, `upper_bounds`, and `lower_bounds` are provided, it iterates over each column
           and applies the corresponding upper and lower bounds to filter the data.
        3. Updates the dataset by removing the original profile data and reintegrating the filtered profile data.

        The method is designed to be flexible, allowing filtering through either a comprehensive set of filters
        or by specifying individual columns and their bounds directly. This makes it adaptable to various
        dataset structures and filtering requirements.

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            filters = zip(['temperature', 'salinity'], [20.0, 35.0], [10.0, 30.0])
            ctd_data.filter_columns_by_range(filters=filters)
            # This will filter the `temperature` column to be between 10.0 and 20.0, and `salinity` to be between 30.0 and 35.0.

        .. code-block:: python

            column = 'temperature'
            upper_bound = 30.0
            lower_bound = 10.0
            ctd_data.filter_columns_by_range(column=column, upper_bound=upper_bound, lower_bound=lower_bound)
            # This will filter the `temperature` column to be between 10.0 and 30.0

        See Also
        --------
        remove_non_positive_samples : Method to remove rows with non-positive values for specific parameters.
        """

        def apply_bounds(profile: pl.DataFrame, column: str, upper_bound: int | float | None,
                         lower_bound: int | float | None) -> pl.DataFrame:
            if upper_bound is not None:
                profile = profile.filter(pl.col(column) <= upper_bound)
            if lower_bound is not None:
                profile = profile.filter(pl.col(column) >= lower_bound)
            return profile

        self.assert_data_not_empty(CTD.filter_columns_by_range.__name__)
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)

            if filters:
                for column, upper_bound, lower_bound in filters:
                    try:
                        profile = apply_bounds(
                            profile, column, upper_bound, lower_bound
                        )
                    except ValueError as e:
                        if strict:
                            raise e
            elif column and upper_bound and lower_bound:
                try:
                    profile = apply_bounds(profile, column, upper_bound, lower_bound)
                except ValueError as e:
                    if strict:
                        raise e
            self._data = self._data.filter(
                pl.col(PROFILE_ID.label) != profile_id
            ).vstack(profile)

    def remove_non_positive_samples(self) -> None:
        r"""
        Removes rows with non-positive values for depth, pressure, practical salinity, absolute salinity, or density.

        Raises
        -------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Notes
        -----
        This method cleans the CTD (Conductivity, Temperature, Depth) dataset by removing any samples
        that have non-positive values for key parameters. Non-positive values in these parameters
        are generally invalid and indicate erroneous measurements.

        Let :math:`( x_i )` represent the value of a parameter (depth, pressure, practical salinity, absolute salinity,
        or density) at the :math:`( i )`-th sampling event. The condition for retaining a data point is given by:

        .. math::

            x_i > 0 \quad \text{and} \quad x_i \neq \text{NaN} \quad \text{and} \quad x_i \neq \text{null}

        Rows not satisfying this condition for any of the parameters are removed.

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.remove_non_positive_samples()
            # This will clean the dataset by removing samples with non-positive, null, or NaN values
            # for the specified key parameters.

        See Also
        --------
        remove_invalid_salinity_values : method to remove salinity values < 10 PSU.

        """
        self.assert_data_not_empty(CTD.remove_non_positive_samples.__name__)
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            cols = list(
                {
                    DEPTH.label,
                    PRESSURE.label,
                    SALINITY.label,
                    ABSOLUTE_SALINITY.label,
                    DENSITY.label,
                }.intersection(profile.collect_schema().names())
            )
            for col in cols:
                profile = profile.filter(
                    pl.col(col) > 0.0, ~pl.col(col).is_null(), pl.col(col).is_not_nan()
                )
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)

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
        NoSamplesError
            When the function is called on a CTD object with no data.

        ValueError
            When the cleaning method is invalid.

        Notes
        -----
        The 'clean_salinity_ai' method uses a Gated Recurrent Unit (GRU) model to correct
        salinity measurements. This model is designed to smooth out unrealistic fluctuations
        in salinity with respect to pressure.

        The procedure for 'clean_salinity_ai' is as follows:

        1. Bin the data every 0.5 dbar of pressure.
        2. Use a GRU model with a loss function that penalizes non-monotonic salinity increases with decreasing pressure.
        3. Train the model on the binned data to predict clean salinity values.
        4. Replace the original salinity values in the profile with the predicted clean values.
        5. Reintegration of the cleaned profile into the main dataset.

        The loss function :math:`( L )` used in training is defined as:

        .. math::

            L = \text{MAE}(y_{\text{true}}, y_{\text{pred}}) + \lambda \cdot \text{mean}(P)

        where :math:`( P )` are penalties for predicted salinity increases with decreasing pressure,
        and :math:`( \lambda )` is a weighting factor.

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.clean('clean_salinity_ai')
            # This will clean the salinity data using the AI-based method, correcting any unrealistic
            # values and ensuring smoother transitions with respect to pressure.

        See Also
        --------
        _AI.clean_salinity_ai : Method used to clean salinity with 'clean_salinity_ai' option.

        """
        self.assert_data_not_empty(CTD.clean.__name__)
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            if method == "clean_salinity_ai":
                profile = ai.AI.clean_salinity_ai(profile, profile_id)
            else:
                raise ValueError(f"Invalid method {method}.")
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = pl.concat([self._data, profile], how=CONCAT_HOW, rechunk=True)

    def add_absolute_salinity(self) -> None:
        r"""
        Calculates and adds absolute salinity to the CTD data using the TEOS-10 salinity conversion formula.

        Raises
        -------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Notes
        -----
        The `gsw.SA_from_SP` function from the Gibbs SeaWater (GSW) Oceanographic Toolbox is utilized
        for this calculation. More information about this function can be found at the
        `TEOS-10 website <https://www.teos-10.org/pubs/gsw/html/gsw_SA_from_SP.html>`__.

        This method computes the absolute salinity from practical salinity for each profile in the dataset
        using the TEOS-10 standard. Absolute salinity provides a more accurate representation of salinity
        by accounting for the variations in seawater composition.

        The TEOS-10 formula for converting practical salinity :math:`( S_P )` to absolute salinity :math:`( S_A )` is used:

        .. math::

            S_A = f(S_P, p, \phi, \lambda)

        where :math:`( p )` is the sea pressure, :math:`( \phi )` is the latitude, and :math:`( \lambda )` is the longitude.

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.add_absolute_salinity()
            # This will add a new column with absolute salinity values to the dataset, calculated using the
            # TEOS-10 formula.

        See Also
        --------
        gsw.conversions.SA_from_SP : Function used for the conversion from practical salinity to absolute salinity.

        """
        self.assert_data_not_empty(CTD.add_absolute_salinity.__name__)
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(ABSOLUTE_SALINITY.label)
        )
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            s = profile.select(pl.col(SALINITY.label)).to_numpy()
            p = profile.select(pl.col(SEA_PRESSURE.label)).to_numpy()
            lat = profile.select(pl.col(LATITUDE.label)).to_numpy()
            long = profile.select(pl.col(LONGITUDE.label)).to_numpy()
            salinity_abs_list = gsw.conversions.SA_from_SP(s, p, lat, long)
            salinity_abs = pl.Series(
                np.array(salinity_abs_list).flatten(), dtype=pl.Float64, strict=False
            ).to_frame(ABSOLUTE_SALINITY.label)
            profile = profile.with_columns(salinity_abs)
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)

    def add_density(self):
        r"""
        Calculates and adds density to CTD data using the TEOS-10 formula.
        If absolute salinity is not present, it is calculated first.

        Raises
        -------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Notes
        -----
        The `gsw.rho_t_exact` function from the Gibbs SeaWater (GSW) Oceanographic Toolbox is utilized
        for this calculation. More information about this function can be found at the
        `TEOS-10 website <https://www.teos-10.org/pubs/gsw/html/gsw_rho_t_exact.html>`__.

        This method computes the density of seawater from absolute salinity, in-situ temperature,
        and sea pressure using the TEOS-10 standard. The density is a critical parameter for
        understanding the physical properties of seawater and its buoyancy characteristics.

        The TEOS-10 formula for calculating density :math:`( \rho )` is used:

        .. math::

            \rho = f(S_A, T, p)

        where :math:`( S_A )` is the absolute salinity, :math:`( T )` is the in-situ temperature, and :math:`( p )` is the sea pressure.

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.add_density()
            # This will add a new column with density values to the dataset, calculated using the
            # TEOS-10 formula.

        See Also
        --------
        gsw.density.rho_t_exact : Function used for the calculation of density.
        add_absolute_salinity : Method to add absolute salinity if it is not already present in the dataset.

        """
        self.assert_data_not_empty(CTD.add_density.__name__)
        if ABSOLUTE_SALINITY.label not in self._data.columns:
            self.add_absolute_salinity()
        self._data = self._data.with_columns(
            pl.lit(None).cast(pl.Float64).alias(DENSITY.label)
        )
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            sa = profile.select(pl.col(ABSOLUTE_SALINITY.label)).to_numpy()
            t = profile.select(pl.col(TEMPERATURE.label)).to_numpy()
            p = profile.select(pl.col(SEA_PRESSURE.label)).to_numpy()
            density = pl.Series(
                np.array(gsw.density.rho_t_exact(sa, t, p)).flatten(),
                dtype=pl.Float64,
                strict=True,
            ).to_frame(DENSITY.label)
            profile = profile.with_columns(density)
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)

    def add_potential_density(self):
        r"""
        Calculates and adds potential density to the CTD data using the TEOS-10 formula.
        If absolute salinity is not present, it is calculated first.

        Raises
        -------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Notes
        -----
        The `gsw.sigma0` function from the Gibbs SeaWater (GSW) Oceanographic Toolbox is utilized
        for this calculation. More information about this function can be found at the
        `TEOS-10 website <https://www.teos-10.org/pubs/gsw/html/gsw_sigma0.html>`__.

        This method computes the potential density of seawater from absolute salinity and in-situ temperature
        using the TEOS-10 standard. Potential density is the density a parcel of seawater would have if
        it were adiabatically brought to the sea surface, which helps in understanding the stability and
        stratification of the water column.

        The TEOS-10 formula for calculating potential density :math:`( \sigma_0 )` is used:

        .. math::

            \sigma_0 = f(S_A, T)

        where :math:`( S_A )` is the absolute salinity and :math:`( T )` is the in-situ temperature.

        Examples
        --------
        .. code-block::
        
            ctd_data = CTD('example.csv')
            ctd_data.add_potential_density()
            # This will add a new column with potential density values to the dataset, calculated using the
            # TEOS-10 formula.

        See Also
        --------
        gsw.sigma0 : Function used for the calculation of potential density.
        add_absolute_salinity : Method to add absolute salinity if it is not already present in the dataset.

        """
        self.assert_data_not_empty(CTD.add_potential_density.__name__)
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(POTENTIAL_DENSITY.label)
        )
        if ABSOLUTE_SALINITY.label not in self._data.columns:
            self.add_absolute_salinity()
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            sa = profile.select(pl.col(ABSOLUTE_SALINITY.label)).to_numpy()
            t = profile.select(pl.col(TEMPERATURE.label)).to_numpy()
            potential_density = pl.Series(
                np.array(gsw.sigma0(sa, t)).flatten()
            ).to_frame(POTENTIAL_DENSITY.label)
            profile = profile.with_columns(pl.Series(potential_density))
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)

    def add_surface_salinity(self, start=10.1325, end=12.1325):
        """
        Calculate and add surface salinity to the dataset.

        This function calculates the surface salinity for each profile in the dataset. The surface salinity
        is determined by selecting the salinity value at the lowest pressure within the specified range.

        Parameters
        ----------
        start : float, default 10.1325
            Upper bound of surface pressure.
        end : float, default 12.1325
            Lower bound of surface pressure.

        Raises
        ------
        NoSamplesError
            If the dataset is empty when the function is called.

        Warning
            If no measurements are found within the specified pressure range for a profile.
        """
        self.assert_data_not_empty(self.add_surface_salinity.__name__)
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(SURFACE_SALINITY.label)
        )
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            surface_data = profile.filter(
                pl.col(PRESSURE.label) > start, pl.col(PRESSURE.label) < end
            )
            if surface_data.is_empty():
                raise_warning_calculatuion(
                    filename=self._filename,
                    message=WARNING_CTD_SURFACE_MEASUREMENT,
                )
                self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
                self._data = self._data.vstack(profile)
                continue

            surface_salinity = surface_data.select(pl.col(SALINITY.label).first()).item()
            profile = profile.with_columns(
                pl.lit(surface_salinity).alias(SURFACE_SALINITY.label)
            )
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)

    def add_surface_temperature(self, start=10.1325, end=12.1325):
        """
        Calculate and add surface temperature to the dataset.

        This function calculates the surface temperature for each profile in the dataset. The surface
        temperature is determined by calculating the mean temperature within the specified pressure range.

        Parameters
        ----------
        start : float, default 10.1325
            Upper bound of surface pressure.
        end : float, default 12.1325
            Lower bound of surface pressure.

        Raises
        ------
        NoSamplesError
            If the dataset is empty when the function is called.

        Warning
            If no measurements are found within the specified pressure range for a profile.
        """
        self.assert_data_not_empty(self.add_surface_temperature.__name__)
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(SURFACE_TEMPERATURE.label)
        )
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            surface_data = profile.filter(
                pl.col(PRESSURE.label) > start, pl.col(PRESSURE.label) < end
            )
            if surface_data.is_empty():
                raise_warning_calculatuion(
                    filename=self._filename,
                    message=WARNING_CTD_SURFACE_MEASUREMENT,
                )
                self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
                self._data = self._data.vstack(profile)
                continue

            surface_temperature = np.array(
                surface_data.select(pl.col(TEMPERATURE.label).mean()).to_numpy()
            ).item(0)
            profile = profile.with_columns(
                pl.lit(surface_temperature).alias(SURFACE_TEMPERATURE.label)
            )
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)

    def add_meltwater_fraction(self):
        """
        Calculate and add meltwater fractions (EQ. 10 and EQ. 11) to the dataset.

        This function calculates the meltwater fractions for each profile based on the surface salinity.
        The calculations are performed using the formulas provided by `Pan et al. 2019 <10.1371/journal.pone.0211107>`__.

        Raises
        ------
        NoSamplesError
            If the dataset is empty when the function is called.

        Notes
        -----
        - Meltwater fraction EQ 11, where :math:`S_0` is surface salinity

        .. math::

            \text{Meltwater Fraction} = (-0.021406 * S_0 + 0.740392) * 100

        - Meltwater fraction EQ 10, where :math:`S_0` is surface salinity

        .. math::

            \text{Meltwater Fraction} = (-0.016 * S_0 + 0.544) * 100

        """
        self.assert_data_not_empty(self.add_meltwater_fraction.__name__)
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(MELTWATER_FRACTION_EQ_10.label),
            pl.lit(None, dtype=pl.Float64).alias(MELTWATER_FRACTION_EQ_11.label)
        )
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            surface_salinity = profile.select(pl.col(SURFACE_SALINITY.label).first()).item()

            mwf10 = (-0.016 * surface_salinity + 0.544) * 100
            mwf11 = (-0.021406 * surface_salinity + 0.740392) * 100

            profile = profile.with_columns(
                pl.lit(mwf10).alias(MELTWATER_FRACTION_EQ_10.label),
                pl.lit(mwf11).alias(MELTWATER_FRACTION_EQ_11.label)
            )
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)


    def add_speed_of_sound(self) -> None:
        """
        Calculates and adds sound speed to the CTD data using the TEOS-10 formula.

        This method computes the speed of sound in seawater, which is influenced by factors such as
        salinity, temperature, and pressure. The sound speed is a critical parameter for various
        oceanographic studies, particularly in understanding acoustic propagation.

        Raises
        ------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Notes
        -----
        The `gsw.sound_speed` function from the Gibbs SeaWater (GSW) Oceanographic Toolbox is utilized
        for this calculation. More information about this function can be found at the
        `TEOS-10 website <https://www.teos-10.org/pubs/gsw/html/gsw_sound_speed.html>`__.

        The speed of sound, :math:`c`, in seawater is calculated using the TEOS-10 equation:

        .. math::

            c = f(S_A, T, p)

        where:
        - :math:`S_A` is the absolute salinity,
        - :math:`T` is the in-situ temperature,
        - :math:`p` is the sea pressure.

        This method adds a new column for sound speed to the dataset.

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.add_speed_of_sound()
            # This will add a new column with sound speed values to the dataset, calculated using the TEOS-10 formula.
            
        """
        self.assert_data_not_empty(CTD.add_speed_of_sound.__name__)
        if ABSOLUTE_SALINITY.label not in self._data.columns:
            self.add_absolute_salinity()
        self._data = self._data.with_columns(
            pl.lit(None).cast(pl.Float64).alias(SPEED_OF_SOUND.label)
        )
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            sa = profile.select(pl.col(ABSOLUTE_SALINITY.label)).to_numpy()
            t = profile.select(pl.col(TEMPERATURE.label)).to_numpy()
            p = profile.select(pl.col(SEA_PRESSURE.label)).to_numpy()
            sound_speed = pl.Series(
                np.array(gsw.sound_speed(sa, t, p)).flatten(),
                dtype=pl.Float64,
                strict=True,
            ).to_frame(SPEED_OF_SOUND.label)
            profile = profile.with_columns(sound_speed)
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)

    def add_potential_temperature(self, p_ref: Union[float, np.ndarray] = 0) -> None:
        r"""
        Calculates and adds potential temperature to the CTD data using the TEOS-10 formula.

        This method computes the potential temperature of seawater, which is the temperature
        a parcel of water would have if moved adiabatically to the sea surface pressure.

        Raises
        ------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Notes
        -----
        The `gsw.pt_from_t` function from the Gibbs SeaWater (GSW) Oceanographic Toolbox is utilized
        for this calculation. More information about this function can be found at the
        `TEOS-10 website <https://www.teos-10.org/pubs/gsw/html/gsw_pt_from_t.html>`__.

        This method adds a new column for potential temperature in the dataset.

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.add_potential_temperature()
            # This will add a new column with potential temperature values to the dataset, calculated using the TEOS-10 formula.
            
        """
        self.assert_data_not_empty(CTD.add_potential_temperature.__name__)
        if ABSOLUTE_SALINITY.label not in self._data.columns:
            self.add_absolute_salinity()

        # Compute potential temperature across all profiles in one go
        sa = self._data.select(pl.col(ABSOLUTE_SALINITY.label)).to_numpy().flatten()
        t = self._data.select(pl.col(TEMPERATURE.label)).to_numpy().flatten()
        p = self._data.select(pl.col(SEA_PRESSURE.label)).to_numpy().flatten()

        # Calculate potential temperature for all data points
        potential_temperature_values = gsw.pt_from_t(sa, t, p, p_ref)

        # Add the calculated potential temperature to the dataframe
        self._data = self._data.with_columns(
            pl.Series(potential_temperature_values, dtype=pl.Float64).alias(
                "potential_temperature"
            )
        )

    def add_mean_surface_density(self, start=10.1325, end=12.1325) -> None:
        """
        Calculates the mean surface density from the density values and adds it as a new column
        to the CTD data table. Requires absolute salinity and absolute density to be calculated first.

        Raises
        ------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Parameters
        ----------
        start : float, default 10.1325
            Upper bound of surface pressure.
        end : float, default 12.1325
            Lower bound of surface pressure.

        Notes
        -----
        Mean surface density is computed as the mean of density values within the specified pressure range (`start` to `end`).

        Examples
        --------
        .. code-block::

            ctd_data = CTD('example.csv')
            ctd_data.add_mean_surface_density(start=10.1325, end=12.1325)
            # This will add a new column with mean surface density values to the dataset, calculated using the
            # specified pressure range.

        See Also
        --------
        add_absolute_salinity : Method to add absolute salinity if it is not already present in the dataset.
        add_density : Method to add density if it is not already present in the dataset.

        """
        self.assert_data_not_empty(CTD.add_mean_surface_density.__name__)
        # Filtering data within the specified pressure range
        self._data = self._data.with_columns(
            pl.lit(None).cast(pl.Float64).alias(SURFACE_DENSITY.label)
        )
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            surface_data = profile.filter(
                pl.col(PRESSURE.label) > start, pl.col(PRESSURE.label) < end
            )
            surface_density = surface_data.select(pl.col(DENSITY.label).mean()).item()
            profile = profile.with_columns(
                pl.lit(surface_density).alias(SURFACE_DENSITY.label)
            )
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)

    def add_conservative_temperature(self) -> None:
        """
        Calculates and adds conservative temperature to the CTD data using the TEOS-10 formula.

        This method computes the conservative temperature, which is a more accurate measure of heat content
        in seawater compared to potential temperature.

        Raises
        ------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Notes
        -----
        The `gsw.CT_from_t` function from the Gibbs SeaWater (GSW) Oceanographic Toolbox is utilized
        for this calculation. More information about this function can be found at the
        `TEOS-10 website <https://www.teos-10.org/pubs/gsw/html/gsw_CT_from_t.html>`__.

        This method adds a new column for conservative temperature in the dataset.

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.add_conservative_temperature()
            # This will add a new column with conservative temperature values to the dataset, calculated using the TEOS-10 formula.

        """
        self.assert_data_not_empty(CTD.add_conservative_temperature.__name__)
        if ABSOLUTE_SALINITY.label not in self._data.columns:
            self.add_absolute_salinity()

        sa = self._data.select(pl.col(ABSOLUTE_SALINITY.label)).to_numpy().flatten()
        t = self._data.select(pl.col(TEMPERATURE.label)).to_numpy().flatten()

        conservative_temperature_values = gsw.CT_from_t(sa, t, 0)

        self._data = self._data.with_columns(
            pl.Series(conservative_temperature_values, dtype=pl.Float64).alias(
                "conservative_temperature"
            )
        )

    def add_dynamic_height(self, p_ref: Union[float, np.ndarray] = 0) -> None:
        r"""
        Calculates and adds dynamic height anomaly to the CTD data using the TEOS-10 formula.

        This method computes the dynamic height anomaly, which represents the geostrophic streamfunction
        that indicates the difference in horizontal velocity between the pressure at the measurement
        point (p) and a reference pressure (p_ref).

        Raises
        ------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Parameters
        ----------
        p_ref : Union[float, np.ndarray], optional
            Reference pressure, in dbar. The default is 0 dbar, corresponding to the sea surface.

        Notes
        -----
        The `gsw.geo_strf_dyn_height` function from the Gibbs SeaWater (GSW) Oceanographic Toolbox is utilized
        for this calculation. More information about this function can be found at the
        `TEOS-10 website <https://www.teos-10.org/pubs/gsw/pdf/geo_strf_dyn_height.pdf>`__.

        Examples
        --------
        .. code-block:: python
        
            ctd_data = CTD('example.csv')
            ctd_data.add_dynamic_height()
            # This will add a new column with dynamic height values to the dataset, calculated using the TEOS-10 formula.
            
        """
        self.assert_data_not_empty(CTD.add_dynamic_height.__name__)
        if ABSOLUTE_SALINITY.label not in self._data.columns:
            self.add_absolute_salinity()

        sa = self._data.select(pl.col(ABSOLUTE_SALINITY.label)).to_numpy().flatten()
        ct = (
            self._data.select(pl.col(CONSERVATIVE_TEMPERATURE.label))
            .to_numpy()
            .flatten()
        )
        p = self._data.select(pl.col(SEA_PRESSURE.label)).to_numpy().flatten()

        dynamic_height = gsw.geo_strf_dyn_height(sa, ct, p, p_ref)

        self._data = self._data.with_columns(
            pl.Series(dynamic_height, dtype=pl.Float64).alias("dynamic_height")
        )


    def add_thermal_expansion_coefficient(self) -> None:
        r"""
        Calculates and adds thermal expansion coefficient to the CTD data using the TEOS-10 formula.

        The thermal expansion coefficient, :math:`\alpha`, is important for understanding how the volume of seawater
        changes with temperature at constant pressure. It is derived from absolute salinity, conservative temperature,
        and sea pressure.

        Raises
        ------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Notes
        -----
        The `gsw.alpha` function from the Gibbs SeaWater (GSW) Oceanographic Toolbox is utilized
        for this calculation. More information about this function can be found at the
        `TEOS-10 website <https://www.teos-10.org/pubs/gsw/html/gsw_alpha.html>`__.

        The thermal expansion coefficient is calculated using the following equation from TEOS-10:

        .. math::

            \alpha^\theta = -\frac{1}{\rho} \frac{\partial \rho}{\partial \theta} \bigg|_{S_A, P}

        where:
        - :math:`\rho` is the in-situ density of seawater,
        - :math:`\theta` is the conservative temperature,
        - :math:`S_{A}` is the absolute salinity,
        - :math:`P` is the sea pressure.

        This method adds a new column for the thermal expansion coefficient in the dataset. It ensures that absolute salinity
        and conservative temperature are present in the data, calculating them if necessary, before computing the thermal expansion coefficient.

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.add_thermal_expansion_coefficient()
            # This will add a new column with thermal expansion coefficient values to the dataset, calculated using the TEOS-10 formula.

        """
        self.assert_data_not_empty(CTD.add_thermal_expansion_coefficient.__name__)
        if ABSOLUTE_SALINITY.label not in self._data.columns:
            self.add_absolute_salinity()
        if CONSERVATIVE_TEMPERATURE.label not in self._data.columns:
            self.add_conservative_temperature()

        sa = self._data.select(pl.col(ABSOLUTE_SALINITY.label)).to_numpy().flatten()
        ct = (
            self._data.select(pl.col(CONSERVATIVE_TEMPERATURE.label))
            .to_numpy()
            .flatten()
        )
        p = self._data.select(pl.col(SEA_PRESSURE.label)).to_numpy().flatten()

        thermal_expansion_coefficient_values = gsw.alpha(sa, ct, p)

        self._data = self._data.with_columns(
            pl.Series(thermal_expansion_coefficient_values, dtype=pl.Float64).alias(
                "thermal_expansion_coefficient"
            )
        )

    def add_haline_contraction_coefficient(self) -> None:
        r"""
        Calculates and adds haline contraction coefficient to the CTD data using the TEOS-10 formula.

        The haline contraction coefficient, :math:`\beta`, is important for understanding how the volume of seawater
        changes with salinity at constant temperature. It is derived from absolute salinity, conservative temperature,
        and sea pressure.

        Raises
        ------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Notes
        -----
        The `gsw.beta` function from the Gibbs SeaWater (GSW) Oceanographic Toolbox is utilized
        for this calculation. More information about this function can be found at the
        `TEOS-10 website <https://www.teos-10.org/pubs/gsw/html/gsw_beta.html>`__.

        The haline contraction coefficient is calculated using the following equation from TEOS-10:

        .. math::

            \beta^\theta = \frac{1}{\rho} \frac{\partial \rho}{\partial S_A} \bigg|_{\theta, P}

        where:
        - :math:`\rho` is the in-situ density of seawater,
        - :math:`S_A` is the absolute salinity,
        - :math:`\theta` is the conservative temperature,
        - :math:`P` is the sea pressure.

        This method adds a new column for the haline contraction coefficient in the dataset. It ensures that absolute salinity
        and conservative temperature are present in the data, calculating them if necessary, before computing the haline contraction coefficient.

        Examples
        --------
        .. code-block:: python
        
            ctd_data = CTD('example.csv')
            ctd_data.add_haline_contraction_coefficient()
            # This will add a new column with haline contraction coefficient values to the dataset, calculated using the TEOS-10 formula.

        """
        self.assert_data_not_empty(CTD.add_haline_contraction_coefficient.__name__)
        if ABSOLUTE_SALINITY.label not in self._data.columns:
            self.add_absolute_salinity()
        if CONSERVATIVE_TEMPERATURE.label not in self._data.columns:
            self.add_conservative_temperature()
        sa = self._data.select(pl.col(ABSOLUTE_SALINITY.label)).to_numpy().flatten()
        ct = (
            self._data.select(pl.col(CONSERVATIVE_TEMPERATURE.label))
            .to_numpy()
            .flatten()
        )
        p = self._data.select(pl.col(SEA_PRESSURE.label)).to_numpy().flatten()

        haline_contraction_coefficient_values = gsw.beta(sa, ct, p)

        self._data = self._data.with_columns(
            pl.Series(haline_contraction_coefficient_values, dtype=pl.Float64).alias(
                "haline_contraction_coefficient"
            )
        )

    def add_mld(
            self, method: str = "potential_density_avg", delta: float | None = 0.05, reference: int | None = 10
    ) -> None:
        r"""
        Calculates and adds the mixed layer depth (MLD) using the density threshold method.

        Parameters
        ----------
        method : str, default "potential_density_avg"
            The MLD calculation method. Options are "abs_density_avg" or "potential_density_avg".
        delta : float or None, default 0.05
            The change in density or potential density from the reference that would define the MLD in units of
            :math:`\frac{kg}{m^3}`.
        reference : int
            The reference depth for MLD calculation.

        Raises
        ------
        NoSamplesError
            When the function is called on a CTD object with no data.
        ValueError
            When the specified method is invalid.


        Notes
        -----
        The mixed layer depth (MLD) is calculated using the density threshold method, defined as the depth
        at which the density increases by a specified amount (delta) from the reference density. The reference
        density is calculated as the mean density up to the reference depth.

        The procedure is as follows for the thresholding methods:

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

        Examples
        --------
        
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.add_mld(reference=10, method="potential_density_avg", delta=0.05)
            # This will add a new column with MLD values to the dataset, calculated using the specified method
            # and parameters.

        See Also
        --------
        add_density : method to calculate derived density.
        add_potential_density : method to calculate derived potential density.

        """
        self.assert_data_not_empty(CTD.add_mld.__name__)
        mld_col_labels = []
        supported_methods = ["abs_density_avg", "potential_density_avg"]
        if method == supported_methods[0] or method == supported_methods[1]:
            mld_col_labels.append(f"MLD_Ref:{reference}_(m)_Thresh_{delta}_(kg/m^3)")
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(mld_col_labels[-1])
        )
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            df_filtered = profile.filter(pl.col(DEPTH.label) <= reference)
            if method == supported_methods[0]:
                reference_density = df_filtered.select(
                    pl.col(DENSITY.label).mean()
                ).item()
                df_filtered = profile.filter(
                    pl.col(DENSITY.label) >= reference_density + delta
                )
            elif method == supported_methods[1]:
                reference_density = df_filtered.select(
                    pl.col(POTENTIAL_DENSITY.label).mean()
                ).item()
                df_filtered = profile.filter(
                    pl.col(POTENTIAL_DENSITY.label) >= reference_density + delta
                )
            else:
                raise ValueError(f'Invalid method "{method}" not in {supported_methods}')
            mld = df_filtered.select(pl.col(DEPTH.label).first()).item()
            profile = profile.with_columns(pl.lit(mld, dtype=pl.Float64).alias(mld_col_labels[-1]))
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)

    def add_profile_classification(self, stratification_threshold=0.5):
        """
        Classifies each profile based on salinity and depth into one of three categories.

        Parameters
        ----------
        stratification_threshold : float, default 0.5
            The threshold for determining if a profile is stratified/type C.

        Raises
        ------
        NoSamplesError
            When the function is called on a CTD object with no data.

        Notes
        -----
        This method adds a new column to the dataset with the following classification:
        A - Very shallow low salinity surface ML
        B - Normal well-mixed MLD
        C - Stratified MLD from surface to bottom of ML

        The classification criteria are:
        - A: if salinity_at_mld - surface_salinity > 0.5 meters
        - C: if max(salinity) - min(salinity) < 0.5 PSU
        - B: otherwise

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.add_profile_classification()
            # This will add a new column with profile classifications to the dataset.

        """

        # Check if required columns exist
        self.assert_data_not_empty(CTD.add_profile_classification.__name__)

        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias(CLASSIFICATION.label)
        )

        # Iterate through each profile
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            mld_column = profile.select(pl.col(r"^.*MLD.*$")).columns[0]
            first_salinity = profile.select(pl.col(SALINITY.label).first()).item()
            mld_salinity = profile.select(pl.col(mld_column).first()).item()
            if type(mld_salinity) is not type(None):
                salinity_diff = float(mld_salinity) - float(first_salinity)
            else:
                salinity_diff = None
            # Classify profile based on the criteria
            if type(salinity_diff) is not type(None) and salinity_diff > 0.5:
                classification = 'A - Very shallow low salinity surface ML'
            elif profile.select((pl.col(SALINITY.label).max() - pl.col(
                    SALINITY.label).min()).abs() < stratification_threshold).item():
                classification = 'C - Stratified MLD from surface to bottom of ML'
            else:
                classification = 'B - Normal well mixed ML'
            # Update the profile with the classification result
            profile = profile.with_columns(pl.lit(classification).alias(CLASSIFICATION.label))

            # Reintegration of the updated profile into the main dataset
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)

    def add_mld_bf(self, min_qi=0.0) -> None:
        r"""
        Calculates the mixed layer depth (MLD) using the max buoyancy frequency (N^2) method based on salinity profiles.

        Parameters
        ----------
        min_qi
            Minimum quality index score to be considered a valid MLD.

        Notes
        -----
        The mixed layer depth (MLD) is determined by identifying the depth where the
        maximum buoyancy frequency occurs. After finding this depth, the Quality Index (QI)
        is calculated to assess the reliability of the MLD. The Quality Index is given by:

        .. math::
            QI = 1 - \frac{\sigma_{A1}}{\sigma_{A2}}

        where:

        * :math:`\sigma_{A1}` is the standard deviation of the potential density within the mixed layer depth
        * :math:`\sigma_{A2}` is the standard deviation of the potential density within 1.5 times the mixed layer depth.

        The calculated QI must be greater than or equal to `min_qi` for the MLD to be considered valid. If invalid
        or incalculable MLD is reported as None.

        Raises
        ------
        NoSamplesError
            When the function is called on a CTD object with no data.

        See Also
        --------
        add_mld : Method to calculate and add MLD to a dataset using different methods, including density threshold.

        """
        # Check if required columns exist
        self.assert_data_not_empty(CTD.add_mld_bf.__name__)
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("MLD_BF_(m)"),
            pl.lit(None, dtype=pl.Float64).alias("Quality_Index")
        )
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            bf_max = profile.filter(pl.col(N2.label) == (pl.col(N2.label).max()))
            mld = bf_max.select(pl.col(DEPTH.label).first()).item()
            profile_up_to_mld = profile.filter(pl.col(DEPTH.label) <= mld)
            if not profile.select(pl.col(DEPTH.label).max() >= mld*1.5).item():
                profile_up_to_mld_1dot5 = profile.filter(pl.col(DEPTH.label) <= mld * 1.5)
            else:
                profile_up_to_mld_1dot5 = profile
            std_A1 = profile_up_to_mld.select(pl.col(POTENTIAL_DENSITY.label).std()).item()
            if std_A1 is None:
                # Reintegration of the updated profile into the main dataset
                self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
                self._data = self._data.vstack(profile)
                continue
            std_A2 = profile_up_to_mld_1dot5.select(pl.col(POTENTIAL_DENSITY.label).std()).item()
            qi = 1 - (std_A1/std_A2)
            if qi >= min_qi:
                profile = profile.with_columns(pl.lit(mld).alias("MLD_BF_(m)"),
                                               pl.lit(qi).alias("Quality_Index"))
            # Reintegration of the updated profile into the main dataset
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)
    def add_n_squared(self) -> None:
        r"""
        Calculates buoyancy frequency squared and adds it to the CTD data.
        Requires potential density to be calculated first.

        Raises
        ------
        NoSamplesError
            When the function is called on a CTD object with no data.
        ValueError
            When buoyancy frequency could not be calculated because of malformed data.

        Notes
        -----
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

        The `gsw.Nsquared` function from the Gibbs SeaWater (GSW) Oceanographic Toolbox is utilized
        for this calculation. More information about this function can be found at the
        `TEOS-10 website <https://www.teos-10.org/pubs/gsw/html/gsw_Nsquared.html>`__.

        The buoyancy frequency squared :math:`( N^2 )` is calculated using the formula:

        .. math::

            N_2 = g_2 \cdot \frac{\beta \cdot d(SA) - \alpha \cdot d(CT)}{\text{specvol_local} \cdot dP}

        This routine uses rho from "gsw_specvol", which is the
        computationally efficient 75-term expression for specific volume in
        terms of SA, CT and p (Roquet et al., 2015).

        Note also that the pressure increment, dP, in the above formula is in
        Pa, so that it is 104 times the pressure increment dp in dbar.

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.add_brunt_vaisala_squared()
            # This will add new columns with buoyancy frequency squared values and mid-pressure values to the dataset,
            # calculated using the TEOS-10 formula.

        See Also
        --------
        gsw.Nsquared : Function used for the calculation of buoyancy frequency squared.

        """
        self._data = self._data.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(N2.label),
            pl.lit(None, dtype=pl.Float64).alias(P_MID.label),
        )
        for profile_id in (
                self._data.select(PROFILE_ID.label)
                        .unique(keep="first")
                        .to_series()
                        .to_list()
        ):
            profile = self._data.filter(pl.col(PROFILE_ID.label) == profile_id)
            sa = profile.select(pl.col(ABSOLUTE_SALINITY.label)).to_numpy().flatten()
            t = profile.select(pl.col(TEMPERATURE.label)).to_numpy().flatten()
            p = profile.select(pl.col(SEA_PRESSURE.label)).to_numpy().flatten()
            lat = profile.select(pl.col(LATITUDE.label)).to_numpy().flatten()
            ct = gsw.CT_from_t(sa, t, p).flatten()
            try:
                n_2, p_mid = gsw.Nsquared(SA=sa, CT=ct, p=p, lat=lat)
            except ValueError:
                raise ValueError(f"Unable to calculate buoyancy frequency, likely due to lat = {lat}",
                                 )
            buoyancy_frequency = (
                pl.Series(np.array(n_2).flatten())
                .extend_constant(None, n=1)
                .to_frame(N2.label)
            )
            p_mid = pl.Series(p_mid).extend_constant(None, n=1).to_frame(P_MID.label)
            profile = profile.with_columns(
                pl.Series(buoyancy_frequency), pl.Series(p_mid)
            )
            self._data = self._data.filter(pl.col(PROFILE_ID.label) != profile_id)
            self._data = self._data.vstack(profile)
        self.assert_data_not_empty(CTD.add_n_squared.__name__)

    def save_to_csv(self, output_file: str, null_value: str | None) -> None:
        """
        Renames the columns of the CTD data table based on a predefined mapping and saves the
        data to the specified CSV file.

        Raises
        ------
        IOError
            If there is an error in writing to the specified file path.

        Parameters
        ----------
        output_file : str
            The output CSV file path.
        null_value : str
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

        Examples
        --------
        .. code-block:: python

            ctd_data = CTD('example.csv')
            ctd_data.save_to_csv(output_file='path/to/output.csv')
            # This will rename the columns of the CTD dataset and save it to 'path/to/output.csv'.
            # Any existing file with the same name at that location will be overwritten.

        See Also
        --------
        utils.save_to_csv : Utility function used to save the data to a CSV file.

        """
        utils.save_to_csv(self._data, output_file=output_file, null_value=null_value)
