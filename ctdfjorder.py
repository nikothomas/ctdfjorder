# -*- coding: utf-8 -*-
import itertools
import openpyxl
import tensorflow as tf
from keras.api.models import Model
from keras.api.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, GRU
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
import pandas as pd
from pyrsktools import RSK, Geo
from pyrsktools import Region
import gsw
import matplotlib.pyplot as plt
import statsmodels.api
from matplotlib.ticker import ScalarFormatter
import logging
from typing import Generator
from typing import Any
from typing import Dict
import warnings
warnings.filterwarnings('ignore')
import loggersetup

mixed_precision.set_global_policy('float64')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.CRITICAL)
logger = logging.getLogger('ctdfjorder')
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

    Usage
    -----
    To create a CTD object for custom pipelines use the following method:

    >>> my_data = CTD('example.rsk')

    You can then run a CTD command on it like removing non-postive sample rows and then viewing your table

    >>> my_data.remove_non_positive_samples()
    >>> my_data.view_table()

    Notes
    -----
    - Filenames must match a correct date in instances where the mastersheet must be consulted.
    """
    rskName_to_label = {
        'temperature_00': 'temperature',
        'chlorophyll_00': 'chlorophyll',
        'seapressure_00': 'sea_pressure',
        'depth_00': 'depth',
        'salinity_00': 'salinity',
        'speedofsound_00': 'speed_of_sound',
        'specificconductivity_00': 'specific_conductivity',
        'conductivity_00': 'conductivity',
        'pressure_00': 'pressure',
    }
    errorType_to_errorMessage = {
        'no_samples': 'No samples in file',
        'no_location':'No location could be found',
        'density_calcultion': 'Could not calculate density on this dataset',
        'salinity_abs': 'Could not calculate salinity absolute on this dataset',
        'no_mastersheet': 'No mastersheet provided, please provide a mastersheet'
    }
    warningType_to_warningMessage = {

    }
    _data = pl.DataFrame()
    _num_profiles = 0
    _rsk = None
    _filename_time = None
    _filename_full = None
    _filepath = None
    _cwd = None
    _cached_master_sheet = pl.DataFrame()
    _original_profile = None
    master_sheet_path = None
    _NO_MASTER_SHEET_ERROR = ( +
                              '')
    _RSK_ERROR = 'File is corrupted and could not be read'
    _TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
    _FILENAME_GPS_ENDING = '_gps'
    _FILENAME_CM_ENDING = 'cm'
    _rsk_file_flag = False
    _add_unique_id = False

    def __init__(self, ctd_file_path='', cached_master_sheet=pl.DataFrame(), master_sheet_path='',
                 add_unique_id=False):
        """
        Initialize a new CTD object.

        Parameters
        ----------
        ctd_file_path : str
            The file path to the RSK or Castaway file.
        """

        def _process_rsk_profile(lf: pl.DataFrame, geo: Generator[Geo, Any, None], num=1):
            lf = lf.with_columns(pl.lit(self._filename_full + self._FILENAME_CM_ENDING).alias('filename'))
            try:
                profile_geodata = next(geo)
                lf = lf.with_columns(
                    pl.lit(profile_geodata.latitude).alias('latitude'),
                    pl.lit(profile_geodata.longitude).alias('longitude')
                )
                return lf
            except StopIteration:
                logger.debug(f"{self._filename_full} - lacks native location data")
                new_filename = self._filename_full + self._FILENAME_CM_ENDING
                _lat, _long, _ = self._process_master_sheet(lf)
                lf = lf.with_columns(
                    pl.lit(_lat).alias('latitude'),
                    pl.lit(_long).alias('longitude'),
                    pl.lit(new_filename).alias('filename')
                )
                return lf

        def _profile_is_empty_init(data: pl.DataFrame):
            if data.is_empty():
                return True
            return False

        # Function to ensure milliseconds in the timestamp
        def ensure_milliseconds(ts: str):
            if '.' not in ts:
                ts += '.000'
            return ts
        self._filepath = ctd_file_path
        self._cwd = CTD.Utility.get_cwd()
        self.master_sheet_path = master_sheet_path
        self._cached_master_sheet = cached_master_sheet
        self._filename_full = path.basename(ctd_file_path)
        self._add_unique_id = add_unique_id
        # Ruskin init
        if 'rsk' in ctd_file_path:
            self._rsk_file_flag = True
            try:
                self._rsk = RSK(ctd_file_path)
            except OperationalError:
                raise CTDError(filename=self._filename_full, message=self._RSK_ERROR)
            rsk_casts_down = self._rsk.casts(Region.CAST_DOWN)
            for i, endpoints in enumerate(rsk_casts_down):
                rsk_numpy_array = np.array(self._rsk.npsamples(endpoints.start_time, endpoints.end_time))
                for x, timestamp in enumerate(rsk_numpy_array['timestamp']):
                    ts = timestamp.strftime()
                    rsk_numpy_array['timestamp'][x] = ensure_milliseconds(ts)

                profile = pl.DataFrame(rsk_numpy_array).rename(self.rsk_relabel)
                profile = profile.drop_nulls()
                profile = profile.with_columns(
                    pl.col('timestamp').cast(pl.String).str.to_datetime(format='%Y-%m-%d %H:%M:%S%.6f', time_zone='UTC', time_unit='ns').cast(
                        pl.Datetime(time_unit='ns')).dt.replace_time_zone('UTC').alias('timestamp'))
                geodata = self._rsk.geodata(endpoints.start_time, endpoints.end_time)
                profile = profile.with_columns(pl.lit(i).alias('Profile_ID'))
                if profile is not None and profile.select('timestamp').height > 3:
                    profile = _process_rsk_profile(profile, geodata)
                    self._data = pl.concat([profile, self._data], how="diagonal_relaxed")
                    self._num_profiles += 1
                else:
                    logger.warning(f'{self._filename_full} No samples in profile number {self._num_profiles}, dropping from analysis')
            if _profile_is_empty_init(self._data):
                rsk_numpy_array = np.array(self._rsk.npsamples())
                for x, timestamp in enumerate(rsk_numpy_array['timestamp']):
                    ts = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
                    rsk_numpy_array['timestamp'][x] = ensure_milliseconds(ts)
                profile = pl.DataFrame(rsk_numpy_array).rename(self.rsk_relabel)
                profile = profile.drop_nulls()
                profile = profile.with_columns(
                    pl.col('timestamp').cast(pl.String).str.to_datetime(format='%Y-%m-%d %H:%M:%S%.6f', time_zone='UTC', time_unit='ns').dt.replace_time_zone('UTC').cast(
                        pl.Datetime(time_unit='ns')).dt.replace_time_zone('UTC').alias('timestamp'))
                geodata = self._rsk.geodata()
                profile = profile.with_columns(pl.lit(0).alias('Profile_ID'))
                if profile is not None and profile.select('timestamp').height > 3:
                    profile = _process_rsk_profile(profile, geodata)
                    self._data = pl.concat([profile, self._data], how="diagonal_relaxed")
                    self._num_profiles += 1
                else:
                    raise CTDError(message="No samples in file", filename=self._filename_full)
        # Castaway init
        else:
            self._rsk_file_flag = False
            # Calculate the time increments for each row
            with open(ctd_file_path) as file:
                profile = pl.read_csv(file, comment_prefix='%')
            num_rows = profile.height
            if num_rows < 2:
                raise CTDError(message="No samples in file", filename=self._filename_full)
            if 'datetime_utc' in profile.columns and not profile.select('datetime_utc').is_empty():
                if pl.Datetime not in profile.collect_schema().dtypes():
                    profile = profile.with_columns(
                        pl.col('datetime_utc').str.to_datetime(format='%Y-%m-%d %H:%M:%S', time_zone='UTC', time_unit='ns').cast(pl.Datetime).alias(
                            'timestamp'))
                start_time = profile.select(pl.col('timestamp').first()).item()
            else:
                start_time = CTD.Utility.extract_utc_cast_time(ctd_file_path)
            timestamps = [start_time + pd.Timedelta(seconds=0.2 * i) for i in range(num_rows)]
            # Make timestamps timezone-aware
            timestamps = pd.to_datetime(timestamps).tz_localize('UTC')
            profile = profile.with_columns(pl.Series(timestamps).dt.replace_time_zone('UTC').dt.cast_time_unit('ns').rename('timestamp'))
            header_mapping = {
                "Pressure (Decibar)": "sea_pressure",
                "Depth (Meter)": "depth",
                "Temperature (Celsius)": "temperature",
                "Conductivity (MicroSiemens per Centimeter)": "conductivity",
                "Specific conductance (MicroSiemens per Centimeter)": "specific_conductivity",
                "Salinity (Practical Salinity Scale)": "salinity",
                "Sound velocity (Meters per Second)": "speed_of_sound",
                "Density (Kilograms per Cubic Meter)": "density"
            }
            for header, maps_to in header_mapping.items():
                if header in profile.columns:
                    profile = profile.rename({header: maps_to})
            if 'file_id' in profile.collect_schema().names():
                profile = profile.drop('file_id')
            profile = profile.with_columns((pl.col('sea_pressure') + 10.1325).alias('pressure'))
            if profile.is_empty():
                logger.info(f"{self._filename_full} - No samples in file")
            if 'latitude' not in profile.collect_schema().names():
                lat, long = self.Utility.extract_lat_long_castaway(ctd_file_path)
                profile = profile.with_columns(pl.lit(lat, dtype=pl.Float64).alias('latitude'),
                                               pl.lit(long, dtype=pl.Float64).alias('longitude'))
            profile = profile.with_columns(pl.lit(0).alias('Profile_ID'),
                                           pl.lit(self._filename_full).alias('filename'))
            if profile is not None:
                self._data = profile
                self._num_profiles +=1
        if self._data.height < 3:
            logger.info(f"{self._filename_full} - No samples in file")
        try:
            self._data = self._data.with_columns(pl.col('latitude').cast(pl.Float64),
                                pl.col('longitude').cast(pl.Float64))
        except pl.exceptions.InvalidOperationError:
            raise CTDError(message="Location data invalid, even after checking mastersheet", filename=self._filename_full)
        if add_unique_id:
            self._data = self._data.with_columns(pl.lit(None, dtype=pl.String).alias('Unique_ID'))
            for profile_id in range(self._data.n_unique('Profile_ID')):
                profile = self._data.filter(pl.col('Profile_ID') == profile_id)
                _, _, unique_id = self._process_master_sheet(profile, for_id=True)
                profile = profile.with_columns(pl.lit(unique_id, dtype=pl.String).alias('Unique_ID'))
                self._data = self._data.filter(pl.col('Profile_ID') != profile_id)
                self._data.vstack(profile, in_place=True)
        logger.debug(f"{self._filename_full} - new CTD object initialized from file")
    def find_master_sheet_file(self):
        """
        Function to find and the master sheet path. Uses the first xlsx file in the current working directory.
        """
        cwd = CTD.Utility.get_cwd()
        xlsx_files = [file for file in os.listdir(cwd) if file.endswith(".xlsx")]
        if len(xlsx_files) > 0:
            self.master_sheet_path = os.path.abspath(xlsx_files[0])

    def get_df(self, pandas=True):
        """
        Exposes the dataframe of the CTD object for custom processes.

        Parameters
        ----------
        pandas : bool, optional
            If True returns a pandas df, if False returns a polars DataFrame. Defaults to True.

        Returns
        -------
        pl.DataFrame
            The pandas df of the CTD object.
        """
        # Convert each DataFrame to a DataFrame and collect them in a list
        return self._data

    def _is_profile_empty(self, func: str):
        if self._data.is_empty():
            raise CTDError(filename=self._filename_full, message=f"No valid samples in file after running {func}")
        return True

    def remove_upcasts(self):
        """
        Removes upcasts based on the rate of change of pressure over time.
        This function calculates the vertical speed of the system through the water
        using the change of pressure with respect to time. It filters out data
        collected in the air or while stationary at the surface or bottom, and
        separates the downcasts from upcasts.
        """
        for profile_id in range(self._data.n_unique('Profile_ID')):
            profile = self._data.filter(pl.col('Profile_ID') == profile_id)
            profile.filter((pl.col('pressure').diff()) > 0.0)
            self._data = self._data.filter(pl.col('Profile_ID') != profile_id)
            self._data.vstack(profile, in_place=True)
        return self._is_profile_empty(CTD.remove_upcasts.__name__)

    def remove_non_positive_samples(self):
        """
        Iterates through the columns of the CTD data table and removes rows with non-positive values
        for depth, pressure, salinity, absolute salinity, or density.
        """
        for profile_id in range(self._data.n_unique('Profile_ID')):
            profile = self._data.filter(pl.col('Profile_ID') == profile_id)
            cols = list({'depth', 'pressure', 'salinity', 'salinity_abs', 'density'}.intersection(
                profile.collect_schema().names()))
            for col in cols:
                profile = profile.filter(pl.col(col) > 0.0)
            self._data = self._data.filter(pl.col('Profile_ID') != profile_id)
            self._data.vstack(profile, in_place=True)
        return self._is_profile_empty(CTD.remove_non_positive_samples.__name__)

    def remove_invalid_salinity_values(self):
        """
        Removes rows with invalid values (<10) for practical salinity.
        """
        #if len(self._data) < 1:
            #raise CTDError(filename=self._filename_full, message=self._NO_SAMPLES_ERROR)
        for profile_id in range(self._data.n_unique('Profile_ID')):
            profile = self._data.filter(pl.col('Profile_ID') == profile_id)
            profile = profile.filter(pl.col('salinity') > 10)
            self._data = self._data.filter(pl.col('Profile_ID') != profile_id)
            self._data.vstack(profile, in_place=True)
        return self._is_profile_empty(CTD.remove_invalid_salinity_values.__name__)

    def clean(self, method='salinity_ai'):
        """
        Applies data cleaning methods to the specified feature using the selected method.
        Currently supports cleaning practical salinity using 'salinity_diff' or 'salinity_ai' methods.

        Parameters
        ----------
        method : str, optional
            The cleaning method to apply, defaults to 'salinity_ai'.
            Options are 'salinity_diff', 'salinity_ai'.
        """
        for profile_id in range(self._data.n_unique('Profile_ID')):
            profile = self._data.filter(pl.col('Profile_ID') == profile_id)
            if method == 'salinity_diff':
                profile = self._Calculate.calculate_and_drop_salinity_spikes(profile)
            if method == 'salinity_ai':
                profile = self.clean_salinity_ai(profile, profile_id)
            self._data = self._data.filter(pl.col('Profile_ID') != profile_id)
            self._data = pl.concat([profile, self._data], how="diagonal_relaxed")
        return self._is_profile_empty(CTD.clean.__name__)

    def add_absolute_salinity(self):
        """
        Calculates absolute salinity from practical salinity, pressure,
        and geographical coordinates using the TEOS-10 salinity conversion formulas.
        """
        self._data = self._data.with_columns(pl.lit(None, dtype=pl.Float64).alias('salinity_abs'))
        for profile_id in range(self._data.n_unique('Profile_ID')):
            profile = self._data.filter(pl.col('Profile_ID') == profile_id)
            s = profile.select(pl.col('salinity')).to_numpy()
            p = profile.select(pl.col('temperature')).to_numpy()
            lat = profile.select(pl.col('latitude')).to_numpy()
            long = profile.select(pl.col('longitude')).to_numpy()
            salinity_abs_list = gsw.conversions.SA_from_SP(s, p, lat, long)
            salinity_abs = pl.Series(np.array(salinity_abs_list).flatten(), dtype=pl.Float64, strict=False).to_frame(
                'salinity_abs')
            profile = profile.with_columns(salinity_abs)
            self._data = self._data.filter(pl.col('Profile_ID') != profile_id)
            self._data.vstack(profile, in_place=True)
        return self._is_profile_empty(CTD.add_absolute_salinity.__name__)

    def add_density(self):
        """
        Calculates the density using the TEOS-10 equations and adds it as a new column to the CTD
        data table. If absolute salinity is not present, it is calculated first.
        """
        self._data = self._data.with_columns(pl.lit(None, dtype=pl.Float64).alias('density'))
        for profile_id in range(self._data.n_unique('Profile_ID')):
            profile = self._data.filter(pl.col('Profile_ID') == profile_id)
            if 'salinity_abs' not in profile.collect_schema().names():
                self.add_absolute_salinity()
            sa = profile.select(pl.col('salinity_abs')).to_numpy()
            t = profile.select(pl.col('temperature')).to_numpy()
            p = profile.select(pl.col('sea_pressure')).to_numpy()
            density = pl.Series(np.array(gsw.density.rho_t_exact(sa, t, p)).flatten(), dtype=pl.Float64,
                                strict=False).to_frame('density')
            profile = profile.with_columns(density)
            self._data = self._data.filter(pl.col('Profile_ID') != profile_id)
            self._data.vstack(profile, in_place=True)
        return self._is_profile_empty(CTD.add_density.__name__)

    def add_potential_density(self):
        """
        Calculates potential density from the CTD data using the TEOS-10 equations,
        ensuring all data points are within the valid oceanographic funnel.
        """
        self._data = self._data.with_columns(pl.lit(None, dtype=pl.Float64).alias('potential_density'))
        for profile_id in range(self._data.n_unique('Profile_ID')):
            profile = self._data.filter(pl.col('Profile_ID') == profile_id)
            sa = profile.select(pl.col('salinity_abs')).to_numpy()
            t = profile.select(pl.col('temperature')).to_numpy()
            p = profile.select(pl.col('sea_pressure')).to_numpy()
            ct = gsw.CT_from_t(sa, t, p)
            potential_density = pl.Series(np.array(gsw.sigma0(sa, t)).flatten()).to_frame('potential_density')
            profile = profile.with_columns(pl.Series(potential_density))
            self._data = self._data.filter(pl.col('Profile_ID') != profile_id)
            self._data.vstack(profile, in_place=True)
        return self._is_profile_empty(CTD.add_potential_density.__name__)

    def add_surface_salinity_temp_meltwater(self, start=10.0, end=15.0):
        """
        Calculates the surface salinity and meltwater fraction of a CTD profile.
        Reports the mean salinity of the first 2 meters of the profile by finding the minimum salinity, and reports
        meltwater fraction as given by (-0.021406 * surface_salinity + 0.740392) * 100.
        """
        self._data = self._data.with_columns(pl.lit(None, dtype=pl.Float64).alias('surface_salinity'),
                                             pl.lit(None, dtype=pl.Float64).alias('surface_temperature'),
                                             pl.lit(None, dtype=pl.Float64).alias('meltwater_fraction'))
        for profile_id in range(self._data.n_unique('Profile_ID')):
            profile = self._data.filter(pl.col('Profile_ID') == profile_id)
            surface_data = profile.filter(pl.col('pressure') > start,
                                          pl.col('pressure') < end)
            if surface_data.height < 1:
                logger.info(
                    f"{self._filename_full} - First measurment lies below {end} dbar, cannot apply surface measurements")
                return self._is_profile_empty(CTD.add_surface_salinity_temp_meltwater.__name__)
            surface_salinity = np.array(surface_data.select(pl.col('salinity')).to_numpy())
            surface_salinity = surface_salinity.item(0)
            surface_temperature = np.array(surface_data.select(pl.col('temperature').mean()).to_numpy()).item(0)
            mwf = (-0.021406 * surface_salinity + 0.740392) * 100
            profile = profile.with_columns(pl.lit(surface_salinity).alias('surface_salinity'),
                                                          pl.lit(surface_temperature).alias('surface_temperature'),
                                                          pl.lit(mwf).alias('meltwater_fraction'))
            self._data = self._data.filter(pl.col('Profile_ID') != profile_id)
            self._data.vstack(profile, in_place=True)
        return self._is_profile_empty(CTD.add_surface_salinity_temp_meltwater.__name__)

    def add_mean_surface_density(self, start=0.0, end=100.0):
        """
        Calculates the mean surface density from the density values and adds it as a new column
        to the CTD data table.

        Parameters
        ----------
        start : float, optional
            Upper pressure bound, defaults to 0.
        end : float, optional
            Lower pressure bound, default to 1.
        """
        # Filtering data within the specified pressure range
        for profile_id in range(self._data.n_unique('Profile_ID')):
            profile = self._data.filter(pl.col('Profile_ID') == profile_id)
            surface_data = profile.filter(pl.col('pressure') > start,
                                          pl.col('pressure') < end)
            surface_density = surface_data.select(pl.col('density').mean()).item()
            profile = profile.with_columns(pl.lit(surface_density).alias('surface_density'))
            self._data = self._data.filter(pl.col('Profile_ID') != profile_id)
            self._data.vstack(profile, in_place=True)
        return self._is_profile_empty(CTD.add_mean_surface_density.__name__)

    def add_mld(self, reference: int, method="potential_density_avg"):
        """
        Calculates the mixed layer depth using the specified method and reference depth.
        Adds the MLD and the actual reference depth used as new columns to the CTD data table.

        Parameters
        ----------
        reference : int
            The reference depth for MLD calculation.
        method : str
            The MLD calculation method options are "abs_density" or "potential_density_avg"
             (default: "potential_density_avg").
        """
        supported_methods = [
            "abs_density"
            "potential_density_avg"
        ]
        self._data = self._data.with_columns(pl.lit(None, dtype=pl.Float64).alias(f'MLD {reference}'))
        for profile_id in range(self._data.n_unique('Profile_ID')):
            profile = self._data.filter(pl.col('Profile_ID') == profile_id)
            unpack = None
            if method == "abs_density":
                mld = self._Calculate.calculate_mld(profile.select(pl.col('salinity_abs')).to_numpy(),
                                                    profile.select(pl.col('depth')).to_numpy(),
                                                    reference)
            elif method == "potential_density_avg":
                mld = self._Calculate.calculate_mld_average(
                    profile.select(pl.col('potential_density')).to_numpy(),
                    profile.select(pl.col('depth')).to_numpy(),
                    reference)
            else:
                raise CTDError(message=f"add_mld: Invalid method \"{method}\" not in {supported_methods}",
                               filename=self._filename_full)
            profile = profile.with_columns(pl.lit(mld).alias(f'MLD {reference}'))
            self._data = self._data.filter(pl.col('Profile_ID') != profile_id)
            self._data.vstack(profile, in_place=True)
        return self._is_profile_empty(CTD.add_mld.__name__)

    def add_stratification(self, depth_range=20):
        """
        Calculates the SI (stratification index) up to the specified depth.
        Adds the SI to the CTD data table.
        Requires potential density to be calculated first.

        Parameters
        ----------
        depth_range : int
            The depth range to calculate SI.
        """
        self._data = self._data.with_columns(pl.lit(None, dtype=pl.Float64).alias('buoyancy_frequency'),
                                             pl.lit(None, dtype=pl.Float64).alias('p_mid'))
        for profile_id in range(self._data.n_unique('Profile_ID')):
            profile = self._data.filter(pl.col('Profile_ID') == profile_id)
            sa = profile.select(pl.col('salinity_abs')).to_numpy().flatten()
            t = profile.select(pl.col('temperature')).to_numpy().flatten()
            p = profile.select(pl.col('sea_pressure')).to_numpy().flatten()
            lat = profile.select(pl.col('latitude')).to_numpy().flatten()
            ct = gsw.CT_from_t(sa, t, p).flatten()
            n_2, p_mid = gsw.Nsquared(SA=sa, CT=ct, p=p, lat=lat)
            buoyancy_frequency = pl.Series(np.array(n_2).flatten()).extend_constant(None, n=1).to_frame('buoyancy_frequency')
            p_mid = pl.Series(p_mid).extend_constant(None, n=1).to_frame('p_mid')
            profile = profile.with_columns(pl.Series(buoyancy_frequency),
                                           pl.Series(p_mid))
            self._data = self._data.filter(pl.col('Profile_ID') != profile_id)
            self._data.vstack(profile, in_place=True)
        return self._is_profile_empty(CTD.add_surface_salinity_temp_meltwater.__name__)

    def plot(self, measurement, plot_type='scatter'):
        """
        Generates a plot of depth vs. specified measurement (salinity, density, temperature).
        Adds horizontal lines indicating the mixed layer depth (MLD) if present.
        Allows for both scatter and line plot types.
        Saves the plot as an image file.

        Parameters
        ----------
        measurement : str
            Options are 'salinity', 'density', 'potential_density, or 'temperature'.
        plot_type : str
            Options are 'scatter' or 'line'.
        """
        plt.rcParams.update({'font.size': 16})
        plot_folder = os.path.join(self._cwd, "plots")
        os.makedirs(plot_folder, exist_ok=True)
        for profile_id in range(self._data.n_unique('Profile_ID')):
            profile = self._data.filter(pl.col('Profile_ID') == profile_id)
            fig, ax1 = plt.subplots(figsize=(18, 18))
            ax1.invert_yaxis()
            ax1.set_ylim([profile.select(pl.col('depth')).max().item(), 0])
            color_map = {'salinity': 'tab:blue',
                         'density': 'tab:red',
                         'potential_density': 'tab:red',
                         'temperature': 'tab:blue'}
            label_map = {'salinity': 'Practical Salinity (PSU)',
                         'density': 'Density (kg/m^3)',
                         'potential_density': 'Potential Density (kg/m^3)',
                         'temperature': 'Temperature (°C)'}
            if plot_type == 'line':
                lowess = statsmodels.api.nonparametric.lowess
                y, x = zip(*lowess(profile.select(pl.col(f"{measurement}")).to_numpy(),
                                   profile.select(pl.col('depth')).to_numpy(), frac=0.1))
            else:
                x, y = profile.select(pl.col(f"{measurement}")).to_numpy(), profile.select(
                    pl.col('depth')).to_numpy()
            ax1.plot(x, y, color=color_map[measurement], label=label_map[measurement]) if plot_type == 'line' \
                else ax1.scatter(x, y, color=color_map[measurement], label=label_map[measurement])
            ax1.set_xlabel(label_map[measurement], color=color_map[measurement])
            ax1.set_ylabel('Depth (m)')
            ax1.tick_params(axis='x', labelcolor=color_map[measurement])
            mld_col = None
            mld = profile.select(pl.col(r"^.*MLD.*$").first()).item()
            if mld is not None:
                # Plot MLD line
                ax1.axhline(y=mld, color='green', linestyle='--', linewidth=2, label=f'{mld}')
                ax1.text(0.95, mld, f'{mld}', va='center', ha='right',
                         backgroundcolor='white', color='green', transform=ax1.get_yaxis_transform())
            plt.title(f"{self._filename_full} \n Profile {profile_id} \n Depth vs. {label_map[measurement]}\n MLD {mld}")
            ax1.grid(True)
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
            plot_path = os.path.join(plot_folder,
                                     f"{self._filename_full}_{profile_id}_depth_{measurement}_{plot_type}_plot.png")
            plt.savefig(plot_path)
            plt.close(fig)

    def _process_master_sheet(self, lf=pl.DataFrame(), for_id=False):
        """
        Extracts the date and time components from the filename and compares them with the data
        in the master sheet. Calculates the absolute differences between the dates and times to
        find the closest match. Returns the estimated latitude, longitude, and updated filename
        based on the closest match.

        Parameters
        ----------
        lf : DataFrame
            CTD profile

        for_id : bool
            Flag for logging purposes, indicates if we processed for location or id.

        Returns
        -------
        tuple
            A tuple containing the estimated latitude, longitude, and updated filename.
        """
        if self._cached_master_sheet.height < 1:
            self._cached_master_sheet = pl.read_excel(self.master_sheet_path, infer_schema_length=None,
                                                      schema_overrides={"time_local": pl.String,
                                                                        "date_local": pl.String,
                                                                        "time (UTC)": pl.String,
                                                                        "date (UTC)": pl.String})
            self._cached_master_sheet = CTD.Utility.load_master_sheet(self._cached_master_sheet)
        if 'timestamp' not in lf.collect_schema().names():
            raise CTDError(message="No timestamp in file, could not get location", filename=self._filename_full)
        if 'datetime' not in self._cached_master_sheet.collect_schema().names():
            raise CTDError(message="No timestamp in mastersheet, could not get location", filename=self._filename_full)
        timestamp_highest = lf.select(pl.last("timestamp").dt.datetime()).item()
        closest_row_overall = self._cached_master_sheet.filter(
            abs(self._cached_master_sheet['datetime'] - timestamp_highest) == abs(
                self._cached_master_sheet['datetime'] - timestamp_highest).min())
        latitude = closest_row_overall.select(pl.col('latitude').first()).item()
        longitude = closest_row_overall.select(pl.col('longitude').first()).item()
        distance = closest_row_overall.select(pl.col('datetime').first()).item() - timestamp_highest
        unique_id = closest_row_overall.select(pl.col('UNIQUE ID CODE ').first()).item()
        # Extract days, hours, and minutes from the time difference
        days = abs(distance.days)
        hours, remainder = divmod(distance.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        if for_id:
            message = f"{self._filename_full} - Guessed Location : Matched to unique ID '{unique_id}' with distance {days} days and time difference {hours}:{minutes}"
            if abs(days) > 2:
                logger.warning(message)
            else:
                logger.info(message)
        else:
            message = f"{self._filename_full} - Guessed Unique ID : Matched to unique ID '{unique_id}' with distance {days} days and time difference {hours}:{minutes}"
            if abs(days) > 2:
                logger.warning(message)
            else:
                logger.info(message)
        return latitude, longitude, unique_id

    def clean_salinity_ai(self, profile: pl.DataFrame, profile_id: int):
        """
        Cleans salinity using a GRU ML model.
        Warning: This process will bin the data every 0.5 dbar of pressure.

        Returns
        -------
        DataFrame
            CTD data with clean salinity values.
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
            return mae + 6.0 * tf.reduce_mean(penalties)  # Adjust weighting of penalty as needed

        def build_gru(input_shape):
            """
            GRU architecture.
            """
            inputs = Input(shape=(input_shape[0], input_shape[1]), name='inputs')
            gru1 = GRU(16, activation='tanh',
                       return_sequences=True)(inputs)
            output = TimeDistributed(Dense(input_shape[1], activation='linear'), name='output')(
                gru1)

            gru = Model(inputs, output)
            optimizer = Adam(learning_rate=0.01)
            gru.compile(optimizer=optimizer, loss=loss_function)  # Specify your loss function here
            return gru

        def plot_original_data(salinity, depths, filename):
            plt.figure(figsize=(10, 6))
            plt.scatter(salinity, -depths, alpha=0.6)
            plt.xlabel('Salinity (PSU)')
            plt.ylabel('Depth (m)')
            plt.title(f'Original Salinity vs. Depth - {filename}')
            plt.grid(True)
            plt.savefig(os.path.join('plots', f'{filename}_original.png'))
            xlim = plt.xlim()
            ylim = plt.ylim()
            # plt.show()
            plt.close()
            return xlim, ylim

        def plot_predicted_data(salinity, depths, xlim, ylim, filename):
            plt.figure(figsize=(10, 6))
            plt.scatter(salinity, -depths, alpha=0.6, color='red')
            title = f'Predicted Salinity vs. Depth - {filename}'
            plt.title(title)
            plt.xlabel('Salinity (PSU)')
            plt.ylabel('Depth (m)')
            plt.grid(True)
            plt.xlim((xlim[0], xlim[1]))
            plt.ylim((ylim[0], ylim[1]))
            plt.savefig(os.path.join('plots', f'{filename}_predicted.png'))
            # plt.show()
            plt.close()

        def run_gru(data: pl.DataFrame, show_plots=True):
            """
            Runs the GRU.

            Parameters
            ----------
            data : DataFrame
                CTD dataframe
            show_plots : bool
                False to not save plots, True to save plots.
            Returns
            -------
            DataFrame
                CTD data with clean salinity values.
            """
            # Binning data
            pandas_df = pd.DataFrame(data.filter(pl.col('depth') > 1).to_pandas())
            original_depth = pandas_df['depth']
            original_salinity = pandas_df['salinity']
            pandas_df.loc[:, 'pressure_bin'] = pandas_df['pressure'] // 0.5 * 0.5
            # Define the desired columns and their aggregation functions
            column_agg_dict = {
                "temperature": "mean",
                "chlorophyll": "mean",
                "sea_pressure": "mean",
                "depth": "mean",
                "salinity": "median",
                "speed_of_sound": "mean",
                "specific_conductivity": "mean",
                "conductivity": "mean",
                "density": "mean",
                "potential_density": "mean",
                "salinity_abs": "mean",
                "timestamp": "first",
                "longitude": "first",
                "latitude": "first",
                "Unique_ID": "first",
                "filename": "first",
                "Profile_ID": "first"
            }
            # Check which columns are present in the DataFrame
            available_columns = {col: agg_func for col, agg_func in column_agg_dict.items() if col in data.columns}
            # Group by and aggregate based on the available columns
            data_binned = pandas_df.groupby('pressure_bin').agg(available_columns)
            # Rename the Pressure_Bin column if it exists
            data_binned = data_binned.reset_index()
            data_binned = data_binned.rename(columns={'pressure_bin': 'pressure'}).reindex()
            data = pl.DataFrame(data_binned)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            if data.height < 2:
                raise CTDError(message="Not enough values to run the autoencoder on this data",
                               filename=self._filename_full)
            logger.debug(f"{self._filename_full} - About to scale salinity")
            df = np.array(data.select(pl.col('salinity')).to_numpy())
            scaled_sequence = scaler.fit_transform(df)
            logger.debug(f"{self._filename_full} - Salinity scaled")
            scaled_seq = np.expand_dims(scaled_sequence, axis=0)
            min_pres = data.select(pl.min('depth')).item()
            max_pres = data.select(pl.max('depth')).item()
            # Calculate ranges
            pres_range = max_pres - min_pres
            epochs = int(pres_range * 16)
            # Build autoencoder and predict on CTD data
            autoencoder = build_gru(scaled_seq.shape[1:])
            logger.debug(f"{self._filename_full} - GRU model built")
            autoencoder.fit(scaled_seq, scaled_seq, epochs=epochs, verbose=0, batch_size=4)
            X_pred = autoencoder.predict(scaled_seq, verbose=None)
            predicted_seq = np.array(scaler.inverse_transform(X_pred[0])).flatten()
            if show_plots == True:
                xlim, ylim = plot_original_data(original_salinity, original_depth,
                                                self._filename_full + str(profile_id))
                plot_predicted_data(salinity=predicted_seq, depths=data_binned['depth'],
                                    filename=self._filename_full + str(profile_id), xlim=xlim, ylim=ylim)
            data = data.with_columns(pl.Series(predicted_seq, dtype=pl.Float64).alias('salinity'))
            return data

        return run_gru(profile)

    def save_to_csv(self, output_file: str):
        """
        Renames the columns of the CTD data table based on a predefined mapping and saves the
        data to the specified CSV file. If the file already exists, the data is appended to it.

        Parameters
        ----------
        output_file : str
            The output CSV file path.
        """
        for profile_id, profile in self._data:
            self.Utility.save_to_csv(profile, output_file=output_file)

    class Utility:
        """
        Utility
        --------
        Utility class for CTD data processing.
        """

        @staticmethod
        def save_to_csv(data: pl.DataFrame, output_file: str):
            """
            Renames the columns of the CTD data table based on a predefined mapping and saves the
            data to the specified CSV file. If the file already exists, the data is appended to it.

            Parameters
            ----------
            data : DataFrame
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
                    "buoyancy_frequency": "Brunt-Vaisala_Frequency_Squared"
                }
                if label in data_label_mapping.keys():
                    return data_label_mapping[label]
                else:
                    return label

            data = data.rename(relabel_ctd_data)
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
            with open(ctd_file_path, 'r') as file:
                for line in file:
                    if line.startswith('% Cast time (UTC)'):
                        # Extract the UTC cast time after the comma
                        parts = line.strip().split(',')
                        if len(parts) > 1:
                            cast_time_str = parts[1].strip()  # Take only the datetime part
                            # Convert the datetime string to ISO format if possible
                            cast_time_utc = datetime.strptime(cast_time_str, "%Y-%m-%d %H:%M:%S")
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
            with open(ctd_file_path, 'r') as file:
                for line in file:
                    if line.startswith('% Start latitude'):
                        # Assume the format is '% description, latitude'
                        parts = line.strip().split(',')
                        if len(parts) > 1:
                            latitude = parts[1].strip()
                            if latitude == '':
                                latitude = np.nan
                    if line.startswith('% Start longitude'):
                        # Assume the format is '% description, longitude'
                        parts = line.strip().split(',')
                        if len(parts) > 1:
                            longitude = parts[1].strip()
                            if longitude == '':
                                longitude = np.nan
            return latitude, longitude

        @staticmethod
        def load_master_sheet(df: pl.DataFrame):
            df = df.drop_nulls('time_local')
            df = df.filter(~pl.col('time_local').eq('-999'))
            df = df.filter(~pl.col('time_local').eq('NA'))
            df = df.filter(~pl.col('date_local').eq('NA'))
            df = df.with_columns(
                pl.col('date_local').str.strptime(format='%Y-%m-%d %H:%M:%S%.3f', dtype=pl.Date, strict=False))
            df = df.drop_nulls('date_local')
            df = df.with_columns(
                pl.col('time_local').str.strptime(format='%Y-%m-%d %H:%M:%S%.3f', dtype=pl.Time, strict=False))
            df = df.drop_nulls('time_local')
            df = df.with_columns(
                (pl.col('date_local').dt.combine(pl.col('time_local').cast(pl.Time))).alias('datetime'))
            return df

        @staticmethod
        def get_cwd():
            working_directory_path = None
            # determine if application is a script file or frozen exe
            if getattr(sys, 'frozen', False):
                working_directory_path = os.path.dirname(sys.executable)
            elif __file__:
                working_directory_path = os.getcwd()
            else:
                working_directory_path = os.getcwd()
            return working_directory_path

    class _Calculate:
        """
        Calculate
        ----------

        Class for CTD data calculations.
        """

        @staticmethod
        def calculate_and_drop_salinity_spikes(data: pl.DataFrame):
            """
            Calculates and removes salinity spikes from the CTD data based on predefined thresholds for acceptable
            changes in salinity with depth.

            Parameters
            ----------
            data : DataFrame
                DataFrame containing depth and salinity data
            """
            min_depth = data.select(pl.loc('depth').min()).item()
            max_depth = data.select(pl.loc('depth').max()).item()
            if abs(min_depth - max_depth) < 5.0:
                logger.info("Insufficient depth range to clean salinity spikes")
                return
            acceptable_delta_salinity_per_depth = [
                (0.0005, 0.001),
                (0.005, 0.01),
                (0.05, 0.1),
                (0.5, 1),
                (1, 2),
            ]
            # Collect depth and salinity arrays
            depth_salinity = data.select(['depth', 'salinity'])
            depths = data.select(pl.loc('depth')).to_numpy()
            salinities = data.select(pl.loc('salinity')).to_numpy()
            # Detect and mark spikes for removal
            spike_mask = np.zeros(len(salinities), dtype=bool)
            for i in range(1, len(salinities)):
                depth_diff = depths[i] - depths[i - 1]
                salinity_diff = abs(salinities[i] - salinities[i - 1])
                for delta, range_depth in acceptable_delta_salinity_per_depth:
                    if depth_diff < range_depth:
                        if salinity_diff / depth_diff > delta:
                            spike_mask[i] = True
                            break

            # Remove spikes
            clean_data = depth_salinity[~spike_mask]
            depths = pl.Series(clean_data['depths'], name='depths')
            salinity = pl.Series(clean_data['salinity'], name='salinity')
            data.with_columns(depths, salinity)
            return data

        @staticmethod
        def calculate_surface_temperature(ctd_array, start=10.0, end=12.0):
            """
            Calculates the surface temperature of a CTD profile. Reports the mean temperature of the first 2 meters
            of the profile.

            Parameters
            ----------
            ctd_array : DataFrame
                DataFrame containing temperature and depth data.
            start : float
                Minimum pressure of the surface, defaults to 10.0.
            end : float
                Maximum pressure of the surface, defaults to 12.0.
            Returns
            -------
            float
                Returns the mean surface temperature.
            """
            # Filtering data within the specified pressure range
            surface_data = ctd_array[(ctd_array['pressure'] >= start) & (ctd_array['pressure'] <= end)]
            surface_temperature = surface_data['temperature_00'].mean()

            return surface_temperature

        @staticmethod
        def calculate_mld(densities, depths, reference_depth, delta=0.5):
            """
            Calculates the mixed layer depth (MLD) using the density threshold method.
            MLD is the depth at which the density exceeds the reference density
            by a predefined amount delta, which defaults to (0.05 kg/m³).

            Parameters
            ----------
            densities : Series
                Series of densities
            depths : Series
                Series of depths corresponding to densities
            reference_depth : float
                The depth at which to anchor the reference density
            delta : float, optional
                The difference in density which would indicate the MLD, defaults to 0.05 kg/m³.

            Returns
            -------
            tuple
                A tuple containing the calculated MLD and the reference depth used to calculate MLD.
            """
            reference_depth = int(reference_depth)
            if len(depths) == 0 or len(densities) == 0:
                return None
            sorted_data = sorted(zip(depths, densities), key=lambda x: x[0])
            sorted_depths, sorted_densities = zip(*sorted_data)
            # Determine reference density
            reference_density = None
            for i, depth in enumerate(sorted_depths):
                if depth >= reference_depth:
                    if depth == reference_depth:
                        reference_density = sorted_densities[i]
                        reference_depth = sorted_depths[i]
                    else:
                        # Linear interpolation
                        try:
                            reference_density = sorted_densities[i - 1] + (
                                    (sorted_densities[i] - sorted_densities[i - 1]) * (
                                    (reference_depth - sorted_depths[i - 1]) /
                                    (sorted_depths[i] - sorted_depths[i - 1])))
                        except:
                            raise CTDError("Insufficient depth range to calculate MLD",
                                           "Unknown")
                    break
            if reference_density is None:
                return None
            # Find the depth where density exceeds the reference density by more than 0.05 kg/m³
            for depth, density in zip(sorted_depths, sorted_densities):
                if density > reference_density + delta and depth >= reference_depth:
                    return depth, reference_depth
            return None  # If no depth meets the criterion

        @staticmethod
        def calculate_mld_average(densities: np.array, depths: np.array, reference_depth=20, delta=0.05):
            """
            Calculates the mixed layer depth (MLD) using the density threshold method.
            Reference density calculated as the average density up to the reference depth.
            MLD is the depth at which the density exceeds the reference density
            by a predefined amount delta, which defaults to (0.05 kg/m³).

            Parameters
            ----------
            densities : Series
                Series of densities
            depths : Series
                Series of depths corresponding to densities
            reference_depth : float
                The depth at which to anchor the reference density, defaults to 20.
            delta : float, optional
                The difference in density which would indicate the MLD, defaults to 0.03 kg/m³.

            Returns
            -------
            tuple
                A tuple containing the calculated MLD and the reference depth used to calculate MLD.
            """
            df = pl.DataFrame({'depth': depths.flatten(), 'density': densities.flatten()})
            df_filtered = df.filter(pl.col('depth') <= reference_depth)
            reference_density = df_filtered.select(pl.col('density').mean()).item()
            # Find the depth where density exceeds the average reference density by more than delta kg/m³
            df_filtered = df.filter(pl.col('density') >= reference_density + delta)
            mld = df_filtered.select(pl.col('depth').first()).item()
            return mld  # If no depth meets the criterion after the reference depth

        @staticmethod
        def calculate_mean_surface_density(df, range_):
            """
            Calculates the mean surface density from the CTD data, for a specified range or the entire dataset if the range is larger.

            Parameters
            ----------
            df : DataFrame
                DataFrame containing density data.
            range_ : tuple or int
                Tuple indicating the (start, end) indices for the range of rows to be included in the calculation,
                or an integer indicating the number of rows from the start.

            Returns
            -------
            float, None
                Mean density value of the specified sample or None if unable to calculate.
            """
            min_depth = df.select(pl.col('depth')).min().item()
            max_depth = df.select(pl.col('depth')).min().item()

            if isinstance(range_, tuple):
                start, end = range_

                # Adjust 'start' to ensure it is within the valid range
                start = max(start, min_depth)

                # Adjust 'end' to ensure it does not exceed the maximum depth value
                end = min(end, max_depth)

                # Ensure start is less than end
                if start <= end:
                    return df.filter(pl.loc('depth') < end,
                                     pl.loc('depth') > start).select('density').mean().item()
                else:
                    return None

            elif isinstance(range_, int):
                # Use 'range_' as the number of rows from the start, adjust if it exceeds the DataFrame length
                return df.filter(pl.loc('depth') < range_).select('density').mean().item()

            else:
                raise ValueError("Invalid range type. Must be tuple or int.")

        @staticmethod
        def get_cwd():
            # determine if application is a script file or frozen exe
            if getattr(sys, 'frozen', False):
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
    filename: input dataset which caused the error
    message: message -- explanation of the error
    """

    def __init__(self, message, filename=None):
        super().__init__(filename + ' - ' + message)
