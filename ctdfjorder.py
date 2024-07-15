# -*- coding: utf-8 -*-
import itertools
import os
import sys
from datetime import datetime
from sqlite3 import OperationalError
from typing import Sized
from os import path

import numpy as np
import pandas
import pandas as pd
from pandas import DataFrame
from pandas.errors import MergeError
from pyrsktools import RSK
from pyrsktools import Region
import gsw
import matplotlib.pyplot as plt
import statsmodels.api
from matplotlib.ticker import ScalarFormatter
from tabulate import tabulate

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, GRU
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

import logging

# Tensorflow, logging, matplotlib setup
mixed_precision.set_global_policy('float32')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logger = logging.getLogger('ctdfjorder')
logger.propagate = 0


def drop_dups(df):
    return df.loc[:, ~df.columns.duplicated()].copy().reindex()


def merge_dataframes(df_list, on=None, how="inner"):
    """
    Merge multiple dataframes in a list into a single dataframe.

    Parameters:
    - df_list: list of pandas DataFrame objects to merge.
    - on: string or list of strings, column(s) to join on, common to all dataframes.
    - how: type of merge to perform. Options include 'left', 'right', 'outer', 'inner' (default: 'inner').

    Returns:
    - DataFrame: merged dataframe.
    """
    # Start with the first dataframe in the list
    if not df_list:
        raise ValueError("The list of dataframes is empty")

    merged_df = df_list[0]

    # Iterate over the remaining dataframes in the list and merge them one by one
    for df in df_list[1:]:
        merged_df = pd.merge(merged_df, df, on=on, how=how)

    return merged_df

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

    >>> my_data = CTD("example.rsk")

    You can then run a CTD command on it like removing non-postive sample rows and then viewing your table

    >>> my_data.remove_non_positive_samples()
    >>> my_data.view_table()

    Notes
    -----
    - Filenames must match a correct date in instances where the mastersheet must be consulted.
    """
    _ctd_array = pandas.DataFrame()
    _rsk = None
    _filename_time = None
    _filename_full = None
    _filepath = None
    _cwd = None
    _cached_master_sheet = pandas.DataFrame()
    _original_profile = None
    master_sheet_path = None
    _NO_SAMPLES_ERROR = "No samples in file"
    _NO_LOCATION_ERROR = "No location could be found"
    _DENSITY_CALCULATION_ERROR = "Could not calculate density on this dataset"
    _SALINITYABS_CALCULATION_ERROR = "Could not calculate salinity on this dataset"
    _DATA_CLEANING_ERROR = "No data remains after data cleaning, reverting to previous CTD"
    _REMOVE_NEGATIVES_ERROR = "No data remains after removing non-positive samples"
    _MLD_ERROR = "No data remains after calculating MLD"
    _STRATIFICATION_ERROR = "No data remains after calculating stratification index"
    _NO_MASTER_SHEET_ERROR = ("No mastersheet provided. Cannot geolocate file, try using find_master_sheet_file " +
                              "before this step.")
    _RSK_ERROR = "File is corrupted and could not be read"
    _rsk_file_flag = False
    unique_id = None

    def __init__(self, ctdfilepath, cached_master_sheet=pandas.DataFrame(), master_sheet_path=None,
                 add_unique_id=False):
        """
        Initialize a new CTD object.

        Parameters
        ----------
        ctdfilepath : str
            The file path to the RSK or Castaway file.
        """
        self._filepath = ctdfilepath
        self._cwd = CTD.Utility.get_cwd()
        self.master_sheet_path = master_sheet_path
        self._cached_master_sheet = cached_master_sheet
        self._filename_full = path.basename(ctdfilepath)
        # Ruskin init
        if "rsk" in ctdfilepath:
            self._rsk_file_flag = True
            try:
                self._rsk = RSK(ctdfilepath)
            except OperationalError:
                raise CTDError(filename=self._filename_full, message=self._RSK_ERROR)
            profiles_df_list = []
            rsk_profiles_list = self._rsk.casts(Region.CAST_DOWN)
            for i, downcast_profile_endpoints in enumerate(rsk_profiles_list):
                tempdf = pd.DataFrame(np.array(self._rsk.npsamples(downcast_profile_endpoints.start_time,
                                                                   downcast_profile_endpoints.end_time)))
                tempdf_geodata = self._rsk.geodata(downcast_profile_endpoints.start_time,
                                                   downcast_profile_endpoints.end_time)
                if not pd.api.types.is_datetime64_any_dtype(tempdf['timestamp']):
                    tempdf['timestamp'] = pd.to_datetime(tempdf['timestamp'])
                tempdf = tempdf.assign(filename=self._filename_full + f"_profile_{i + 1}" + "_gps")
                try:
                    tempdf = tempdf.assign(latitude=next(tempdf_geodata).latitude)
                    tempdf = tempdf.assign(longitude=next(tempdf_geodata).longitude)
                except StopIteration:
                    logger.debug(f"{self._filename_full} - lacks native location data")
                    lat, long, self.unique_id = self._process_master_sheet(tempdf)
                    tempdf = tempdf.assign(latitude=lat)
                    tempdf = tempdf.assign(longitude=long)
                    tempdf['filename'] += "cm"
                tempdf = drop_dups(tempdf)
                profiles_df_list.append(tempdf)
            if len(profiles_df_list) > 1:
                logger.debug(f"{self._filename_full} - contains {len(profiles_df_list)} profile/s")
                ctd_array = merge_dataframes(profiles_df_list)
            elif len(profiles_df_list) == 1:
                logger.debug(f"{self._filename_full} - contains 1 profile/s")
                ctd_array = profiles_df_list[0]
            else:
                logger.debug(f"{self._filename_full} - contains no profile object, "
                             f"ctdfjorder will treat all data as one sampling event")
                tempdf_geodata = self._rsk.geodata()
                tempdf = pd.DataFrame(np.array(self._rsk.npsamples()))
                tempdf = tempdf.assign(filename=self._filename_full + f"_profile_1" + "_gps")
                if not pd.api.types.is_datetime64_any_dtype(tempdf['timestamp']):
                    tempdf['timestamp'] = pd.to_datetime(tempdf['timestamp'])
                try:
                    tempdf = tempdf.assign(latitude=next(tempdf_geodata).latitude)
                    tempdf = tempdf.assign(longitude=next(tempdf_geodata).longitude)
                except StopIteration:
                    logger.debug(f"{self._filename_full} - lacks native location data")
                    lat, long, self.unique_id = self._process_master_sheet(tempdf)
                    tempdf = tempdf.assign(latitude=lat)
                    tempdf = tempdf.assign(longitude=long)
                    tempdf['filename'] += "cm"
                ctd_array = tempdf.copy()
            if len(ctd_array) < 1:
                raise CTDError(message="No valid samples present in file", filename=self._filename_full)
            self._ctd_array = ctd_array.copy()
            self._original_profile = self._ctd_array.copy()
        # Castaway init
        else:
            self._rsk_file_flag = False
            # Calculate the time increments for each row
            pd_csv_castaway = pd.read_csv(ctdfilepath, comment='%')
            # Convert UTC cast time to datetime and create timestamp column
            try:
                start_time = pd.to_datetime(pd_csv_castaway.datetime_utc[0])
            except KeyError:
                start_time = pd.to_datetime(CTD.Utility.extract_utc_cast_time(ctdfilepath))
            except AttributeError:
                start_time = pd.to_datetime(CTD.Utility.extract_utc_cast_time(ctdfilepath))
            timestamps = [start_time + pd.Timedelta(seconds=0.2 * i) for i in range(len(pd_csv_castaway))]
            # Make timestamps timezone-aware
            timestamps = pd.to_datetime(timestamps).tz_localize('UTC')
            header_mapping = {
                "Pressure (Decibar)": "seapressure_00",
                "Depth (Meter)": "depth_00",
                "Temperature (Celsius)": "temperature_00",
                "Conductivity (MicroSiemens per Centimeter)": "conductivity_00",
                "Specific conductance (MicroSiemens per Centimeter)": "specificconductivity_00",
                "Salinity (Practical Salinity Scale)": "salinity_00",
                "Sound velocity (Meters per Second)": "speedofsound_00",
                "latitude": "latitude",
                "longitude": "longitude"
            }
            pd_csv_castaway.rename(columns=header_mapping, inplace=True)
            pd_csv_castaway.drop('file_id', axis=1, inplace=True)
            pd_csv_castaway.drop('datetime_utc', axis=1, inplace=True)
            # Insert the timestamp column as the first column
            pd_csv_castaway['pressure_00'] = pd_csv_castaway['seapressure_00'].copy()
            pd_csv_castaway['pressure_00'] += 10.1325
            pd_csv_castaway['conductivity_00'] /= 1000
            pd_csv_castaway.insert(0, 'timestamp', timestamps)
            pd_csv_castaway = pd_csv_castaway.assign(filename=self._filename_full + "_profile0" + "_gps")
            pd_csv_castaway.drop('Density (Kilograms per Cubic Meter)', axis=1, inplace=True)
            self._ctd_array = pd_csv_castaway
            self._original_profile = pd_csv_castaway.copy()
        if add_unique_id:
            profiles = []
            for filename, profile in self._ctd_array.groupby('filename'):
                if 'Unique_ID' in self._ctd_array.columns:
                    profiles.append(profile)
                    continue
                if self.Utility.no_values_in_object(profile):
                    continue
                if not self.unique_id:
                    _, _, self.unique_id = self._process_master_sheet(profile, for_id=True)
                profile.assign(Unique_ID=self.unique_id)
                profiles.append(profile)
            if len(profiles) < 1:
                raise CTDError(message="No valid samples present in file", filename=self._filename_full)
            self._ctd_array = merge_dataframes(profiles)
        logger.debug(f"{self._filename_full} - new CTD object initialized from file")

    def find_master_sheet_file(self):
        """
        Function to find and the master sheet path. Uses the first xlsx file in the current working directory.
        """
        cwd = CTD.Utility.get_cwd()
        xlsx_files = [file for file in os.listdir(cwd) if file.endswith(".xlsx")]
        if len(xlsx_files) > 0:
            self.master_sheet_path = os.path.abspath(xlsx_files[0])

    def view_table(self):
        """
        Print the CTD data table.
        """
        print(tabulate(self._ctd_array, headers='keys', tablefmt='psql'))

    def get_pandas_df(self, copy=True):
        """
        Exposes the dataframe of the CTD object for custom processes.

        Parameters
        ----------
        copy : bool, optional
            If True returns a copy, if False returns the actual DataFrame internal to the CTD object. Defaults to True.

        Returns
        -------
        DataFrame
            The pandas df of the CTD object.
        """
        return self._ctd_array.copy() if copy is True else self._ctd_array

    def remove_upcasts(self):
        """
        Removes upcasts based on the rate of change of pressure over time.
        This function calculates the vertical speed of the system through the water
        using the derivative of pressure with respect to time. It filters out data
        collected in the air or while stationary at the surface or bottom, and
        separates the downcasts from upcasts.
        """
        profiles = []
        for filename, profile in self._ctd_array.groupby('filename'):
            # Ensure 'timestamp' is a datetime object for correct differentiation
            if not pd.api.types.is_datetime64_any_dtype(profile['timestamp']):
                profile['timestamp'] = pd.to_datetime(profile['timestamp'])
            # Calculate the time differences in seconds
            time_diffs = profile['timestamp'].diff().dt.total_seconds()
            # Calculate the rate of change of pressure over time (dP/dt)
            pressure_diffs = profile['pressure_00'].diff()
            vertical_speed = pressure_diffs / time_diffs
            # Add vertical speed to the DataFrame
            profile['vertical_speed'] = vertical_speed
            # Filter out data where the rate of change of pressure indicates upcasting or stationary periods
            # Assuming positive vertical speed indicates downcasting (i.e., increasing pressure)
            profile_filtered = profile[
                (profile['vertical_speed'] > 0) & (~profile['vertical_speed'].isna())]
            profiles.append(profile_filtered)
        self._ctd_array = merge_dataframes(profiles)
    def remove_non_positive_samples(self):
        """
        Iterates through the columns of the CTD data table and removes rows with non-positive values
        for depth, pressure, salinity, absolute salinity, or density.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._NO_SAMPLES_ERROR)
        for column in self._ctd_array.columns:
            match column:
                case 'depth_00':
                    self._ctd_array = self.Utility.remove_rows_with_negative_depth(self._ctd_array)
                case 'pressure_00':
                    self._ctd_array = self.Utility.remove_rows_with_negative_pressure(self._ctd_array)
                case 'salinity_00':
                    self._ctd_array = self.Utility.remove_rows_with_negative_salinity(self._ctd_array)
                case 'salinityabs':
                    self._ctd_array = self.Utility.remove_rows_with_negative_salinityabs(self._ctd_array)
                case 'density':
                    self._ctd_array = self.Utility.remove_rows_with_negative_density(self._ctd_array)
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._REMOVE_NEGATIVES_ERROR)

    def remove_invalid_salinity_values(self):
        """
        Removes rows with invalid values (<10) for practical salinity.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._NO_SAMPLES_ERROR)
        for column in self._ctd_array.columns:
            match column:
                case 'salinity_00':
                    self._ctd_array = self.Utility.remove_rows_with_invalid_salinity(self._ctd_array)
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._REMOVE_NEGATIVES_ERROR)

    def bin_average(self, bin_size=0.5):
        """
        Bins by pressure and averages numeric columns.

        Parameters
        ----------
        bin_size: float
            Size of pressure bins, defaults to 0.5.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._NO_SAMPLES_ERROR)
        # Check if 'Pressure_Bin' already exists and drop it if it does
        if 'Pressure_Bin' in self._ctd_array.columns:
            self._ctd_array.drop(columns=['Pressure_Bin'], inplace=True)
        profiles = []
        for filename, profile in self._ctd_array.groupby('filename'):
            # Binning data using the specified bin size
            profile['Pressure_Bin'] = (profile['pressure_00'] // bin_size) * bin_size
            # Identifying numeric and non-numeric columns
            numeric_cols = profile.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols.remove('pressure_00')  # Remove the original pressure column if it's still in the list
            non_numeric_cols = profile.select_dtypes(exclude=[np.number]).columns.tolist()
            # Creating aggregation dictionary
            agg_dict = {col: 'mean' for col in numeric_cols}
            agg_dict.update({col: 'first' for col in non_numeric_cols})
            # Aggregating data by the new pressure bins
            data_binned = profile.groupby('Pressure_Bin').agg(agg_dict).reindex()
            # Cleaning up DataFrame, resetting the index and column names
            data_binned.rename(columns={'Pressure_Bin': 'pressure_00'}, inplace=True)
            profile = data_binned.copy()
            profiles.append(profile)

        self._ctd_array = merge_dataframes(profiles)
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._NO_SAMPLES_ERROR)

    def clean(self, feature, method='autoencoder'):
        """
        Applies data cleaning methods to the specified feature using the selected method.
        Currently supports cleaning practical salinity using 'salinitydiff' or 'autoencoder' methods.

        Parameters
        ----------
        feature : str
            The feature to clean (e.g., 'practicalsalinity').
            Options are 'practicalsalinity'.
        method : str, optional
            The cleaning method to apply, defaults to 'autoencoder'.
            Options are 'salinitydiff', 'autoencoder'.
        """
        profiles = []
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._NO_SAMPLES_ERROR)
        for filename, profile in self._ctd_array.groupby('filename'):
            if method == 'salinitydiff':
                profile = self._Calculate.calculate_and_drop_salinity_spikes(profile)
            if method == 'autoencoder':
                profile = self.clean_salinity_autoencoder(profile)
            profiles.append(profile)

        self._ctd_array = merge_dataframes(profiles)
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._DATA_CLEANING_ERROR)

    def add_absolute_salinity(self):
        """
        Calculates the absolute salinity using the TEOS-10 equations and adds it as a new column
        to the CTD data table. Removes rows with negative absolute salinity values.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._NO_SAMPLES_ERROR)
        self._ctd_array['salinityabs'] = self._Calculate.calculate_absolute_salinity(
            self._ctd_array)
        self._ctd_array = self.Utility.remove_rows_with_negative_salinityabs(self._ctd_array)
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._SALINITYABS_CALCULATION_ERROR)

    def add_density(self):
        """
        Calculates the density using the TEOS-10 equations and adds it as a new column to the CTD
        data table. If absolute salinity is not present, it is calculated first.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._NO_SAMPLES_ERROR)
        if 'salinityabs' in self._ctd_array.columns:
            densities = self._Calculate.calculate_absolute_density(
                self._ctd_array)
            if len(densities.dropna()) < 1:
                raise CTDError(filename=self._filename_full,
                               message="All densities were NaN, further processing may fail")
            self._ctd_array.loc[self._ctd_array.index, 'density'] = densities
        else:
            self.add_absolute_salinity()
            self._ctd_array.loc[self._ctd_array.index, 'density'] = self._Calculate.calculate_absolute_density(
                self._ctd_array)
            self._ctd_array.drop('salinityabs')
            if self.Utility.no_values_in_object(self._ctd_array):
                raise CTDError(filename=self._filename_full, message=self._DENSITY_CALCULATION_ERROR)

    def add_potential_density(self):
        """
        Calculates the density using the TEOS-10 equations and adds it as a new column to the CTD
        data table. If absolute salinity is not present, it is calculated first.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._NO_SAMPLES_ERROR)
        profiles = []
        for filename, profile in self._ctd_array.groupby('filename'):
            if 'salinityabs' in profile.columns:
                densities = self._Calculate.calculate_potential_density(
                    profile)
                if len(densities.dropna()) < 1:
                    raise CTDError(filename=self._filename_full,
                                   message="All densities were NaN, further processing may fail")
                profile.loc[profile.index, 'potentialdensity'] = densities
            else:
                self.add_absolute_salinity()
                profile.loc[
                    profile.index, 'potentialdensity'] = self._Calculate.calculate_potential_density(
                    profile)
                profile.drop('salinityabs')
            profiles.append(profile)

        self._ctd_array = merge_dataframes(profiles)
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._DENSITY_CALCULATION_ERROR)

    def add_overturns(self):
        """
        Calculates density changes between consecutive measurements and identifies overturns where
        denser water lies above less dense water. Adds an 'overturn' column to the CTD data table.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._NO_SAMPLES_ERROR)
        profiles = []
        for filename, profile in self._ctd_array.groupby('filename'):
            profile = self._Calculate.calculate_overturns(profile.copy())
            profiles.append(profile)

        self._ctd_array = merge_dataframes(profiles)

    def add_surface_salinity_temp_meltwater(self, start=10.0, end=12.0):
        """
        Calculates the surface salinity, temperature, and meltwater fraction and adds them as a new column to the CTD
        data table

        Parameters
        ----------
        start : float, optional
            Minimum pressure bound, defaults to 10.0.
        end : float, optional
            Maximum pressure bound, defaults to 12.0.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._NO_SAMPLES_ERROR)
        profiles = []
        for filename, profile in self._ctd_array.groupby('filename'):
            surface_salinity, meltwater_fraction = (self._Calculate.calculate_surface_salinity_and_meltwater
                                                    (profile, start, end))
            surface_temperature = (self._Calculate.calculate_surface_temperature(self._ctd_array, start, end))
            profile = profile.assign(surface_salinity=surface_salinity)
            profile = profile.assign(meltwater_fraction=meltwater_fraction)
            profile = profile.assign(surface_temperature=surface_temperature)
            profiles.append(profile)

        self._ctd_array = merge_dataframes(profiles)

    def add_mean_surface_density(self, start=0.0, end=100.0):
        """
        Calculates the mean surface density from the density values and adds it as a new column
        to the CTD data table.

        Parameters
        ----------
        start : float, optional
            Depth bound, defaults to 0.
        end : float, optional
            Depth bound, default to 1.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._NO_SAMPLES_ERROR)
        profiles = []
        for filename, profile in self._ctd_array.groupby('filename'):
            mean_surface_density = self._Calculate.calculate_mean_surface_density(profile.copy(), (start, end))
            profile = profile.assign(mean_surface_density=mean_surface_density)
            profiles.append(profile)
        
        self._ctd_array = merge_dataframes(profiles)

    def add_mld(self, reference, method="potentialdensityavg"):
        """
        Calculates the mixed layer depth using the specified method and reference depth.
        Adds the MLD and the actual reference depth used as new columns to the CTD data table.

        Parameters
        ----------
        reference : int
            The reference depth for MLD calculation.
        method : str
            The MLD calculation method options are "absdensity", "potentialdensity" or "potentialdensityavg"
             (default: "absdensity").
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._NO_SAMPLES_ERROR)
        supported_methods = [
            "absdensity"
            "potentialdensity"
        ]
        profiles = []
        for filename, profile in self._ctd_array.groupby('filename'):
            unpack = None
            if method == "absdensity":
                unpack = self._Calculate.calculate_mld(profile['absdensity'], profile['depth_00'],
                                                       reference)
            elif method == "potentialdensity":
                unpack = self._Calculate.calculate_mld(profile['potentialdensity'], profile['depth_00'],
                                                       reference)
            elif method == "potentialdensityavg":
                unpack = self._Calculate.calculate_mld_average(profile['potentialdensity'], profile['depth_00'],
                                                               reference)
            else:
                raise CTDError(message=f"add_mld: Invalid method \"{method}\" not in {supported_methods}",
                               filename=self._filename_full)
            if unpack is None:
                logger.debug(f"{self._filename_full} - Couldn't calculate mld")
                unpack = [None, None]
            MLD = unpack[0]
            depth_used_as_reference = unpack[1]
            profile.loc[:, f'MLD {reference}'] = MLD
            profile.loc[:, f'MLD {reference} Actual Reference Depth'] = depth_used_as_reference
            profiles.append(profile)

        self._ctd_array = merge_dataframes(profiles)
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._MLD_ERROR)

    def add_stratification(self, depth_range=20):
        """
        Calculates the SI (stratification index) within the specified depth range through the profile.
        Adds the SI to the CTD data table.
        Requires potential density to be calculated first.

        Parameters
        ----------
        depth_range : int
            The depth range to calculate SI.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._NO_SAMPLES_ERROR)
        if 'potentialdensity' not in self._ctd_array.columns:
            raise CTDError(filename=self._filename_full,
                           message="Cannot calculate stratification index without potential density")
        profiles = []
        for filename, profile in self._ctd_array.groupby('filename'):
            depths = profile['depth_00'].values
            potential_densities = profile['potentialdensity'].values
            si_values = np.full(depths.shape, np.nan)
            for start_depth in np.arange(depths.min(), depths.max(), depth_range):
                end_depth = start_depth + depth_range
                if end_depth > depths.max():
                    continue
                si = self._Calculate.stratification_index(depths, potential_densities, start_depth, end_depth)
                mask = (depths >= start_depth) & (depths < end_depth)
                si_values[mask] = si
            profile['SI'] = si_values
            profiles.append(profile)

        self._ctd_array = merge_dataframes(profiles)
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(filename=self._filename_full, message=self._MLD_ERROR)

    def plot(self, measurement, plot_type='scatter'):
        """
        Generates a plot of depth vs. specified measurement (salinity, density, temperature).
        Adds horizontal lines indicating the mixed layer depth (MLD) if present.
        Allows for both scatter and line plot types.
        Saves the plot as an image file.

        Parameters
        ----------
        measurement : str
            Options are 'salinity', 'density', 'potentialdensity, or 'temperature'.
        plot_type : str
            Options are 'scatter' or 'line'.
        """
        plt.rcParams.update({'font.size': 16})
        plot_folder = os.path.join(self._cwd, "plots")
        os.makedirs(plot_folder, exist_ok=True)
        ids = self._ctd_array['depth_00']
        for filename, profile in self._ctd_array.groupby('filename'):
            ids = profile['depth_00']
            fig, ax1 = plt.subplots(figsize=(18, 18))
            ax1.invert_yaxis()
            ax1.set_ylim([profile['depth_00'].max(), 0])
            color_map = {'salinity': 'tab:blue',
                         'density': 'tab:red',
                         'potentialdensity': 'tab:red',
                         'temperature': 'tab:blue'}
            label_map = {'salinity': 'Practical Salinity (PSU)',
                         'density': 'Density (kg/m^3)',
                         'potentialdensity': 'Potential Density (kg/m^3)',
                         'temperature': 'Temperature (°C)'}
            if measurement == 'salinity' or measurement == 'temperature':
                measurement_col = f'{measurement}_00'
            else:
                measurement_col = measurement
            if plot_type == 'line':
                lowess = statsmodels.api.nonparametric.lowess
                y, x = zip(*lowess(profile[measurement_col], profile['depth_00'], frac=0.1))
            else:
                x, y = profile[measurement_col], profile['depth_00']
            ax1.plot(x, y, color=color_map[measurement], label=label_map[measurement]) if plot_type == 'line' \
                else ax1.scatter(x, y, color=color_map[measurement], label=label_map[measurement])
            ax1.set_xlabel(label_map[measurement], color=color_map[measurement])
            ax1.set_ylabel('Depth (m)')
            ax1.tick_params(axis='x', labelcolor=color_map[measurement])
            mld_cols = [profile[col] for col in profile.columns if 'MLD' in col and 'Actual' not in col]
            refdepth_cols = [profile[col] for col in profile.columns if 'Actual' in col]
            for idx, mld_col in enumerate(mld_cols):
                if mld_col.empty or type(mld_col.iloc[0]) is type(None):
                    break
                ax1.axhline(y=mld_col.iloc[0], color='green', linestyle='--',
                            label=f'MLD {refdepth_cols[idx].iloc[0]} Ref')
                ax1.text(0.95, mld_col.iloc[0], f'MLD with respect to {refdepth_cols[idx].iloc[0]}m', va='center',
                         ha='right', backgroundcolor='white', color='green', transform=ax1.get_yaxis_transform())
                break
            plt.title(f"{filename}\n Depth vs. {label_map[measurement]}\n")
            ax1.grid(True)
            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
            plot_path = os.path.join(plot_folder, f"{filename}_depth_{measurement}_{plot_type}_plot.png")
            plt.savefig(plot_path)
            plt.close(fig)
            plt.close('all')

    def _process_master_sheet(self, ctd_df, for_id=False):
        """
        Extracts the date and time components from the filename and compares them with the data
        in the master sheet. Calculates the absolute differences between the dates and times to
        find the closest match. Returns the estimated latitude, longitude, and updated filename
        based on the closest match.

        Parameters
        ----------
        master_sheet_path : str
            The path to the master sheet Excel file.

        filename : str
            The filename of the RSK file.

        ctd_df : DataFrame
            CTD profile

        for_id : bool
            Flag for logging purposes, indicates if we processed for location or id.

        Returns
        -------
        tuple
            A tuple containing the estimated latitude, longitude, and updated filename.
        """
        if self._cached_master_sheet.empty:
            self._cached_master_sheet = pd.read_csv(self.master_sheet_path)
            self._cached_master_sheet = CTD.Utility.load_master_sheet(self._cached_master_sheet)
        if 'timestamp' not in ctd_df.columns:
            raise CTDError(message="No timestamp in file, could not get location", filename=self._filename_full)
        master_df = self._cached_master_sheet
        timestamp = ctd_df['timestamp']
        if timestamp.empty:
            raise CTDError(message="Timestamp column empty, could not get location", filename=self._filename_full)
        timestamp = timestamp.iloc[0]
        closest_row_overall = master_df.iloc[abs((master_df['datetime'] - timestamp)).idxmin(), :]
        latitude = closest_row_overall['latitude']
        longitude = closest_row_overall['longitude']
        distance = closest_row_overall['datetime'] - timestamp
        unique_id = closest_row_overall.iloc[0]
        # Extract days, hours, and minutes from the time difference
        days = abs(distance.days)
        hours, remainder = divmod(distance.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        if for_id:
            message = f"{self._filename_full} - Guessed Location : Matched to unique ID '{unique_id}' with distance {days} days {hours}:{minutes}"
            if abs(days) > 2:
                logger.warning(message)
            else:
                logger.info(message)
        else:
            message = f"{self._filename_full} - Guessed Unique ID : Matched to unique ID '{unique_id}' with distance {days} days {hours}:{minutes}"
            if abs(days) > 2:
                logger.warning(message)
            else:
                logger.info(message)
        return latitude, longitude, unique_id

    def clean_salinity_autoencoder(self, profile):
        """
        Cleans salinity using an LSTM autoencoder ML model.
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

        def build_gru_autoencoder(input_shape):
            """
            GRU autoencoder architecture.
            """
            # Encoder
            encoder_inputs = Input(shape=(input_shape[0], input_shape[1]), name='encoder_inputs')
            encoder_gru1 = GRU(16, activation='tanh',
                               return_sequences=True)(encoder_inputs)
            #encoder_gru2 = GRU(32, activation='tanh',
                               #return_sequences=False)(encoder_gru1)
            #encoder_gru3 = GRU(8, activation='tanh',
                               #return_sequences=False)(encoder_gru2)
            # encoder_gru4 = GRU(8, activation='tanh',
            # return_sequences=False)(encoder_gru3)
            #encoder_output = Dense(16, activation='linear')(encoder_gru1)

            # Decoder
            #decoder_inputs = RepeatVector(input_shape[0])(encoder_output)
            #decoder_gru1 = GRU(64, activation='tanh',
                               #return_sequences=True)(decoder_inputs)
            #decoder_gru2 = GRU(32, activation='tanh',
                               #return_sequences=True)(decoder_gru1)
            #decoder_gru3 = GRU(16, activation='tanh',
                               #return_sequences=True)(decoder_gru2)
            # decoder_gru4 = GRU(16, activation='tanh',
            # return_sequences=True)(decoder_gru3)
            decoder_output = TimeDistributed(Dense(input_shape[1], activation='linear'), name='decoded_output')(
                encoder_gru1)

            autoencoder = Model(encoder_inputs, decoder_output)
            optimizer = Adam(learning_rate=0.01)
            autoencoder.compile(optimizer=optimizer, loss=loss_function)  # Specify your loss function here
            return autoencoder

        def plot_original_data(data, filename):
            plt.figure(figsize=(10, 6))
            plt.scatter(data['salinity_00'], -data['depth_00'], alpha=0.6)
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

        def plot_predicted_data(predicted_df, xlim, ylim, filename):
            plt.figure(figsize=(10, 6))
            plt.scatter(predicted_df['salinity_00'], -predicted_df['depth_00'], alpha=0.6, color='red')
            title = f'Predicted Salinity vs. Depth - {filename}'
            plt.title(title)
            plt.xlabel('Salinity (PSU)')
            plt.ylabel('Depth (m)')
            plt.grid(True)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.savefig(os.path.join('plots', f'{filename}_predicted.png'))
            # plt.show()
            plt.close()

        def run_autoencoder(data, show_plots=False):
            """
            Runs the autoencoder.

            Parameters
            ----------
            data : DataFrame
                CTD dataframe
            filename : str
                Name of profile
            show_plots : bool
                False to not save plots, True to save plots.
            Returns
            -------
            DataFrame
                CTD data with clean salinity values.
            """
            # Binning data
            filename = data.filename.iloc[-1]
            data = data[data['depth_00'] >= 1].reindex()
            data.loc[:, 'Pressure_Bin'] = data['pressure_00'] // 1.0 * 1.0
            # Define the desired columns and their aggregation functions
            column_agg_dict = {
                "temperature_00": "mean",
                "chlorophyll_00": "mean",
                "seapressure_00": "mean",
                "depth_00": "mean",
                "salinity_00": "median",
                "speedofsound_00": "mean",
                "specificconductivity_00": "mean",
                "conductivity_00": "mean",
                "density": "mean",
                "densitypotential": "mean",
                "salinityabs": "mean",
                "timestamp": "first",
                "MLD_Zero": "first",
                "MLD_Ten": "first",
                "stratification": "first",
                "mean_surface_density": "first",
                "surface_salinity": "first",
                "surface_temperature": "first",
                "meltwater_fraction": "first",
                "longitude": "first",
                "latitude": "first",
                "overturn": "first",
                "Unique_ID": "first",
                "filename": "first"
            }
            # Check which columns are present in the DataFrame
            available_columns = {col: agg_func for col, agg_func in column_agg_dict.items() if col in data.columns}
            # Group by and aggregate based on the available columns
            data_binned = data.groupby('Pressure_Bin').agg(available_columns)
            # Rename the Pressure_Bin column if it exists
            data_binned = data_binned.reset_index()
            data_binned = data_binned.rename(columns={'Pressure_Bin': 'pressure_00'}).reindex()
            # Scaling data
            numerical_columns = ['salinity_00']
            scaler = MinMaxScaler(feature_range=(-1, 1))
            if len(data_binned[numerical_columns]) < 2:
                raise CTDError(message="Not enough values to run the autoencoder on this data",
                               filename=self._filename_full)
            scaled_sequence = scaler.fit_transform(data_binned[numerical_columns])
            scaled_seq = np.expand_dims(scaled_sequence, axis=0)
            min_pres = data_binned['depth_00'].min()
            max_pres = data_binned['depth_00'].max()
            # Calculate ranges
            pres_range = max_pres - min_pres
            epochs = int(pres_range * 16)
            # Build autoencoder and predict on CTD data
            autoencoder = build_gru_autoencoder(scaled_seq.shape[1:])
            autoencoder.fit(scaled_seq, scaled_seq, epochs=epochs, verbose=0, batch_size=4)
            X_pred = autoencoder.predict(scaled_seq, verbose=None)
            # Revert scaling
            predicted_seq = scaler.inverse_transform(X_pred[0])
            data_binned['salinity_00'] = predicted_seq[:, 0]
            if show_plots:
                xlim, ylim = plot_original_data(data, filename)
                plot_predicted_data(data_binned, xlim, ylim, filename)
            data_binned = data_binned.loc[:, ~data_binned.columns.duplicated()].copy()
            return data_binned.reset_index()
        return run_autoencoder(profile)

    def save_to_csv(self, output_file):
        """
        Renames the columns of the CTD data table based on a predefined mapping and saves the
        data to the specified CSV file. If the file already exists, the data is appended to it.

        Parameters
        ----------
        output_file : str
            The output CSV file path.
        """
        self.Utility.save_to_csv(self._ctd_array, output_file=output_file)

    class Utility:
        """
        Utility
        --------
        Utility class for CTD data processing.

        Attributes
        ----------
        filename :  str
            Filename of the RSK file.
        """

        @staticmethod
        def no_values_in_object(object_to_check):
            """
            Checks if the given object is None, empty, or has a length greater than 0.
            Returns True if the object has no values, False otherwise.

            Parameters
            ----------
            object_to_check : object
                The object to check for values.
            Returns
            --------
            bool
                True if the object has no values, False otherwise.
            """
            if isinstance(object_to_check, type(None)):
                return True
            if isinstance(object_to_check, DataFrame) and object_to_check.empty:
                return True
            if isinstance(object_to_check, Sized) and len(object_to_check) > 0:
                return False

        @staticmethod
        def remove_rows_with_negative_depth(df):
            """
            Removes rows from the given DataFrame where the 'depth_00' column has negative values.
            Returns the updated DataFrame.

            Parameter
            ---------
            df : DataFrame
                The DataFrame to process.

            Returns
            -------
            DataFrame
                The updated DataFrame with rows containing negative depth values removed.
            """
            if CTD.Utility.no_values_in_object(df):
                return None
            if 'depth_00' in df.columns:
                df = df[df['depth_00'] >= 0].reindex()
            else:
                return None
            if CTD.Utility.no_values_in_object(df):
                return None
            return df.copy()

        @staticmethod
        def remove_rows_with_negative_salinity(df):
            """
            Removes rows from the given DataFrame where the 'salinity_00' column has negative values.
            Returns the updated DataFrame.

            Parameters
            ----------
            df: DataFrame
                The DataFrame to process.

            Returns
            -------
            DataFrame
                The updated DataFrame with rows containing negative salinity values removed.
            """
            if CTD.Utility.no_values_in_object(df):
                return None
            if 'salinity_00' in df.columns:
                df = df[df['salinity_00'] >= 0].reindex()
            else:
                return None
            if CTD.Utility.no_values_in_object(df):
                return None
            return df.copy()

        @staticmethod
        def remove_rows_with_invalid_salinity(df):
            """
            Removes rows from the given DataFrame where the practical salinity column has values < 10.
            Returns the updated DataFrame.

            Parameters
            ----------
            df: DataFrame
                The DataFrame to process.

            Returns
            -------
            DataFrame
                The updated DataFrame with rows containing negative salinity values removed.
            """
            if CTD.Utility.no_values_in_object(df):
                return None
            if 'salinity_00' in df.columns:
                df = df[df['salinity_00'] >= 10].reindex()
            else:
                return None
            if CTD.Utility.no_values_in_object(df):
                return None
            return df.copy()

        @staticmethod
        def remove_rows_with_negative_pressure(df):
            """
            Removes rows from the given DataFrame where the 'pressure_00' column has negative values.
            Returns the updated DataFrame.

            Parameters
            ----------
            df: DataFrame
                The DataFrame to process.

            Returns
            -------
            DataFrame
                The updated DataFrame with rows containing negative pressure values removed.
            """
            if CTD.Utility.no_values_in_object(df):
                return None
            if 'pressure_00' in df.columns:
                df = df[df['pressure_00'] >= 0].reindex()
            else:
                return None
            if CTD.Utility.no_values_in_object(df):
                return None
            return df.copy()

        @staticmethod
        def remove_rows_with_negative_salinityabs(df):
            """
            Removes rows from the given DataFrame where the 'salinityabs' column has negative values.
            Returns the updated DataFrame.

            Parameters
            ----------
            df: DataFrame
                The DataFrame to process.

            Returns
            -------
            DataFrame
                The updated DataFrame with rows containing negative absolute salinity values removed.
            """
            if CTD.Utility.no_values_in_object(df):
                return None
            if 'salinityabs' in df.columns:
                df = df[df['salinityabs'] >= 0].reindex()
            else:
                return None
            if CTD.Utility.no_values_in_object(df):
                return None
            return df.copy()

        @staticmethod
        def remove_rows_with_negative_density(df):
            """
            Removes rows from the given DataFrame where the 'density' column has negative values.
            Returns the updated DataFrame.

            Parameters
            ----------
            df: DataFrame
                The DataFrame to process.

            Returns
            -------
            DataFrame
                The updated DataFrame with rows containing negative density values removed.
            """
            if CTD.Utility.no_values_in_object(df):
                return None
            if 'density' in df.columns:
                df = df[df['density'] >= 0].reindex()
            else:
                return None
            if CTD.Utility.no_values_in_object(df):
                return None
            return df.copy()

        @staticmethod
        def save_to_csv(input_df, output_file):
            """
            Renames the columns of the CTD data table based on a predefined mapping and saves the
            data to the specified CSV file. If the file already exists, the data is appended to it.

            Parameters
            ----------
            input_df : DataFrame
                The output CSV file path.
            output_file : str
                The output CSV file path.
            """
            rsk_labels = {
                "temperature_00": "Temperature_(°C)",
                "pressure_00": "Pressure_(dbar)",
                "chlorophyll_00": "Chlorophyll_a_(µg/l)",
                "seapressure_00": "Sea Pressure_(dbar)",
                "depth_00": "Depth_(m)",
                "salinity_00": "Salinity_(PSU)",
                "speedofsound_00": "Speed of Sound_(m/s)",
                "specificconductivity_00": "Specific Conductivity_(µS/cm)",
                "conductivity_00": "Conductivity_(mS/cm)",
                "density": "Density_(kg/m^3)",
                "potentialdensity": "Potential_Density_(kg/m^3)",
                "salinityabs": "Absolute Salinity_(g/kg)",
                "MLD_Zero": "MLD_Zero_(m)",
                "MLD_Ten": "MLD_Ten_(m)",
                "stratification": "Stratification_(J/m^2)",
                "mean_surface_density": "Mean_Surface_Density_(kg/m^3)",
                "surface_salinity": "Surface_Salinity_(PSU)",
                "surface_temperature": "Surface_Temperature_(°C)",
                "meltwater_fraction": "Meltwater_Fraction_(%)",
                "overturn": "Overturn_(Δρ<-0.05)"
            }
            data = input_df.copy()
            data.rename(columns=rsk_labels, inplace=True)
            data.reset_index(drop=True, inplace=True)
            # Handle CSV file reading and merging
            try:
                csv_df = pd.read_csv(output_file)
                csv_df.reset_index(drop=True, inplace=True)
                try:
                    # Merge the existing DataFrame with the new DataFrame
                    merged_df = merge_dataframes([csv_df, data])
                    # Overwrite the original CSV file with the merged DataFrame
                    merged_df.to_csv(output_file, index=False)
                except MergeError as e:
                    raise CTDError("CSV file and CTD object have different columns, cannot merge and save", "")
            except FileNotFoundError:
                logger.debug(f"The file {output_file} does not exist, a new file will be created")
                data.to_csv(output_file, index=False)



        @staticmethod
        def extract_utc_cast_time(ctdfilepath):
            """
            Function to extract the UTC cast time from a castaway file and convert it to ISO 8601 format.

            Parameters
            ----------
            ctdfilepath : str
                The file path of the castaway file to extract the time from.

            Returns
            -------
            str
                Cast time (UTC) of the castaway file in ISO 8601 format.
            """
            # Initialize variable to store UTC cast time
            cast_time_utc = None
            # Open the file and read line by line
            with open(ctdfilepath, 'r') as file:
                for line in file:
                    if line.startswith('% Cast time (UTC)'):
                        # Extract the UTC cast time after the comma
                        parts = line.strip().split(',')
                        if len(parts) > 1:
                            cast_time_str = parts[1].strip()  # Take only the datetime part
                            # Convert the datetime string to ISO format if possible
                            try:
                                cast_time_utc = datetime.strptime(cast_time_str, "%Y-%m-%d %H:%M:%S").isoformat()
                            except ValueError:
                                # Handle the case where date format does not match expected format
                                logger.info("Date format does not match expected '%Y-%m-%d %H:%M:%S'")
                            break  # Stop reading once the timestamp is found

            return cast_time_utc

        @staticmethod
        def extract_lat_long_castaway(ctdfilepath):
            """
            Function extract start lat/long from castaway file.

            Parameters
            ----------
            ctdfilepath : str
                Filepath to castaway ctd file.

            Returns
            -------
            tuple
                (latitude, longitude)
            """
            latitude = None
            longitude = None
            # Open the file and read line by line
            with open(ctdfilepath, 'r') as file:
                for line in file:
                    if line.startswith('% Start latitude'):
                        # Assume the format is '% description, latitude'
                        parts = line.strip().split(',')
                        if len(parts) > 1:
                            latitude = parts[1].strip()
                    if line.startswith('% Start longitude'):
                        # Assume the format is '% description, longitude'
                        parts = line.strip().split(',')
                        if len(parts) > 1:
                            longitude = parts[1].strip()
            return latitude, longitude

        @staticmethod
        def load_master_sheet(master_df):
            df = master_df
            df['date'] = df['date_local'].astype(str)
            df['time'] = df['time_local'].astype(str).str.split(' ').str[0]
            # Replace non-time values with NaN and fill them with a default time
            df['time'] = (
                df['time'].apply(lambda x: x if pd.to_datetime(
                    x, errors='coerce', format='mixed') is not pd.NaT else np.nan))
            df['time'] = df['time'].fillna('00:00:00')
            df['time'] = df['time'].astype(str)
            # Combine date and time columns into a single datetime column
            df['datetime'] = pd.to_datetime(
                df['date'] + ' ' + df['time'], errors='coerce', format='mixed')
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
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
        def gsw_infunnel(SA, CT, p):
            """
            Check if the given Absolute Salinity (SA), Conservative Temperature (CT),
            and pressure (p) are within the "oceanographic funnel" for the TEOS-10 75-term equation.

            Parameters
            ----------
            SA : Series
                Absolute Salinity in g/kg.
            CT : Series
                Conservative Temperature in degrees Celsius.
            p : Series or scalar
                Sea pressure in dbar (absolute pressure minus 10.1325 dbar).

            Returns
            -------
            Series of bool
                A boolean array where True indicates the values are inside the "oceanographic funnel".
            """
            # Ensure all inputs are Series and aligned
            if not (isinstance(SA, pd.Series) and isinstance(CT, pd.Series) and
                    (isinstance(p, pd.Series) or np.isscalar(p))):
                raise CTDError(message="SA, CT, and p must be pandas Series or p a scalar", filename="Unknown")

            if isinstance(p, pd.Series):
                if not (SA.index.equals(CT.index) and SA.index.equals(p.index)):
                    raise CTDError(message="Indices of SA, CT, and p must be aligned", filename="Unknown")
            else:
                p = pd.Series(p, index=SA.index)

            # Define the funnel conditions
            CT_freezing_p = gsw.CT_freezing(SA, p, 0)
            CT_freezing_500 = gsw.CT_freezing(SA, 500, 0)

            in_funnel = pd.Series(True, index=SA.index)  # Default all to True
            condition = (
                    (p > 8000) |
                    (SA < 0) | (SA > 42) |
                    ((p < 500) & (CT < CT_freezing_p)) |
                    ((p >= 500) & (p < 6500) & (SA < p * 5e-3 - 2.5)) |
                    ((p >= 500) & (p < 6500) & (CT > (31.66666666666667 - p * 3.333333333333334e-3))) |
                    ((p >= 500) & (CT < CT_freezing_500)) |
                    ((p >= 6500) & (SA < 30)) |
                    ((p >= 6500) & (CT > 10.0)) |
                    SA.isna() | CT.isna() | p.isna()
            )
            in_funnel[condition] = False

            return in_funnel

        @staticmethod
        def calculate_and_drop_salinity_spikes(df):
            """
            Calculates and removes salinity spikes from the CTD data based on predefined thresholds for acceptable
            changes in salinity with depth.

            Parameters
            ----------
            df : DataFrame
                DataFrame containing depth and salinity data

            Returns
            -------
            DataFrame
                DataFrame after removing salinity spikes
            """
            if df.empty:
                return None

            df['depth_00'] = pd.to_numeric(df['depth_00'], errors='coerce')
            df['salinity_00'] = pd.to_numeric(df['salinity_00'], errors='coerce')
            df = df.dropna(subset=['depth_00', 'salinity_00'])

            min_depth = df['depth_00'].min()
            max_depth = df['depth_00'].max()
            if min_depth == max_depth:
                logger.info("Insufficient depth range to clean salinity spikes")
                return df

            acceptable_delta_salinity_per_depth = [
                (0.0005, 0.001),
                (0.005, 0.01),
                (0.05, 0.1),
                (0.5, 1),
                (1, 2),

            ]

            for acceptable_delta, depth_range in acceptable_delta_salinity_per_depth:
                num_points = int((max_depth - min_depth) / depth_range)
                bins = np.linspace(min_depth, max_depth, num=num_points)
                indices = np.digitize(df['depth_00'], bins)

                # Group by indices and filter
                grouped = df.groupby(indices)
                df = grouped.filter(lambda x: abs(x['salinity_00'].max() - x['salinity_00'].min()) <= acceptable_delta)

            return df

        @staticmethod
        def calculate_overturns(ctd_array):
            """
            Calculates density overturns in the CTD data where denser water lies above lighter water with density
            difference of at least 0.05 kg/m³, which may indicate mixing or other dynamic processes.

            Parameters
            ----------
            ctd_array : DataFrame
                DataFrame containing depth, density, and timestamp data

            Returns
            -------
            DataFrame
                DataFrame with identified density overturns
            """
            # Sort DataFrame by depth in ascending order
            ctd_array = ctd_array.sort_values(by='depth_00', ascending=True)
            # Calculate density change and identify overturns
            ctd_array['density_change'] = ctd_array[
                'density'].diff()  # Difference in density between consecutive measurements
            ctd_array['overturn'] = ctd_array['density_change'] < -0.05
            ctd_array = ctd_array.sort_values(by='timestamp', ascending=True)
            if 'density_change' in ctd_array.columns:
                ctd_array = ctd_array.drop('density_change', axis=1)
            return ctd_array

        @staticmethod
        def calculate_surface_salinity_and_meltwater(ctd_array, start=10.0, end=12.0):
            """
            Calculates the surface salinity and meltwater fraction of a CTD profile.
            Reports the mean salinity of the first 2 meters of the profile by finding the minimum salinity, and reports
            meltwater fraction as given by (-0.021406 * surface_salinity + 0.740392) * 100.

            Parameters
            ----------
            ctd_array : DataFrame
                DataFrame containing salinity and depth data.
            start : float
                Minimum pressure of the surface, defaults to 10.0.
            end : float
                Maximum pressure of the surface, defaults to 12.0.

            Returns
            -------
            tuple
                Returns a tuple of (surface salinity, meltwater fraction).
            """
            # Filtering data within the specified pressure range
            surface_data = ctd_array[(ctd_array['pressure_00'] >= start) & (ctd_array['pressure_00'] <= end)]
            surface_salinity = surface_data['salinity_00'].min()
            mwf = (-0.021406 * surface_salinity + 0.740392) * 100

            return surface_salinity, mwf

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
            surface_data = ctd_array[(ctd_array['pressure_00'] >= start) & (ctd_array['pressure_00'] <= end)]
            surface_temperature = surface_data['temperature_00'].mean()

            return surface_temperature

        @staticmethod
        def calculate_absolute_density(ctd_array):
            """
            Calculates absolute density from the CTD data using the TEOS-10 equations,
            ensuring all data points are within the valid oceanographic funnel.

            Parameters
            ----------
            ctd_array : DataFrame
                DataFrame containing salinity, temperature, and pressure data

            Returns
            -------
            Series
                Series with calculated absolute density
            """
            sa = ctd_array['salinityabs']
            t = ctd_array['temperature_00']
            p = ctd_array['seapressure_00']
            ct = gsw.CT_from_t(sa, t, p)
            if CTD._Calculate.gsw_infunnel(sa, ct, p).all():
                return gsw.density.rho_t_exact(sa, t, p)
            else:
                raise CTDError(message="Sample not in funnel, could not calculate density", filename="Unknown")

        @staticmethod
        def calculate_potential_density(ctd_array):
            """
            Calculates potential density from the CTD data using the TEOS-10 equations,
            ensuring all data points are within the valid oceanographic funnel.

            Parameters
            ----------
            ctd_array : DataFrame
                DataFrame containing salinity, temperature, and pressure data

            Returns
            -------
            Series
                Series with calculated absolute density
            """
            sa = ctd_array['salinityabs']
            t = ctd_array['temperature_00']
            p = ctd_array['seapressure_00']
            ct = gsw.CT_from_t(sa, t, p)
            if CTD._Calculate.gsw_infunnel(sa, ct, p).all():
                return gsw.sigma0(sa, ct)
            else:
                raise CTDError(message="Sample not in funnel, could not calculate density", filename="Unknown")

        @staticmethod
        def calculate_absolute_salinity(ctd_array):
            """
            Calculates absolute salinity from practical salinity, pressure,
            and geographical coordinates using the TEOS-10 salinity conversion formulas.

            Parameters
            ----------
            ctd_array : DataFrame
                DataFrame containing practical salinity, pressure, longitude, and latitude data

            Returns
            -------
            Series
                Series with calculated absolute salinity
            """
            sp = ctd_array['salinity_00'].to_numpy(copy=True)
            p = ctd_array['seapressure_00'].to_numpy(copy=True)
            lon = ctd_array['longitude'].to_numpy(copy=True)
            lat = ctd_array['latitude'].to_numpy(copy=True)
            return gsw.conversions.SA_from_SP(sp, p, lon, lat)

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
            # Convert to numeric and ensure no NaNs remain
            densities = densities.apply(pd.to_numeric, errors='coerce')
            depths = depths.apply(pd.to_numeric, errors='coerce')
            densities = densities.dropna(how='any').reset_index(drop=True)
            depths = depths.dropna(how='any').reset_index(drop=True)
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
        def calculate_mld_average(densities, depths, reference_depth=20, delta=0.05):
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
            # Convert to numeric and ensure no NaNs remain
            densities = densities.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            depths = depths.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
            densities = densities.dropna(how='any').reset_index(drop=True)
            depths = depths.dropna(how='any').reset_index(drop=True)
            reference_depth = int(reference_depth)
            if len(depths) == 0 or len(densities) == 0:
                return None
            sorted_data = sorted(zip(depths, densities), key=lambda x: x[0])
            sorted_depths, sorted_densities = zip(*sorted_data)

            # Calculate the average density up to the reference depth
            reference_density = []
            for depth, density in zip(sorted_depths, sorted_densities):
                if depth <= reference_depth:
                    reference_density.append(density)
                else:
                    break

            if not reference_density:
                return None  # No data up to the reference depth

            average_reference_density = sum(reference_density) / len(reference_density)

            # Find the depth where density exceeds the average reference density by more than delta kg/m³
            for depth, density in zip(sorted_depths, sorted_densities):
                if density > average_reference_density + delta:
                    return depth, reference_depth

            return None  # If no depth meets the criterion after the reference depth

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
            min_depth = df.index.min()
            max_depth = df.index.max()

            if isinstance(range_, tuple):
                start, end = range_

                # Adjust 'start' to ensure it is within the valid range
                start = max(start, min_depth)

                # Adjust 'end' to ensure it does not exceed the maximum depth value
                end = min(end, max_depth)

                # Ensure start is less than end
                if start <= end:
                    return df.loc[start:end, 'density'].mean()
                else:
                    return None

            elif isinstance(range_, int):
                # Use 'range_' as the number of rows from the start, adjust if it exceeds the DataFrame length
                range_ = min(range_, len(df))
                return df.iloc[:range_, df.columns.get_loc('density')].mean()

            else:
                raise ValueError("Invalid range type. Must be tuple or int.")

        @staticmethod
        def stratification_index(depths, potential_densities, h1, h2):
            """
            Calculate the stratification index (SI) between depths -h1 and -h2.

            Parameters
            ----------
            depths : array-like
                Array of depth values.
            potential_densities : array-like
                Array of potential densities corresponding to the depths.
            h1 : float
                Upper depth limit (positive value, will be converted to negative).
            h2 : float
                Lower depth limit (positive value, will be converted to negative).

            Returns
            -------
            float:
                Stratification index (SI).
            """

            # Ensure h1 and h2 are within the bounds of the depth array
            if not (0 <= h1 < h2):
                raise ValueError("Ensure that 0 <= h1 < h2")

            # Convert h1 and h2 to negative
            h1 = -h1
            h2 = -h2

            # Ensure depths and potential_densities are numpy arrays
            depths = np.array(depths)
            potential_densities = np.array(potential_densities)

            # Interpolate potential density values at depths using numpy
            potential_density_interp = np.interp(depths, depths, potential_densities)

            # Calculate the potential density gradient
            density_gradient = np.gradient(potential_density_interp, depths)
            density_gradient_interp = np.interp(depths, depths, density_gradient)

            # Define the integrand function for SI
            def integrand(z):
                pd_grad = np.interp(z, depths, density_gradient_interp)
                return pd_grad * z

            # Perform the integration using numpy's trapezoidal rule
            mask = (depths >= h2) & (depths <= h1)
            integral = np.trapz(integrand(depths[mask]), depths[mask])

            return integral

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
