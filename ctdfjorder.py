# -*- coding: utf-8 -*-
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas
import pandas as pd
from pandas import DataFrame
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
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("library.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
# Disable logging for specific third-party libraries
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
# Tensorflow setup
policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)


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
    - Filenames must match a correct date, or else in instances where the mastersheet
    must be consulted CTDFjorder will return an error.

    """
    _ctd_array = pandas.DataFrame()
    _rsk = None
    _filename = None
    _calculator = None
    _cwd = None
    _cached_master_sheet = pandas.DataFrame()
    _original_profile = None
    master_sheet_path = None
    _NO_SAMPLES_ERROR = "No samples in file."
    _NO_LOCATION_ERROR = "No location could be found."
    _DENSITY_CALCULATION_ERROR = "Could not calculate density on this dataset."
    _SALINITYABS_CALCULATION_ERROR = "Could not calculate salinity on this dataset."
    _DATA_CLEANING_ERROR = "No data remains after data cleaning, reverting to previous CTD"
    _REMOVE_NEGATIVES_ERROR = "No data remains after removing non-positive samples."
    _MLD_ERROR = "No data remains after calculating MLD."
    _NO_MASTER_SHEET_ERROR = ("No mastersheet provided. Cannot geolocate file, try using find_master_sheet_file " +
                             "before this step.")
    _rsk_file_flag = False

    def __init__(self, ctdfilepath, cached_master_sheet=pandas.DataFrame(), master_sheet_path = None):
        """
        Initialize a new CTD object.

        Parameters
        ----------
        ctdfilepath : str
            The file path to the RSK or Castaway file.
        """
        if not cached_master_sheet.empty:
            self._cached_master_sheet = cached_master_sheet
        if "rsk" in ctdfilepath:
            self._rsk_file_flag = True
            self._rsk = RSK(ctdfilepath)
            self._filename = ('_'.join(ctdfilepath.split("/")[-1].split("_")[0:3]).split(".rsk")[0])
            num_profiles = sum(1 for _ in self._rsk.profiles())
            rsk_downcast_times = self._rsk.casts(Region.CAST_DOWN)
            if num_profiles < 1:
                self._ctd_array = np.array(self._rsk.npsamples())
            else:
                region = next(rsk_downcast_times)
                start_time = region.start_time
                end_time = region.end_time
                try:
                    self._ctd_array = np.array(self._rsk.npsamples(start_time, end_time))
                except Exception:
                    raise CTDError(self._filename, "No downcasts in file.")
            self._ctd_array = pd.DataFrame(self._ctd_array)
            self._original_profile = pd.DataFrame(self._ctd_array)
        else:
            self._rsk_file_flag = False
            self._filename = ('_'.join(ctdfilepath.split("/")[-1].split("_")[0:3]).split(".csv")[0])
            utc_cast_time = self.Utility.extract_utc_cast_time(ctdfilepath=ctdfilepath)
            latitude, longitude = self.Utility.extract_lat_long_castaway(ctdfilepath=ctdfilepath)
            try:
                latitude = float(latitude)
                longitude = float(longitude)
            except:
                latitude = None
                longitude = None
            # Convert UTC cast time to datetime and create timestamp column
            start_time = pd.to_datetime(utc_cast_time)
            # Calculate the time increments for each row
            pd_csv_castaway = pd.read_csv(ctdfilepath, comment='%')
            timestamps = [start_time + pd.Timedelta(seconds=0.2 * i) for i in range(len(pd_csv_castaway))]
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
            # Insert the timestamp column as the first column
            pd_csv_castaway['pressure_00'] = pd_csv_castaway['seapressure_00']
            pd_csv_castaway['pressure_00'] += 10
            pd_csv_castaway['conductivity_00'] /= 1000
            pd_csv_castaway.insert(0, 'timestamp', timestamps)
            pd_csv_castaway = pd_csv_castaway.assign(latitude=latitude)
            pd_csv_castaway = pd_csv_castaway.assign(longitude=longitude)
            pd_csv_castaway.drop('Density (Kilograms per Cubic Meter)', axis=1, inplace=True)
            self._ctd_array = pd_csv_castaway
            self._original_profile = pd_csv_castaway.copy(deep=True)
        self.Utility = self.Utility(self._filename)
        self._cwd = _get_cwd()
        self.master_sheet_path = master_sheet_path
        self._cached_master_sheet=cached_master_sheet
        logger.info("New CTD Object Created from : " + self._filename)

    def find_master_sheet_file(self):
        """
        Function to find and the master sheet path. Uses the first xlsx file in the current working directory.
        """
        cwd = _get_cwd()
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

    def add_filename_to_table(self):
        """
        Add the filename to the CTD data table.
        """
        self._ctd_array = self._ctd_array.assign(filename=self._filename)

    def remove_timezone_indicator(self):
        """
        Removes the timezone indicator in the CTD data table 'timestamp' column.
        """
        self._ctd_array = self.Utility.remove_sample_timezone_indicator(self._ctd_array)

    def add_location_to_table(self):
        """
        Retrieves the sample location data from the RSK file and adds it to the CTD data table.
        If no location data is found, it attempts to estimate the location using the master sheet.
        """
        # Check if file is castaway
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(self._filename, self._NO_SAMPLES_ERROR)
        if not self._rsk_file_flag:
            if self._ctd_array.latitude[0] == '':
                if self.master_sheet_path is None:
                    raise CTDError(self._filename, self._NO_MASTER_SHEET_ERROR)
                else:
                    location_data = self._process_master_sheet(self.master_sheet_path, self._filename)
            else:
                return
        # Process RSK file
        else:
            location_data = self.get_sample_location(self._rsk, self._filename)
        try:
            self._ctd_array = self._ctd_array.assign(latitude=location_data[0],
                                                     longitude=location_data[1])
        except Exception:
            self._ctd_array.loc['latitude'] = None
            self._ctd_array.loc['longitude'] = None
            self._ctd_array.loc['filename'] = None
            raise CTDError(self._filename, self._NO_LOCATION_ERROR)

    def remove_upcasts(self):
        """
        Removes upcasts based on the rate of change of pressure over time.
        This function calculates the vertical speed of the system through the water
        using the derivative of pressure with respect to time. It filters out data
        collected in the air or while stationary at the surface or bottom, and
        separates the downcasts from upcasts.
        """
        # Ensure 'timestamp' is a datetime object for correct differentiation
        if not pd.api.types.is_datetime64_any_dtype(self._ctd_array['timestamp']):
            self._ctd_array['timestamp'] = pd.to_datetime(self._ctd_array['timestamp'])

        # Calculate the time differences in seconds
        time_diffs = self._ctd_array['timestamp'].diff().dt.total_seconds()

        # Calculate the rate of change of pressure over time (dP/dt)
        pressure_diffs = self._ctd_array['pressure_00'].diff()
        vertical_speed = pressure_diffs / time_diffs

        # Add vertical speed to the DataFrame
        self._ctd_array['vertical_speed'] = vertical_speed

        # Filter out data where the rate of change of pressure indicates upcasting or stationary periods
        # Assuming positive vertical speed indicates downcasting (i.e., increasing pressure)
        # You might need to adjust this logic based on the actual data characteristics
        self._ctd_array = self._ctd_array[
            (self._ctd_array['vertical_speed'] > 0) & (~self._ctd_array['vertical_speed'].isna())]
        self._ctd_array.drop('vertical_speed', axis=1)

    def remove_non_positive_samples(self):
        """
        Iterates through the columns of the CTD data table and removes rows with non-positive values
        for depth, pressure, salinity, absolute salinity, or density.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(self._filename, self._NO_SAMPLES_ERROR)
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
            raise CTDError(self._filename, self._REMOVE_NEGATIVES_ERROR)

    def remove_invalid_salinity_values(self):
        """
        Removes rows with invalid values (<10) for practical salinity.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(self._filename, self._NO_SAMPLES_ERROR)
        for column in self._ctd_array.columns:
            match column:
                case 'salinity_00':
                    self._ctd_array = self.Utility.remove_rows_with_invalid_salinity(self._ctd_array)
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(self._filename, self._REMOVE_NEGATIVES_ERROR)

    def bin_average_pressure(self, bin_size=0.5):
        """
        Bins by pressure and averages numeric columns.

        Parameters
        ----------
        bin_size: float
            Size of pressure bins, defaults to 0.5.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(self._filename, self._NO_SAMPLES_ERROR)
        # Check if 'Pressure_Bin' already exists and drop it if it does
        if 'Pressure_Bin' in self._ctd_array.columns:
            self._ctd_array.drop(columns=['Pressure_Bin'], inplace=True)

        # Binning data using the specified bin size
        self._ctd_array['Pressure_Bin'] = (self._ctd_array['pressure_00'] // bin_size) * bin_size

        # Identifying numeric and non-numeric columns
        numeric_cols = self._ctd_array.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('pressure_00')  # Remove the original pressure column if it's still in the list
        non_numeric_cols = ['timestamp', 'filename', 'latitude',
                            'longitude']  # specify any non-numeric column you expect to handle specially
        # Creating aggregation dictionary
        agg_dict = {col: 'mean' for col in numeric_cols}
        agg_dict.update({col: 'first' for col in non_numeric_cols})
        # Aggregating data by the new pressure bins
        data_binned = self._ctd_array.groupby('Pressure_Bin').agg(agg_dict).reset_index(drop=True)

        # Cleaning up DataFrame, resetting the index and column names
        data_binned.rename(columns={'Pressure_Bin': 'pressure_00'}, inplace=True)
        self._ctd_array = data_binned.copy(deep=True)

        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(self._filename, self._NO_SAMPLES_ERROR)

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
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(self._filename, self._NO_SAMPLES_ERROR)
        supported_features = {
            "practicalsalinity": "salinity_00"
        }
        supported_methods = {
            "salinitydiff": Calculate.calculate_and_drop_salinity_spikes(self._ctd_array),
            "autoencoder": self.clean_salinity_autoencoder()
        }
        if feature in supported_features.keys():
            if method in supported_methods.keys():
                self._ctd_array = supported_methods[method]
            else:
                logger.error(f"clean: Invalid method \"{method}\" not in {supported_methods.keys()}", exc_info=True)
        else:
            logger.error(f"clean: Invalid feature \"{feature}\" not in {supported_features.keys()}.", exc_info=True)
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(self._filename, self._DATA_CLEANING_ERROR)

    def add_absolute_salinity(self):
        """
        Calculates the absolute salinity using the TEOS-10 equations and adds it as a new column
        to the CTD data table. Removes rows with negative absolute salinity values.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(self._filename, self._NO_SAMPLES_ERROR)
        self._ctd_array.loc[self._ctd_array.index, 'salinityabs'] = Calculate.calculate_absolute_salinity(
            self._ctd_array)
        self._ctd_array = self.Utility.remove_rows_with_negative_salinityabs(self._ctd_array)
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(self._filename, self._SALINITYABS_CALCULATION_ERROR)

    def add_density(self):
        """
        Calculates the density using the TEOS-10 equations and adds it as a new column to the CTD
        data table. If absolute salinity is not present, it is calculated first.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(self._filename, self._NO_SAMPLES_ERROR)
        if 'salinityabs' in self._ctd_array.columns:
            self._ctd_array.loc[self._ctd_array.index, 'density'] = Calculate.calculate_absolute_density(
                self._ctd_array)
        else:
            self.add_absolute_salinity()
            self._ctd_array.loc[self._ctd_array.index, 'density'] = Calculate.calculate_absolute_density(
                self._ctd_array)
            self._ctd_array.drop('salinityabs')
            if self.Utility.no_values_in_object(self._ctd_array):
                raise CTDError(self._filename, self._DENSITY_CALCULATION_ERROR)

    def add_overturns(self):
        """
        Calculates density changes between consecutive measurements and identifies overturns where
        denser water lies above less dense water. Adds an 'overturn' column to the CTD data table.
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(self._filename, self._NO_SAMPLES_ERROR)
        self._ctd_array = Calculate.calculate_overturns(self._ctd_array.copy())

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
            raise CTDError(self._filename, self._NO_SAMPLES_ERROR)
        surface_salinity, meltwater_fraction = (Calculate.calculate_surface_salinity_and_meltwater
                                                (self._ctd_array, start, end))
        surface_temperature = (Calculate.calculate_surface_temperature(self._ctd_array, start, end))
        self._ctd_array = self._ctd_array.assign(surface_salinity=surface_salinity)
        self._ctd_array = self._ctd_array.assign(meltwater_fraction=meltwater_fraction)
        self._ctd_array = self._ctd_array.assign(surface_temperature=surface_temperature)

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
            raise CTDError(self._filename, self._NO_SAMPLES_ERROR)
        mean_surface_density = Calculate.calculate_mean_surface_density(self._ctd_array.copy(), (start, end))
        self._ctd_array = self._ctd_array.assign(mean_surface_density=mean_surface_density)

    def add_mld(self, reference, method="default"):
        """
        Calculates the mixed layer depth using the specified method and reference depth.
        Adds the MLD and the actual reference depth used as new columns to the CTD data table.

        Parameters
        ----------
        reference : int
            The reference depth for MLD calculation.
        method : int
            The MLD calculation method (default: "default").
        """
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(self._filename, self._NO_SAMPLES_ERROR)
        copy_ctd_array = self._ctd_array.copy()
        supported_methods = [
            "default"
        ]
        unpack = None

        if method == "default":
            unpack = Calculate.calculate_mld(copy_ctd_array['density'], copy_ctd_array['depth_00'],
                                             reference)
        else:
            logger.error(f"add_mld: Invalid method \"{method}\" not in {supported_methods}", exc_info=True)
            unpack = [None, None]
        if unpack is None:
            unpack = [None, None]
            raise CTDError(self._filename, "MLD could not be calculated.")
        MLD = unpack[0]
        depth_used_as_reference = unpack[1]
        self._ctd_array.loc[self._ctd_array.index, f'MLD {reference}'] = MLD
        self._ctd_array.loc[
            self._ctd_array.index, f'MLD {reference} Actual Reference Depth'] = depth_used_as_reference
        if self.Utility.no_values_in_object(self._ctd_array):
            raise CTDError(self._filename, self._MLD_ERROR)

    def plot_depth_salinity_density_mld_line(self):
        """
        Generates a plot of depth vs. salinity and density, applying LOESS smoothing to the data.
        Adds horizontal lines indicating the mixed layer depth (MLD) at different reference depths.
        Saves the plot as an image file.
        """
        df = self._ctd_array.copy()
        filename = self._filename
        plt.rcParams.update({'font.size': 16})
        df_filtered = df
        if df_filtered.isnull().values.any():
            df_filtered.dropna(inplace=True)  # Drop rows with NaNs
        df_filtered = df_filtered.reset_index(drop=True)
        if len(df_filtered) < 1:
            return
        surface_salinity = df_filtered.loc['surface_salinity'][0]
        surface_temperature = df_filtered.loc['surface_temperature'][0]
        meltwater_fraction = df_filtered.loc['meltwater_fraction'][0]
        fig, ax1 = plt.subplots(figsize=(18, 18))
        ax1.invert_yaxis()
        # Dynamically set y-axis limits based on depth data
        max_depth = df_filtered['depth_00'].max()
        ax1.set_ylim([max_depth, 0])  # Assuming depth increases downwards
        lowess = statsmodels.api.nonparametric.lowess
        salinity_lowess = lowess(df_filtered['salinity_00'], df_filtered['depth_00'], frac=0.1)
        salinity_depths, salinity_smooth = zip(*salinity_lowess)
        color_salinity = 'tab:blue'
        ax1.plot(salinity_smooth, salinity_depths, color=color_salinity, label='Practical Salinity')
        ax1.set_xlabel('Practical Salinity (PSU)', color=color_salinity)
        ax1.set_ylabel('Depth (m)')
        ax1.tick_params(axis='x', labelcolor=color_salinity)
        density_lowess = lowess(df_filtered['density'], df_filtered['depth_00'], frac=0.1)
        density_depths, density_smooth = zip(*density_lowess)
        ax2 = ax1.twiny()
        color_density = 'tab:red'
        ax2.plot(density_smooth, density_depths, color=color_density, label='Density (kg/m^3)')
        ax2.set_xlabel('Density (kg/m^3)', color=color_density)
        ax2.tick_params(axis='x', labelcolor=color_density)
        ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        mld_cols = []
        for col in df.columns:
            if 'MLD' in col and 'Actual' not in col:
                mld_cols.append(df[col])
        refdepth_cols = []
        for col in df.columns:
            if 'Actual' in col:
                refdepth_cols.append(df[col])
        for idx, mld_col in enumerate(mld_cols):
            ax1.axhline(y=mld_col.iloc[0], color='green', linestyle='--',
                        label=f'MLD {refdepth_cols[idx].iloc[0]} Ref')
            ax1.text(0.95, mld_col.iloc[0], f'MLD with respect to {refdepth_cols[idx].iloc[0]}m', va='center',
                     ha='right', backgroundcolor='white', color='green', transform=ax1.get_yaxis_transform())
        if df_filtered['overturn'].any():
            plt.title(
                f"{filename}\n Depth vs. Salinity and Density with LOESS Transform "
                f"\n THIS IS AN UNSTABLE WATER COLUMN "
                f"\n(Higher density fluid lies above lower density fluid)")
        else:
            plt.title(
                f"{filename}\n Depth vs. Salinity and Density with LOESS Transform \n Surface Salinity {surface_salinity}\n, Surface Temperature {surface_temperature}\n, Meltwater Fraction {meltwater_fraction}\n")

        ax1.grid(True)
        lines, labels = ax1.get_legend_handles_labels()
        ax2_legend = ax2.get_legend_handles_labels()
        ax1.legend(lines + ax2_legend[0], labels + ax2_legend[1], loc='lower center', bbox_to_anchor=(0.5, -0.15),
                   ncol=3)
        plot_path = os.path.join(self._cwd, f"plots/{filename}_salinity_density_depth_plot_dual_x_axes_line.png")
        plot_folder = os.path.join(self._cwd, "plots")
        if not (os.path.isdir(plot_folder)):
            os.mkdir(plot_folder)
        plt.savefig(plot_path)
        plt.close(fig)

    def plot_depth_density_salinity_mld_scatter(self):
        """
        Generates a scatter plot of depth vs. salinity and density.
        Adds horizontal lines indicating the mixed layer depth (MLD) at different reference depths.
        Saves the plot as an image file.
        """
        df = self._ctd_array.copy()
        filename = self._filename
        plt.rcParams.update({'font.size': 16})
        df_filtered = df
        if df_filtered.empty:
            plt.close()
            return
        df_filtered = df_filtered.reset_index(drop=True)
        surface_salinity = df_filtered.surface_salinity[0]
        surface_temperature = df_filtered.surface_temperature[0]
        meltwater_fraction = df_filtered.meltwater_fraction[0]
        fig, ax1 = plt.subplots(figsize=(18, 18))
        ax1.invert_yaxis()
        # Dynamically set y-axis limits based on depth data
        max_depth = df_filtered['depth_00'].max()
        ax1.set_ylim([max_depth, 0])  # Assuming depth increases downwards
        color_salinity = 'tab:blue'
        ax1.scatter(df_filtered['salinity_00'], df_filtered['depth_00'], color=color_salinity,
                    label='Practical Salinity')
        ax1.set_xlabel('Practical Salinity (PSU)', color=color_salinity)
        ax1.set_ylabel('Depth (m)')
        ax1.tick_params(axis='x', labelcolor=color_salinity)
        ax2 = ax1.twiny()
        color_density = 'tab:red'
        ax2.scatter(df_filtered['density'], df_filtered['depth_00'], color=color_density, label='Density (kg/m^3)')
        ax2.set_xlabel('Density (kg/m^3)', color=color_density)
        ax2.tick_params(axis='x', labelcolor=color_density)
        ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        mld_cols = []
        for col in df.columns:
            if 'MLD' in col and 'Actual' not in col:
                mld_cols.append(df[col])
        refdepth_cols = []
        for col in df.columns:
            if 'Actual' in col:
                refdepth_cols.append(df[col])
        for idx, mld_col in enumerate(mld_cols):
            ax1.axhline(y=mld_col.iloc[0], color='green', linestyle='--',
                        label=f'MLD {refdepth_cols[idx].iloc[0]} Ref')
            ax1.text(0.95, mld_col.iloc[0], f'MLD with respect to {refdepth_cols[idx].iloc[0]}m', va='center',
                     ha='right', backgroundcolor='white', color='green', transform=ax1.get_yaxis_transform())
        if df_filtered['overturn'].any():
            plt.title(
                f"{filename}\n Depth vs. Salinity and Density "
                f"\n THIS IS AN UNSTABLE WATER COLUMN "
                f"\n(Higher density fluid lies above lower density fluid)")
        else:
            plt.title(
                f"{filename}\n Depth vs. Salinity and Density \n Surface Salinity {surface_salinity}, Meltwater Fraction {meltwater_fraction}\n")

        ax1.grid(True)
        lines, labels = ax1.get_legend_handles_labels()
        ax2_legend = ax2.get_legend_handles_labels()
        ax1.legend(lines + ax2_legend[0], labels + ax2_legend[1], loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   ncol=3)
        plot_path = os.path.join(self._cwd, f"plots/{filename}_salinity_density_depth_plot_dual_x_axes.png")
        plot_folder = os.path.join(self._cwd, "plots")
        if not (os.path.isdir(plot_folder)):
            os.mkdir(plot_folder)
        plt.savefig(plot_path)
        plt.close(fig)

    def plot_depth_temperature_scatter(self):
        """
        Generates a scatter plot of depth vs. temperature.
        Adds horizontal lines indicating the mixed layer depth (MLD) at different reference depths.
        Saves the plot as an image file.
        """
        df = self._ctd_array.copy()
        filename = self._filename
        plt.rcParams.update({'font.size': 16})
        df_filtered = df
        if df_filtered.empty:
            plt.close()
            return
        df_filtered = df_filtered.reset_index(drop=True)
        surface_salinity = df_filtered.surface_salinity[0]
        surface_temperature = df_filtered.surface_temperature[0]
        meltwater_fraction = df_filtered.meltwater_fraction[0]
        fig, ax1 = plt.subplots(figsize=(18, 18))
        ax1.invert_yaxis()
        # Dynamically set y-axis limits based on depth data
        max_depth = df_filtered['depth_00'].max()
        ax1.set_ylim([max_depth, 0])  # Assuming depth increases downwards

        color_temp = 'tab:blue'
        ax1.scatter(df_filtered['temperature_00'], df_filtered['depth_00'], color=color_temp,
                    label="Temperature (°C)")
        ax1.set_xlabel("Temperature (°C)", color=color_temp)
        ax1.set_ylabel('Depth (m)')
        ax1.tick_params(axis='x', labelcolor=color_temp)
        mld_cols = []
        for col in df.columns:
            if "MLD" in col and "Actual" not in col:
                mld_cols.append(df[col])
        refdepth_cols = []
        for col in df.columns:
            if "Reference Depth" in col:
                refdepth_cols.append(df[col])
        for idx, mld_col in enumerate(mld_cols):
            ax1.axhline(y=mld_col.iloc[0], color='green', linestyle='--',
                        label=f'MLD {refdepth_cols[idx].iloc[0]} Ref')
            ax1.text(0.95, mld_col.iloc[0], f'MLD with respect to {refdepth_cols[idx].iloc[0]}m', va='center',
                     ha='right', backgroundcolor='white', color='green', transform=ax1.get_yaxis_transform())
        if df_filtered['overturn'].any():
            plt.title(
                f"{filename}\n Depth vs. Temperature \n "
                f"THIS IS AN UNSTABLE WATER COLUMN \n"
                f"(Higher density fluid lies above lower density fluid)")
        else:
            plt.title(
                f"{filename}\n Depth vs. Temperature \n Surface Temperature {surface_temperature}\n")
        ax1.grid(True)
        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plot_path = os.path.join(self._cwd, f"plots/{filename}_temperature_depth_plot.png")
        plot_folder = os.path.join(self._cwd, "plots")
        if not (os.path.isdir(plot_folder)):
            os.mkdir(plot_folder)
        plt.savefig(plot_path)
        plt.close(fig)

    def _process_master_sheet(self, master_sheet_path, filename):
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

        Returns
        -------
        tuple
            A tuple containing the estimated latitude, longitude, and updated filename.
        """

        def get_date_from_string(filename):
            try:
                year = filename.split('_')[1][:4]
                month = filename.split('_')[1][4:6]
                day = filename.split('_')[1][6:]
                hour = filename.split('_')[2][0:2]
                minute = filename.split('_')[2][2:4]
                time = f"{hour}:{minute}"
                return float(year), float(month), float(day), time
            except:
                return None, None, None, None

        # Function to calculate the absolute difference between two dates
        def date_difference(row, target_year, target_month, target_day):
            return abs(row['year'] - target_year) + abs(row['month'] - target_month) + abs(
                row['day'] - target_day)

        # Function to calculate the absolute difference between two times
        def time_difference(target_time, df_time):
            df_time_str = str(df_time)
            try:
                target_hour, target_minute = [int(target_time.split(':')[0]), int(target_time.split(':')[1])]
            except ValueError as e:
                raise CTDError(filename, "Malformed filename, could not parse time.")
            try:
                df_hour, df_minute = [int(df_time_str.split(':')[0]), int(df_time_str.split(':')[1])]
            except:
                # Invalid time
                return 1000000
            return abs((target_hour * 60 + target_minute) - (df_hour * 60 + df_minute))

        # Check if the master sheet is already cached
        if self._cached_master_sheet.empty:
            # Load the master sheet and cache it
            logger.info("Reloading master sheet.")
            self._cached_master_sheet = pd.read_excel(master_sheet_path)
        master_df = self._cached_master_sheet
        # Get date and time components from the filename
        year, month, day, time = get_date_from_string(filename)
        if year is None:
            return None, None, None
        # Calculate absolute differences for each row in 'master_df'
        master_df['date_difference'] = master_df.apply(date_difference, args=(year, month, day), axis=1)
        master_df['time_difference'] = master_df['time_local'].apply(lambda x: time_difference(time, x))
        # Find the rows with the smallest total difference for date
        smallest_date_difference = master_df['date_difference'].min()
        closest_date_rows = master_df[master_df['date_difference'] == smallest_date_difference]
        # Check if time_difference returns None
        if closest_date_rows['time_difference'].isnull().any():
            closest_time_time = None
            closest_row_overall = closest_date_rows.iloc[0]
        else:
            # If there are multiple rows with the smallest date difference, select the row with the smallest time difference
            if len(closest_date_rows) > 1:
                try:
                    closest_time_row = closest_date_rows.loc[closest_date_rows['time_difference'].idxmin()]
                except TypeError as e:
                    raise CTDError(filename, "Malformed time input.")
                closest_row_overall = closest_time_row
                closest_time_time = closest_row_overall['time_local']
            else:
                closest_row_overall = closest_date_rows.iloc[0]
                closest_time_time = closest_row_overall['time_local']
        latitude = closest_row_overall['latitude']
        longitude = closest_row_overall['longitude']
        unique_id = closest_row_overall.iloc[0]
        RBRfilename = filename + "_gpscm"
        # Access the closest date components
        closest_date_year = closest_row_overall['year']
        closest_date_month = closest_row_overall['month']
        closest_date_day = closest_row_overall['day']
        # Print the closest date and time
        logger.warning("|-ESTIMATION ALERT-|")
        logger.warning("Had to guess location on file: " + filename)
        logger.warning("Unique ID: " + unique_id)
        logger.warning("Closest Date (Year, Month, Day):" + str(closest_date_year) + str(closest_date_month) + str(
            closest_date_day))
        logger.warning("Lat: " + str(latitude))
        logger.warning("Long: " + str(longitude))
        if closest_time_time:
            logger.warning("Closest Time:" + str(closest_time_time))
        logger.warning("====================")
        return latitude, longitude, RBRfilename

    def get_sample_location(self, rsk, filename):
        """
        Retrieves the sample location data from the RSK file. If no location data is found,
        it attempts to estimate the location using the master sheet. Returns the latitude,
        longitude, and updated filename.

        Parameters
        ----------
        rsk : RSK
            Ruskin object of the RSK file.
        filename : str
            The filename of the RSK file.

        Returns
        -------
        tuple
            A tuple containing the latitude associated with the sample, longitude associated with the sample,
            and the filename, adds _gps if the location was in the ruskin file,
            _gpscm if located via mastersheet, or _gpserror if unable to locate.
        """
        # Adding geo data, assumes no drift and uses the first lat long in the file if there is one
        geo_data_length = len(list(itertools.islice(rsk.geodata(), None)))
        if geo_data_length < 1:
            latitude_intermediate, longitude_intermediate, filename = self._process_master_sheet(
                self.master_sheet_path, filename)
            return latitude_intermediate, longitude_intermediate, filename
        else:
            for geo in itertools.islice(rsk.geodata(), None):
                # Is there geo data?
                if geo.latitude is not None:
                    # If there is, is it from the southern ocean?
                    if not (geo.latitude > -60):
                        try:
                            latitude_intermediate = geo.latitude[0]
                            longitude_intermediate = geo.longitude[0]
                            filename += "_gps"
                            return latitude_intermediate, longitude_intermediate, filename
                        except:
                            latitude_intermediate = geo.latitude
                            longitude_intermediate = geo.longitude
                            filename += "_gps"
                            return latitude_intermediate, longitude_intermediate, filename
                    else:
                        latitude_intermediate, longitude_intermediate, filename = self._process_master_sheet(
                            self.master_sheet_path, filename)
                        return latitude_intermediate, longitude_intermediate, filename
                else:
                    return None, None, filename + 'gpserror'

    def clean_salinity_autoencoder(self):
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

        def build_lstm_autoencoder(input_shape):
            """
            LSTM autoencoder architecture.
            """
            # Encoder
            encoder_inputs = Input(shape=(input_shape[0], input_shape[1]), name='encoder_inputs')
            encoder_lstm1 = LSTM(16, activation='tanh',
                                 return_sequences=True)(encoder_inputs)
            encoder_lstm2 = LSTM(8, activation='tanh',
                                 return_sequences=True)(encoder_lstm1)
            encoder_lstm3 = LSTM(8, activation='tanh',
                                 return_sequences=True)(encoder_lstm2)
            encoder_lstm4 = LSTM(8, activation='tanh',
                                 return_sequences=False)(encoder_lstm3)
            encoder_output = Dense(8, activation='linear')(encoder_lstm4)

            # Decoder
            decoder_inputs = RepeatVector(input_shape[0])(encoder_output)
            decoder_lstm1 = LSTM(8, activation='tanh',
                                 return_sequences=True)(decoder_inputs)
            decoder_lstm2 = LSTM(8, activation='tanh',
                                 return_sequences=True)(decoder_lstm1)
            decoder_lstm3 = LSTM(8, activation='tanh',
                                 return_sequences=True)(decoder_lstm2)
            decoder_lstm4 = LSTM(16, activation='tanh',
                                 return_sequences=True)(decoder_lstm3)
            decoder_output = TimeDistributed(Dense(input_shape[1], activation='linear'), name='decoded_output')(
                decoder_lstm4)

            autoencoder = Model(encoder_inputs, decoder_output)
            optimizer = Adam(learning_rate=0.03)
            autoencoder.compile(optimizer=optimizer, loss=loss_function)
            return autoencoder

        def plot_original_data(data):
            filename = data['filename'].iloc[0]
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

        def plot_predicted_data(predicted_df, xlim, ylim):
            filename = predicted_df['filename'].iloc[0]
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

        def run_autoencoder(show_plots=True):
            """
            Runs the autoencoder.

            Returns
            -------
            DataFrame
                CTD data with clean salinity values.
            """
            # Preprocess data
            data = self._ctd_array.copy()
            # Binning data
            data.loc[:, 'Pressure_Bin'] = data['pressure_00'] // 0.25 * 0.25
            # Define the desired columns and their aggregation functions
            column_agg_dict = {
                "temperature_00": "mean",
                "pressure_00": "median",
                "chlorophyll_00": "mean",
                "seapressure_00": "median",
                "depth_00": "median",
                "salinity_00": "mean",
                "speedofsound_00": "mean",
                "specificconductivity_00": "mean",
                "conductivity_00": "mean",
                "density": "mean",
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
                "overturn": "first"
            }
            # Check which columns are present in the DataFrame
            available_columns = {col: agg_func for col, agg_func in column_agg_dict.items() if col in data.columns}
            # Group by and aggregate based on the available columns
            data_binned = data.groupby(['filename', 'Pressure_Bin']).agg(available_columns).reset_index()
            # Rename the Pressure_Bin column if it exists
            if 'Pressure_Bin' in data_binned.columns:
                data_binned.rename(columns={'Pressure_Bin': 'pressure_00'}, inplace=True)
            # Scaling data
            numerical_columns = ['salinity_00']
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_sequence = scaler.fit_transform(data_binned[numerical_columns])
            if len(scaled_sequence) < 2:
                raise CTDError(self._filename, "Not enough values to run the autoencoder on this data.")
            scaled_seq = np.expand_dims(scaled_sequence, axis=0)
            min_pres = data_binned['depth_00'].min()
            max_pres = data_binned['depth_00'].max()
            # Calculate ranges
            pres_range = max_pres - min_pres
            epochs = int(pres_range * 18)
            # Build autoencoder and predict on CTD data
            autoencoder = build_lstm_autoencoder(scaled_seq.shape[1:])
            autoencoder.fit(scaled_seq, scaled_seq, epochs=epochs, verbose=0, batch_size=4)
            X_pred = autoencoder.predict(scaled_seq, verbose=None)
            # Revert scaling
            predicted_seq = scaler.inverse_transform(X_pred[0])
            data_binned['salinity_00'] = predicted_seq[:, 0]
            if show_plots:
                xlim, ylim = plot_original_data(self._ctd_array)
                plot_predicted_data(data_binned, xlim, ylim)
            data_binned = data_binned.loc[:, ~data_binned.columns.duplicated()].copy()
            return data_binned.reindex()

        return run_autoencoder()

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

        def __init__(self, filename):
            """
            Initialize a new Utility object.
            Parameters
            ----------
            filename : str
                The filename of the RSK file.
            """
            self.filename = filename

        def no_values_in_object(self, object_to_check):
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
            if object_to_check.empty:
                return True
            if len(object_to_check) > 0:
                return False

        def remove_sample_timezone_indicator(self, df):
            """
            Removes the timezone indicator (e.g., '+00:00') from the 'timestamp' column of the
            given DataFrame. Returns the updated DataFrame.

            Parameters
            ----------
            df : DataFrame
                The DataFrame to process.

            Returns
            -------
            DataFrame
                The updated DataFrame with the timezone indicator removed.
            """
            if self.no_values_in_object(df):
                return None
            if 'timestamp' in df.columns:
                df['timestamp'] = df['timestamp'].astype(str).str.split('+').str[0]
                return df
            else:
                return df

        def remove_rows_with_negative_depth(self, df):
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
            if self.no_values_in_object(df):
                return None
            if 'depth_00' in df.columns:
                df = df[df['depth_00'] >= 0].reset_index(drop=True)
            else:
                return None
            if self.no_values_in_object(df):
                return None
            return df.copy()

        def remove_rows_with_negative_salinity(self, df):
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
            if self.no_values_in_object(df):
                return None
            if 'salinity_00' in df.columns:
                df = df[df['salinity_00'] >= 0].reset_index(drop=True)
            else:
                return None
            if self.no_values_in_object(df):
                return None
            return df.copy()

        def remove_rows_with_invalid_salinity(self, df):
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
            if self.no_values_in_object(df):
                return None
            if 'salinity_00' in df.columns:
                df = df[df['salinity_00'] >= 10].reset_index(drop=True)
            else:
                return None
            if self.no_values_in_object(df):
                return None
            return df.copy()

        def remove_rows_with_negative_pressure(self, df):
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
            if self.no_values_in_object(df):
                return None
            if 'pressure_00' in df.columns:
                df = df[df['pressure_00'] >= 0].reset_index(drop=True)
            else:
                return None
            if self.no_values_in_object(df):
                return None
            return df.copy()

        def remove_rows_with_negative_salinityabs(self, df):
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
            if self.no_values_in_object(df):
                return None
            if 'salinityabs' in df.columns:
                df = df[df['salinityabs'] >= 0].reset_index(drop=True)
            else:
                return None
            if self.no_values_in_object(df):
                return None
            return df.copy()

        def remove_rows_with_negative_density(self, df):
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
            if self.no_values_in_object(df):
                return None
            if 'density' in df.columns:
                df = df[df['density'] >= 0].reset_index(drop=True)
            else:
                return None
            if self.no_values_in_object(df):
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
                "salinityabs": "Absolute Salinity_(g/kg)",
                "MLD_Zero": "MLD Zero_(m)",
                "MLD_Ten": "MLD Ten_(m)",
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
            except FileNotFoundError:
                logger.info(f"The file {output_file} does not exist. A new file will be created.")
                csv_df = pd.DataFrame()

            # Merge the existing DataFrame with the new DataFrame
            merged_df = pd.concat((csv_df, data), ignore_index=True)
            # Overwrite the original CSV file with the merged DataFrame
            merged_df.to_csv(output_file, index=False)

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
                                logger.error("Date format does not match expected '%Y-%m-%d %H:%M:%S'")
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


class Calculate:
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
        p : Series
            Sea pressure in dbar (absolute pressure minus 10.1325 dbar).

        Returns
        -------
        Series of bool
            A boolean array where True indicates the values are inside the "oceanographic funnel".
        """
        # Ensure all inputs are Series and aligned
        if not (isinstance(SA, pd.Series) and isinstance(CT, pd.Series) and (
                isinstance(p, pd.Series) or np.isscalar(p))):
            raise CTDError("", "SA, CT, and p must be pandas Series or p a scalar")

        if isinstance(p, pd.Series) and (SA.index.equals(CT.index) and SA.index.equals(p.index)) is False:
            raise CTDError("", "Indices of SA, CT, and p must be aligned")

        if np.isscalar(p):
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
            logger.error("Insufficient depth range to calculate.")
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
        if Calculate.gsw_infunnel(sa, ct, p).all():
            return gsw.density.rho_t_exact(sa, t, p)
        else:
            raise CTDError("", "Sample not in funnel, could not calculate density.")

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
    def calculate_mld(densities, depths, reference_depth, delta=0.15):
        """
        Calculates the mixed layer depth (MLD) using the density threshold method.
        MLD is the depth at which the density exceeds the reference density
        by a predefined amount delta, which defaults to (0.03 kg/m³).

        Parameters
        ----------
        densities : Series
            Series of densities
        depths : Series
            Series of depths corresponding to densities
        reference_depth : float
            The depth at which to anchor the reference density
        delta : float, optional
            The difference in density which would indicate the MLD, defaults to 0.1 kg/m³.

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
                        raise CTDError("",
                                       f"Insufficient depth range to calculate MLD. "
                                       f"Maximum sample depth is "f"{depths.max()}, minimum is {depths.min()}")
                break
        if reference_density is None:
            return None
        # Find the depth where density exceeds the reference density by more than 0.05 kg/m³
        for depth, density in zip(sorted_depths, sorted_densities):
            if density > reference_density + delta and depth >= reference_depth:
                return depth, reference_depth
        return None  # If no depth meets the criterion

    @staticmethod
    def calculate_mld_loess(densities, depths, reference_depth, delta=0.15):
        """
        Calculates the mixed layer depth (MLD) using LOESS smoothing to first smooth the density profile and
        then determine the depth where the smoothed density exceeds the reference density
        by a predefined amount which defaults to 0.03 kg/m³.

        Parameters
        ----------
        densities : Series
            Series of densities
        depths : Series
            Series of depths corresponding to densities
        reference_depth :
            The depth at which to anchor the reference density
        delta : float, optional
            The difference in density which would indicate the MLD, defaults to 0.03 kg/m.

        Returns
        -------
        tuple
            A tuple containing the calculated MLD and the reference depth used to calculate MLD.
        """
        # Ensure input is pandas Series and drop NA values
        if isinstance(densities, pd.Series) and isinstance(depths, pd.Series):
            densities = densities.dropna().reset_index(drop=True)
            depths = depths.dropna().reset_index(drop=True)

        # Convert to numeric and ensure no NaNs remain
        densities = densities.apply(pd.to_numeric, errors='coerce')
        depths = depths.apply(pd.to_numeric, errors='coerce')
        densities = densities.dropna().reset_index(drop=True)
        depths = depths.dropna().reset_index(drop=True)
        if densities.empty or depths.empty:
            return None, None

        # Convert pandas Series to numpy arrays for NumPy operations
        densities = densities.to_numpy()
        depths = depths.to_numpy()

        # Remove duplicates by averaging densities at the same depth
        unique_depths, indices = np.unique(depths, return_inverse=True)
        average_densities = np.zeros_like(unique_depths)
        np.add.at(average_densities, indices, densities)
        counts = np.zeros_like(unique_depths)
        np.add.at(counts, indices, 1)
        average_densities /= counts

        # Apply LOESS smoothing
        lowess = statsmodels.api.nonparametric.lowess
        smoothed = lowess(average_densities, unique_depths, frac=0.1)
        smoothed_depths, smoothed_densities = zip(*smoothed)
        reference_density = np.interp(reference_depth, smoothed_depths, smoothed_densities)

        # Find the depth where density exceeds the reference density by more than 0.05 kg/m³
        exceeding_indices = np.where(np.array(smoothed_densities) > reference_density + delta
                                     and np.array(smoothed_densities) > reference_depth)[0]
        if exceeding_indices.size > 0:
            mld_depth = smoothed_depths[exceeding_indices[0]]
            return mld_depth, reference_depth

        return None, None  # If no depth meets the criterion

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


class CTDError(Exception):
    """
    Exception raised for CTD related errors.

    Parameters
    ----------
    filename: input dataset which caused the error
    message: message -- explanation of the error
    """

    def __init__(self, filename, message=" Unknown, check to make sure your mastersheet is in your current directory."):
        self.filename = filename
        self.message = f"{filename}: {message}"
        logger.error(f"{filename}: {message}")
        super().__init__(self.message)


def process_ctd_file(file, plot=False, cached_master_sheet=pd.DataFrame(), master_sheet_path=None):
    try:
        my_data = CTD(file, cached_master_sheet=cached_master_sheet, master_sheet_path=master_sheet_path)
        my_data.add_filename_to_table()
        my_data.add_location_to_table()
        my_data.remove_upcasts()
        my_data.remove_non_positive_samples()
        my_data.remove_invalid_salinity_values()
        my_data.clean("practicalsalinity", 'autoencoder')
        my_data.add_surface_salinity_temp_meltwater()
        my_data.add_absolute_salinity()
        my_data.add_density()
        my_data.add_overturns()
        my_data.add_mld(1)
        my_data.add_mld(5)
        if plot:
            my_data.plot_depth_density_salinity_mld_scatter()
            my_data.plot_depth_temperature_scatter()
    except CTDError as e:
        return pd.DataFrame()
    return my_data.get_pandas_df()


def run_default(plot=False, master_sheet_path=None, max_workers=1):
    ctd_files_list = get_rsk_filenames_in_dir(_get_cwd())
    ctd_files_list.extend(get_csv_filenames_in_dir(_get_cwd()))
    cached_master_sheet = pd.read_excel(master_sheet_path)
    if not ctd_files_list:
        logger.info("No files to process.")
        return

    # Process the first file
    first_file = ctd_files_list[0]
    df = process_ctd_file(first_file, plot, cached_master_sheet=cached_master_sheet,
                          master_sheet_path=master_sheet_path)

    # Process the rest of the files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_ctd_file, file, plot, cached_master_sheet=cached_master_sheet,
                                   master_sheet_path=master_sheet_path): file for file
                   in ctd_files_list[1:]}
        for future in as_completed(futures):
            result = future.result()
            if not result.empty:
                df = pd.concat((df, result), ignore_index=True)
    CTD.Utility.save_to_csv(df, "outputclean.csv")


def merge_all_in_folder():
    ctd_files_list = get_rsk_filenames_in_dir(_get_cwd())
    ctd_files_list.extend(get_csv_filenames_in_dir(_get_cwd()))
    error_out = []
    for file in ctd_files_list:
        try:
            my_data = CTD(file)
            my_data.add_filename_to_table()
            my_data.add_location_to_table()
            my_data.remove_upcasts()
            my_data.remove_non_positive_samples()
            my_data.save_to_csv("output.csv")
        except Exception:
            continue


def get_rsk_filenames_in_dir(working_directory):
    rsk_files_list = []
    rsk_filenames_no_path = []
    for filename in os.listdir(working_directory):
        if filename.endswith('.rsk'):
            for filepath in rsk_files_list:
                filename_no_path = ('_'.join(filepath.split("/")[-1].split("_")[0:3]).split('.rsk')[0])
                if filename_no_path in rsk_filenames_no_path:
                    continue
                rsk_filenames_no_path.append(filename_no_path)
            file_path = os.path.join(working_directory, filename)
            rsk_files_list.append(file_path)
    return rsk_files_list


def get_csv_filenames_in_dir(working_directory):
    csv_files_list = []
    rsk_filenames_no_path = []
    for filename in os.listdir(working_directory):
        if filename.endswith('.csv') and 'output' not in filename:
            for filepath in csv_files_list:
                filename_no_path = ('_'.join(filepath.split("/")[-1].split("_")[0:3]).split('.csv')[0])
                if filename_no_path in rsk_filenames_no_path:
                    continue
                rsk_filenames_no_path.append(filename_no_path)
            file_path = os.path.join(working_directory, filename)
            csv_files_list.append(file_path)
    return csv_files_list


def _get_cwd():
    working_directory_path = None
    # determine if application is a script file or frozen exe
    if getattr(sys, 'frozen', False):
        working_directory_path = os.path.dirname(sys.executable)
    elif __file__:
        working_directory_path = os.getcwd()
    else:
        working_directory_path = os.getcwd()
    return working_directory_path


def _get_filename(filepath):
    return '_'.join(filepath.split("/")[-1].split("_")[0:3]).split('.rsk')[0]


def _reset_file_environment():
    output_file_csv = "output.csv"
    output_file_csv_clean = "outputclean.csv"
    cwd = _get_cwd()
    output_file_csv = os.path.join(cwd, output_file_csv)
    output_file_csv_clean = os.path.join(cwd, output_file_csv_clean)
    if cwd is None:
        raise CTDError("", "Couldn't get working directory.")
    if os.path.isfile(output_file_csv):
        os.remove(output_file_csv)
    if os.path.isfile(output_file_csv_clean):
        os.remove(output_file_csv_clean)
    if not os.path.isdir("./plots"):
        os.mkdir("./plots")


def main():
    parser = argparse.ArgumentParser(
        description="CTD Fjorder Processing Script",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subparser for the 'merge' command
    parser_merge = subparsers.add_parser('merge', help='Merge all RSK files in the current folder')
    parser_merge.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    # Subparser for the 'default' command
    parser_default = subparsers.add_parser('default', help='Run the default processing pipeline')
    parser_default.add_argument('-p', '--plot', action='store_true',
                                help='Generate plots during the default processing pipeline')
    parser_default.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser_default.add_argument("-r", "--reset", help="Resets file environment (DELETES FILES)",
                                action="store_true")
    parser_default.add_argument('-o', '--output', type=str, help='Path to output file')
    parser_default.add_argument('-m', '--mastersheet', type=str, help='Path to mastersheet')
    parser_default.add_argument('-w', '--workers', type=int, nargs='?', const=1, help='Sets max workers for parallel processing')
    args = parser.parse_args()

    if args.command == 'merge':
        if args.verbose:
            logger.setLevel(logging.INFO)
        merge_all_in_folder()
        print("Merging completed successfully.")

    elif args.command == 'default':
        if args.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.FATAL)
        if args.reset:
            _reset_file_environment()
        if not args.mastersheet:
            run_default(plot=args.plot, max_workers=args.workers)
        else:
            run_default(plot=args.plot, master_sheet_path=args.mastersheet, max_workers=args.workers)
        if args.plot:
            print("Default processing with plotting completed successfully.")
        else:
            print("Default processing completed successfully.")
    sys.exit(1)


if __name__ == '__main__':
    main()
