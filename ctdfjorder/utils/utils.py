import os.path
from datetime import datetime
import getpass

from ctdfjorder.constants.constants import *
import polars as pl
import sys
from os import path, getcwd


def save_to_csv(data: pl.DataFrame, output_file: str, null_value: str | None):
    """
    Renames the columns of the CTD data table based on a predefined mapping and saves the
    data to the specified CSV file.

    Parameters
    ----------
    data : pl.DataFrame
        The CTD data table.
    output_file : str
        The output CSV file path.
    null_value : str
        The value that will fill blank cells in the data.
    """

    def relabel_ctd_data(label: str) -> str:
        data_label_mapping = {
            TIMESTAMP_LABEL: EXPORT_TIMESTAMP_LABEL,
            YEAR_LABEL: EXPORT_YEAR_LABEL,
            MONTH_LABEL: EXPORT_MONTH_LABEL,
            TEMPERATURE_LABEL: EXPORT_TEMPERATURE_LABEL,
            PRESSURE_LABEL: EXPORT_PRESSURE_LABEL,
            CHLOROPHYLL_LABEL: EXPORT_CHLOROPHYLL_LABEL,
            SEA_PRESSURE_LABEL: EXPORT_SEA_PRESSURE_LABEL,
            DEPTH_LABEL: EXPORT_DEPTH_LABEL,
            SALINITY_LABEL: EXPORT_SALINITY_LABEL,
            SPEED_OF_SOUND_LABEL: EXPORT_SPEED_OF_SOUND_LABEL,
            SPECIFIC_CONDUCTIVITY_LABEL: EXPORT_SPECIFIC_CONDUCTIVITY_LABEL,
            CONDUCTIVITY_LABEL: EXPORT_CONDUCTIVITY_LABEL,
            DENSITY_LABEL: EXPORT_DENSITY_LABEL,
            POTENTIAL_DENSITY_LABEL: EXPORT_POTENTIAL_DENSITY_LABEL,
            SALINITY_ABS_LABEL: EXPORT_SALINITY_ABS_LABEL,
            SURFACE_DENSITY_LABEL: EXPORT_SURFACE_DENSITY_LABEL,
            SURFACE_SALINITY_LABEL: EXPORT_SURFACE_SALINITY_LABEL,
            SURFACE_TEMPERATURE_LABEL: EXPORT_SURFACE_TEMPERATURE_LABEL,
            MELTWATER_FRACTION_LABEL: EXPORT_MELTWATER_FRACTION_LABEL,
            LONGITUDE_LABEL: EXPORT_LONGITUDE_LABEL,
            LATITUDE_LABEL: EXPORT_LATITUDE_LABEL,
            FILENAME_LABEL: EXPORT_FILENAME_LABEL,
            PROFILE_ID_LABEL: EXPORT_PROFILE_ID_LABEL,
            UNIQUE_ID_LABEL: EXPORT_UNIQUE_ID_LABEL,
            BV_LABEL: EXPORT_BV_LABEL,
            P_MID_LABEL: EXPORT_P_MID_LABEL,
            SECCHI_DEPTH_LABEL: EXPORT_SECCHI_DEPTH_LABEL,
            SITE_NAME_LABEL: EXPORT_SITE_NAME_LABEL,
            SITE_ID_LABEL: EXPORT_SITE_ID_LABEL
        }
        return data_label_mapping.get(label, label)

    # Rename columns
    renamed_data = data.rename(relabel_ctd_data)

    # Define the desired column order based on the mapping values
    ordered_columns = [
        EXPORT_FILENAME_LABEL,
        EXPORT_PROFILE_ID_LABEL,
        EXPORT_UNIQUE_ID_LABEL,
        EXPORT_SITE_NAME_LABEL,
        EXPORT_SITE_ID_LABEL,
        EXPORT_LONGITUDE_LABEL,
        EXPORT_LATITUDE_LABEL,
        EXPORT_TIMESTAMP_LABEL,
        EXPORT_YEAR_LABEL,
        EXPORT_MONTH_LABEL,
        EXPORT_TEMPERATURE_LABEL,
        EXPORT_PRESSURE_LABEL,
        EXPORT_DEPTH_LABEL,
        EXPORT_SEA_PRESSURE_LABEL,
        EXPORT_CHLOROPHYLL_LABEL,
        EXPORT_SALINITY_LABEL,
        EXPORT_SPECIFIC_CONDUCTIVITY_LABEL,
        EXPORT_CONDUCTIVITY_LABEL,
        EXPORT_DENSITY_LABEL,
        EXPORT_POTENTIAL_DENSITY_LABEL,
        EXPORT_SALINITY_ABS_LABEL,
        EXPORT_SURFACE_DENSITY_LABEL,
        EXPORT_SPEED_OF_SOUND_LABEL,
        EXPORT_SURFACE_SALINITY_LABEL,
        EXPORT_SURFACE_TEMPERATURE_LABEL,
        EXPORT_MELTWATER_FRACTION_LABEL,
        EXPORT_BV_LABEL,
        EXPORT_P_MID_LABEL,
        EXPORT_SECCHI_DEPTH_LABEL,
    ]

    # Reorder columns if they are present in the DataFrame
    present_columns = [col for col in ordered_columns if col in renamed_data.columns]
    reordered_data = renamed_data.select(present_columns)

    # Append any missing columns that were not in the specified order
    missing_columns = [
        col for col in renamed_data.columns if col not in present_columns
    ]
    if missing_columns:
        missing_data = renamed_data.select(missing_columns)
        reordered_data = pl.concat([reordered_data, missing_data], how="horizontal")

    # Create metadata
    if output_file == "ctdfjorder_data.csv":
        creation_date = datetime.now().strftime('%Y%m%d%H%M%S')
        user = getpass.getuser()
        metadata = f"_{creation_date}_{user}.csv"
        file = os.path.splitext(output_file)[0]
        reordered_data.write_csv(file+metadata, null_value=null_value)
    else:
        reordered_data.write_csv(output_file, null_value=null_value)
    return reordered_data


def get_cwd():
    """
    Gets the current working directory.

    Returns
    -------
    str
        The current working directory.
    """
    # Determine if application is a script file or frozen exe
    if getattr(sys, "frozen", False):
        working_directory_path = path.dirname(sys.executable)
    elif __file__:
        working_directory_path = getcwd()
    else:
        working_directory_path = getcwd()
    return working_directory_path
