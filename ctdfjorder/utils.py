from ctdfjorder.constants import *
import polars as pl
import sys
from os import path, getcwd


def save_to_csv(data: pl.DataFrame, output_file: str):
    """
    Renames the columns of the CTD data table based on a predefined mapping and saves the
    data to the specified CSV file.

    Parameters
    ----------
    data : pl.DataFrame
        The CTD data table.
    output_file : str
        The output CSV file path.
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
        }
        return data_label_mapping.get(label, label)

    # Rename columns
    renamed_data = data.rename(relabel_ctd_data)

    # Define the desired column order based on the mapping values
    ordered_columns = [
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
        EXPORT_LONGITUDE_LABEL,
        EXPORT_LATITUDE_LABEL,
        EXPORT_FILENAME_LABEL,
        EXPORT_PROFILE_ID_LABEL,
        EXPORT_UNIQUE_ID_LABEL,
    ]

    # Reorder columns if all ordered_columns are present in the DataFrame
    present_columns = [col for col in ordered_columns if col in renamed_data.columns]
    reordered_data = renamed_data.select(present_columns)

    # Save to CSV
    reordered_data.write_csv(output_file)
    return reordered_data


def get_cwd():
    working_directory_path = None
    # determine if application is a script file or frozen exe
    if getattr(sys, "frozen", False):
        working_directory_path = path.dirname(sys.executable)
    elif __file__:
        working_directory_path = getcwd()
    else:
        working_directory_path = getcwd()
    return working_directory_path
