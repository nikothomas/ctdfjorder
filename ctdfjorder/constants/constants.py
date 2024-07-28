"""
Constants used throughout the CTDFjorder package.

This module contains various constant values used for labeling columns,
time formatting, error messages, warning messages, and other configuration
settings within the CTDFjorder package.
"""

from typing import Literal

# Column labels for internal use
TIMESTAMP_LABEL: str = "timestamp"
"""str: Label for timestamp column."""
YEAR_LABEL: str = "year"
"""str: Label for year column."""
MONTH_LABEL: str = "month"
"""str: Label for month column."""
FILENAME_LABEL: str = "filename"
"""str: Label for filename column."""
CHLOROPHYLL_LABEL: str = "chlorophyll"
"""str: Label for chlorophyll column."""
TEMPERATURE_LABEL: str = "temperature"
"""str: Label for temperature column."""
SEA_PRESSURE_LABEL: str = "sea_pressure"
"""str: Label for sea pressure column."""
DEPTH_LABEL: str = "depth"
"""str: Label for depth column."""
SALINITY_LABEL: str = "salinity"
"""str: Label for salinity column."""
SPEED_OF_SOUND_LABEL: str = "speed_of_sound"
"""str: Label for speed of sound column."""
SPECIFIC_CONDUCTIVITY_LABEL: str = "specific_conductivity"
"""str: Label for specific conductivity column."""
CONDUCTIVITY_LABEL: str = "conductivity"
"""str: Label for conductivity column."""
PRESSURE_LABEL: str = "pressure"
"""str: Label for pressure column."""
SALINITY_ABS_LABEL: str = "salinity_abs"
"""str: Label for absolute salinity column."""
SURFACE_SALINITY_LABEL: str = "surface_salinity"
"""str: Label for surface salinity column."""
SURFACE_TEMPERATURE_LABEL: str = "surface_temperature"
"""str: Label for surface temperature column."""
SURFACE_DENSITY_LABEL: str = "surface_density"
"""str: Label for surface density column."""
MELTWATER_FRACTION_LABEL: str = "meltwater_fraction"
"""str: Label for meltwater fraction column."""
DENSITY_LABEL: str = "density"
"""str: Label for density column."""
POTENTIAL_DENSITY_LABEL: str = "potential_density"
"""str: Label for potential density column."""
BV_LABEL: str = "brunt_vaisala_frequency_squared"
"""str: Label for Brunt-Väisälä frequency squared column."""
P_MID_LABEL: str = "p_mid"
"""str: Label for mid pressure used for Brunt-Väisälä calculation column."""
SECCHI_DEPTH_LABEL: str = "secchi_depth"
"""str: Label for Secchi depth column."""
LATITUDE_LABEL: str = "latitude"
"""str: Label for latitude column."""
LONGITUDE_LABEL: str = "longitude"
"""str: Label for longitude column."""
UNIQUE_ID_LABEL: str = "unique_id"
"""str: Label for unique ID column."""
PROFILE_ID_LABEL: str = "profile_id"
"""str: Label for profile ID column."""
SITE_NAME_LABEL: str = "site_name"
"""str: Label for site name column."""
SITE_ID_LABEL: str = "site_id"
"""str: Label for the site id column."""
LIST_LABELS: list[str] = [
    TIMESTAMP_LABEL,
    YEAR_LABEL,
    MONTH_LABEL,
    FILENAME_LABEL,
    CHLOROPHYLL_LABEL,
    TEMPERATURE_LABEL,
    SEA_PRESSURE_LABEL,
    DEPTH_LABEL,
    SALINITY_LABEL,
    SPEED_OF_SOUND_LABEL,
    SPECIFIC_CONDUCTIVITY_LABEL,
    CONDUCTIVITY_LABEL,
    PRESSURE_LABEL,
    SALINITY_ABS_LABEL,
    SURFACE_SALINITY_LABEL,
    SURFACE_TEMPERATURE_LABEL,
    SURFACE_DENSITY_LABEL,
    MELTWATER_FRACTION_LABEL,
    DENSITY_LABEL,
    POTENTIAL_DENSITY_LABEL,
    BV_LABEL,
    P_MID_LABEL,
    SECCHI_DEPTH_LABEL,
    LATITUDE_LABEL,
    LONGITUDE_LABEL,
    UNIQUE_ID_LABEL,
    PROFILE_ID_LABEL,
    SITE_ID_LABEL,
    SITE_NAME_LABEL
]
"""list[str]: List of all internal column labels."""

# Export labels
EXPORT_TIMESTAMP_LABEL = "timestamp"
"""str: Export label for timestamp."""
EXPORT_YEAR_LABEL = "Year"
"""str: Export label for year."""
EXPORT_MONTH_LABEL = "Month"
"""str: Export label for month."""
EXPORT_TEMPERATURE_LABEL = "Temperature_(°C)"
"""str: Export label for temperature."""
EXPORT_PRESSURE_LABEL = "Pressure_(dbar)"
"""str: Export label for pressure."""
EXPORT_DEPTH_LABEL = "Depth_(m)"
"""str: Export label for depth."""
EXPORT_SEA_PRESSURE_LABEL = "Sea_Pressure_(dbar)"
"""str: Export label for sea pressure."""
EXPORT_CHLOROPHYLL_LABEL = "Chlorophyll_a_(µg/l)"
"""str: Export label for chlorophyll-a."""
EXPORT_SALINITY_LABEL = "Salinity_(PSU)"
"""str: Export label for salinity."""
EXPORT_SPECIFIC_CONDUCTIVITY_LABEL = "Specific_Conductivity_(µS/cm)"
"""str: Export label for specific conductivity."""
EXPORT_CONDUCTIVITY_LABEL = "Conductivity_(mS/cm)"
"""str: Export label for conductivity."""
EXPORT_DENSITY_LABEL = "Density_(kg/m^3)"
"""str: Export label for density."""
EXPORT_POTENTIAL_DENSITY_LABEL = "Potential_Density_(kg/m^3)"
"""str: Export label for potential density."""
EXPORT_SALINITY_ABS_LABEL = "Absolute_Salinity_(g/kg)"
"""str: Export label for absolute salinity."""
EXPORT_SURFACE_DENSITY_LABEL = "Mean_Surface_Density_(kg/m^3)"
"""str: Export label for mean surface density."""
EXPORT_SPEED_OF_SOUND_LABEL = "Speed_of_Sound_(m/s)"
"""str: Export label for speed of sound."""
EXPORT_SURFACE_SALINITY_LABEL = "Surface_Salinity_(PSU)"
"""str: Export label for surface salinity."""
EXPORT_SURFACE_TEMPERATURE_LABEL = "Surface_Temperature_(°C)"
"""str: Export label for surface temperature."""
EXPORT_MELTWATER_FRACTION_LABEL = "Meltwater_Fraction_(%)"
"""str: Export label for meltwater fraction."""
EXPORT_LONGITUDE_LABEL = "longitude"
"""str: Export label for longitude."""
EXPORT_LATITUDE_LABEL = "latitude"
"""str: Export label for latitude."""
EXPORT_FILENAME_LABEL = "filename"
"""str: Export label for filename."""
EXPORT_PROFILE_ID_LABEL = "Profile_ID"
"""str: Export label for profile ID."""
EXPORT_UNIQUE_ID_LABEL = "Unique_ID"
"""str: Export label for unique ID."""
EXPORT_BV_LABEL = "Brunt_Vaisala_Frequency_Squared"
"""str: Export label for Brunt-Väisälä frequency squared."""
EXPORT_P_MID_LABEL = "Mid_Pressure_Used_For_BV_Calc"
"""str: Export label for mid pressure used for Brunt-Väisälä calculation."""
EXPORT_SECCHI_DEPTH_LABEL = "Secchi_Depth_(m)"
"""str: Export label for Secchi depth."""
EXPORT_SITE_NAME_LABEL = "Site_Name"
"""str: Export label for site name."""
EXPORT_SITE_ID_LABEL = "Site_ID"
"""str: Export label for site id."""

# Time string constants
TIME_ZONE: str = "UTC"
"""str: Time zone used for timestamps."""
TIME_UNIT: Literal["ns", "us", "ms"] = "ns"
"""Literal["ns", "us", "ms"]: Time unit for timestamps."""
TIME_FORMAT: str = "%Y-%m-%d %H:%M:%S.%f"
"""str: Format for timestamps."""

# Error messages
ERROR_NO_SAMPLES: str = "No samples in file"
"""str: Error message for no samples in file."""
ERROR_NO_LOCATION: str = "No location could be found"
"""str: Error message for no location found."""
ERROR_DENSITY_CALCULATION: str = "Could not calculate density on this dataset"
"""str: Error message for density calculation failure."""
ERROR_SALINITY_ABS: str = "Could not calculate salinity absolute on this dataset"
"""str: Error message for salinity absolute calculation failure."""
ERROR_NO_MASTER_SHEET: str = (
    "No mastersheet provided, could not update the file's missing location data"
)
"""str: Error message for missing master sheet."""
ERROR_RSK_CORRUPT: str = "Ruskin file is corrupted and could not be read"
"""str: Error message for corrupted Ruskin file."""
ERROR_LOCATION_DATA_INVALID: str = (
    "Location data invalid, probably due to malformed master sheet data"
)
"""str: Error message for invalid location data."""
ERROR_NO_TIMESTAMP_IN_FILE: str = "No timestamp in file, could not get location"
"""str: Error message for missing timestamp in file."""
ERROR_NO_TIMESTAMP_IN_MASTER_SHEET: str = (
    "No timestamp in master sheet, could not get location"
)
"""str: Error message for missing timestamp in master sheet."""
ERROR_MLD_DEPTH_RANGE: str = "Insufficient depth range to calculate MLD"
"""str: Error message for insufficient depth range to calculate MLD."""
ERROR_GRU_INSUFFICIENT_DATA: str = "Not enough values to run the GRU on this data"
"""str: Error message for insufficient data to run the GRU."""
ERROR_CASTAWAY_START_TIME: str = "Castaway file has no time data"
"""str: Error message for missing time data in Castaway file."""
ERROR_CTD_FILENAME_ENDING: str = "CTD filename must end in '.rsk' or '.csv'"
"""str: Error message for invalid CTD filename ending."""

# Warning messages
WARNING_DROPPED_PROFILE: str = "No samples in profile number "
"""str: Warning message for dropped profile due to no samples."""
WARNING_CTD_SURFACE_MEASUREMENT: str = (
    "First measurment lies below {end} dbar, cannot compute surface measurements"
)
"""str: Warning message for invalid surface measurement."""
WARNING_FILE_LACKS_LOCATION: str = "File lacks native location data"
"""str: Warning message for file lacking native location data."""

# Filename constants
FILENAME_GPS_ENDING: str = "_gps"
"""str: Suffix for GPS filename."""
FILENAME_CM_ENDING: str = "cm"
"""str: Suffix for CM filename."""
RSK_FILE_MARKER: str = ".rsk"
"""str: Marker for RSK file."""
CASTAWAY_FILE_MARKER: str = ".csv"
"""str: Marker for Castaway file."""

# Castaway column labels
CASTAWAY_DATETIME_LABEL: str = "datetime_utc"
"""str: Column label for datetime in Castaway files."""
CASTAWAY_FILE_ID_LABEL: str = "file_id"
"""str: Column label for file ID in Castaway files."""

# Concatenation parameters
CONCAT_HOW: Literal["diagonal_relaxed"] = "diagonal_relaxed"
"""Literal["diagonal_relaxed"]: Concatenation method for merging data."""

# Sea pressure to pressure difference
SEA_PRESSURE_TO_PRESSURE_DIFF: float = 10.1325
"""float: Conversion factor from sea pressure to pressure difference."""

# Library logger name
LIB_LOGGER_NAME = "ctdfjorder"
"""str: Logger name for the CTDFjorder library."""

# Default output file
DEFAULT_OUTPUT_FILE = "ctdfjorder_data.csv"
"""str: Name of the default output file."""