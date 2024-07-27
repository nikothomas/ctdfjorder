# Column labels for internal use
from typing import Literal

TIMESTAMP_LABEL: str = "timestamp"
YEAR_LABEL: str = "year"
MONTH_LABEL: str = "month"
FILENAME_LABEL: str = "filename"
CHLOROPHYLL_LABEL: str = "chlorophyll"
TEMPERATURE_LABEL: str = "temperature"
SEA_PRESSURE_LABEL: str = "sea_pressure"
DEPTH_LABEL: str = "depth"
SALINITY_LABEL: str = "salinity"
SPEED_OF_SOUND_LABEL: str = "speed_of_sound"
SPECIFIC_CONDUCTIVITY_LABEL: str = "specific_conductivity"
CONDUCTIVITY_LABEL: str = "conductivity"
PRESSURE_LABEL: str = "pressure"
SALINITY_ABS_LABEL: str = "salinity_abs"
SURFACE_SALINITY_LABEL: str = "surface_salinity"
SURFACE_TEMPERATURE_LABEL: str = "surface_temperature"
SURFACE_DENSITY_LABEL: str = "surface_density"
MELTWATER_FRACTION_LABEL: str = "meltwater_fraction"
DENSITY_LABEL: str = "density"
POTENTIAL_DENSITY_LABEL: str = "potential_density"
BV_LABEL: str = "brunt_vaisala_frequency_squared"
P_MID_LABEL: str = "p_mid"
SECCHI_DEPTH_LABEL: str = "secchi_depth"
LATITUDE_LABEL: str = "latitude"
LONGITUDE_LABEL: str = "longitude"
UNIQUE_ID_LABEL: str = "unique_id"
PROFILE_ID_LABEL: str = "profile_id"

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
    PROFILE_ID_LABEL
]

# Export labels
EXPORT_TIMESTAMP_LABEL = "timestamp"
EXPORT_YEAR_LABEL = "Year"
EXPORT_MONTH_LABEL = "Month"
EXPORT_TEMPERATURE_LABEL = "Temperature_(°C)"
EXPORT_PRESSURE_LABEL = "Pressure_(dbar)"
EXPORT_DEPTH_LABEL = "Depth_(m)"
EXPORT_SEA_PRESSURE_LABEL = "Sea_Pressure_(dbar)"
EXPORT_CHLOROPHYLL_LABEL = "Chlorophyll_a_(µg/l)"
EXPORT_SALINITY_LABEL = "Salinity_(PSU)"
EXPORT_SPECIFIC_CONDUCTIVITY_LABEL = "Specific_Conductivity_(µS/cm)"
EXPORT_CONDUCTIVITY_LABEL = "Conductivity_(mS/cm)"
EXPORT_DENSITY_LABEL = "Density_(kg/m^3)"
EXPORT_POTENTIAL_DENSITY_LABEL = "Potential_Density_(kg/m^3)"
EXPORT_SALINITY_ABS_LABEL = "Absolute_Salinity_(g/kg)"
EXPORT_SURFACE_DENSITY_LABEL = "Mean_Surface_Density_(kg/m^3)"
EXPORT_SPEED_OF_SOUND_LABEL = "Speed_of_Sound_(m/s)"
EXPORT_SURFACE_SALINITY_LABEL = "Surface_Salinity_(PSU)"
EXPORT_SURFACE_TEMPERATURE_LABEL = "Surface_Temperature_(°C)"
EXPORT_MELTWATER_FRACTION_LABEL = "Meltwater_Fraction_(%)"
EXPORT_LONGITUDE_LABEL = "longitude"
EXPORT_LATITUDE_LABEL = "latitude"
EXPORT_FILENAME_LABEL = "filename"
EXPORT_PROFILE_ID_LABEL = "Profile_ID"
EXPORT_UNIQUE_ID_LABEL = "Unique_ID"
EXPORT_BV_LABEL = "Brunt_Vaisala_Frequency_Squared"
EXPORT_P_MID_LABEL = "Mid_Pressure_Used_For_BV_Calc"
EXPORT_SECCHI_DEPTH_LABEL = "Secchi_Depth_(m)"

# Time string constants
TIME_ZONE: str = "UTC"
TIME_UNIT: Literal["ns", "us", "ms"] = "ns"
TIME_FORMAT: str = "%Y-%m-%d %H:%M:%S.%f"

# Error messages
ERROR_NO_SAMPLES: str = "No samples in file"
ERROR_NO_LOCATION: str = "No location could be found"
ERROR_DENSITY_CALCULATION: str = "Could not calculate density on this dataset"
ERROR_SALINITY_ABS: str = "Could not calculate salinity absolute on this dataset"
ERROR_NO_MASTER_SHEET: str = (
    "No mastersheet provided, could not update the file's missing location data"
)
ERROR_RSK_CORRUPT: str = "Ruskin file is corrupted and could not be read"
ERROR_LOCATION_DATA_INVALID: str = (
    "Location data invalid, probably due to malformed master sheet data"
)
ERROR_NO_TIMESTAMP_IN_FILE: str = "No timestamp in file, could not get location"
ERROR_NO_TIMESTAMP_IN_MASTER_SHEET: str = (
    "No timestamp in master sheet, could not get location"
)
ERROR_MLD_DEPTH_RANGE: str = "Insufficient depth range to calculate MLD"
ERROR_GRU_INSUFFICIENT_DATA: str = "Not enough values to run the GRU on this data"
ERROR_CASTAWAY_START_TIME: str = "Castaway file has no time data"
ERROR_CTD_FILENAME_ENDING: str = "CTD filename must end in '.rsk' or '.csv'"

# Warning messages
WARNING_DROPPED_PROFILE: str = "No samples in profile number "
WARNING_CTD_SURFACE_MEASUREMENT: str = (
    "First measurment lies below {end} dbar, cannot compute surface measurements"
)
WARNING_FILE_LACKS_LOCATION: str = "File lacks native location data"

# Filename constants
FILENAME_GPS_ENDING: str = "_gps"
FILENAME_CM_ENDING: str = "cm"
RSK_FILE_MARKER: str = ".rsk"
CASTAWAY_FILE_MARKER: str = ".csv"

# Castaway column labels
CASTAWAY_DATETIME_LABEL: str = "datetime_utc"
CASTAWAY_FILE_ID_LABEL: str = "file_id"

# Concatenation parameters
CONCAT_HOW: Literal["diagonal_relaxed"] = "diagonal_relaxed"

# Sea pressure to pressure difference
SEA_PRESSURE_TO_PRESSURE_DIFF: float = 10.1325

# Library logger name
LIB_LOGGER_NAME = "ctdfjorder"
