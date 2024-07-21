# Column labels for internal use
from typing import Literal

TIMESTAMP_LABEL: str = "timestamp"
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
LATITUDE_LABEL: str = "latitude"
LONGITUDE_LABEL: str = "longitude"
UNIQUE_ID_LABEL: str = "unique_id"
PROFILE_ID_LABEL: str = "profile_id"
BV_LABEL: str = "brunt_vaisala_frequency_squared"
P_MID_LABEL: str = "p_mid"
SECCHI_DEPTH_LABEL: str = "secchi_depth"
PROFILE_ID_LABEL: str = "profile_id"

# Column label mapping from rsk to internal
rskLabels_to_labelInternal: dict[str, str] = {
    "temperature_00": TEMPERATURE_LABEL,
    "chlorophyll_00": CHLOROPHYLL_LABEL,
    "seapressure_00": SEA_PRESSURE_LABEL,
    "depth_00": DEPTH_LABEL,
    "salinity_00": SALINITY_LABEL,
    "speedofsound_00": SPEED_OF_SOUND_LABEL,
    "specificconductivity_00": SPECIFIC_CONDUCTIVITY_LABEL,
    "conductivity_00": CONDUCTIVITY_LABEL,
    "pressure_00": PRESSURE_LABEL,
}

# Column label mapping from castaway to internal
csvLabels_to_labelInternal: dict[str, str] = {
    "Pressure (Decibar)": SEA_PRESSURE_LABEL,
    "Depth (Meter)": DEPTH_LABEL,
    "Temperature (Celsius)": TEMPERATURE_LABEL,
    "Conductivity (MicroSiemens per Centimeter)": CONDUCTIVITY_LABEL,
    "Specific conductance (MicroSiemens per Centimeter)": SPECIFIC_CONDUCTIVITY_LABEL,
    "Salinity (Practical Salinity Scale)": SALINITY_LABEL,
    "Sound velocity (Meters per Second)": SPEED_OF_SOUND_LABEL,
    "Density (Kilograms per Cubic Meter)": DENSITY_LABEL,
}
# Column labels of master sheet
MASTER_SHEET_TIME_LOCAL_LABEL = "time_local"
MASTER_SHEET_DATE_LOCAL_LABEL = "date_local"
MASTER_SHEET_TIME_UTC_LABEL = "time (UTC)"
MASTER_SHEET_DATE_UTC_LABEL = "date (UTC)"
MASTER_SHEET_DATETIME_LABEL = "datetime"
MASTER_SHEET_SECCHI_DEPTH_LABEL = "secchi depth"

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
# Warning messages
WARNING_DROPPED_PROFILE: str = "No samples in profile number "

# Info messages
INFO_CTD_SURFACE_MEASUREMENT: str = (
    "First measurment lies below {end} dbar, cannot compute surface measurements"
)
# Debug messages
DEBUG_FILE_LACKS_LOCATION: str = "File lacks native location data"
DEBUG_CTD_OBJECT_INITITALIZED: str = "New CTD object initialized from file"

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
