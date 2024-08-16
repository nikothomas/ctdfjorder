"""
Constants used throughout the CTDFjorder package.

This module contains various constant values used for labeling columns,
time formatting, error messages, warning messages, and other configuration
settings within the CTDFjorder package.
"""

from typing import Literal

# ------------------------------------
# Column labels for internal use
# ------------------------------------
TIMESTAMP_LABEL: str = "timestamp"
"""str: Label for timestamp column."""

YEAR_LABEL: str = "year"
"""str: Label for year column."""

MONTH_LABEL: str = "month"
"""str: Label for month column."""

DAY_LABEL: str = "day"
"""str: Label for day column."""

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

MELTWATER_FRACTION_EQ_10_LABEL: str = "meltwater_fraction_eq_10"
"""str: Label for meltwater fraction equation 10 column."""

MELTWATER_FRACTION_EQ_11_LABEL: str = "meltwater_fraction_eq_11"
"""str: Label for meltwater fraction equation 11 column."""

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

CONSERVATIVE_TEMPERATURE_LABEL: str = "conservative_temperature"
"""str: Label for the conservative temperature column."""

OXYGEN_CONCENTRATION_LABEL: str = "oxygen_concentration"
"""str: Label for dissolved oxygen concentration column."""

OXYGEN_SATURATION_LABEL: str = "oxygen_saturation"
"""str: Label for oxygen saturation column."""

NITRATE_LABEL: str = "nitrate"
"""str: Label for nitrate concentration column."""

PHOSPHATE_LABEL: str = "phosphate"
"""str: Label for phosphate concentration column."""

SILICATE_LABEL: str = "silicate"
"""str: Label for silicate concentration column."""

PH_LABEL: str = "ph"
"""str: Label for pH value column."""

ALKALINITY_LABEL: str = "alkalinity"
"""str: Label for alkalinity column."""

TURBIDITY_LABEL: str = "turbidity"
"""str: Label for turbidity column."""

PARTICULATE_ORGANIC_CARBON_LABEL: str = "particulate_organic_carbon"
"""str: Label for particulate organic carbon column."""

TOTAL_ORGANIC_CARBON_LABEL: str = "total_organic_carbon"
"""str: Label for total organic carbon column."""

PARTICULATE_INORGANIC_CARBON_LABEL: str = "particulate_inorganic_carbon"
"""str: Label for particulate inorganic carbon column."""

DISSOLVED_INORGANIC_CARBON_LABEL: str = "dissolved_inorganic_carbon"
"""str: Label for dissolved inorganic carbon column."""

CHLOROPHYLL_FLUORESCENCE_LABEL: str = "chlorophyll_fluorescence"
"""str: Label for chlorophyll fluorescence column."""

PAR_LABEL: str = "par"
"""str: Label for photosynthetically active radiation (PAR) column."""

AMMONIUM_LABEL: str = "ammonium"
"""str: Label for ammonium concentration column."""

ORP_LABEL: str = "orp"
"""str: Label for oxidation-reduction potential (ORP) column."""

CLASSIFICATION_LABEL: str = "profile_type"
"""str: Label for profile type (A, B, C) column."""

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
    MELTWATER_FRACTION_EQ_10_LABEL,
    MELTWATER_FRACTION_EQ_11_LABEL,
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
    SITE_NAME_LABEL,
    OXYGEN_CONCENTRATION_LABEL,
    OXYGEN_SATURATION_LABEL,
    NITRATE_LABEL,
    PHOSPHATE_LABEL,
    SILICATE_LABEL,
    PH_LABEL,
    ALKALINITY_LABEL,
    TURBIDITY_LABEL,
    PARTICULATE_ORGANIC_CARBON_LABEL,
    TOTAL_ORGANIC_CARBON_LABEL,
    PARTICULATE_INORGANIC_CARBON_LABEL,
    DISSOLVED_INORGANIC_CARBON_LABEL,
    CHLOROPHYLL_FLUORESCENCE_LABEL,
    PAR_LABEL,
    AMMONIUM_LABEL,
    ORP_LABEL,
    CLASSIFICATION_LABEL
]
"""list[str]: List of all internal column labels."""

# ------------------------------------
# Export labels
# ------------------------------------
EXPORT_TIMESTAMP_LABEL = "timestamp"
"""str: Export label for timestamp."""

EXPORT_YEAR_LABEL = "Year"
"""str: Export label for year."""

EXPORT_MONTH_LABEL = "Month"
"""str: Export label for month."""

EXPORT_DAY_LABEL: str = "Day"
"""str: Label for day column."""

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

EXPORT_MELTWATER_FRACTION_EQ_10_LABEL = "Meltwater_Fraction_EQ_10_(%)"
"""str: Export label for meltwater fraction equation 10."""

EXPORT_MELTWATER_FRACTION_EQ_11_LABEL = "Meltwater_Fraction_EQ_11_(%)"
"""str: Export label for meltwater fraction equation 11."""

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

EXPORT_OXYGEN_CONCENTRATION_LABEL = "Oxygen_Concentration_(µmol/kg)"
"""str: Export label for dissolved oxygen concentration."""

EXPORT_OXYGEN_SATURATION_LABEL = "Oxygen_Saturation_(%)"
"""str: Export label for oxygen saturation."""

EXPORT_NITRATE_LABEL = "Nitrate_(µmol/L)"
"""str: Export label for nitrate concentration."""

EXPORT_PHOSPHATE_LABEL = "Phosphate_(µmol/L)"
"""str: Export label for phosphate concentration."""

EXPORT_SILICATE_LABEL = "Silicate_(µmol/L)"
"""str: Export label for silicate concentration."""

EXPORT_PH_LABEL = "pH"
"""str: Export label for pH value."""

EXPORT_ALKALINITY_LABEL = "Alkalinity_(µmol/kg)"
"""str: Export label for alkalinity."""

EXPORT_TURBIDITY_LABEL = "Turbidity_(NTU)"
"""str: Export label for turbidity."""

EXPORT_PARTICULATE_ORGANIC_CARBON_LABEL = "Particulate_Organic_Carbon_(µmol/L)"
"""str: Export label for particulate organic carbon."""

EXPORT_TOTAL_ORGANIC_CARBON_LABEL = "Total_Organic_Carbon_(µmol/L)"
"""str: Export label for total organic carbon."""

EXPORT_PARTICULATE_INORGANIC_CARBON_LABEL = "Particulate_Inorganic_Carbon_(µmol/L)"
"""str: Export label for particulate inorganic carbon."""

EXPORT_DISSOLVED_INORGANIC_CARBON_LABEL = "Dissolved_Inorganic_Carbon_(µmol/kg)"
"""str: Export label for dissolved inorganic carbon."""

EXPORT_CHLOROPHYLL_FLUORESCENCE_LABEL = "Chlorophyll_Fluorescence_(RFU)"
"""str: Export label for chlorophyll fluorescence."""

EXPORT_PAR_LABEL = "PAR_(µmol/m²/s)"
"""str: Export label for photosynthetically active radiation (PAR)."""

EXPORT_AMMONIUM_LABEL = "Ammonium_(µmol/L)"
"""str: Export label for ammonium concentration."""

EXPORT_ORP_LABEL = "ORP_(mV)"
"""str: Export label for oxidation-reduction potential (ORP)."""

EXPORT_CLASSIFICATION_LABEL: str = "Profile_Type"
"""str: Export label for profile type (A, B, C) column."""

# ------------------------------------
# Mapping of internal labels to export labels
# ------------------------------------
DATA_LABEL_MAPPING: dict[str, str] = {
    TIMESTAMP_LABEL: EXPORT_TIMESTAMP_LABEL,
    YEAR_LABEL: EXPORT_YEAR_LABEL,
    MONTH_LABEL: EXPORT_MONTH_LABEL,
    DAY_LABEL: EXPORT_DAY_LABEL,
    FILENAME_LABEL: EXPORT_FILENAME_LABEL,
    CHLOROPHYLL_LABEL: EXPORT_CHLOROPHYLL_LABEL,
    TEMPERATURE_LABEL: EXPORT_TEMPERATURE_LABEL,
    PRESSURE_LABEL: EXPORT_PRESSURE_LABEL,
    SEA_PRESSURE_LABEL: EXPORT_SEA_PRESSURE_LABEL,
    DEPTH_LABEL: EXPORT_DEPTH_LABEL,
    SALINITY_LABEL: EXPORT_SALINITY_LABEL,
    SPEED_OF_SOUND_LABEL: EXPORT_SPEED_OF_SOUND_LABEL,
    SPECIFIC_CONDUCTIVITY_LABEL: EXPORT_SPECIFIC_CONDUCTIVITY_LABEL,
    CONDUCTIVITY_LABEL: EXPORT_CONDUCTIVITY_LABEL,
    DENSITY_LABEL: EXPORT_DENSITY_LABEL,
    POTENTIAL_DENSITY_LABEL: EXPORT_POTENTIAL_DENSITY_LABEL,
    SALINITY_ABS_LABEL: EXPORT_SALINITY_ABS_LABEL,
    SURFACE_SALINITY_LABEL: EXPORT_SURFACE_SALINITY_LABEL,
    SURFACE_TEMPERATURE_LABEL: EXPORT_SURFACE_TEMPERATURE_LABEL,
    SURFACE_DENSITY_LABEL: EXPORT_SURFACE_DENSITY_LABEL,
    MELTWATER_FRACTION_EQ_10_LABEL: EXPORT_MELTWATER_FRACTION_EQ_10_LABEL,
    MELTWATER_FRACTION_EQ_11_LABEL: EXPORT_MELTWATER_FRACTION_EQ_11_LABEL,
    LONGITUDE_LABEL: EXPORT_LONGITUDE_LABEL,
    LATITUDE_LABEL: EXPORT_LATITUDE_LABEL,
    UNIQUE_ID_LABEL: EXPORT_UNIQUE_ID_LABEL,
    PROFILE_ID_LABEL: EXPORT_PROFILE_ID_LABEL,
    SITE_NAME_LABEL: EXPORT_SITE_NAME_LABEL,
    SITE_ID_LABEL: EXPORT_SITE_ID_LABEL,
    BV_LABEL: EXPORT_BV_LABEL,
    P_MID_LABEL: EXPORT_P_MID_LABEL,
    SECCHI_DEPTH_LABEL: EXPORT_SECCHI_DEPTH_LABEL,
    OXYGEN_CONCENTRATION_LABEL: EXPORT_OXYGEN_CONCENTRATION_LABEL,
    OXYGEN_SATURATION_LABEL: EXPORT_OXYGEN_SATURATION_LABEL,
    NITRATE_LABEL: EXPORT_NITRATE_LABEL,
    PHOSPHATE_LABEL: EXPORT_PHOSPHATE_LABEL,
    SILICATE_LABEL: EXPORT_SILICATE_LABEL,
    PH_LABEL: EXPORT_PH_LABEL,
    ALKALINITY_LABEL: EXPORT_ALKALINITY_LABEL,
    TURBIDITY_LABEL: EXPORT_TURBIDITY_LABEL,
    PARTICULATE_ORGANIC_CARBON_LABEL: EXPORT_PARTICULATE_ORGANIC_CARBON_LABEL,
    TOTAL_ORGANIC_CARBON_LABEL: EXPORT_TOTAL_ORGANIC_CARBON_LABEL,
    PARTICULATE_INORGANIC_CARBON_LABEL: EXPORT_PARTICULATE_INORGANIC_CARBON_LABEL,
    DISSOLVED_INORGANIC_CARBON_LABEL: EXPORT_DISSOLVED_INORGANIC_CARBON_LABEL,
    CHLOROPHYLL_FLUORESCENCE_LABEL: EXPORT_CHLOROPHYLL_FLUORESCENCE_LABEL,
    PAR_LABEL: EXPORT_PAR_LABEL,
    AMMONIUM_LABEL: EXPORT_AMMONIUM_LABEL,
    ORP_LABEL: EXPORT_ORP_LABEL,
    CLASSIFICATION_LABEL: EXPORT_CLASSIFICATION_LABEL
}
"""dict[str, str]: Mapping of internal column labels to export column labels."""

# ------------------------------------
# Desired column order for CSV export
# ------------------------------------
EXPORT_COLUMN_ORDER: list[str] = [
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
    EXPORT_MELTWATER_FRACTION_EQ_10_LABEL,
    EXPORT_MELTWATER_FRACTION_EQ_11_LABEL,
    EXPORT_BV_LABEL,
    EXPORT_P_MID_LABEL,
    EXPORT_SECCHI_DEPTH_LABEL,
    EXPORT_OXYGEN_CONCENTRATION_LABEL,
    EXPORT_OXYGEN_SATURATION_LABEL,
    EXPORT_NITRATE_LABEL,
    EXPORT_PHOSPHATE_LABEL,
    EXPORT_SILICATE_LABEL,
    EXPORT_PH_LABEL,
    EXPORT_ALKALINITY_LABEL,
    EXPORT_TURBIDITY_LABEL,
    EXPORT_PARTICULATE_ORGANIC_CARBON_LABEL,
    EXPORT_TOTAL_ORGANIC_CARBON_LABEL,
    EXPORT_PARTICULATE_INORGANIC_CARBON_LABEL,
    EXPORT_DISSOLVED_INORGANIC_CARBON_LABEL,
    EXPORT_CHLOROPHYLL_FLUORESCENCE_LABEL,
    EXPORT_PAR_LABEL,
    EXPORT_AMMONIUM_LABEL,
    EXPORT_ORP_LABEL,
    EXPORT_CLASSIFICATION_LABEL
]
"""list[str]: Desired column order for CSV export."""

# ------------------------------------
# Time string constants
# ------------------------------------
TIME_ZONE: str = "UTC"
"""str: Time zone used for timestamps."""

TIME_UNIT: Literal["ns", "us", "ms"] = "ns"
"""Literal["ns", "us", "ms"]: Time unit for timestamps."""

TIME_FORMAT: str = "%Y-%m-%d %H:%M:%S.%f"
"""str: Format for timestamps."""

# ------------------------------------
# Warning messages
# ------------------------------------
WARNING_DROPPED_PROFILE: str = "No samples in profile number "
"""str: Warning message for dropped profile due to no samples."""

WARNING_CTD_SURFACE_MEASUREMENT: str = (
    "First measurment lies below 'end'parameter, cannot compute surface measurements"
)
"""str: Warning message for invalid surface measurement."""

WARNING_FILE_LACKS_LOCATION: str = "File lacks native location data"
"""str: Warning message for file lacking native location data."""

# ------------------------------------
# Filename constants
# ------------------------------------
FILENAME_GPS_ENDING: str = "_gps"
"""str: Suffix for GPS filename."""

FILENAME_CM_ENDING: str = "cm"
"""str: Suffix for CM filename."""

RSK_FILE_MARKER: str = ".rsk"
"""str: Marker for RSK file."""

CASTAWAY_FILE_MARKER: str = ".csv"
"""str: Marker for Castaway file."""

SEABIRD_FILE_MARKER: str = ".cnv"
"""str: Marker for Seabird file."""

# ------------------------------------
# Castaway column labels
# ------------------------------------
CASTAWAY_DATETIME_LABEL: str = "datetime_utc"
"""str: Column label for datetime in Castaway files."""

CASTAWAY_FILE_ID_LABEL: str = "file_id"
"""str: Column label for file ID in Castaway files."""

# ------------------------------------
# Concatenation parameters
# ------------------------------------
CONCAT_HOW: Literal["diagonal_relaxed"] = "diagonal_relaxed"
"""Literal["diagonal_relaxed"]: Concatenation method for merging data."""

# ------------------------------------
# Sea pressure to pressure difference
# ------------------------------------
SEA_PRESSURE_TO_PRESSURE_DIFF: float = 10.1325
"""float: Conversion factor from sea pressure to pressure difference."""

# ------------------------------------
# Library logger name
# ------------------------------------
LIB_LOGGER_NAME = "ctdfjorder"
"""str: Logger name for the CTDFjorder library."""

# ------------------------------------
# Default output file
# ------------------------------------
DEFAULT_OUTPUT_FILE = "ctdfjorder_data.csv"
"""str: Name of the default output file."""
