"""
Constants used throughout the CTDFjorder package.

This module contains various constant values used for labeling columns,
time formatting, error messages, warning messages, and other configuration
settings within the CTDFjorder package.
"""
from typing import Literal
from ctdfjorder.dataclasses.dataclasses import SampleFeature

# --------------
# Column labels
# --------------

from typing import NamedTuple
import polars as pl

# Create SampleFeature instances for all labeled features
TIMESTAMP = SampleFeature(label="timestamp", export_label="timestamp", unit="ISO8601", pl_unit=pl.Datetime)
YEAR = SampleFeature(label="year", export_label="Year", unit="", pl_unit=pl.Int16)
MONTH = SampleFeature(label="month", export_label="Month", unit="", pl_unit=pl.Int8)
DAY = SampleFeature(label="day", export_label="Day", unit="", pl_unit=pl.Int8)
FILENAME = SampleFeature(label="filename", export_label="filename", unit="", pl_unit=pl.Utf8)
CHLOROPHYLL = SampleFeature(label="chlorophyll", export_label="Chlorophyll_a_(µg/l)", unit="µg/l", pl_unit=pl.Float32)
TEMPERATURE = SampleFeature(label="temperature", export_label="Temperature_(°C)", unit="°C", pl_unit=pl.Float32)
SEA_PRESSURE = SampleFeature(label="sea_pressure", export_label="Sea_Pressure_(dbar)", unit="dbar", pl_unit=pl.Float32)
DEPTH = SampleFeature(label="depth", export_label="Depth_(m)", unit="m", pl_unit=pl.Float32)
SALINITY = SampleFeature(label="salinity", export_label="Salinity_(PSU)", unit="PSU", pl_unit=pl.Float32)
SPEED_OF_SOUND = SampleFeature(label="speed_of_sound", export_label="Speed_of_Sound_(m/s)", unit="m/s",
                               pl_unit=pl.Float32)
SPECIFIC_CONDUCTIVITY = SampleFeature(label="specific_conductivity", export_label="Specific_Conductivity_(µS/cm)",
                                      unit="µS/cm", pl_unit=pl.Float32)
CONDUCTIVITY = SampleFeature(label="conductivity", export_label="Conductivity_(mS/cm)", unit="mS/cm",
                             pl_unit=pl.Float32)
PRESSURE = SampleFeature(label="pressure", export_label="Pressure_(dbar)", unit="dbar", pl_unit=pl.Float32)
ABSOLUTE_SALINITY = SampleFeature(label="salinity_abs", export_label="Absolute_Salinity_(g/kg)", unit="g/kg",
                                  pl_unit=pl.Float32)
SURFACE_SALINITY = SampleFeature(label="surface_salinity", export_label="Surface_Salinity_(PSU)", unit="PSU",
                                 pl_unit=pl.Float32)
SURFACE_TEMPERATURE = SampleFeature(label="surface_temperature", export_label="Surface_Temperature_(°C)", unit="°C",
                                    pl_unit=pl.Float32)
SURFACE_DENSITY = SampleFeature(label="surface_density", export_label="Mean_Surface_Density_(kg/m^3)", unit="kg/m^3",
                                pl_unit=pl.Float32)
MELTWATER_FRACTION_EQ_10 = SampleFeature(label="meltwater_fraction_eq_10", export_label="Meltwater_Fraction_EQ_10_(%)",
                                         unit="%", pl_unit=pl.Float32)
MELTWATER_FRACTION_EQ_11 = SampleFeature(label="meltwater_fraction_eq_11", export_label="Meltwater_Fraction_EQ_11_(%)",
                                         unit="%", pl_unit=pl.Float32)
DENSITY = SampleFeature(label="density", export_label="Density_(kg/m^3)", unit="kg/m^3", pl_unit=pl.Float32)
POTENTIAL_DENSITY = SampleFeature(label="potential_density", export_label="Potential_Density_(kg/m^3)", unit="kg/m^3",
                                  pl_unit=pl.Float32)
N2 = SampleFeature(label="brunt_vaisala_frequency_squared", export_label="N^2", unit="s^-2", pl_unit=pl.Float32)
P_MID = SampleFeature(label="p_mid", export_label="Mid_Pressure_Used_For_BV_Calc", unit="dbar", pl_unit=pl.Float32)
SECCHI_DEPTH = SampleFeature(label="secchi_depth", export_label="Secchi_Depth_(m)", unit="m", pl_unit=pl.Float32)
LATITUDE = SampleFeature(label="latitude", export_label="latitude", unit="decimal_degrees", pl_unit=pl.Float64)
LONGITUDE = SampleFeature(label="longitude", export_label="longitude", unit="decimal_degrees", pl_unit=pl.Float64)
UNIQUE_ID = SampleFeature(label="unique_id", export_label="Unique_ID", unit="", pl_unit=pl.Utf8)
PROFILE_ID = SampleFeature(label="profile_id", export_label="Profile_ID", unit="", pl_unit=pl.Utf8)
SITE_NAME = SampleFeature(label="site_name", export_label="Site_Name", unit="", pl_unit=pl.Utf8)
SITE_ID = SampleFeature(label="site_id", export_label="Site_ID", unit="", pl_unit=pl.Utf8)
CONSERVATIVE_TEMPERATURE = SampleFeature(label="conservative_temperature", export_label="Conservative_Temperature_(°C)",
                                         unit="°C", pl_unit=pl.Float32)
OXYGEN_CONCENTRATION = SampleFeature(label="oxygen_concentration", export_label="Oxygen_Concentration_(µmol/kg)",
                                     unit="µmol/kg", pl_unit=pl.Float32)
OXYGEN_SATURATION = SampleFeature(label="oxygen_saturation", export_label="Oxygen_Saturation_(%)", unit="%",
                                  pl_unit=pl.Float32)
NITRATE = SampleFeature(label="nitrate", export_label="Nitrate_(µmol/L)", unit="µmol/L", pl_unit=pl.Float32)
PHOSPHATE = SampleFeature(label="phosphate", export_label="Phosphate_(µmol/L)", unit="µmol/L", pl_unit=pl.Float32)
SILICATE = SampleFeature(label="silicate", export_label="Silicate_(µmol/L)", unit="µmol/L", pl_unit=pl.Float32)
PH = SampleFeature(label="ph", export_label="pH", unit="", pl_unit=pl.Float32)
ALKALINITY = SampleFeature(label="alkalinity", export_label="Alkalinity_(µmol/kg)", unit="µmol/kg", pl_unit=pl.Float32)
TURBIDITY = SampleFeature(label="turbidity", export_label="Turbidity_(NTU)", unit="NTU", pl_unit=pl.Float32)
PARTICULATE_ORGANIC_CARBON = SampleFeature(label="particulate_organic_carbon",
                                           export_label="Particulate_Organic_Carbon_(µmol/L)", unit="µmol/L",
                                           pl_unit=pl.Float32)
TOTAL_ORGANIC_CARBON = SampleFeature(label="total_organic_carbon", export_label="Total_Organic_Carbon_(µmol/L)",
                                     unit="µmol/L", pl_unit=pl.Float32)
PARTICULATE_INORGANIC_CARBON = SampleFeature(label="particulate_inorganic_carbon",
                                             export_label="Particulate_Inorganic_Carbon_(µmol/L)", unit="µmol/L",
                                             pl_unit=pl.Float32)
DISSOLVED_INORGANIC_CARBON = SampleFeature(label="dissolved_inorganic_carbon",
                                           export_label="Dissolved_Inorganic_Carbon_(µmol/kg)", unit="µmol/kg",
                                           pl_unit=pl.Float32)
CHLOROPHYLL_FLUORESCENCE = SampleFeature(label="chlorophyll_fluorescence",
                                         export_label="Chlorophyll_Fluorescence_(RFU)", unit="RFU", pl_unit=pl.Float32)
PAR = SampleFeature(label="par", export_label="PAR_(µmol/m²/s)", unit="µmol/m²/s", pl_unit=pl.Float32)
AMMONIUM = SampleFeature(label="ammonium", export_label="Ammonium_(µmol/L)", unit="µmol/L", pl_unit=pl.Float32)
ORP = SampleFeature(label="orp", export_label="ORP_(mV)", unit="mV", pl_unit=pl.Float32)
CLASSIFICATION = SampleFeature(label="profile_type", export_label="Profile_Type", unit="", pl_unit=pl.Categorical)
QUALITY_INDEX = SampleFeature(label="mld_quality_index", export_label="MLD_Quality_Index", unit="", pl_unit=pl.Float32)
MLD_N2 = SampleFeature(label="MLD_BF", export_label="MLD_BF_(m)", unit="(m)", pl_unit=pl.Float32)
# List of all Sample_Feature instances
ALL_SAMPLE_FEATURES = [
    # Identification and metadata
    FILENAME, UNIQUE_ID, PROFILE_ID, SITE_ID, SITE_NAME,

    # Temporal data
    TIMESTAMP, YEAR, MONTH, DAY,

    # Spatial data
    LATITUDE, LONGITUDE,

    # Primary measurements
    DEPTH, PRESSURE, SEA_PRESSURE, P_MID,
    TEMPERATURE, CONSERVATIVE_TEMPERATURE,
    SALINITY, ABSOLUTE_SALINITY,
    DENSITY, POTENTIAL_DENSITY,

    # Derived or calculated values
    SURFACE_TEMPERATURE, SURFACE_SALINITY, SURFACE_DENSITY,
    MELTWATER_FRACTION_EQ_10, MELTWATER_FRACTION_EQ_11,
    N2, CLASSIFICATION,

    # Additional physical properties
    CONDUCTIVITY, SPECIFIC_CONDUCTIVITY,
    SPEED_OF_SOUND,

    # Chemical properties
    OXYGEN_CONCENTRATION, OXYGEN_SATURATION,
    PH, ALKALINITY,
    NITRATE, PHOSPHATE, SILICATE, AMMONIUM,

    # Organic content
    PARTICULATE_ORGANIC_CARBON, TOTAL_ORGANIC_CARBON,
    PARTICULATE_INORGANIC_CARBON, DISSOLVED_INORGANIC_CARBON,

    # Optical properties
    SECCHI_DEPTH, TURBIDITY,
    CHLOROPHYLL, CHLOROPHYLL_FLUORESCENCE,
    PAR,  # Photosynthetically Active Radiation

    # Other
    ORP  # Oxidation-Reduction Potential
]

RELABEL_DICT = {
    # Identification and metadata
    FILENAME.label: FILENAME.export_label,
    UNIQUE_ID.label: UNIQUE_ID.export_label,
    PROFILE_ID.label: PROFILE_ID.export_label,
    SITE_ID.label: SITE_ID.export_label,
    SITE_NAME.label: SITE_NAME.export_label,

    # Temporal data
    TIMESTAMP.label: TIMESTAMP.export_label,
    YEAR.label: YEAR.export_label,
    MONTH.label: MONTH.export_label,
    DAY.label: DAY.export_label,

    # Spatial data
    LATITUDE.label: LATITUDE.export_label,
    LONGITUDE.label: LONGITUDE.export_label,

    # Primary measurements
    DEPTH.label: DEPTH.export_label,
    PRESSURE.label: PRESSURE.export_label,
    SEA_PRESSURE.label: SEA_PRESSURE.export_label,
    P_MID.label: P_MID.export_label,
    TEMPERATURE.label: TEMPERATURE.export_label,
    CONSERVATIVE_TEMPERATURE.label: CONSERVATIVE_TEMPERATURE.export_label,
    SALINITY.label: SALINITY.export_label,
    ABSOLUTE_SALINITY.label: ABSOLUTE_SALINITY.export_label,
    DENSITY.label: DENSITY.export_label,
    POTENTIAL_DENSITY.label: POTENTIAL_DENSITY.export_label,

    # Derived or calculated values
    SURFACE_TEMPERATURE.label: SURFACE_TEMPERATURE.export_label,
    SURFACE_SALINITY.label: SURFACE_SALINITY.export_label,
    SURFACE_DENSITY.label: SURFACE_DENSITY.export_label,
    MELTWATER_FRACTION_EQ_10.label: MELTWATER_FRACTION_EQ_10.export_label,
    MELTWATER_FRACTION_EQ_11.label: MELTWATER_FRACTION_EQ_11.export_label,
    N2.label: N2.export_label,
    CLASSIFICATION.label: CLASSIFICATION.export_label,

    # Additional physical properties
    CONDUCTIVITY.label: CONDUCTIVITY.export_label,
    SPECIFIC_CONDUCTIVITY.label: SPECIFIC_CONDUCTIVITY.export_label,
    SPEED_OF_SOUND.label: SPEED_OF_SOUND.export_label,

    # Chemical properties
    OXYGEN_CONCENTRATION.label: OXYGEN_CONCENTRATION.export_label,
    OXYGEN_SATURATION.label: OXYGEN_SATURATION.export_label,
    PH.label: PH.export_label,
    ALKALINITY.label: ALKALINITY.export_label,
    NITRATE.label: NITRATE.export_label,
    PHOSPHATE.label: PHOSPHATE.export_label,
    SILICATE.label: SILICATE.export_label,
    AMMONIUM.label: AMMONIUM.export_label,

    # Organic content
    PARTICULATE_ORGANIC_CARBON.label: PARTICULATE_ORGANIC_CARBON.export_label,
    TOTAL_ORGANIC_CARBON.label: TOTAL_ORGANIC_CARBON.export_label,
    PARTICULATE_INORGANIC_CARBON.label: PARTICULATE_INORGANIC_CARBON.export_label,
    DISSOLVED_INORGANIC_CARBON.label: DISSOLVED_INORGANIC_CARBON.export_label,

    # Optical properties
    SECCHI_DEPTH.label: SECCHI_DEPTH.export_label,
    TURBIDITY.label: TURBIDITY.export_label,
    CHLOROPHYLL.label: CHLOROPHYLL.export_label,
    CHLOROPHYLL_FLUORESCENCE.label: CHLOROPHYLL_FLUORESCENCE.export_label,
    PAR.label: PAR.export_label,  # Photosynthetically Active Radiation

    # Other
    ORP.label: ORP.export_label  # Oxidation-Reduction Potential
}


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
# Default output files/directories
# ------------------------------------
DEFAULT_OUTPUT_FILE = "ctdfjorder_data.csv"
"""str: Name of the default output file."""

DEFAULT_PLOTS_FOLDER = "ctdplots"
"""str: Name of the default plots output folder."""
