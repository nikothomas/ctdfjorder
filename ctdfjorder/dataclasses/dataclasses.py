from dataclasses import dataclass

from dataclasses import dataclass, field
from collections import namedtuple
import polars as pl

from typing import NamedTuple


@dataclass
class Measurement:
    """
    Represents metadata for a measurement.

    Attributes
    ----------
    label : str
        The internal label used in CTDFjorder, if available.
    unit : str | None
        The unit of measurement, if available.
    polars_type : pl.DataType
        The Polars data type for this measurement.
    export_label : str
        The exported label for this measurement.

    """
    label: str
    unit: str | None
    polars_type: pl.DataType()
    export_label: str = field(init=False)

    def __post_init__(self):
        if self.unit:
            self.export_label = f"{self.label}_({self.unit})"
        else:
            self.export_label = self.label


class VarMetadata:
    """
    Represents variable metadata for a CTD cast, including variable labels, units, and Polars types.
    """
    timestamp: Measurement = Measurement(label="timestamp", unit=None, polars_type=pl.Datetime)
    year: Measurement = Measurement(label="year", unit=None, polars_type=pl.Int32)
    month: Measurement = Measurement(label="month", unit=None, polars_type=pl.Int32)
    day: Measurement = Measurement(label="day", unit=None, polars_type=pl.Int32)
    filename: Measurement = Measurement(label="filename", unit=None, polars_type=pl.Utf8)
    chlorophyll: Measurement = Measurement(label="chlorophyll", unit="µg/l", polars_type=pl.Float64)
    temperature: Measurement = Measurement(label="temperature", unit="°C", polars_type=pl.Float64)
    sea_pressure: Measurement = Measurement(label="sea_pressure", unit="dbar", polars_type=pl.Float64)
    depth: Measurement = Measurement(label="depth", unit="m", polars_type=pl.Float64)
    salinity: Measurement = Measurement(label="salinity", unit="PSU", polars_type=pl.Float64)
    speed_of_sound: Measurement = Measurement(label="speed_of_sound", unit="m/s", polars_type=pl.Float64)
    specific_conductivity: Measurement = Measurement(label="specific_conductivity", unit="µS/cm",
                                                     polars_type=pl.Float64)
    conductivity: Measurement = Measurement(label="conductivity", unit="mS/cm", polars_type=pl.Float64)
    pressure: Measurement = Measurement(label="pressure", unit="dbar", polars_type=pl.Float64)
    salinity_abs: Measurement = Measurement(label="salinity_abs", unit="g/kg", polars_type=pl.Float64)
    surface_salinity: Measurement = Measurement(label="surface_salinity", unit="PSU", polars_type=pl.Float64)
    surface_temperature: Measurement = Measurement(label="surface_temperature", unit="°C", polars_type=pl.Float64)
    surface_density: Measurement = Measurement(label="surface_density", unit="kg/m^3", polars_type=pl.Float64)
    meltwater_fraction: Measurement = Measurement(label="meltwater_fraction", unit="%", polars_type=pl.Float64)
    density: Measurement = Measurement(label="density", unit="kg/m^3", polars_type=pl.Float64)
    potential_density: Measurement = Measurement(label="potential_density", unit="kg/m^3", polars_type=pl.Float64)
    brunt_vaisala_frequency_squared: Measurement = Measurement(label="brunt_vaisala_frequency_squared", unit=None,
                                                               polars_type=pl.Float64)
    p_mid: Measurement = Measurement(label="p_mid", unit="dbar", polars_type=pl.Float64)
    secchi_depth: Measurement = Measurement(label="secchi_depth", unit="m", polars_type=pl.Float64)
    latitude: Measurement = Measurement(label="latitude", unit=None, polars_type=pl.Float64)
    longitude: Measurement = Measurement(label="longitude", unit=None, polars_type=pl.Float64)
    unique_id: Measurement = Measurement(label="unique_id", unit=None, polars_type=pl.Utf8)
    profile_id: Measurement = Measurement(label="profile_id", unit=None, polars_type=pl.Utf8)
    site_name: Measurement = Measurement(label="site_name", unit=None, polars_type=pl.Utf8)
    site_id: Measurement = Measurement(label="site_id", unit=None, polars_type=pl.Utf8)
    conservative_temperature: Measurement = Measurement(label="conservative_temperature", unit="°C",
                                                        polars_type=pl.Float64)
    oxygen_concentration: Measurement = Measurement(label="oxygen_concentration", unit="µmol/kg",
                                                    polars_type=pl.Float64)
    oxygen_saturation: Measurement = Measurement(label="oxygen_saturation", unit="%", polars_type=pl.Float64)
    nitrate: Measurement = Measurement(label="nitrate", unit="µmol/L", polars_type=pl.Float64)
    phosphate: Measurement = Measurement(label="phosphate", unit="µmol/L", polars_type=pl.Float64)
    silicate: Measurement = Measurement(label="silicate", unit="µmol/L", polars_type=pl.Float64)
    ph: Measurement = Measurement(label="ph", unit=None, polars_type=pl.Float64)
    alkalinity: Measurement = Measurement(label="alkalinity", unit="µmol/kg", polars_type=pl.Float64)
    turbidity: Measurement = Measurement(label="turbidity", unit="NTU", polars_type=pl.Float64)
    particulate_organic_carbon: Measurement = Measurement(label="particulate_organic_carbon", unit="µmol/L",
                                                          polars_type=pl.Float64)
    total_organic_carbon: Measurement = Measurement(label="total_organic_carbon", unit="µmol/L", polars_type=pl.Float64)
    particulate_inorganic_carbon: Measurement = Measurement(label="particulate_inorganic_carbon", unit="µmol/L",
                                                            polars_type=pl.Float64)
    dissolved_inorganic_carbon: Measurement = Measurement(label="dissolved_inorganic_carbon", unit="µmol/kg",
                                                          polars_type=pl.Float64)
    chlorophyll_fluorescence: Measurement = Measurement(label="chlorophyll_fluorescence", unit="RFU",
                                                        polars_type=pl.Float64)
    par: Measurement = Measurement(label="par", unit="µmol/m²/s", polars_type=pl.Float64)
    ammonium: Measurement = Measurement(label="ammonium", unit="µmol/L", polars_type=pl.Float64)
    orp: Measurement = Measurement(label="orp", unit="mV", polars_type=pl.Float64)


@dataclass
class SamplingEvent:
    """
    Represents metadata for a sampling event from the master sheet, including its coordinates, unique ID, and secchi depth.

    Attributes
    ----------
    latitude : float | None
        The latitude of the site, if available.
    longitude : float | None
        The longitude of the site, if available.
    unique_id : str | None
        The unique identifier for the site, if available.
    secchi_depth : float | None
        The secchi depth measurement for the site, if available.
    site_name : str | None
        The name of the site, if available.
    site_id : str | None
        The short name the site, if available.
    """

    latitude: float | None
    longitude: float | None
    unique_id: str | None
    secchi_depth: float | None
    site_name: str | None
    site_id: str | None
