from dataclasses import dataclass
from collections import namedtuple
from typing import NamedTuple
from dataclasses import dataclass, field
import polars as pl

from typing import NamedTuple


class SampleFeature(NamedTuple):
    label: str
    export_label: str
    unit: str
    pl_unit: pl.DataType()


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
