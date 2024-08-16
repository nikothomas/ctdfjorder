import polars as pl
from os import path

import ctdfjorder.exceptions.exceptions
from ctdfjorder.exceptions.exceptions import CTDError
from ctdfjorder.constants.constants import *
from ctdfjorder.dataclasses.dataclasses import SamplingEvent
from typing import Literal

# Try to import optional dependencies
try:
    from ctdfjorder.phyto.scardb.scardb import generate_sites_database
except ImportError:
    generate_sites_database = None


class MasterSheet:
    """
    Represents a master sheet for research sample data, allowing for the integration and cross-checking of sample data.

    Attributes
    ----------
    TIME_UNIT : Literal["ns", "us", "ms"], default "ns"
        The time unit for datetime operations.
    TIME_ZONE : str, default "UTC"
        The time zone for datetime operations.
    data : polars.DataFrame
        The data from the master sheet.
    secchi_depth_label : str
        The label for secchi depth in the master sheet.
    latitude_label : str
        The label for latitude in the master sheet.
    longitude_label : str
        The label for longitude in the master sheet.
    date_utc_label : str
        The label for UTC date in the master sheet.
    time_utc_label : str
        The label for UTC time in the master sheet.
    site_names_label : str
        The label for site names in the master sheet.
    site_names_short_label : str
        The label for short site names in the master sheet.

    Parameters
    ----------
    master_sheet_path : str
        The file path to the master sheet.
    secchi_depth_label : str, default "secchi depth"
        The label for secchi depth in the master sheet.
    latitude_label : str, default "latitude"
        The label for latitude in the master sheet.
    longitude_label : str, default "longitude"
        The label for longitude in the master sheet.
    datetime_utc_label : str, default "date/time (ISO)"
        The label for UTC date and time in the master sheet.
    unique_id_label : str, default "UNIQUE ID CODE "
        The label for unique ID in the master sheet.
    site_names_label : str, default "location"
        The label for site names in the master sheet.
    site_names_short_label : str, default "loc id"
        The label for short site names in the master sheet.
    null_values : list[str], default None
        Specifies which values should be considered null while reading master sheet.
    Raises
    ------
    IOError
        If the master sheet file type is not supported or if the data cannot be read.
    CTDError
        If there are issues with the data, such as missing or invalid values.

    """

    TIME_UNIT: Literal["ns", "us", "ms"] = "ns"
    TIME_ZONE: str = "UTC"
    data = None
    # Column labels of master sheet
    secchi_depth_label: str
    latitude_label: str
    longitude_label: str
    date_utc_label: str
    time_utc_label: str
    filename_label: str
    site_names_label: str
    site_names_short_label: str
    with_crosschecked_site_names: bool

    def __init__(
        self,
        master_sheet_path: str,
        secchi_depth_label: str = "secchi depth",
        latitude_label: str = "nominal latitude",
        longitude_label: str = "nominal longitude",
        datetime_utc_label: str = "date/time (ISO)",
        unique_id_label: str = "UNIQUE ID CODE ",
        filename_label: str = "CTD cast file name",
        site_names_label: str = "location",
        site_names_short_label: str = "loc id",
        null_values: list[str] = None,
    ):
        self.secchi_depth_label: str = secchi_depth_label
        self.latitude_label: str = latitude_label
        self.longitude_label: str = longitude_label
        self.datetime_utc_label: str = datetime_utc_label
        self.unique_id_label: str = unique_id_label
        self.site_names_label: str = site_names_label
        self.site_names_short_label: str = site_names_short_label
        self.filename_label: str = filename_label
        list_of_cols = [
            self.secchi_depth_label,
            self.latitude_label,
            self.longitude_label,
            self.datetime_utc_label,
            self.unique_id_label,
            self.site_names_label,
            self.site_names_short_label,
            self.filename_label,
        ]

        # Generating polars dataframe representation of the master sheet
        df = None
        if ".xlsx" in path.basename(master_sheet_path):
            df = pl.read_excel(master_sheet_path, engine="calamine")
            df = df.select(list_of_cols)
            df = df.with_columns(
                pl.col(self.datetime_utc_label).replace(old=list_of_cols, new=None)
            )
            df = df.with_columns(
                pl.col(self.datetime_utc_label)
                .str.to_datetime(
                    format="%Y-%m-%dT%H:%M:%S.%f%z", strict=False, exact=False
                )
                .cast(pl.Datetime)
                .dt.cast_time_unit(TIME_UNIT)
                .dt.replace_time_zone(TIME_ZONE)
                .alias(self.datetime_utc_label)
            )
            df = df.drop_nulls(self.datetime_utc_label)
            if df.is_empty():
                raise ctdfjorder.exceptions.exceptions.CorruptMasterSheetError(filename=master_sheet_path)

        if ".csv" in path.basename(master_sheet_path):
            df = pl.read_csv(
                master_sheet_path,
                columns=list_of_cols,
                try_parse_dates=True,
                rechunk=True,
                null_values=null_values,
            )
            try:
                df.select(pl.col(self.datetime_utc_label).dt.time())
            except pl.exceptions.SchemaError:
                raise ctdfjorder.exceptions.exceptions.CorruptMasterSheetError(filename=master_sheet_path)
        if type(df) is type(None):
            raise IOError(
                f"Invalid master sheet filetype. {master_sheet_path} not an xlsx or csv file."
            )
        self.data = df.drop_nulls(self.datetime_utc_label)
        self.data = self.data.with_columns(
            pl.col(latitude_label).cast(pl.Float64).alias(latitude_label),
            pl.col(longitude_label).cast(pl.Float64).alias(longitude_label),
        )

    def find_match(
        self,
        profile: pl.DataFrame,
    ) -> SamplingEvent:
        """
        Locates the row in the master sheet with a filename value that matches the profile parameter.
        Returns the time, latitude, longitude, unique id, loc id, location  and secchi depth from that row.

        Parameters
        ----------
        profile : pl.DataFrame
            Profile to match to master sheet.

        Returns
        -------
        MetadataMastersheet
            An object containing the estimated latitude, longitude, unique id, and secchi depth.

        Raises
        ------
        CTDError
            When there is no timestamp data in the master sheet and/or CTD file.
        """
        filename = profile.select(pl.first(FILENAME_LABEL)).item()
        closest_row_overall = self.data.filter(
            pl.col(self.filename_label).str.contains(filename)
        )
        if closest_row_overall.height < 1:
            raise CTDError(
                message="No unique ID's associated with this cast", filename=filename
            )
        if closest_row_overall.height > 1:
            raise CTDError(
                message="Multiple unique ID's associated with this cast",
                filename=filename,
            )
        latitude = closest_row_overall.item(row=0, column=self.latitude_label)
        longitude = closest_row_overall.item(row=0, column=self.longitude_label)
        unique_id = closest_row_overall.select(pl.col(self.unique_id_label)).item(
            row=0, column=0
        )
        secchi_depth = closest_row_overall.select(
            pl.col(self.secchi_depth_label).cast(pl.Float32, strict=False).first()
        ).item(row=0, column=0)
        site_name = closest_row_overall.select(
            pl.col(self.site_names_label).cast(pl.String).first()
        ).item(row=0, column=0)
        site_id = closest_row_overall.select(
            pl.col(self.site_names_short_label).cast(pl.String).first()
        ).item(row=0, column=0)
        return SamplingEvent(
            latitude=latitude,
            longitude=longitude,
            unique_id=unique_id,
            secchi_depth=secchi_depth,
            site_name=site_name,
            site_id=site_id,
        )
