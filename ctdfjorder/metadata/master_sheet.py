import polars as pl
from os import path

import ctdfjorder.exceptions.exceptions
from ctdfjorder.exceptions.exceptions import (
    CTDError,
    raise_warning_improbable_match,
)
from ctdfjorder.constants.constants import *
from ctdfjorder.dataclasses.dataclasses import SitesDatabase, Metadata
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
    with_crosschecked_site_names : bool
        Indicates whether site names should be cross-checked with a database, requires 'phyto' optional dependency.
    site_names_db : SitesDatabase
        The database of site names for cross-checking, requires 'phyto' optional dependency.

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
    with_crosschecked_site_names : bool, default False
        Indicates whether site names should be cross-checked with a database, if true requires 'phyto' optional dependency.

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
    site_names_db: SitesDatabase

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
            with_crosschecked_site_names: bool = False,
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
        self.with_crosschecked_site_names: bool = with_crosschecked_site_names

        # Generating polars dataframe representation of the master sheet
        df = None
        null_values = [
            "-999",
            "NA",
            "#N/A",
            "",
            "2022-10-29 -999",
            "14-11-2022 11:50",
            "14-11-2022 12:20",
            "2022-11-28 -999",
            "28-11-2022 13:00",
            "23-12-2022 20:30",
            "16-1-2023 11:37",
            "19-1-2023 13:23",
            "2023-01-22 -999",
            "17-2-2023 12:01",
            "17-2-2023 17:10",
            "18-2-2023 17:05",
            "19-2-2023 12:03",
            "20-2-2023 12:05",
            "20-2-2023 22:00",
            "20-2-2023 16:00",
            "22-2-2023 11:30",
            "22-2-2023 18:30",
            "24-2-2023 18:00",
            "25-2-2023 17:00",
            "26-2-2023 11:28",
            "27-2-2023 11:06",
            "27-2-2023 18:00",
            "28-Jan--22 18:30",
            "OCt-NOV ",
            " ",
        ]
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
                raise ctdfjorder.exceptions.exceptions.Critical(
                    f"Could not read mastersheet data from {master_sheet_path}."
                    f" If on mac download your mastersheet as a csv not an xlsx."
                )

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
                raise ctdfjorder.exceptions.exceptions.Critical(
                    f"Could not read mastersheet data from {master_sheet_path}."
                    f" If on mac download your mastersheet as a csv not an xlsx."
                )
        if type(df) is type(None):
            raise IOError(
                f"Invalid master sheet filetype. {master_sheet_path} not an xlsx or csv file."
            )
        self.data = df.drop_nulls(self.datetime_utc_label)
        self.data = self.data.with_columns(
            pl.col(latitude_label).cast(pl.Float64).alias(latitude_label),
            pl.col(longitude_label).cast(pl.Float64).alias(longitude_label),
        )
        if self.with_crosschecked_site_names and generate_sites_database is not None:
            sites_in_master_sheet = (
                df.select(pl.col(site_names_label)).to_series().to_list()
            )
            self.site_names_db = generate_sites_database(sites_in_master_sheet)

    def find_match(
            self,
            profile: pl.DataFrame,
    ) -> Metadata:
        """
        Locates the row in the master sheet with a filename value that matches the profile parameter.
        Returns the latitude, longitude, unique id, and secchi depth from that row.
        based on the closest match.

        Parameters
        ----------
        profile : pl.DataFrame
            Profile to match to master sheet.

        Returns
        -------
        Metadata
            An object containing the estimated latitude, longitude, unique id, and secchi depth.

        Raises
        ------
        CTDError
            When there is no timestamp data in the master sheet and/or CTD file.
        Warning
            When the guessed unique ID or latitude is improbable or inconsistent.
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
        if not isinstance(latitude, float):
            raise_warning_improbable_match(
                f"Latitude: {latitude} is not float", filename=filename
            )
        if self.with_crosschecked_site_names and generate_sites_database is not None:
            latitude_from_profile = profile.select(
                pl.col(LATITUDE_LABEL).first()
            ).item()
            longitude_from_profile = profile.select(
                pl.col(LONGITUDE_LABEL).first()
            ).item()
            if latitude_from_profile is not None and longitude_from_profile is not None:
                site_name_of_sample_master_sheet = closest_row_overall.select(
                    (pl.col(self.site_names_label))
                ).item()
                for site in self.site_names_db.sites:
                    if site_name_of_sample_master_sheet == site.name:
                        if (
                                not abs(site.latitude - latitude_from_profile) < 0.2
                                or not abs(site.longitude - longitude_from_profile) < 0.2
                        ):
                            message = (
                                f"Matched to Unique ID '{unique_id}' but crosschecking "
                                f"site location from SCAR with location from file yields "
                                f"inconsistent results."
                            )
                            raise CTDError(message=message, filename=filename)
        return Metadata(
            latitude=latitude,
            longitude=longitude,
            unique_id=unique_id,
            secchi_depth=secchi_depth,
            site_name=site_name,
            site_id=site_id,
        )
