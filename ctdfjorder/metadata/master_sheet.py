import polars as pl
from os import path
from ctdfjorder.exceptions.ctd_exceptions import (
    CTDError,
    raise_warning_improbable_match,
)
from ctdfjorder.constants.constants import *
from ctdfjorder.scardb.generate_sites_db import generate_sites_database
from ctdfjorder.ctddataclasses.ctddataclasses import SitesDatabase, Metadata

from typing import Literal


class MasterSheet:
    TIME_UNIT: Literal["ns", "us", "ms"] = "ns"
    TIME_ZONE: str = "UTC"
    data = None
    # Column labels of master sheet
    secchi_depth_label: str
    latitude_label: str
    longitude_label: str
    date_utc_label: str
    time_utc_label: str
    site_names_label: str
    site_names_short_label: str
    with_crosschecked_site_names: bool
    site_names_db: SitesDatabase

    def __init__(
        self,
        master_sheet_path: str,
        secchi_depth_label: str = "secchi depth",
        latitude_label: str = "latitude",
        longitude_label: str = "longitude",
        datetime_utc_label: str = "date/time (ISO)",
        unique_id_label: str = "UNIQUE ID CODE ",
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
        list_of_cols = [
            self.secchi_depth_label,
            self.latitude_label,
            self.longitude_label,
            self.datetime_utc_label,
            self.unique_id_label,
            self.site_names_label,
            self.site_names_short_label,
        ]
        self.with_crosschecked_site_names: bool = with_crosschecked_site_names

        # Generating polars dataframe representation of the master sheet
        df = None
        if ".xlsx" in path.basename(master_sheet_path):
            df = pl.read_excel(
                master_sheet_path,
                infer_schema_length=None,
            )
        if ".csv" in path.basename(master_sheet_path):
            df = pl.io.csv.read_csv(
                master_sheet_path,
                columns=list_of_cols,
                try_parse_dates=True,
                rechunk=True,
                null_values=[
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
                ],
            )
        if type(df) is type(None):
            raise IOError(
                f"Invalid master sheet filetype. {master_sheet_path} not an xlsx or csv file.",
            )
        self.data = df.drop_nulls(self.datetime_utc_label)
        self.data = self.data.with_columns(
            pl.col(latitude_label).cast(pl.Float64).alias(latitude_label),
            pl.col(longitude_label).cast(pl.Float64).alias(longitude_label),
        )
        if self.with_crosschecked_site_names:
            sites_in_master_sheet = (
                df.select(pl.col(site_names_label)).to_series().to_list()
            )
            self.site_names_db = generate_sites_database(sites_in_master_sheet)

    def find_match(
        self,
        profile: pl.DataFrame,
    ) -> Metadata:
        """
        Extracts the date and time components from the filename and compares them with the data
        in the master sheet. Calculates the absolute differences between the dates and times to
        find the closest match. Returns the estimated latitude, longitude, unique id, and secchi depth
        based on the closest match.

        Parameters
        ----------
        profile : pl.Dataframe
            Profile to match to master sheet.

        Returns
        -------
        tuple
            A tuple containing the estimated latitude, longitude, unique id, and secchi depth.

        Raises
        ------
        CTDError
            When there is no timestamp data in the master sheet and/or CTD file.

        """
        filename = profile.select(pl.first(FILENAME_LABEL)).item()
        if TIMESTAMP_LABEL not in profile.collect_schema().names():
            raise CTDError(message=ERROR_NO_TIMESTAMP_IN_FILE, filename=filename)
        timestamp_in_profile = profile.select(
            pl.last(TIMESTAMP_LABEL)
            .dt.convert_time_zone(TIME_ZONE)
            .cast(pl.Datetime(time_unit=TIME_UNIT, time_zone=TIME_ZONE))
        ).item()
        closest_row_overall = self.data.select(
            pl.all()
            .sort_by((pl.col(self.datetime_utc_label) - timestamp_in_profile).abs())
            .first()
        )
        latitude = closest_row_overall.select(pl.col(self.latitude_label)).item()
        longitude = closest_row_overall.select(pl.col(self.longitude_label)).item()
        distance = (
            closest_row_overall.select(pl.col(self.datetime_utc_label)).item()
            - timestamp_in_profile
        )
        unique_id = closest_row_overall.select(pl.col("UNIQUE ID CODE ")).item()
        secchi_depth = closest_row_overall.select(
            pl.col(self.secchi_depth_label).cast(pl.Float32, strict=False).first()
        ).item()
        # Extract days, hours, and minutes from the time difference
        days = abs(distance.days)
        hours, remainder = divmod(distance.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        message = (
            f"Guessed Unique ID : Matched to unique ID '{unique_id}' with "
            f"distance {days} days and time difference {hours}:{minutes}"
        )
        if abs(days) > 2:
            raise_warning_improbable_match(filename=filename, message=message)
        if abs(days) > 30:
            CTDError(filename=filename, message=message)
        if not isinstance(latitude, float):
            raise_warning_improbable_match(
                f"Latitude: {latitude} is not float", filename=filename
            )
        if self.with_crosschecked_site_names:
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
                    if site_name_of_sample_master_sheet is site.name:
                        if (
                            not abs(site.latitude - latitude_from_profile) < 0.2
                            and not abs(site.longitude - longitude_from_profile) < 0.2
                        ):
                            message = (
                                f"Matched to Unique ID '{unique_id}' but crosschecking "
                                f"site location from SCAR with location from file yields "
                                f"inconsistent results."
                            )
                            raise_warning_improbable_match(
                                message=message, filename=filename
                            )
        return Metadata(
            latitude=latitude,
            longitude=longitude,
            unique_id=unique_id,
            secchi_depth=secchi_depth,
        )