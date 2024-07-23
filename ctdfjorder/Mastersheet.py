from ctdfjorder.constants import *
import polars as pl
from os import path
from ctdfjorder.CTDExceptions.CTDExceptions import (
    CTDError,
    raise_warning_improbable_match,
)
from typing import Tuple
from typing import Any


class Mastersheet:
    data = None

    def __init__(self, master_sheet_path: str = None):
        def parse_date_column(column, formats):
            for fmt in formats:
                try:
                    column = column.str.strptime(
                        pl.Date, format=fmt, strict=False, exact=False
                    )
                    return column
                except:
                    pass
            raise ValueError(
                f"Date format not matched for any of the formats: {formats}"
            )

        _masterSheetLabels_to_dtypeInternal: dict[str, type(pl.String)] = {
            MASTER_SHEET_SECCHI_DEPTH_LABEL: pl.String,
            LATITUDE_LABEL: pl.String,
            LONGITUDE_LABEL: pl.String,
            MASTER_SHEET_DATE_UTC_LABEL: pl.String,
            MASTER_SHEET_TIME_UTC_LABEL: pl.String,
        }
        df = None
        if ".xlsx" in path.basename(master_sheet_path):
            df = pl.read_excel(
                master_sheet_path,
                infer_schema_length=None,
                schema_overrides=_masterSheetLabels_to_dtypeInternal,
            )
        if ".csv" in path.basename(master_sheet_path):
            df = pl.io.csv.read_csv(
                master_sheet_path,
                schema_overrides=_masterSheetLabels_to_dtypeInternal,
            )
        if type(df) is type(None):
            raise IOError(
                f"Invalid master sheet filetype. {master_sheet_path} not an xlsx or csv file.",
            )
        df = df.filter(
            ~pl.col(MASTER_SHEET_TIME_UTC_LABEL).eq("-999"),
            ~pl.col(MASTER_SHEET_TIME_UTC_LABEL).eq("NA"),
            ~pl.col(MASTER_SHEET_DATE_UTC_LABEL).eq("NA"),
        )
        date_formats = ["%Y-%m-%d", "%d/%m/%Y"]
        df = df.with_columns(
            parse_date_column(
                pl.col(MASTER_SHEET_DATE_UTC_LABEL), formats=date_formats
            ),
            pl.col(MASTER_SHEET_TIME_UTC_LABEL).str.strptime(
                format="%H:%M", dtype=pl.Time, strict=False, exact=False
            ),
            pl.col(MASTER_SHEET_SECCHI_DEPTH_LABEL).cast(pl.Float64, strict=False),
        )
        df = df.drop_nulls(MASTER_SHEET_DATE_UTC_LABEL)
        df = df.drop_nulls(MASTER_SHEET_TIME_UTC_LABEL)
        df = df.with_columns(
            (
                pl.col(MASTER_SHEET_DATE_UTC_LABEL).dt.combine(
                    pl.col(MASTER_SHEET_TIME_UTC_LABEL), time_unit=TIME_UNIT
                )
            )
            .dt.cast_time_unit(TIME_UNIT)
            .alias(MASTER_SHEET_DATETIME_LABEL)
            .dt.replace_time_zone(TIME_ZONE),
            pl.when(pl.col(MASTER_SHEET_SECCHI_DEPTH_LABEL) == -999)
            .then(None)
            .otherwise(pl.col(MASTER_SHEET_SECCHI_DEPTH_LABEL))
            .alias(MASTER_SHEET_SECCHI_DEPTH_LABEL),
        )
        df = df.drop_nulls(MASTER_SHEET_DATETIME_LABEL)
        self.data = df

    def find_match(
        self,
        profile: pl.DataFrame,
    ) -> Tuple[Any, Any, str, float | None]:
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
        if "datetime" not in self.data.collect_schema().names():
            raise CTDError(
                message=ERROR_NO_TIMESTAMP_IN_MASTER_SHEET,
                filename=filename,
            )
        timestamp_highest = profile.select(
            pl.last(TIMESTAMP_LABEL)
            .dt.convert_time_zone(TIME_ZONE)
            .cast(pl.Datetime(time_unit=TIME_UNIT, time_zone=TIME_ZONE))
        ).item()
        closest_row_overall = self.data.select(
            pl.all().sort_by(
                (pl.col(MASTER_SHEET_DATETIME_LABEL) - timestamp_highest).abs()
            )
        )
        latitude = closest_row_overall.select(pl.col(LATITUDE_LABEL).first()).item()
        longitude = closest_row_overall.select(pl.col(LONGITUDE_LABEL).first()).item()
        distance = (
            closest_row_overall.select(
                pl.col(MASTER_SHEET_DATETIME_LABEL).first()
            ).item()
            - timestamp_highest
        )
        unique_id = closest_row_overall.select(pl.col("UNIQUE ID CODE ").first()).item()
        secchi_depth = closest_row_overall.select(
            pl.col(MASTER_SHEET_SECCHI_DEPTH_LABEL)
            .cast(pl.Float32, strict=False)
            .first()
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
        return latitude, longitude, unique_id, secchi_depth
