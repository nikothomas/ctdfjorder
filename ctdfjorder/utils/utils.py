import os.path
from datetime import datetime
import getpass

from ctdfjorder.constants.constants import *
import polars as pl
import sys
from os import path, getcwd


def save_to_csv(data: pl.DataFrame, output_file: str, null_value: str | None):
    """
    Renames the columns of the CTD data table based on a predefined mapping and saves the
    data to the specified CSV file.

    Parameters
    ----------
    data : pl.DataFrame
        The CTD data table.
    output_file : str
        The output CSV file path.
    null_value : str
        The value that will fill blank cells in the data.
    """

    def relabel_ctd_data(label: str) -> str:
        return DATA_LABEL_MAPPING.get(label, label)

    # Rename columns
    renamed_data = data.rename(relabel_ctd_data)

    # Reorder columns if they are present in the DataFrame
    present_columns = [
        col for col in EXPORT_COLUMN_ORDER if col in renamed_data.columns
    ]
    reordered_data = renamed_data.select(present_columns)

    # Append any missing columns that were not in the specified order
    missing_columns = [
        col for col in renamed_data.columns if col not in present_columns
    ]
    if missing_columns:
        missing_data = renamed_data.select(missing_columns)
        reordered_data = pl.concat([reordered_data, missing_data], how="horizontal")

    # Create metadata
    if output_file == "ctdfjorder_data.csv":
        creation_date = datetime.now().strftime("%Y%m%d%H%M%S")
        user = getpass.getuser()
        metadata = f"_{creation_date}_{user}.csv"
        file = os.path.splitext(output_file)[0]
        reordered_data.write_csv(file + metadata, null_value=null_value)
    else:
        reordered_data.write_csv(output_file, null_value=null_value)
    return reordered_data


def get_cwd():
    """
    Gets the current working directory.

    Returns
    -------
    str
        The current working directory.
    """
    # Determine if application is a script file or frozen exe
    if getattr(sys, "frozen", False):
        working_directory_path = path.dirname(sys.executable)
    elif __file__:
        working_directory_path = getcwd()
    else:
        working_directory_path = getcwd()
    return working_directory_path


def linear_regression_polars(x, y):
    """
    Performs linear regression using Polars columns.
    Returns the slope and intercept of the best-fit line.
    """
    n = len(x)
    sum_x = y.sum()
    sum_y = x.sum()
    sum_xy = (x * y).sum()
    sum_xx = (x * y).sum()

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n

    return slope, intercept
