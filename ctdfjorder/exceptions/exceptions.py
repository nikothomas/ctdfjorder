import logging
import warnings

logger = logging.getLogger("ctdfjorder")


class CTDError(Exception):
    """
    Base exception class for CTD-related errors.

    Parameters
    ----------
    message : str
        Explanation of the error.
    filename : str, optional
        Input dataset which caused the error.
    """

    def __init__(self, message, filename=None):
        full_message = f"{filename} - {message}" if filename else message
        super().__init__(full_message)


class NoSamplesError(CTDError):
    """
    Exception raised when a function that requires samples as input is called on a CTD object with no samples.

    Parameters
    ----------
    filename : str, optional
        Input dataset which caused the error.
    """

    def __init__(self, filename=None, func=None):
        super().__init__(message=f"Cannot call {func} on a CTD object with no samples.", filename=filename)


class NoLocationError(CTDError):
    """
    Exception raised when no location could be found.

    Parameters
    ----------
    filename : str, optional
        Input dataset which caused the error.
    """

    def __init__(self, filename=None):
        super().__init__("No location could be found", filename)


class DensityCalculationError(CTDError):
    """
    Exception raised when density calculation fails.

    Parameters
    ----------
    filename : str, optional
        Input dataset which caused the error.
    """

    def __init__(self, filename=None):
        super().__init__("Could not calculate density on this dataset", filename)


class SalinityCalculationError(CTDError):
    """
    Exception raised when salinity absolute calculation fails.

    Parameters
    ----------
    filename : str, optional
        Input dataset which caused the error.
    """

    def __init__(self, filename=None):
        super().__init__("Could not calculate salinity absolute on this dataset", filename)


class MissingMasterSheetError(CTDError):
    """
    Exception raised when no master sheet is provided.

    Parameters
    ----------
    filename : str, optional
        Input dataset which caused the error.
    """

    def __init__(self, filename=None):
        super().__init__("No mastersheet provided, could not update the file's missing location data", filename)


class CorruptMasterSheetError(CTDError):
    """
    Exception raised when master sheet is corrupt.

    Parameters
    ----------
    filename : str, optional
        Input dataset which caused the error.
    """

    def __init__(self, filename=None):
        super().__init__(f"Could not read mastersheet data from mastersheet."
                         f" If on mac download your mastersheet as a csv not an xlsx.", filename)


class CTDCorruptError(CTDError):
    """
    Exception raised when a Ruskin file is corrupted and cannot be read.

    Parameters
    ----------
    filename : str, optional
        Input dataset which caused the error.
    """

    def __init__(self, filename=None):
        super().__init__("File is corrupted or incomplete and could not be read", filename)


class InvalidLocationDataError(CTDError):
    """
    Exception raised when location data is invalid, possibly due to malformed master sheet data.

    Parameters
    ----------
    filename : str, optional
        Input dataset which caused the error.
    """

    def __init__(self, filename=None):
        super().__init__("Location data invalid, probably due to malformed master sheet data", filename)


class MissingTimestampError(CTDError):
    """
    Exception raised when there is no timestamp in the file or master sheet.

    Parameters
    ----------
    filename : str, optional
        Input dataset which caused the error.
    """

    def __init__(self, filename=None, source="file"):
        message = (
            "No timestamp in file, could not get location"
            if source == "file"
            else "No timestamp in master sheet, could not get location"
        )
        super().__init__(message, filename)


class MLDDepthRangeError(CTDError):
    """
    Exception raised when the depth range is insufficient to calculate Mixed Layer Depth (MLD).

    Parameters
    ----------
    filename : str, optional
        Input dataset which caused the error.
    """

    def __init__(self, filename=None):
        super().__init__("Insufficient depth range to calculate MLD", filename)


class InvalidCTDFilenameError(CTDError):
    """
    Exception raised when a CTD filename has an invalid ending.

    Parameters
    ----------
    filename : str, optional
        Input dataset which caused the error.
    """

    def __init__(self, filename=None):
        super().__init__("CTD filename must end in '.rsk' or '.csv'", filename)


def raise_warning_calculatuion(message, filename=None):
    """
    CTD calculation warning function.

    Parameters
    ----------
    filename : str, default None
        Input dataset which caused the error.
    message : str
        Explanation of the error.
    """
    warnings.warn(message=f"{filename} - {message}", category=RuntimeWarning)
    logger.warning(f"{filename} - {message}")


def raise_warning_native_location(message, filename=None):
    """
    CTD location warning function.

    Parameters
    ----------
    filename : str, default None
        Input dataset which caused the error.
    message : str
        Explanation of the error.
    """
    warnings.warn(message=f"{filename} - {message}", category=RuntimeWarning)
    logger.warning(f"{filename} - {message}")
