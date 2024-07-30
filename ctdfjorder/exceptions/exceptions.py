import logging
import warnings

logger = logging.getLogger("ctdfjorder")


class CTDError(Exception):
    """
    Exception raised for CTD related errors.

    Parameters
    ----------
    filename : str, default None
        Input dataset which caused the error.
    message : str
        Explanation of the error.
    """

    def __init__(self, message, filename=None):
        super().__init__(filename + " - " + message)


class Critical(Exception):
    """
    Exception raised for CTDFjorder critical errors.

    Parameters
    ----------
    message : str
        Explanation of the error.
    """

    def __init__(self, message):
        super().__init__(message)


class NativeLocation(Warning):
    def __init__(self, message):
        self.message = message


class Mastersheet(Warning):
    def __init__(self, message):
        self.message = message


class Calculation(Warning):
    def __init__(self, message):
        self.message = message


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
    warnings.warn(message=f"{filename} - {message}", category=Calculation)
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
    warnings.warn(message=f"{filename} - {message}", category=NativeLocation)
    logger.warning(f"{filename} - {message}")


def raise_warning_improbable_match(message, filename=None):
    """
    CTD location warning function.

    Parameters
    ----------
    filename : str, default None
        Input dataset which caused the error.
    message : str
        Explanation of the error.
    """
    warnings.warn(message=f"{filename} - {message}", category=Mastersheet)
    logger.warning(f"{filename} - {message}")


def raise_warning_site_location(message, filename=None):
    """
    CTD site location warning function.

    Parameters
    ----------
    filename : str, default None
        Input dataset which caused the error.
    message : str
        Explanation of the error.
    """
    warnings.warn(message=f"{filename} - {message}", category=Mastersheet)
    logger.warning(f"{filename} - {message}")
