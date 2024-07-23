import warnings


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
