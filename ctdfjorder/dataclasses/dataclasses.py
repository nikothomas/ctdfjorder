from dataclasses import dataclass


@dataclass
class Metadata:
    """
    Represents metadata for a research site, including its coordinates, unique ID, and secchi depth.

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
