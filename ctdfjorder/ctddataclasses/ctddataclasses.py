from dataclasses import dataclass

@dataclass
class ResearchSite:
    """
    Represents a research site with its name, coordinates, and narrative description.

    Attributes
    ----------
    name : str
        The full name of the research site.
    short_name : str | None
        The short name or identifier for the research site, if available.
    latitude : float
        The latitude of the research site.
    longitude : float
        The longitude of the research site.
    narrative : str
        A descriptive narrative about the research site.
    """
    name: str
    short_name: str | None
    latitude: float
    longitude: float
    narrative: str

@dataclass
class SitesDatabase:
    """
    Represents a database of research sites.

    Attributes
    ----------
    sites : list[ResearchSite]
        A list of ResearchSite objects that make up the database.
    """
    sites: list[ResearchSite]

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