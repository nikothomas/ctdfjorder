from dataclasses import dataclass


@dataclass
class ResearchSite:
    name: str
    short_name: str | None
    latitude: float
    longitude: float
    narrative: str


@dataclass
class SitesDatabase:
    sites: list[ResearchSite]

@dataclass
class Metadata:
    latitude: float | None
    longitude: float | None
    unique_id: str | None
    secchi_depth: float | None