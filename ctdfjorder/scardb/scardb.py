# List of site names
import os
from ctdfjorder.dataclasses.dataclasses import SitesDatabase, ResearchSite
import polars as pl
from ctdfjorder.exceptions.exceptions import raise_warning_site_location


def generate_sites_database(site_names: list[str]) -> SitesDatabase:
    """
    Generates a database of research sites based on given site names, using data from various SCAR (Scientific Committee on Antarctic Research) datasets.

    Parameters
    ----------
    site_names : list[str]
        A list of site names to search for in the SCAR datasets.

    Returns
    -------
    SitesDatabase
        An object containing a list of ResearchSite objects for each matching site found in the datasets.

    Notes
    -----
    The function attempts to find exact matches for the provided site names in the SCAR datasets. If no exact match is found,
    it calculates the Levenshtein similarity between the provided site name and the names in the datasets to suggest potential matches.

    Examples
    --------
    Generate a database with a list of site names:

    >>> site_names = ['Site A', 'Site B', 'Site C']
    >>> database = generate_sites_database(site_names)
    >>> print(database)
    SitesDatabase with 3 sites

    Details
    -------
    - The function uses Levenshtein distance to compute similarities between site names.
    - It reads data from SCAR datasets stored in Parquet files.
    - If an exact match is found, the corresponding site information is added to the `SitesDatabase`.
    - If no exact match is found, it suggests potential matches with the highest similarity score.
    - Raises a warning if no similar site name is found in any of the datasets.

    Raises
    ------
    Warning
        If a provided site name does not match exactly or similarly with any site name in the SCAR datasets.

    """
    def levenshtein_distance(s1, s2):
        s1 = str.lower(s1)
        s2 = str.lower(s2)
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize the matrix
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Compute the Levenshtein distance
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,  # Insertion
                    dp[i - 1][j - 1] + cost,
                )  # Substitution

        return dp[m][n]

    def levenshtein_similarity(s1, s2):
        if type(s1) is type(None):
            return 0.0
        if type(s2) is type(None):
            return 0.0
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        distance = levenshtein_distance(s1, s2)
        return float((max_len - distance) / max_len)

    def get_parquet_file_path(filename: str):
        # Get the absolute path of the Parquet file
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, "sitenamesdb")
        file_path = os.path.join(data_dir, filename)
        return file_path

    site_names_SCAR_USA = pl.read_parquet(get_parquet_file_path("USA.parquet"))
    site_names_SCAR_UK = pl.read_parquet(get_parquet_file_path("UK.parquet"))
    site_names_SCAR_Argentina = pl.read_parquet(
        get_parquet_file_path("Argentina.parquet")
    )
    site_names_SCAR_Chile = pl.read_parquet(get_parquet_file_path("Chile.parquet"))

    priority_dfs = [
        site_names_SCAR_USA,
        site_names_SCAR_UK,
        site_names_SCAR_Argentina,
        site_names_SCAR_Chile,
    ]
    sites_found_list: list[ResearchSite] = []
    for site_name in site_names:
        found = False
        for df in priority_dfs:
            matching_rows = df.filter(pl.col("place_name_mapping") == site_name)
            if not matching_rows.is_empty():
                site_name = matching_rows.select(
                    pl.col("place_name_mapping").first()
                ).item()
                site_lat = matching_rows.select(pl.col("latitude").first()).item()
                site_long = matching_rows.select(pl.col("longitude").first()).item()
                site_narrative = matching_rows.select(
                    pl.col("narrative").first()
                ).item()
                new_site = ResearchSite(
                    name=site_name,
                    short_name=None,
                    latitude=site_lat,
                    longitude=site_long,
                    narrative=site_narrative,
                )
                sites_found_list.append(new_site)
                found = True
                break
        if not found:
            name = pl.col("place_name_mapping")
            sim = "similarity"
            sim_col = pl.col("similarity")
            SCAR_with_similarity_USA = site_names_SCAR_USA.select(
                name.map_elements(lambda x: levenshtein_similarity(x, site_name))
            )
            SCAR_with_similarity_UK = site_names_SCAR_UK.select(
                name.map_elements(lambda x: levenshtein_similarity(x, site_name))
            )
            SCAR_with_similarity_Argentina = site_names_SCAR_Argentina.select(
                name.map_elements(lambda x: levenshtein_similarity(x, site_name))
            )
            SCAR_with_similarity_Chile = site_names_SCAR_Chile.select(
                name.map_elements(lambda x: levenshtein_similarity(x, site_name))
            )
            sim_USA = site_names_SCAR_USA.with_columns(
                SCAR_with_similarity_USA.to_series().alias(sim)
            )
            sim_UK = site_names_SCAR_UK.with_columns(
                SCAR_with_similarity_UK.to_series().alias(sim)
            )
            sim_Arg = site_names_SCAR_Argentina.with_columns(
                SCAR_with_similarity_Argentina.to_series().alias(sim)
            )
            sim_Chile = site_names_SCAR_Chile.with_columns(
                SCAR_with_similarity_Chile.to_series().alias(sim)
            )
            closest_USA = sim_USA.filter(sim_col == sim_col.max()).limit(2)
            closest_UK = sim_UK.filter(sim_col == sim_col.max()).limit(2)
            closest_Arg = sim_Arg.filter(sim_col == sim_col.max()).limit(2)
            closest_Chile = sim_Chile.filter(sim_col == sim_col.max()).limit(2)
            sn_USA = closest_USA.select(name).to_series().to_list()
            sn_UK = closest_UK.select(name).to_series().to_list()
            sn_Arg = closest_Arg.select(name).to_series().to_list()
            sn_Chile = closest_Chile.select(name).to_series().to_list()
            ss_USA = closest_USA.select(sim_col).to_series().to_list()
            ss_UK = closest_UK.select(sim_col).to_series().to_list()
            ss_Arg = closest_Arg.select(sim_col).to_series().to_list()
            ss_Chile = closest_Chile.select(sim_col).to_series().to_list()
            if 0.0 in ss_USA:
                sn_USA = None
            if 0.0 in ss_UK:
                sn_UK = None
            if 0.0 in ss_Arg:
                sn_UK = None
            if 0.0 in ss_Chile:
                sn_UK = None
            raise_warning_site_location(
                message=f"Site name '{site_name}' may be one of US: {sn_USA} UK: {sn_UK} AR: {sn_Arg} CH: {sn_Chile} "
            )
    return SitesDatabase(sites_found_list)
