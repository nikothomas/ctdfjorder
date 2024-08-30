from matplotlib import pyplot as plt
import numpy as np
import os
from ctdfjorder.constants.constants import *
import polars as pl
from dash import dcc, html, dash
from dash.dependencies import Input, Output, State
import plotly.express as px
import logging
from flask import Flask
import re
import matplotlib.colors as mcolors


def plot_map(df: pl.DataFrame, mapbox_access_token):
    """
    Generates an interactive map using Plotly and Dash to visualize CTD profiles.

    Parameters
    ----------
    df : pl.DataFrame
        The CTD data to plot on the map.
    mapbox_access_token : str
        The Mapbox access token for rendering the map.

    Notes
    -----
    This function creates a Dash web application with interactive controls for filtering the data by year,
    month, unique ID, latitude, longitude, and date range. The filtered data is displayed on a Mapbox map.
    """
    px.set_mapbox_access_token(mapbox_access_token)

    lat_median = df.select(pl.col(LATITUDE.export_label).median().first()).item()
    long_median = df.select(pl.col(LONGITUDE.export_label).median().first()).item()
    if UNIQUE_ID.export_label not in df.columns:
        df = df.with_columns((pl.col(FILENAME.export_label).cast(UNIQUE_ID.pl_unit) + pl.col(PROFILE_ID.export_label).cast(UNIQUE_ID.pl_unit)).alias(UNIQUE_ID.export_label));
    df = df.filter(pl.col(YEAR.export_label) >= 2017)
    pd_df = df.to_pandas()

    server = Flask(__name__)
    app = dash.Dash(__name__, server=server)
    app.logger.setLevel(logging.ERROR)
    server.logger.setLevel(logging.ERROR)

    app.layout = html.Div(
        [
            html.Div(
                [
                    dcc.Dropdown(
                        id="year-dropdown",
                        options=[
                            {"label": str(year), "value": year}
                            for year in pd_df[YEAR.export_label].unique()
                        ],
                        placeholder="Select a year",
                        multi=True,
                    ),
                    dcc.Dropdown(
                        id="month-dropdown",
                        options=[
                            {"label": month, "value": index}
                            for index, month in enumerate(
                                [
                                    "January",
                                    "February",
                                    "March",
                                    "April",
                                    "May",
                                    "June",
                                    "July",
                                    "August",
                                    "September",
                                    "October",
                                    "November",
                                    "December",
                                ],
                                1,
                            )
                        ],
                        placeholder="Select a month",
                        multi=True,
                    ),
                    dcc.Input(
                        id="unique-id-input",
                        type="text",
                        placeholder="Enter Unique ID or regex",
                        style={"width": "100%"},
                    ),
                    dcc.Input(
                        id="lat-range",
                        type="text",
                        placeholder="Enter latitude range (e.g. -60, -55)",
                        style={"width": "100%"},
                    ),
                    dcc.Input(
                        id="lon-range",
                        type="text",
                        placeholder="Enter longitude range (e.g. -70, -65)",
                        style={"width": "100%"},
                    ),
                    dcc.DatePickerRange(
                        id="date-picker-range",
                        start_date=pd_df["timestamp"].min(),
                        end_date=pd_df["timestamp"].max(),
                        display_format="YYYY-MM-DD",
                        style={"width": "100%"},
                    ),
                    html.Div(id="total-profiles", style={"padding": "10px 0"}),
                    html.Button("Download CSV", id="download-button"),
                    dcc.Download(id="download-dataframe-csv"),
                ],
                style={
                    "width": "20%",
                    "display": "inline-block",
                    "verticalAlign": "top",
                },
            ),
            html.Div(
                [
                    dcc.Graph(id="map", style={"height": "100vh", "width": "100vw"}),
                ],
                style={
                    "width": "80%",
                    "display": "inline-block",
                    "verticalAlign": "top",
                },
            ),
        ],
        style={"height": "100vh", "width": "100vw", "margin": 0, "padding": 0},
    )

    @app.callback(
        [Output("map", "figure"), Output("total-profiles", "children")],
        [
            Input("year-dropdown", "value"),
            Input("month-dropdown", "value"),
            Input("unique-id-input", "value"),
            Input("lat-range", "value"),
            Input("lon-range", "value"),
            Input("date-picker-range", "start_date"),
            Input("date-picker-range", "end_date"),
        ],
    )
    def update_map(
            selected_years,
            selected_months,
            unique_id_filter,
            lat_range,
            lon_range,
            start_date,
            end_date,
    ):
        filtered_df = pd_df.copy()
        if selected_years:
            filtered_df = filtered_df[
                filtered_df[YEAR.export_label].isin(selected_years)
            ]
        if selected_months:
            filtered_df = filtered_df[filtered_df[MONTH.export_label].isin(selected_months)]
        if unique_id_filter:
            unique_id_list = unique_id_filter.split()
            try:
                filtered_df = filtered_df[
                    filtered_df["Unique_ID"].str.contains(
                        "|".join(unique_id_list), regex=True
                    )
                ]
            except re.error:
                pass
        if lat_range:
            try:
                lat_min, lat_max = map(float, lat_range.split(","))
                filtered_df = filtered_df[
                    (filtered_df["latitude"] >= lat_min)
                    & (filtered_df["latitude"] <= lat_max)
                    ]
            except ValueError:
                pass
        if lon_range:
            try:
                lon_min, lon_max = map(float, lon_range.split(","))
                filtered_df = filtered_df[
                    (filtered_df["longitude"] >= lon_min)
                    & (filtered_df["longitude"] <= lon_max)
                    ]
            except ValueError:
                pass
        if start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df["timestamp"] >= start_date)
                & (filtered_df["timestamp"] <= end_date)
                ]

        total_profiles = filtered_df["Unique_ID"].nunique()
        try:
            lat_median = filtered_df[LATITUDE.export_label].median()
            long_median = filtered_df[LONGITUDE.export_label].median()
        except Exception as e:
            pass
        fig = px.scatter_mapbox(
            filtered_df,
            lat="latitude",
            lon="longitude",
            hover_name="Unique_ID",
            hover_data={
                "timestamp": True,
                "filename": True,
                "latitude": True,
                "longitude": True,
            },
            mapbox_style="light",
            zoom=5,
            center={"lat": lat_median, "lon": long_median},
        )

        fig.update_layout(
            mapbox_accesstoken=mapbox_access_token, margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig, f"Total Profiles: {total_profiles}"

    @app.callback(
        Output("download-dataframe-csv", "data"),
        [Input("download-button", "n_clicks")],
        [
            State("year-dropdown", "value"),
            State("month-dropdown", "value"),
            State("unique-id-input", "value"),
            State("lat-range", "value"),
            State("lon-range", "value"),
            State("date-picker-range", "start_date"),
            State("date-picker-range", "end_date"),
        ],
    )
    def download_csv(
            n_clicks,
            selected_years,
            selected_months,
            unique_id_filter,
            lat_range,
            lon_range,
            start_date,
            end_date,
    ):
        if n_clicks is None:
            raise dash.PreventUpdate

        filtered_df = pd_df.copy()
        if selected_years:
            filtered_df = filtered_df[
                filtered_df[YEAR.export_label].isin(selected_years)
            ]
        if selected_months:
            filtered_df = filtered_df[filtered_df[MONTH.export_label].isin(selected_months)]
        if unique_id_filter:
            unique_id_list = unique_id_filter.split()
            try:
                filtered_df = filtered_df[
                    filtered_df["Unique_ID"].str.contains(
                        "|".join(unique_id_list), regex=True
                    )
                ]
            except re.error:
                pass
        if lat_range:
            try:
                lat_min, lat_max = map(float, lat_range.split(","))
                filtered_df = filtered_df[
                    (filtered_df["latitude"] >= lat_min)
                    & (filtered_df["latitude"] <= lat_max)
                    ]
            except ValueError:
                pass
        if lon_range:
            try:
                lon_min, lon_max = map(float, lon_range.split(","))
                filtered_df = filtered_df[
                    (filtered_df["longitude"] >= lon_min)
                    & (filtered_df["longitude"] <= lon_max)
                    ]
            except ValueError:
                pass
        if start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df["timestamp"] >= start_date)
                & (filtered_df["timestamp"] <= end_date)
                ]

        return dcc.send_data_frame(filtered_df.to_csv, "selected_profiles.csv")

    import webbrowser
    from threading import Timer

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8050/")

    Timer(1, open_browser).start()
    app.run_server(debug=False, host='10.70.204.98', port=8050)


def plot_depth_vs(
        df: pl.DataFrame, measurement: str, plot_folder: str, plot_type: str = "scatter"
):
    """
    Generates a plot of depth vs. specified measurement (salinity, density, temperature).

    Parameters
    ----------
    df : pl.DataFrame
        The CTD dataframe containing the data to plot.
    measurement : str
        The measurement to plot against depth. Options are 'salinity', 'density', 'potential_density', or 'temperature'.
    plot_folder : str
        The path to the folder where plots will be saved.
    plot_type : str, optional
        The type of plot to generate. Options are 'scatter'. Defaults to 'scatter'.

    Notes
    -----
    - Adds horizontal lines indicating the mixed layer depth (MLD) if present.
    - Allows for both scatter and line plot types.
    - Saves the plot as an image file in the specified folder.
    """
    plt.rcParams.update({"font.size": 16})
    os.makedirs(plot_folder, exist_ok=True)
    for profile_id in (
            df.select(PROFILE_ID.label).unique(keep="first").to_series().to_list()
    ):
        profile = df.filter(pl.col(PROFILE_ID.label) == profile_id)
        # Calculate the standard deviation of the brunt_vaisala column if it exists
        if QUALITY_INDEX.label in profile.columns:
            quality_index = profile.select(pl.col(QUALITY_INDEX.label).first()).item()
        else:
            quality_index = None
        if CLASSIFICATION.label in profile.columns:
            profile_type = profile.select(pl.col(CLASSIFICATION.label).first()).item()
        else:
            profile_type = None

        filename = profile.select(pl.first(FILENAME.label)).item()
        fig, ax1 = plt.subplots(figsize=(18, 18))
        ax1.invert_yaxis()
        ax1.set_ylim([profile.select(pl.col(DEPTH.label)).max().item(), 0])
        color_map = {
            SALINITY.label: "tab:blue",
            DENSITY.label: "tab:red",
            POTENTIAL_DENSITY.label: "tab:red",
            TEMPERATURE.label: "tab:blue",
        }
        label_map = {
            SALINITY.label: "Practical Salinity (PSU)",
            DENSITY.label: "Density (kg/m^3)",
            POTENTIAL_DENSITY.label: "Potential Density (kg/m^3)",
            TEMPERATURE.label: "Temperature (°C)",
        }
        x, y = (
            profile.select(pl.col(f"{measurement}")).to_numpy(),
            profile.select(pl.col(DEPTH.label)).to_numpy(),
        )
        ax1.scatter(
            x, y, color=color_map[measurement], label=label_map[measurement]
        )
        ax1.set_xlabel(label_map[measurement], color=color_map[measurement])
        ax1.set_ylabel("Depth (m)")
        ax1.tick_params(axis="x", labelcolor=color_map[measurement])

        # Select all columns that match the MLD pattern
        mld_columns = profile.select(pl.col(r"^.*MLD.*$"))

        # Extract and filter out None or NaN MLD values
        mld_values = [
            profile.select(pl.col(col).first()).item()
            for col in mld_columns.columns
            if profile.select(pl.col(col).first()).item() is not None
               and not np.isnan(profile.select(pl.col(col).first()).item())
        ]

        if mld_values:  # Proceed only if there are valid MLD values
            # Determine the minimum and maximum MLD values for normalization
            min_mld = min(mld_values)
            max_mld = max(mld_values)

            # Normalize function to map MLD values to a 0-1 range
            normalize = mcolors.Normalize(vmin=min_mld, vmax=max_mld)

            # Base color (light green) to start with
            base_color = (0.2, 0.8, 0.2)  # RGB for a medium green

            def adjust_color(base_color, intensity):
                """Darken the base color based on intensity."""
                return tuple(c * (1 - intensity * 0.5) for c in base_color)

            # Iterate over each MLD column and plot the corresponding MLD line
            for col in mld_columns.columns:
                mld = profile.select(pl.col(col).first()).item()
                if mld is not None and not np.isnan(mld):
                    # Normalize MLD value to get intensity
                    intensity = normalize(mld)

                    # Adjust base color according to intensity
                    color_intensity = adjust_color(base_color, intensity)

                    # Plot MLD line
                    ax1.axhline(
                        y=mld,
                        color=color_intensity,
                        linestyle="--",
                        linewidth=2,
                        label=f"{col}: {mld}",
                    )
                    ax1.text(
                        0.95,
                        mld,
                        f"{col}: {mld}",
                        va="center",
                        ha="right",
                        backgroundcolor="white",
                        color=color_intensity,
                        transform=ax1.get_yaxis_transform(),
                    )
        plt.title(
            f"{filename} \n Profile {profile_id} \n Depth vs. {label_map[measurement]} \n MLD Quality Index: {quality_index} \n Profile Type: {profile_type}"
        )
        ax1.grid(True)
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
        plot_path = os.path.join(
            plot_folder,
            f"{filename}_{profile_id}_depth_{measurement}_{plot_type}_plot.png",
        )
        plt.savefig(plot_path)
        plt.close(fig)


def plot_original_data(salinity, depths, filename, plot_path):
    """
    Generates a scatter plot of original salinity vs. depth.

    Parameters
    ----------
    salinity : array-like
        The salinity data to plot.
    depths : array-like
        The corresponding depth data.
    filename : str
        The name of the file being plotted.
    plot_path : str
        The path to save the plot image.

    Returns
    -------
    tuple
        The x and y limits of the plot.

    Notes
    -----
    - Creates a scatter plot with salinity on the x-axis and depth on the y-axis (inverted).
    - Adds a title, labels, and grid to the plot.
    - Saves the plot as an image file and returns the plot limits.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(salinity, -depths, alpha=0.6)
    plt.xlabel("Salinity (PSU)")
    plt.ylabel("Depth (m)")
    plt.title(f"Original Salinity vs. Depth - {filename}")
    plt.grid(True)
    plt.savefig(plot_path)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.close()
    return xlim, ylim


def plot_predicted_data(salinity, depths, xlim, ylim, filename, plot_path):
    """
    Generates a scatter plot of predicted salinity vs. depth.

    Parameters
    ----------
    salinity : array-like
        The predicted salinity data to plot.
    depths : array-like
        The corresponding depth data.
    xlim : tuple
        The x-axis limits from the original data plot.
    ylim : tuple
        The y-axis limits from the original data plot.
    filename : str
        The name of the file being plotted.
    plot_path : str
        The path to save the plot image.

    Notes
    -----
    - Creates a scatter plot with predicted salinity on the x-axis and depth on the y-axis (inverted).
    - Adds a title, labels, and grid to the plot.
    - Uses the same axis limits as the original data plot for comparison.
    - Saves the plot as an image file.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(salinity, -depths, alpha=0.6, color="red")
    title = f"Predicted Salinity vs. Depth - {filename}"
    plt.title(title)
    plt.xlabel("Salinity (PSU)")
    plt.ylabel("Depth (m)")
    plt.grid(True)
    plt.xlim((xlim[0], xlim[1]))
    plt.ylim((ylim[0], ylim[1]))
    plt.savefig(plot_path)
    plt.close()
