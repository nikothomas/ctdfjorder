import logging
import polars as pl
from flask import Flask
from matplotlib import pyplot as plt
import numpy as np
import os
from ctdfjorder.constants import *
from statsmodels.api import nonparametric
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, dash_table


def plot_map(df: pl.DataFrame, mapbox_access_token):
    px.set_mapbox_access_token(mapbox_access_token)
    df_pd = df.to_pandas()

    # Create the Dash app with external JavaScript to handle window close
    server = Flask(__name__)
    app = dash.Dash(__name__, server=server)
    app.logger.setLevel(logging.ERROR)
    server.logger.setLevel(logging.ERROR)

    # Create the map plot using plotly.express.scatter_mapbox
    fig = px.scatter_mapbox(
        df_pd,
        lat="latitude",
        lon="longitude",
        hover_name="Unique_ID",
        hover_data={
            "timestamp": True,
            "filename": True,
            "latitude": True,  # Lat and lon are not included in hover as they are already displayed
            "longitude": True,
        },
        mapbox_style="dark",
        zoom=5,
        center={"lat": -62.0, "lon": -60.0},
    )

    # Update layout with Mapbox access token
    fig.update_layout(
        mapbox_accesstoken=mapbox_access_token, margin=dict(l=0, r=0, t=0, b=0)
    )

    # App layout with CSS for dark theme and stacking map over table
    app.layout = html.Div(
        [
            html.Div(
                dcc.Graph(id="map", figure=fig),
                style={
                    "flex": "1",
                    "height": "calc(100vh - 50vh)",
                    "min-height": "50vh",
                },  # Dynamic height map
            ),
            html.Div(
                id="table-container",
                style={
                    "flex": "1",
                    "overflowY": "auto",
                    "overflowX": "auto",
                    "height": "50vh",
                },
                # Scrollable table container
            ),
            dcc.Interval(id="interval-component", interval=1 * 1000, n_intervals=0),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "height": "100vh",
            "backgroundColor": "#1e1e1e",
            "color": "#ffffff",
        },
    )

    # Callback to update table based on clicked point
    @app.callback(Output("table-container", "children"), Input("map", "clickData"))
    def display_profile_data(clickData):
        if clickData is None:
            return html.P(
                "Click on a point to see profile data.", style={"color": "#ffffff"}
            )

        unique_id = clickData["points"][0][
            "hovertext"
        ]  # Get the unique_id from hovertext
        filtered_df = df_pd[df_pd["Unique_ID"] == unique_id]

        return dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in filtered_df.columns],
            data=filtered_df.to_dict("records"),
            style_table={"overflowX": "auto", "overflowY": "auto"},
            style_header={"backgroundColor": "#333333", "color": "#ffffff"},
            style_cell={
                "backgroundColor": "#1e1e1e",
                "color": "#ffffff",
                "whiteSpace": "normal",
            },
        )

    # Open the browser automatically
    import webbrowser
    from threading import Timer

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8050/")

    Timer(1, open_browser).start()
    app.run(debug=False)


def plot_depth_vs(
    df: pl.DataFrame, measurement: str, plot_folder: str, plot_type: str = "scatter"
):
    """
    Generates a plot of depth vs. specified measurement (salinity, density, temperature).

    Parameters
    ----------
    df : pl.Dataframe
        CTD dataframe
    measurement : str
        Options are self.SALINITY_LABEL, self.DENSITY_LABEL, 'potential_density, or self.TEMPERATURE_LABEL.
    plot_folder : str
        Full path to plots folder.
    plot_type : str
        Options are 'line' or 'scatter', defaults to 'scatter'.

    Notes
    -----
    - Adds horizontal lines indicating the mixed layer depth (MLD) if present.
    - Allows for both scatter and line plot types.
    - Saves the plot as an image file.

    """
    plt.rcParams.update({"font.size": 16})
    os.makedirs(plot_folder, exist_ok=True)
    for profile_id in (
        df.select(PROFILE_ID_LABEL).unique(keep="first").to_series().to_list()
    ):
        profile = df.filter(pl.col(PROFILE_ID_LABEL) == profile_id)
        filename = profile.select(pl.first(FILENAME_LABEL)).item()
        fig, ax1 = plt.subplots(figsize=(18, 18))
        ax1.invert_yaxis()
        ax1.set_ylim([profile.select(pl.col(DEPTH_LABEL)).max().item(), 0])
        color_map = {
            SALINITY_LABEL: "tab:blue",
            DENSITY_LABEL: "tab:red",
            POTENTIAL_DENSITY_LABEL: "tab:red",
            TEMPERATURE_LABEL: "tab:blue",
        }
        label_map = {
            SALINITY_LABEL: "Practical Salinity (PSU)",
            DENSITY_LABEL: "Density (kg/m^3)",
            POTENTIAL_DENSITY_LABEL: "Potential Density (kg/m^3)",
            TEMPERATURE_LABEL: "Temperature (°C)",
        }
        if plot_type == "line":
            lowess = nonparametric.lowess
            y, x = zip(
                *lowess(
                    profile.select(pl.col(f"{measurement}")).to_numpy(),
                    profile.select(pl.col(DEPTH_LABEL)).to_numpy(),
                    frac=0.1,
                )
            )
        else:
            x, y = (
                profile.select(pl.col(f"{measurement}")).to_numpy(),
                profile.select(pl.col(DEPTH_LABEL)).to_numpy(),
            )
        (
            ax1.plot(x, y, color=color_map[measurement], label=label_map[measurement])
            if plot_type == "line"
            else ax1.scatter(
                x, y, color=color_map[measurement], label=label_map[measurement]
            )
        )
        ax1.set_xlabel(label_map[measurement], color=color_map[measurement])
        ax1.set_ylabel("Depth (m)")
        ax1.tick_params(axis="x", labelcolor=color_map[measurement])
        mld = profile.select(pl.col(r"^.*MLD.*$").first()).item()
        if mld is not None:
            # Plot MLD line
            ax1.axhline(
                y=mld, color="green", linestyle="--", linewidth=2, label=f"{mld}"
            )
            ax1.text(
                0.95,
                mld,
                f"{mld}",
                va="center",
                ha="right",
                backgroundcolor="white",
                color="green",
                transform=ax1.get_yaxis_transform(),
            )
        plt.title(
            f"{filename} \n Profile {profile_id} \n Depth vs. {label_map[measurement]}\n MLD {mld}"
        )
        ax1.grid(True)
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
        plot_path = os.path.join(
            plot_folder,
            f"{filename}_{profile_id}_depth_{measurement}_{plot_type}_plot.png",
        )
        plt.savefig(plot_path)
        plt.close(fig)


def plot_secchi_chla(df: pl.DataFrame, plot_folder: str):
    os.makedirs(plot_folder, exist_ok=True)
    df = df.filter(
        pl.col("secchi_depth").is_not_null(), pl.col("chlorophyll").is_not_null()
    )
    data_secchi_chla = df.group_by("profile_id", "filename", maintain_order=True).agg(
        pl.first("secchi_depth"), pl.max("chlorophyll")
    )
    secchi_depths = data_secchi_chla.select(pl.col("secchi_depth")).to_series()
    chlas = data_secchi_chla.select(pl.col("chlorophyll")).to_series()
    log_secchi_depth = np.log10(np.array(secchi_depths.to_numpy()))
    log_chla = np.log10(np.array(chlas.to_numpy()))
    # Plotting
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(log_secchi_depth, log_chla, color="b", label="Data Points")
    plt.title("Log10 of Secchi Depth vs Log10 of Chlorophyll-a")
    plt.xlabel("Log10 of Secchi Depth (m)")
    plt.ylabel("Log10 of Chlorophyll-a (µg/l)")
    plt.grid(True)
    plt.legend()
    fig.savefig(os.path.join(plot_folder, "secchi_depth_vs_chla.png"))
    plt.close(fig)


def plot_original_data(salinity, depths, filename, plot_path):
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
