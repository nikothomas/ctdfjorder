import polars as pl
from matplotlib import pyplot as plt
import numpy as np
import os
from ctdfjorder.constants import *
from statsmodels.api import nonparametric


def plot_depth_vs(df: pl.DataFrame, measurement: str, plot_folder: str, plot_type: str = "scatter"):
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
            df.select(PROFILE_ID_LABEL)
                    .unique(keep="first")
                    .to_series()
                    .to_list()
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
            ax1.plot(
                x, y, color=color_map[measurement], label=label_map[measurement]
            )
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