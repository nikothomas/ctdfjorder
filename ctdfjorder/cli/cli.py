import logging
import shutil
import sys
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from os import path, listdir, remove, mkdir, getcwd
from warnings import catch_warnings
import signal

import polars as pl
import psutil
from polars.exceptions import ChronoFormatWarning
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table, box
from rich.status import Status
from rich.prompt import Confirm
from rich import print as richprint
from rich_argparse import RichHelpFormatter

from ctdfjorder.visualize import ctd_plot
from ctdfjorder.exceptions.exceptions import CTDError, Critical
from ctdfjorder.CTD import CTD
from ctdfjorder.utils.utils import save_to_csv
from ctdfjorder.constants.constants import *
from ctdfjorder.metadata.master_sheet import MasterSheet

console = Console(color_system="windows")


def process_ctd_file(
    file, plot, cached_master_sheet, master_sheet_path, verbosity, plots_folder, filters
):
    """
    Processes a CTD file through a series of data cleaning and analysis steps.

    Parameters
    ----------
    file : str
        The path to the CTD file.
    plot : bool
        Flag indicating whether to generate plots.
    cached_master_sheet : MasterSheet
        A cached instance of the master sheet for cross-checking site names.
    master_sheet_path : str
        The path to the master sheet.
    verbosity : int
        The verbosity level for logging.
    plots_folder : str
        The folder to save plots in.
    filters : list
        List of filters to apply to the data.

    Returns
    -------
    pl.DataFrame | None
        The processed data as a Polars DataFrame or None if an error occurred.
    list[str]
        The status of each processing step, represented by color codes ("green", "yellow", "red").

    Notes
    -----
    The function goes through several steps to process the CTD file, including filtering,
    removing upcasts and non-positive samples, cleaning salinity data, adding surface measurements,
    absolute salinity, density, potential density, mixed layer depth (MLD), and Brunt-Väisälä frequency squared (BF Squared).
    If plotting is enabled, it generates plots for potential density and salinity versus depth.
    """
    logger = setup_logging(verbosity)
    steps = [
        (
            "Load File",
            lambda: CTD(
                file,
                plot=plot,
                cached_master_sheet=cached_master_sheet,
                master_sheet_path=master_sheet_path,
            ),
        ),
        ("Filter", lambda data: data.filter_columns_by_range(filters=filters)),
        ("Remove Upcasts", lambda data: data.remove_upcasts()),
        ("Remove Negative", lambda data: data.remove_non_positive_samples()),
        (
            "Remove Invalid Salinity Values",
            lambda data: data.remove_invalid_salinity_values(),
        ),
        ("Clean Salinity AI", lambda data: data.clean("clean_salinity_ai")),
        (
            "Add Surface Measurements",
            lambda data: data.add_surface_salinity_temp_meltwater(),
        ),
        ("Add Absolute Salinity", lambda data: data.add_absolute_salinity()),
        ("Add Density", lambda data: data.add_density()),
        ("Add Potential Density", lambda data: data.add_potential_density()),
        ("Add MLD", lambda data: data.add_mld(20, "potential_density_avg")),
        ("Add BF Squared", lambda data: data.add_bf_squared()),
        ("Plot", lambda data: plot_data(data.get_df(), plots_folder)),
        ("Exit", lambda data: data.get_df()),
    ]

    status = []
    data = None
    with catch_warnings(record=True, action="once") as warning_list:
        warning_list_length = len(warning_list)
        for step_name, step_function in steps:
            try:
                if step_name == "Load File":
                    data = step_function()
                elif step_name == "Exit":
                    return step_function(data), status + ["green"]
                else:
                    step_function(data)
                status.append(
                    "yellow" if len(warning_list) > warning_list_length
                    and warning_list[-1].category is not ChronoFormatWarning else "green"
                )
                warning_list_length = len(warning_list)
            except CTDError as error:
                logger.error(error)
                status.extend(["red"] * (len(steps) - len(status)))
                return None, status
            except Exception as e:
                print(e)
                logger.exception(e)
                status.extend(["red"] * (len(steps) - len(status)))
                return None, status
    return None, status


def plot_data(my_data, plots_folder):
    """
    Generates plots for the given CTD data.

    Parameters
    ----------
    my_data : pl.DataFrame
        The CTD data to plot.
    plots_folder : str
        The folder to save plots in.

    Notes
    -----
    This function generates two types of plots: depth vs potential density and depth vs salinity.
    The plots are saved in the specified folder.
    """
    ctd_plot.plot_depth_vs(my_data, POTENTIAL_DENSITY_LABEL, plot_folder=plots_folder)
    ctd_plot.plot_depth_vs(my_data, SALINITY_LABEL, plot_folder=plots_folder)


def generate_status_table(status_table):
    """
    Generates a status table displaying the processing status for each file.

    Parameters
    ----------
    status_table : list[tuple]
        A list of tuples containing the file name and its processing status.

    Returns
    -------
    rich.table.Table
        A rich Table object displaying the status of each file.

    Notes
    -----
    The table includes columns for each processing step and uses color codes to indicate the status:
    green for success, yellow for warnings, and red for errors.
    """
    steps = [
        "Load File",
        "Filter",
        "Remove Upcasts",
        "Remove Negative Values",
        "Remove Invalid Salinity Values",
        "Clean Salinity Data",
        "Add Surface Measurements",
        "Add Absolute Salinity",
        "Add Density",
        "Add Potential Density",
        "Add MLD",
        "Add BF Squared",
        "Plot",
        "Exit",
    ]
    table = Table(box=box.SQUARE)
    table.add_column("File", width=30)
    for step in steps:
        table.add_column(step)
    for file, status in status_table:
        table.add_row(file, *[f"[{color}]•[/{color}]" for color in status])
    return table


def run_default(
    plot,
    master_sheet_path,
    max_workers,
    verbosity,
    output_file,
    debug_run,
    status_show,
    mapbox_access_token,
    filters,
):
    """
    Runs the default processing pipeline for CTD files.

    Parameters
    ----------
    plot : bool
        Flag indicating whether to generate plots.
    master_sheet_path : str
        The path to the master sheet.
    max_workers : int
        The maximum number of worker processes to use.
    verbosity : int
        The verbosity level for logging.
    output_file : str
        The path to the output CSV file.
    debug_run : bool
        Flag indicating whether to run in debug mode (processes only 20 files).
    status_show : bool
        Flag indicating whether to show the processing status.
    mapbox_access_token : str
        The Mapbox access token for generating interactive maps.
    filters : list or None
        List of filters to apply to the data.

    Notes
    -----
    This function orchestrates the entire CTD file processing pipeline. It sets up logging,
    processes the master sheet, and uses a process pool to handle multiple CTD files concurrently.
    The processing status of each file is tracked, and results are compiled and saved to a CSV file.
    If plotting is enabled, it generates plots and an interactive map.
    """
    logger = setup_logging(verbosity)
    status_table, results = [], []
    cached_master_sheet = None
    if not status_show:
        Console(quiet=True)

    plots_folder = path.join(get_cwd(), "ctdplots")
    files = get_ctd_filenames_in_dir(get_cwd(), [".rsk", ".csv"])[
        : 20 if debug_run else None
    ]
    total_files = len(files)
    remaining_files = total_files

    if master_sheet_path in files:
        total_files -= 1
        files.remove(master_sheet_path)

    if not files:
        raise CTDError(message="No '.rsk' or '.csv' found in this folder", filename="")

    with Status(
        f"Processing master sheet, this might take awhile",
        spinner="earth",
        console=console,
    ) as status_master_sheet:
        try:
            status_master_sheet.start()
            cached_master_sheet = (
                MasterSheet(master_sheet_path, with_crosschecked_site_names=True)
                if master_sheet_path
                else None
            )
        except Critical as e:
            status_master_sheet.stop()
            console.print(e, style="white on red")
            continue_no_mastersheet = Confirm.ask("Continue without mastersheet?")
            if not continue_no_mastersheet:
                sys.exit()


    with ExitStack() as stack:
        executor = stack.enter_context(ProcessPoolExecutor(max_workers=max_workers))
        status_spinner_processing = stack.enter_context(
            Status(
                f"Processing {total_files} files. Press CTRL+Z to shutdown.",
                spinner="earth",
                console=console,
            )
        )

        try:
            futures = {
                executor.submit(
                    process_ctd_file,
                    file,
                    plot,
                    cached_master_sheet,
                    master_sheet_path,
                    verbosity,
                    plots_folder,
                    filters,
                ): file
                for file in files
            }
            for future in as_completed(futures):
                file = futures[future]
                result, status = future.result()
                if isinstance(result, pl.DataFrame):
                    results.append(result)
                    remaining_files -= 1
                else:
                    remaining_files -= 1
                if status_show:
                    status_table.append((path.basename(file), status))
                    status_spinner_processing.update(
                        status=f"Processing {remaining_files} files"
                    )

        except KeyboardInterrupt:
            status_spinner_processing.stop()
            with Status(
                "Shutdown message received, terminating open profile pipelines",
                spinner_style="red",
            ) as status_spinner:
                status_spinner.start()
                executor.shutdown(wait=True, cancel_futures=True)
                status_spinner.stop()
                for proc in psutil.process_iter():
                    if proc.name == "Python":
                        proc.kill()
        finally:
            stack.close()
            if status_show:
                console.print(generate_status_table(status_table))
            process_results(
                results,
                total_files,
                output_file,
                plot,
                plots_folder,
                mapbox_access_token,
            )


def process_results(
    results, total_files, output_file, plot, plots_folder, mapbox_access_token
):
    """
    Processes the results of the CTD file processing pipeline.

    Parameters
    ----------
    results : list[pl.DataFrame]
        The list of processed CTD data.
    total_files : int
        The total number of files processed.
    output_file : str
        The path to the output CSV file.
    plot : bool
        Flag indicating whether to generate plots.
    plots_folder : str
        The folder to save plots in.
    mapbox_access_token : str
        The Mapbox access token for generating interactive maps.

    Notes
    -----
    This function combines the processed data from all files, generates a summary of the data,
    and saves the combined data to a CSV file. If plotting is enabled, it generates various plots
    and an interactive map using Mapbox.
    """
    with console.screen():
        with Status(
            "Combining CTD profiles", spinner="earth", console=console
        ) as status_spinner_combining:
            df = pl.concat(results, how="diagonal")
            panel = Panel(
                Pretty(df.select("pressure", "salinity", "temperature").describe()),
                title="Overall stats",
                subtitle=f"Errors/Total: {total_files - len(results)}/{total_files}",
            )
            pl.Config.set_tbl_rows(-1)
            df_test = df.unique("filename", keep="first").select(
                pl.col("filename"), pl.col("unique_id")
            )
            df_test.filter(~pl.col("unique_id").is_unique()).write_csv(
                path.join(get_cwd(), "ISUNIQUE.csv")
            )
            richprint(panel)
            df = save_to_csv(df, output_file, None)

        if plot:
            plot_results(df, mapbox_access_token)


def plot_results(df, mapbox_access_token):
    """
    Generates plots for the processed CTD data and an interactive map view.

    Parameters
    ----------
    df : pl.DataFrame
        The processed CTD data.
    mapbox_access_token : str
        The Mapbox access token for generating interactive maps.

    Notes
    -----
    This function generates an interactive map to visualize the data.
    """
    with Status(
        "Running interactive map view. To shutdown press CTRL+Z.",
        spinner="earth",
        console=console,
    ) as status_spinner_map_view:
        if mapbox_access_token:
            try:
                ctd_plot.plot_map(df, mapbox_access_token)
            except KeyboardInterrupt:
                for proc in psutil.process_iter():
                    if proc.name == "Python":
                        proc.kill()


def get_ctd_filenames_in_dir(directory, types):
    """
    Gets the filenames of CTD files in a directory.

    Parameters
    ----------
    directory : str
        The directory to search for CTD files.
    types : list[str]
        A list of file extensions to search for.

    Returns
    -------
    list[str]
        A list of CTD file paths in the directory.

    Notes
    -----
    This function searches the specified directory for files with the given extensions (e.g., .rsk, .csv)
    and returns a list of matching file paths.
    """
    return [
        path.join(directory, f)
        for f in listdir(directory)
        if any(f.endswith(t) for t in types)
    ]


def get_cwd():
    """
    Gets the current working directory.

    Returns
    -------
    str
        The current working directory.

    Notes
    -----
    If the script is running in a frozen state (e.g., packaged with PyInstaller), it returns the directory
    of the executable. Otherwise, it returns the standard current working directory.
    """
    return path.dirname(sys.executable) if getattr(sys, "frozen", False) else getcwd()


def reset_file_environment():
    """
    Resets the file environment by removing existing output files and creating necessary directories.

    Notes
    -----
    This function removes old output files and directories to ensure a clean environment for running
    the CTD file processing pipeline. It creates a new directory for plots.
    """
    cwd = get_cwd()
    for filename in ["output.csv", "ctdfjorder.log"]:
        file_path = path.join(cwd, filename)
        if path.isfile(file_path):
            remove(file_path)
    if path.isdir(path.join(get_cwd(), "ctdplots")):
        shutil.rmtree(path.join(get_cwd(), "ctdplots"))
    mkdir(path.join(get_cwd(), "ctdplots"))


def setup_logging(verbosity):
    """
    Sets up logging for the application.

    Parameters
    ----------
    verbosity : int
        The verbosity level for logging.

    Returns
    -------
    logging.Logger
        The configured logger.

    Notes
    -----
    This function configures the logging settings for the application, including setting the log level
    based on verbosity, formatting log messages, and handling signals for graceful termination.
    """
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    level = max(30 - (verbosity * 10), 10)
    for logger_name in [
        "tensorflow",
        "matplotlib",
        "sklearn",
        "werkzeug",
        "dash",
        "flask",
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    logger = logging.getLogger("ctdfjorder")
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", "%H:%M:%S"
    )
    file_log = logging.FileHandler(path.join(get_cwd(), "ctdfjorder.log"))
    logging.basicConfig(level=level)
    file_log.setFormatter(formatter)
    file_log.setLevel(level)
    logger.addHandler(file_log)
    return logger


def signal_handler(signal_received, frame):
    r"""
    Handles system signals to gracefully terminate the application.

    Parameters
    ----------
    signal_received : Any
        The received signal.
    frame : frame
        The current stack frame.

    Notes
    -----
    This function handles system signals such as SIGINT and SIGTERM to allow for graceful
    termination of the application, cleaning up resources as needed.
    """
    if signal_received == signal.SIGINT:
        return
    raise KeyboardInterrupt


def build_parser():
    """
    Builds the argument parser for the command-line interface.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.

    Notes
    -----
    This function sets up the argument parser for the command-line interface, defining the available
    commands and their respective options. It uses RichHelpFormatterPlus for enhanced help text formatting.
    """
    parser = ArgumentParser(
        description="CTDFjorder", formatter_class=RichHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser_default = subparsers.add_parser(
        "default",
        help="Run the default processing pipeline",
        formatter_class=RichHelpFormatter,
    )
    parser_fjord = subparsers.add_parser(
        "fjord",
        help="Run the Fjord Phyto processing pipeline",
    )
    parser_fjord.add_argument(
        "-t",
        "--token",
        type=str,
        default=None,
        help="MapBox token to enable interactive map plot",
    )
    add_arguments(parser_default)
    return parser


def build_parser_docs():
    """
    Builds the argument parser for the documentation.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.

    Notes
    -----
    This function sets up the argument parser specifically for generating documentation, defining
    the available commands and their respective options without the enhanced formatting.
    """
    parser = ArgumentParser(
        description="CTDFjorder"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser_default = subparsers.add_parser(
        "default",
        help="Run the default processing pipeline",
    )
    parser_fjord = subparsers.add_parser(
        "fjord",
        help="Run the Fjord Phyto processing pipeline",
    )
    parser_fjord.add_argument(
        "-t",
        "--token",
        type=str,
        default=None,
        help="MapBox token to enable interactive map plot",
    )
    add_arguments(parser_default)
    return parser

def add_arguments(parser):
    """
    Adds arguments to the argument parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to add arguments to.

    Notes
    -----
    This function adds various command-line arguments to the parser, including options for plotting,
    verbosity, resetting the file environment, showing processing status, running in debug mode,
    specifying the master sheet path, setting the number of worker processes, providing a Mapbox token,
    and defining filters for data columns.
    """
    parser.add_argument("-p", "--plot", action="store_true", help="Generate plots")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        dest="verbosity",
        default=0,
        help="Verbose logger output to ctdfjorder.log (repeat for increased verbosity)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        const=-1,
        default=0,
        dest="verbosity",
        help="Quiet output (show errors only)",
    )
    parser.add_argument(
        "-r", "--reset", action="store_true", help="Reset file environment"
    )
    parser.add_argument(
        "-s",
        "--show-status",
        action="store_true",
        help="Show processing status and pipeline status",
    )
    parser.add_argument(
        "-d", "--debug-run", action="store_true", help="Run 20 files for testing"
    )
    parser.add_argument("-m", "--mastersheet", type=str, help="Path to mastersheet")
    parser.add_argument(
        "-w", "--workers", type=int, nargs="?", const=1, help="Max workers"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="MapBox token to enable interactive map plot",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=str(DEFAULT_OUTPUT_FILE), help="Output file path"
    )
    parser.add_argument(
        "--filtercolumns",
        nargs="*",
        type=str,
        required=False,
        default=None,
        help="List of columns to filter",
        choices=LIST_LABELS,
    )
    parser.add_argument(
        "--filterupper",
        nargs="*",
        type=float,
        required=False,
        default=None,
        help="Upper bounds for the filtered columns",
    )
    parser.add_argument(
        "--filterlower",
        nargs="*",
        type=float,
        required=False,
        default=None,
        help="Lower bounds for the filtered columns",
    )


def main():
    """
    The main entry point for the application. Parses arguments and runs the appropriate processing pipeline.

    Notes
    -----
    This function initializes signal handlers, builds the argument parser, and parses the command-line arguments.
    Depending on the specified command, it runs the default or Fjord Phyto processing pipeline.
    """
    for sig in [signal.SIGINT, signal.SIGTSTP, signal.SIGTERM]:
        signal.signal(sig, signal_handler)
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "default":
        filters = create_filters(args)
        if args.reset:
            reset_file_environment()
        display_config(args)
        run_default(
            plot=args.plot,
            master_sheet_path=args.mastersheet,
            max_workers=args.workers,
            verbosity=args.verbosity,
            output_file=args.output,
            debug_run=args.debug_run,
            status_show=args.show_status,
            mapbox_access_token=args.token,
            filters=filters,
        )
        sys.exit()
    if args.command == "fjord":
        reset_file_environment()
        display_config(args)
        run_default(
            plot=True,
            master_sheet_path="mastersheet.csv",
            max_workers=4,
            verbosity=3,
            output_file=DEFAULT_OUTPUT_FILE,
            debug_run=False,
            status_show=True,
            mapbox_access_token=args.token,
            filters=None,
        )


def create_filters(args):
    """
    Creates filters from command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command-line arguments.

    Returns
    -------
    list[tuple] | None
        A list of filters or None if no filters are specified.

    Notes
    -----
    This function generates a list of filters based on the specified columns, upper bounds, and lower bounds
    provided via command-line arguments. If not all required filter arguments are provided, it returns None.
    """
    if all(
        arg is not None
        for arg in [args.filtercolumns, args.filterupper, args.filterlower]
    ):
        return zip(args.filtercolumns, args.filterupper, args.filterlower)
    return None


def display_config(args):
    """
    Displays the configuration of the processing pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command-line arguments.

    Notes
    -----
    This function generates a table displaying the configuration options specified via command-line arguments,
    providing a clear overview of the pipeline settings before execution.
    """
    table = Table(title="Processing Pipeline Config")
    table.add_column("Argument", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for arg in vars(args):
        table.add_row(arg.replace("_", " ").title(), str(getattr(args, arg)))
    console.print(table)


if __name__ == "__main__":
    main()
