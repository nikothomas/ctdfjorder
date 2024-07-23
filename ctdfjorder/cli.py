import logging
import shutil
from signal import signal, SIGTERM, SIGINT, SIGTSTP
from sys import executable, exit
import sys
from warnings import catch_warnings
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import polars as pl
from polars.exceptions import ChronoFormatWarning
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table, box
from rich.status import Status
from rich import print as richprint
from ctdfjorder.ctdplot import plot_depth_vs, plot_map, plot_secchi_chla
from ctdfjorder.CTDExceptions.CTDExceptions import CTDError
from ctdfjorder.CTD import CTD
from ctdfjorder.utils import save_to_csv
from ctdfjorder.constants import *
from ctdfjorder.Mastersheet import Mastersheet
import rich_argparse
from os import path, listdir, remove, mkdir, getcwd
from os.path import isfile
console = Console()


def process_ctd_file(
    file, plot: bool = False, cached_master_sheet: Mastersheet = None, master_sheet_path: str = None, verbosity: int = 0, plots_folder: str = None
):
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
        ("Plot", lambda data: plot_data(data)),
        ("Exit", lambda data: data.get_df()),
    ]

    def plot_data(my_data):
        plot_depth_vs(
            my_data.get_df(), POTENTIAL_DENSITY_LABEL, plot_folder=plots_folder
        )
        plot_depth_vs(my_data.get_df(), SALINITY_LABEL, plot_folder=plots_folder)

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
                if (
                    len(warning_list) > warning_list_length
                    and warning_list[-1].category != RuntimeWarning
                    and warning_list[-1].category != ChronoFormatWarning
                ):
                    warning_list_length = len(warning_list)
                    logger.warning(warning_list[-1].message)
                    status.append("yellow")
                else:
                    status.append("green")
            except (CTDError) as error:
                logger.error(error)
                status.extend(["red"] * (len(steps) - len(status)))
                return None, status
            except(Exception) as e:
                print(e)
                logger.exception(e)
                status.extend(["red"] * (len(steps) - len(status)))
                return None, status

def generate_status_table(status_table):
    steps = [
        "Load File",
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
    plot=False,
    master_sheet_path=None,
    max_workers=1,
    verbosity=0,
    output_file=None,
    debug_run=False,
    table_show=False,
    mapbox_access_token=None,
):
    plots_folder = path.join(get_cwd(), "ctdplots")
    files = get_ctd_filenames_in_dir(get_cwd(), [".rsk", ".csv"])[
        : 20 if debug_run else None
    ]
    if master_sheet_path in files:
        files.remove(master_sheet_path)
    if not files:
        raise CTDError(message="No '.rsk' or '.csv' found in this folder")
    cached_master_sheet = (
        Mastersheet(master_sheet_path) if master_sheet_path else None
    )
    status_table, results = [], []
    live_console = console if table_show else Console(quiet=True)
    live = Live(
        generate_status_table(status_table),
        auto_refresh=False,
        console=live_console,
        vertical_overflow="visible",
    )
    executor = ProcessPoolExecutor(max_workers=max_workers)
    status_spinner_combining = Status("Combining CTD profiles", spinner="earth")
    status_spinner_cleaning_up = Status("Cleaning up", spinner="earth")
    status_spinner_map_view = Status(
        "Running interactive map view. To shutdown press CTRL+Z.", spinner="earth"
    )
    error_count = 0
    total_count = len(files)
    with ExitStack() as stack:
        stack.enter_context(live)
        try:
            stack.enter_context(executor)
            futures = {
                executor.submit(
                    process_ctd_file,
                    file,
                    plot,
                    cached_master_sheet,
                    master_sheet_path,
                    verbosity,
                    plots_folder,
                ): file
                for file in files
            }

            for future in as_completed(futures):
                file = futures[future]
                result, status = future.result()
                if isinstance(result, pl.DataFrame):
                    results.append(result)
                else:
                    error_count += 1
                if table_show:
                    status_table.append((path.basename(file), status))
                    live.update(generate_status_table(status_table), refresh=True)

        except KeyboardInterrupt:
            live.stop()
            with Status(
                "Shutdown message received, terminating open profile pipelines",
                spinner_style="red",
            ) as status_spinner:
                status_spinner.start()
                executor.shutdown(wait=True, cancel_futures=True)
                status_spinner.stop()

        finally:
            live.stop()
            with console.screen():
                status_spinner_cleaning_up.start()
                executor.shutdown(wait=True, cancel_futures=True)
                status_spinner_cleaning_up.stop()
                status_spinner_combining.start()
                df = pl.concat(results, how="diagonal")
                panel = Panel(
                    Pretty(df.select("pressure", "salinity", "temperature").describe()),
                    title="Overall stats",
                    subtitle=f"Errors/Total: {error_count}/{total_count}",
                )
                pl.Config.set_tbl_rows(-1)
                df_test = df.unique("filename", keep="first").select(
                    pl.col("filename"), pl.col("unique_id")
                )
                df_test.filter(~pl.col("unique_id").is_unique()).write_csv(
                    path.join(get_cwd(), "ISUNIQUE.csv")
                )
                richprint(panel)
                df_exported = save_to_csv(df, output_file)
                status_spinner_combining.stop()
                if plot:
                    if CHLOROPHYLL_LABEL in df.collect_schema():
                        plot_secchi_chla(df, plots_folder)
                    status_spinner_map_view.start()
                    if mapbox_access_token:
                        plot_map(df_exported, mapbox_access_token)
                        status_spinner_map_view.stop()


def get_ctd_filenames_in_dir(directory, types):
    return [
        path.join(directory, f)
        for f in listdir(directory)
        if any(f.endswith(t) for t in types)
    ]


def get_cwd():
    return path.dirname(executable) if getattr(sys, "frozen", False) else getcwd()


def _reset_file_environment():
    cwd = get_cwd()
    for filename in ["output.csv", "ctdfjorder.log"]:
        file_path = path.join(cwd, filename)
        if isfile(file_path):
            remove(file_path)
    if path.isdir("ctdplots"):
        shutil.rmtree("ctdplots")
    mkdir("ctdplots")


def setup_logging(verbosity):
    level = max(30 - (verbosity * 10), 10)
    signal(SIGTERM, handler)
    signal(SIGINT, handler)
    signal(SIGTSTP, handler)
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
    file_log = logging.FileHandler(path.join(get_cwd(), "ctdfjorder.log"))
    logging.basicConfig(level=level)
    file_log.setLevel(level)
    logger.addHandler(file_log)
    return logger


def handler(signal_received, frame):
    if signal_received == SIGINT:
        return
    raise KeyboardInterrupt


def cli():
    global console
    console.print(Panel("CTDFjorder", title="CTDFjorder CLI"))

    parser = ArgumentParser(
        description="Default Pipeline", formatter_class=rich_argparse.RichHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_default = subparsers.add_parser(
        "default",
        help="Run the default processing pipeline",
        formatter_class=rich_argparse.RichHelpFormatter,
    )
    parser_default.add_argument(
        "-p", "--plot", action="store_true", help="Generate plots"
    )
    parser_default.add_argument(
        "-v",
        "--verbose",
        action="count",
        dest="verbosity",
        default=0,
        help="Verbose logger output to ctdfjorder.log (repeat for increased verbosity)",
    )
    parser_default.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        const=-1,
        default=0,
        dest="verbosity",
        help="Quiet output (show errors only)",
    )
    parser_default.add_argument(
        "-r", "--reset", action="store_true", help="Reset file environment"
    )
    parser_default.add_argument(
        "-t", "--show-table", action="store_true", help="Show live progress table"
    )
    parser_default.add_argument(
        "-d", "--debug-run", action="store_true", help="Run 20 files for testing"
    )
    parser_default.add_argument(
        "-m", "--mastersheet", type=str, help="Path to mastersheet"
    )
    parser_default.add_argument(
        "-w", "--workers", type=int, nargs="?", const=1, help="Max workers"
    )
    parser_default.add_argument(
        "--token",
        type=str,
        default=None,
        help="MapBox token to enable interactive map plot",
    )
    parser_default.add_argument(
        "-o", "--output", type=str, default="output.csv", help="Output file path"
    )

    args = parser.parse_args()

    if args.command == "default":
        if args.reset:
            _reset_file_environment()

        table = Table(title="Processing Pipeline Config")
        table.add_column("Argument", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        for arg in vars(args):
            table.add_row(arg.replace("_", " ").title(), str(getattr(args, arg)))
        console.print(table)

        run_default(
            plot=args.plot,
            master_sheet_path=args.mastersheet,
            max_workers=args.workers,
            verbosity=args.verbosity,
            output_file=args.output,
            debug_run=args.debug_run,
            table_show=args.show_table,
            mapbox_access_token=args.token,
        )
        exit()