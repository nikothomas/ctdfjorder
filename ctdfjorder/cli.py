import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import argparse
import polars as pl
import signal
import psutil
import os
import enlighten
import numpy as np
from matplotlib import pyplot as plt
import importlib
from . import ctdfjorder
from . import loggersetup

def handler(signal_received, frame):
    if signal_received == signal.SIGINT:
        return
    else:
        raise KeyboardInterrupt


signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTSTP, handler)
manager = enlighten.get_manager()


def _process_ctd_file(
        file,
        plot=False,
        cached_master_sheet=None,
        master_sheet_path=None,
        verbosity=0,
        add_unique_id=False,
):
    logger = loggersetup.setup_logging(verbosity)
    try:
        my_data = ctdfjorder.CTD(
            file,
            plot=plot,
            cached_master_sheet=cached_master_sheet,
            master_sheet_path=master_sheet_path,
            add_unique_id=add_unique_id,
        )
        my_data.remove_upcasts()
        my_data.remove_non_positive_samples()
        my_data.remove_invalid_salinity_values()
        my_data.clean("clean_salinity_ai")
        my_data.add_surface_salinity_temp_meltwater()
        my_data.add_absolute_salinity()
        my_data.add_density()
        my_data.add_potential_density()
        my_data.add_mld(20, "potential_density_avg")
        my_data.add_bf_squared()
        if plot:
            my_data.plot("potential_density")
            my_data.plot("salinity")
        return my_data.get_df()
    except ctdfjorder.CTDError as e:
        logger.error(e)
    except Exception as e:
        logger.exception(e)


def _run_default(
        plot=False,
        master_sheet_path=None,
        max_workers=1,
        verbosity=0,
        output_file=None,
        add_unique_id=False,
        plot_secchi_chla_flag=False,
        debug_run=False
):
    df = None
    logger = loggersetup.setup_logging(verbosity)
    if debug_run:
        rsk_files = _get_rsk_filenames_in_dir(get_cwd())[:10]
        csv_files = _get_csv_filenames_in_dir(get_cwd())[:10]
    else:
        # Retrieve and slice the first 10 items of each list
        rsk_files = _get_rsk_filenames_in_dir(get_cwd())
        csv_files = _get_csv_filenames_in_dir(get_cwd())
    # Initialize the ctd_files_list and extend it with the sliced lists
    ctd_files_list = rsk_files
    ctd_files_list.extend(csv_files)
    if master_sheet_path:
        cached_master_sheet = ctdfjorder.CTD.Utility.load_master_sheet(master_sheet_path)
    else:
        cached_master_sheet = None
    total_files = len(ctd_files_list)
    bar_format = (
            "{desc}{desc_pad}{percentage:3.0f}%|{bar}| "
            + "S:{count_0:{len_total}d} "
            + "E:{count_1:{len_total}d} "
            + "[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]"
    )
    success = manager.counter(
        total=total_files,
        desc="Processing Files",
        unit="Files",
        color="green",
        bar_format=bar_format,
    )
    errors = success.add_subcounter("red")
    executor = ProcessPoolExecutor(max_workers=max_workers)
    results: list[pl.DataFrame] = []
    if not ctd_files_list:
        logger.debug("No files to process")
        return
    # Process the rest of the files in parallel
    with ExitStack() as stack:
        stack.enter_context(executor)
        stack.enter_context(success)
        try:
            futures = {
                executor.submit(
                    _process_ctd_file,
                    file,
                    plot=plot,
                    cached_master_sheet=cached_master_sheet,
                    master_sheet_path=master_sheet_path,
                    verbosity=verbosity,
                    add_unique_id=add_unique_id,
                ): file
                for file in ctd_files_list
            }
            for future in as_completed(futures):
                result = future.result()
                if type(result) is pl.DataFrame:
                    results.append(result)
                    success.update(1)
                else:
                    errors.update(1)
            df = pl.concat(results, how="diagonal")
            ctdfjorder.CTD.Utility.save_to_csv(df, output_file)
            if plot_secchi_chla_flag:
                plot_secchi_chla(df)
        except KeyboardInterrupt:
            loggersetup.setup_logging(0)
            print()
            logger.critical(
                "Received shutdown command from keyboard, flushing logs and killing child processes"
            )
            for proc in psutil.process_iter():
                if proc.name().startswith("python"):
                    proc.kill()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def plot_secchi_chla(df: pl.DataFrame):
    df = df.filter(pl.col("secchi_depth").is_not_null(),
                   pl.col("chlorophyll").is_not_null())
    data_secchi_chla = df.group_by(
        "unique_id", maintain_order=True
    ).agg(pl.first('secchi_depth'), pl.max("chlorophyll"))
    secchi_depths = data_secchi_chla.select(pl.col('secchi_depth')).to_series()
    chlas = data_secchi_chla.select(pl.col('secchi_depth')).to_series()
    # Calculate log10 of the values
    log_secchi_depth = np.array(secchi_depths.to_numpy())
    log_chla = np.array(chlas.to_numpy())

    # Plotting
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(log_secchi_depth, log_chla, color='b', label='Data Points')
    plt.plot(log_secchi_depth, log_chla, color='b')

    # Adding titles and labels
    plt.title('Log10 of Secchi Depth vs Log10 of Chlorophyll-a')
    plt.xlabel('Log10 of Secchi Depth (m)')
    plt.ylabel('Log10 of Chlorophyll-a (mg/m³)')
    plt.grid(True)
    plt.legend()
    fig.savefig(os.path.join(get_cwd(), 'plots', 'secchi_depth_vs_chla.png'))
    plt.close(fig)


def _get_rsk_filenames_in_dir(working_directory):
    rsk_files_list = []
    rsk_filenames_no_path = []
    for filename in os.listdir(working_directory):
        if filename.endswith(".rsk"):
            for filepath in rsk_files_list:
                filename_no_path = "_".join(
                    filepath.split("/")[-1].split("_")[0:3]
                ).split(".rsk")[0]
                if filename_no_path in rsk_filenames_no_path:
                    continue
                rsk_filenames_no_path.append(filename_no_path)
            file_path = os.path.join(working_directory, filename)
            rsk_files_list.append(file_path)
    return rsk_files_list


def _get_csv_filenames_in_dir(working_directory):
    csv_files_list = []
    rsk_filenames_no_path = []
    for filename in os.listdir(working_directory):
        if filename.endswith(".csv") and "output" not in filename:
            for filepath in csv_files_list:
                filename_no_path = "_".join(
                    filepath.split("/")[-1].split("_")[0:3]
                ).split(".csv")[0]
                if filename_no_path in rsk_filenames_no_path:
                    continue
                rsk_filenames_no_path.append(filename_no_path)
            file_path = os.path.join(working_directory, filename)
            csv_files_list.append(file_path)
    return csv_files_list


def get_cwd():
    # determine if application is a script file or frozen exe
    if getattr(sys, "frozen", False):
        working_directory_path = os.path.dirname(sys.executable)
    elif __file__:
        working_directory_path = os.getcwd()
    else:
        working_directory_path = os.getcwd()
    return working_directory_path


def _get_filename(filepath):
    return "_".join(filepath.split("/")[-1].split("_")[0:3]).split(".rsk")[0]


def _reset_file_environment():
    output_file_csv = "output.csv"
    output_file_csv_clean = "outputclean.csv"
    output_log = "ctdfjorder.log"
    cwd = get_cwd()
    output_file_csv = os.path.join(cwd, output_file_csv)
    output_file_csv_clean = os.path.join(cwd, output_file_csv_clean)
    if cwd is None:
        raise ctdfjorder.CTDError("Unknown", "Couldn't get working directory.")
    if os.path.isfile(output_file_csv):
        os.remove(output_file_csv)
    if os.path.isfile(output_file_csv_clean):
        os.remove(output_file_csv_clean)
    if os.path.isfile(output_log):
        os.remove(output_log)
    if os.path.isdir("../../plots"):
        shutil.rmtree("../../plots")
    if not os.path.isdir("../../plots"):
        os.mkdir("../../plots")


def main():
    parser = argparse.ArgumentParser(
        description="CTD Fjorder Processing Script",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for the 'default' command
    parser_default = subparsers.add_parser(
        "default", help="Run the default processing pipeline"
    )
    parser_default.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Generate plots during the default processing pipeline",
    )
    parser_default.add_argument(
        "--plot-secchi-chla",
        action="store_true",
        help="Generate plot for secchi depth vs chla",
    )
    parser_default.add_argument(
        "-v",
        "--verbose",
        action="count",
        dest="verbosity",
        default=0,
        help="verbose output (repeat for increased verbosity)",
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
        "-r",
        "--reset",
        help="Resets file environment (DELETES FILES)",
        action="store_true",
    )
    parser_default.add_argument(
        "--add-unique-id",
        help="Add unique id to CTD data from master sheet",
        action="store_true",
    )
    parser_default.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output file, default output.csv",
        default="output.csv",
    )
    parser_default.add_argument(
        "--debug-run",
        help="Runs 20 files total for testing",
        action="store_true",
    )
    parser_default.add_argument(
        "-m", "--mastersheet", type=str, help="Path to mastersheet", default=None
    )
    parser_default.add_argument(
        "-w",
        "--workers",
        type=int,
        nargs="?",
        const=1,
        help="Sets max workers for parallel processing",
    )
    args = parser.parse_args()

    if args.command == "default":
        if args.reset:
            _reset_file_environment()
        _run_default(
            plot=args.plot,
            master_sheet_path=args.mastersheet,
            max_workers=args.workers,
            verbosity=args.verbosity,
            output_file=args.output,
            add_unique_id=args.add_unique_id,
            plot_secchi_chla_flag=args.plot_secchi_chla,
            debug_run=args.debug_run
        )


if __name__ == 'main':
    main()
    sys.exit()