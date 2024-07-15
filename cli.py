import gc
import logging
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import argparse
import pandas as pd
import loggersetup
import ctdfjorder
from ctdfjorder import CTDError
import signal
import psutil
import os
import enlighten


def handler(signal_received, frame):
    if signal_received == signal.SIGINT:
        return
    else:
        raise KeyboardInterrupt


signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTSTP, handler)
manager = enlighten.get_manager()


def _process_ctd_file(file, plot=False, cached_master_sheet=pd.DataFrame(), master_sheet_path=None, verbosity=0):
    logger = loggersetup.setup_logging(verbosity)
    try:
        my_data = ctdfjorder.CTD(file, cached_master_sheet=cached_master_sheet, master_sheet_path=master_sheet_path,
                                 add_unique_id=True)
        file = os.path.basename(file)
        my_data.remove_upcasts()
        if my_data.get_pandas_df().duplicated().any():
            logger.critical(f"{file} - Duplicated after remove_upcasts")
        my_data.remove_non_positive_samples()
        if my_data.get_pandas_df().duplicated().any():
            logger.critical(f"{file} - Duplicated after remove_non_positive_samples")
        my_data.remove_invalid_salinity_values()
        if my_data.get_pandas_df().duplicated().any():
            logger.critical(f"{file} - Duplicated after remove_invalid_salinity_values")
        my_data.clean("practicalsalinity", 'autoencoder')
        if my_data.get_pandas_df().duplicated().any():
            logger.critical(f"{file} - Duplicated after clean")
        my_data.add_surface_salinity_temp_meltwater()
        if my_data.get_pandas_df().duplicated().any():
            logger.critical(f"{file} - Duplicated duplicated after surface_density")
        my_data.add_absolute_salinity()
        if my_data.get_pandas_df().duplicated().any():
            logger.critical(f"{file} - Duplicated after add_absolute_salinity")
        my_data.add_density()
        if my_data.get_pandas_df().duplicated().any():
            logger.critical(f"{file} - Duplicated after add_density")
        my_data.add_potential_density()
        if my_data.get_pandas_df()['depth_00'].duplicated().any():
            logger.critical(f"{file} - Duplicated after add_potential density")
        my_data.add_mld(20, "potentialdensityavg")
        if my_data.get_pandas_df()['depth_00'].duplicated().any():
            logger.critical(f"{file} - Duplicated after add_mld")
        my_data.add_stratification(20)
        if my_data.get_pandas_df()['depth_00'].duplicated().any():
            logger.critical(f"{file} - Duplicated after add_stratification")
        if plot:
            my_data.plot('potentialdensity')
            my_data.plot('salinity')
        if my_data.get_pandas_df()['depth_00'].duplicated().any():
            logger.critical(f"{file} - Duplicated after plotting")
        return my_data.get_pandas_df()
    except CTDError as e:
        logger.error(e)
    except Exception as e:
        logger.exception(e)


def _run_default(plot=False, master_sheet_path=None, max_workers=1, verbosity=0):
    logger = loggersetup.setup_logging(verbosity)
    ctd_files_list = _get_rsk_filenames_in_dir(get_cwd())
    ctd_files_list.extend(_get_csv_filenames_in_dir(get_cwd()))
    cached_master_sheet = pd.read_excel(master_sheet_path)
    cached_master_sheet = ctdfjorder.CTD.Utility.load_master_sheet(cached_master_sheet)
    total_files = len(ctd_files_list)
    bar_format = u'{desc}{desc_pad}{percentage:3.0f}%|{bar}| ' + \
                 u'S:{count_0:{len_total}d} ' + \
                 u'E:{count_1:{len_total}d} ' + \
                 u'[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]'
    success = manager.counter(total=total_files, desc='Processing Files', unit='Files',
                              color='green', bar_format=bar_format)
    errors = success.add_subcounter('red')
    executor = ProcessPoolExecutor(max_workers=max_workers)
    results = []
    if not ctd_files_list:
        logger.debug("No files to process")
        return
    # Process the rest of the files in parallel
    with ExitStack() as stack:
        stack.enter_context(executor)
        stack.enter_context(success)
        try:
            futures = {executor.submit(_process_ctd_file, file,
                                       plot=plot,
                                       cached_master_sheet=cached_master_sheet,
                                       master_sheet_path=master_sheet_path,
                                       verbosity=verbosity): file for file in ctd_files_list}
            for future in as_completed(futures):
                result = future.result()
                if type(result) is not type(None):
                    results.append(result)
                    success.update(1)
                else:
                    errors.update(1)
                futures.pop(future)
                gc.collect()
            df = ctdfjorder.merge_dataframes(results)
            ctdfjorder.CTD.Utility.save_to_csv(df, "outputclean.csv")
        except KeyboardInterrupt:
            loggersetup.setup_logging(0)
            print()
            logger.critical("Received shutdown command from keyboard, flushing logs and killing child processes")
            for proc in psutil.process_iter():
                if proc.name().startswith('python'):
                    proc.kill()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def _merge_all_in_folder():
    ctd_files_list = _get_rsk_filenames_in_dir(get_cwd())
    ctd_files_list.extend(_get_csv_filenames_in_dir(get_cwd()))
    error_out = []
    for file in ctd_files_list:
        try:
            my_data = ctdfjorder.CTD(file)
            my_data.remove_upcasts()
            my_data.remove_non_positive_samples()
            my_data.save_to_csv("output.csv")
        except Exception as e:
            continue


def _get_rsk_filenames_in_dir(working_directory):
    rsk_files_list = []
    rsk_filenames_no_path = []
    for filename in os.listdir(working_directory):
        if filename.endswith('.rsk'):
            for filepath in rsk_files_list:
                filename_no_path = ('_'.join(filepath.split("/")[-1].split("_")[0:3]).split('.rsk')[0])
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
        if filename.endswith('.csv') and 'output' not in filename:
            for filepath in csv_files_list:
                filename_no_path = ('_'.join(filepath.split("/")[-1].split("_")[0:3]).split('.csv')[0])
                if filename_no_path in rsk_filenames_no_path:
                    continue
                rsk_filenames_no_path.append(filename_no_path)
            file_path = os.path.join(working_directory, filename)
            csv_files_list.append(file_path)
    return csv_files_list


def get_cwd():
    # determine if application is a script file or frozen exe
    if getattr(sys, 'frozen', False):
        working_directory_path = os.path.dirname(sys.executable)
    elif __file__:
        working_directory_path = os.getcwd()
    else:
        working_directory_path = os.getcwd()
    return working_directory_path


def _get_filename(filepath):
    return '_'.join(filepath.split("/")[-1].split("_")[0:3]).split('.rsk')[0]


def _reset_file_environment():
    output_file_csv = "output.csv"
    output_file_csv_clean = "outputclean.csv"
    output_log = "ctdfjorder.log"
    cwd = get_cwd()
    output_file_csv = os.path.join(cwd, output_file_csv)
    output_file_csv_clean = os.path.join(cwd, output_file_csv_clean)
    if cwd is None:
        raise CTDError("Unknown", "Couldn't get working directory.")
    if os.path.isfile(output_file_csv):
        os.remove(output_file_csv)
    if os.path.isfile(output_file_csv_clean):
        os.remove(output_file_csv_clean)
    if os.path.isfile(output_log):
        os.remove(output_log)
    if os.path.isdir("./plots"):
        shutil.rmtree('./plots')
    if not os.path.isdir("./plots"):
        os.mkdir("./plots")


def main():
    parser = argparse.ArgumentParser(
        description="CTD Fjorder Processing Script",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subparser for the 'merge' command
    parser_merge = subparsers.add_parser('merge', help='Merge all RSK files in the current folder')
    parser_merge.add_argument('-v', '--verbose',
                              action='count',
                              dest='verbosity',
                              default=0,
                              help="verbose output (repeat for increased verbosity)")
    parser_merge.add_argument('-q', '--quiet',
                              action='store_const',
                              const=-1,
                              default=0,
                              dest='verbosity',
                              help="quiet output (show errors only)")

    # Subparser for the 'default' command
    parser_default = subparsers.add_parser('default', help='Run the default processing pipeline')
    parser_default.add_argument('-p', '--plot', action='store_true',
                                help='Generate plots during the default processing pipeline')
    parser_default.add_argument('-v', '--verbose',
                                action='count',
                                dest='verbosity',
                                default=0,
                                help="verbose output (repeat for increased verbosity)")
    parser_default.add_argument('-q', '--quiet',
                                action='store_const',
                                const=-1,
                                default=0,
                                dest='verbosity',
                                help="quiet output (show errors only)")
    parser_default.add_argument("-r", "--reset", help="Resets file environment (DELETES FILES)",
                                action="store_true")
    parser_default.add_argument('-o', '--output', type=str, help='Path to output file')
    parser_default.add_argument('-m', '--mastersheet', type=str, help='Path to mastersheet', default=None)
    parser_default.add_argument('-w', '--workers', type=int, nargs='?', const=1,
                                help='Sets max workers for parallel processing')
    args = parser.parse_args()
    if args.command == 'merge':
        print("Merging completed successfully.")

    elif args.command == 'default':
        if args.reset:
            _reset_file_environment()
        _run_default(plot=args.plot, master_sheet_path=args.mastersheet, max_workers=args.workers,
                     verbosity=args.verbosity)


if __name__ == '__main__':
    main()
