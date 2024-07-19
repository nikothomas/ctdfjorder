# CTDFjorder

ctdfjorder is a Python package for processing and analyzing CTD (Conductivity, Temperature, Depth) data.
Documentation: [Read the docs](https://nikothomas.github.io/docs/CTDFjorder.html)

## Features

- Read RSK (.rsk) and Castaway (.csv) files and extract CTD data
- Process CTD data, including removing non-positive samples and cleaning data
- Calculate derived quantities such as absolute salinity, density, and potential density
- Determine mixed layer depth (MLD) using different methods
- Generate plots for visualizing CTD profiles and derived quantities
- Command-line interface (CLI) for easy processing and merging of RSK files

## Installation
It's recommended that you create a new environment just for ctdfjorder. This can be done in conda with the following
command.
```shell
conda create --name ctdfjorder -c conda-forge python=3.11
conda activate ctdfjorder
```
To install ctdfjorder you can use pip:
```shell
pip install ctdfjorder
```

## Usage

CTDFjorder provides a command-line interface (CLI) for processing and analyzing CTD data. Here are the available commands:

```shell
ctdfjorder-cli default -r -v
```

This command runs the default processing pipeline on all RSK files found in the current folder.

## Configuration

CTDFjorder looks for a master sheet Excel file named "FjordPhyto MASTER SHEET.xlsx" in the current working directory. This file is used for estimating location information when it's not available in the RSK files. You can change this by modifying the CTDFjorder.master_sheet_path field to the name of your own spreadsheet.

## Examples

Here are a few examples of how to use CTDFjorder:

- Process a single RSK file:

```shell
ctdfjorder-cli process path/to/rskfile.rsk
```

- Merge all RSK files in the current folder:

```shell
ctdfjorder-cli merge
```

- Run the default processing pipeline on all RSK files in the current folder:

```shell
ctdfjorder-cli default
```

- Write your own script:
```
import CTDFjorder
import os
for file in get_rsk_filenames_in_dir(os.getcwd()):
    try:
        my_data = CTD(file)
        my_data.add_filename_to_table()
        my_data.save_to_csv("output.csv")
        my_data.add_location_to_table()
        my_data.remove_non_positive_samples()
        my_data.clean("practicalsalinity", 'salinitydiff')
        my_data.add_absolute_salinity()
        my_data.add_density()
        my_data.add_overturns()
        my_data.add_mld(1)
        my_data.add_mld(5)
        my_data.save_to_csv("outputclean.csv")
        my_data.plot_depth_density_salinity_mld_scatter()
        my_data.plot_depth_temperature_scatter()
        my_data.plot_depth_salinity_density_mld_line()
    except Exception as e:
        print(f"Error processing file: '{file}' {e}")
        continue
```

## Contributing

Contributions to CTDFjorder are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the [GitHub repository](https://github.com/nikothomas/CTDFjorder).

## License

CTDFjorder is released under the MIT License.

## Acknowledgments

CTDFjorder was developed by Nikolas Yanek-Chrones for the Fjord Phyto project. The gsw library was used for certain dervied calculations.

## Citations
McDougall, T. J., & Barker, P. M. (2011). Getting started with TEOS-10 and the Gibbs Seawater (GSW) Oceanographic Toolbox. SCOR/IAPSO WG127.

