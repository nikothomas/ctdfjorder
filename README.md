# CTDFjorder

CTDFjorder is a Python package for processing and analyzing CTD (Conductivity, Temperature, Depth) data.
Documentation: [Read the docs](https://nikothomas.github.io/ctdfjorder/)

## Features

- Read RSK (.rsk) and Castaway (.csv) files and extract CTD data
- Process CTD data, including removing non-positive samples and cleaning data
- Calculate derived quantities such as absolute salinity, density, and potential density
- Determine mixed layer depth (MLD) using different methods
- Generate plots for visualizing CTD profiles and derived quantities
- Command-line interface (CLI) for easy processing and merging of RSK files

## Installation
It's recommended that you create a new environment just for CTDFjorder. This can be done in conda with the following
command.
```shell
conda create --name ctdfjorder python=3.12
conda activate ctdfjorder
```
To install ctdfjorder you can use pip:
```shell
pip install ctdfjorder
```

## Usage

CTDFjorder provides a command-line interface (CLI) for processing and analyzing CTD data.
This command runs the default processing pipeline and creates plots on all RSK and .csv files found in the current folder with master sheet Fjord Phyto MASTERSHEET.xlsx:

```shell
ctdfjorder-cli default -r -p -m "FjordPhyto MASTERSHEET.xlsx" -w 4 --show-table --token 'MATBOX_TOKEN'
```

## Contributing

Contributions to CTDFjorder are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the [GitHub repository](https://github.com/nikothomas/CTDFjorder).

## License

CTDFjorder is released under the MIT License.

## Acknowledgments

CTDFjorder was developed by Nikolas Yanek-Chrones for the Fjord Phyto project. The gsw library was used for certain dervied calculations.

## Citations
McDougall, T. J., & Barker, P. M. (2011). Getting started with TEOS-10 and the Gibbs Seawater (GSW) Oceanographic Toolbox. SCOR/IAPSO WG127.

