
[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/BqQeY7gNQzLD7veRpM5Fpj/4nijPG3CZqyE8SAoqtPWZr/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/circleci/BqQeY7gNQzLD7veRpM5Fpj/4nijPG3CZqyE8SAoqtPWZr/tree/main)

# CTDFjorder
CTDFjorder is a Python package for processing and analyzing CTD (Conductivity, Temperature, Depth) data.
Documentation: [Read the docs](https://nikothomas.github.io/ctdfjorder/)

- [<code>⭐ Features</code>](#-features)
- [<code>📦 Installation</code>](#-installation)
- [<code>🗺️ Mapping</code>](#-mapping)
- [<code>🚀 Usage</code>](#-usage)
- [<code>📝 License</code>](#-license)
- [<code>📢 Acknowledgments</code>](#-acknowledgments)

## ⭐ Features
- Read RSK (.rsk) and Castaway (.csv) files and extract CTD data
- Process CTD data, including removing non-positive samples and cleaning data
- Calculate derived quantities such as absolute salinity, density, and potential density
- Determine mixed layer depth (MLD) using different methods
- Generate plots for visualizing CTD profiles and derived quantities
- Command-line interface (CLI) for easy processing and merging of CTD data files

## 📦 Installation
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

## 🗺️ Mapping
If you want to enable map plotting at the end of the pipeline you'll need a default token from [MapBox](https://www.mapbox.com).

## 🚀 Usage
CTDFjorder provides a command-line interface (CLI) for processing and analyzing CTD data in addition to serving
as a python library. A tutorial has been setup [here](https://nikothomas.github.io/ctdfjorder/Getting%20Started.html)
for running the CLI.

## 📝 License
CTDFjorder is released under the MIT License.

## 📢 Acknowledgments
CTDFjorder was developed for the Fjord Phyto project. The gsw library was used for certain dervied calculations.

## Citations
McDougall, T. J., & Barker, P. M. (2011). Getting started with TEOS-10 and the Gibbs Seawater (GSW) Oceanographic Toolbox. SCOR/IAPSO WG127.

