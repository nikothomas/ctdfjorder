[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13293665.svg#3)](https://doi.org/10.5281/zenodo.13293665)
![Tests](https://github.com/nikothomas/ctdfjorder/actions/workflows/tests.yml/badge.svg?branch=main)
![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/nikothomas/5ceb7e714da5151b751f668c4cc49013/raw/answer.json)
[![PyPI version](https://badge.fury.io/py/ctdfjorder.svg#4)](https://badge.fury.io/py/ctdfjorder)

<p style=text-align:center;>
  <img src="https://raw.githubusercontent.com/nikothomas/ctdfjorder/main/logo.png" alt="CTDFjorder Logo"/>
</p>

CTDFjorder is a Python package for processing and analyzing CTD (Conductivity, Temperature, Depth) data.
Documentation: [Read the docs](https://ctdfjorder.readthedocs.io/en/latest)

- [<code>⭐ Features</code>](#-features)
- [<code>📦 Installation</code>](#-installation)
- [<code>🚀 Usage</code>](#-usage)
- [<code>⚠️ Issues</code>](#-issues)
- [<code>📝 License</code>](#-license)
- [<code>📑 Citing CTDFjorder</code>](#-license)
- [<code>📢 Acknowledgments</code>](#-acknowledgments)

## ⭐ Features
- Read RBR (.rsk) and SonTek Castaway (.csv) files and extract CTD data
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

## 🚀 Usage
CTDFjorder provides a command-line interface (CLI) for processing and analyzing CTD data in addition to serving
as a python library. A tutorial has been setup [here](https://ctdfjorder.readthedocs.io/en/latest/Getting%20Started.html)
for running the CLI.

## ⚠️ Issues
If you encounter an issue while running or using CTDFjorder, please submit a report on the 
[Issues](https://github.com/nikothomas/ctdfjorder/issues) page. This is the fastest way to get your problem resolved.

## 📝 License
CTDFjorder is released under the MIT License.

## 📑 Citing CTDFjorder
Citation information for CTDFjorder can be found here [zenodo.13293665](https://dx.doi.org10/zenodo.13293665>).

## 📢 Acknowledgments
CTDFjorder was developed for the [Fjord Phyto](https://fjordphyto.ucsd.edu) project. The [GSW-python](https://github.com/TEOS-10/GSW-Python) library was used for certain dervied calculations.

## Citations
McDougall, T. J., & Barker, P. M. (2011). Getting started with TEOS-10 and the Gibbs Seawater (GSW) Oceanographic Toolbox. SCOR/IAPSO WG127.

Pan, B.J.; Vernet, M.; Reynolds, R.A.; Mitchell, B.G.: The optical and biological properties of glacial meltwater in an Antarctic fjord. PLOS ONE 14(2): e0211107 (2019). https://doi.org/10.1371/journal.pone.0211107
