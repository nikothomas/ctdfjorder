ctdfjorder
==========

Welcome to the **CTDFjorder** documentation! This guide will help you get started with using the CTDFjorder library,
designed to process and analyze CTD (Conductivity, Temperature, Depth) data efficiently.

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

.. image:: https://dl.circleci.com/status-badge/img/circleci/BqQeY7gNQzLD7veRpM5Fpj/4nijPG3CZqyE8SAoqtPWZr/tree/main.svg?style=shield
   :target: https://dl.circleci.com/status-badge/redirect/circleci/BqQeY7gNQzLD7veRpM5Fpj/4nijPG3CZqyE8SAoqtPWZr/tree/main
   :align: center

Introduction
------------

**CTDFjorder** is a Python package that simplifies the processing and analysis of CTD data. It supports reading data
from various file formats such as RSK (.rsk) and Castaway (.csv), and offers tools for data cleaning, filtering, and
calculation of derived oceanographic quantities like absolute salinity, density, and potential density. Additionally,
CTDFjorder includes capabilities for visualizing CTD profiles and derived quantities through advanced plotting tools.

Getting Started
---------------

If you're new to **CTDFjorder**, we recommend starting with the :doc:`Getting Started` guide. It provides a
comprehensive introduction to the library, including installation instructions, basic usage examples,
and an overview of the key features.

Interacting with the API
------------------------

The primary way to interact with **CTDFjorder** is through the `CTD` class. This class provides a comprehensive
interface for loading, processing, and analyzing CTD data.
Here’s a brief overview of what you can do with the `CTD` class:

- **Load Data**: Read CTD data from RSK (.rsk) and Castaway (.csv) files.
- **Process Data**: Clean and filter data, remove non-positive samples, and handle various preprocessing tasks.
- **Analyze Data**: Calculate derived quantities such as absolute salinity, density, and potential density.
- **Visualize Data**: Generate plots to visualize CTD profiles and derived quantities.

For detailed information on using the `CTD` class and other functionalities, refer to the :doc:`API` reference.

Features
--------

**CTDFjorder** offers a wide range of features to help you process and analyze CTD data:

- Read RSK (.rsk) and Castaway (.csv) files and extract CTD data.
- Process CTD data, including removing non-positive samples and cleaning data.
- Calculate derived quantities such as absolute salinity, density, and potential density.
- Determine mixed layer depth (MLD) using different methods.
- Generate plots for visualizing CTD profiles and derived quantities.
- Command-line interface (CLI) for easy processing and merging of CTD data files.

Installation
------------

To get started with **CTDFjorder**, we recommend creating a new environment. You can do this using conda with the following commands:

.. code-block:: bash

   conda create --name ctdfjorder python=3.12
   conda activate ctdfjorder

To install **CTDFjorder**, simply use pip:

.. code-block:: bash

   pip install ctdfjorder

Mapping
-------

To enable map plotting you will need a token from `MapBox <https://www.mapbox.com>`_.

Usage
-----

The **CTDFjorder** library includes a command-line interface (CLI) for processing and analyzing CTD data.
The primary way to interact with the API is through the `CTD` class, which provides a comprehensive set of methods
for handling CTD data. Below are the key functions from the `CTD` class used in the fjord processing CLI:

.. currentmodule:: ctdfjorder

.. autosummary::
    :toctree: generated/
    :caption: CTD Class Methods:
    :nosignatures:

    CTD.__init__
    CTD.get_df
    CTD.remove_upcasts
    CTD.filter_columns_by_range
    CTD.remove_non_positive_samples
    CTD.remove_invalid_salinity_values
    CTD.clean
    CTD.add_absolute_salinity
    CTD.add_density
    CTD.add_potential_density
    CTD.add_surface_salinity_temp_meltwater
    CTD.add_mean_surface_density
    CTD.add_mld
    CTD.add_bf_squared
    CTD.save_to_csv

License
-------

**CTDFjorder** is released under the MIT License.

Acknowledgments
---------------

**CTDFjorder** was developed for the Fjord Phyto project. The gsw library was used for certain derived calculations.

Citations
---------

McDougall, T. J., & Barker, P. M. (2011). Getting started with TEOS-10 and the Gibbs Seawater (GSW) Oceanographic Toolbox. SCOR/IAPSO WG127.

Contents
--------

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Contents:

   Getting Started
   API
