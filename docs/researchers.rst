Researchers
===========

Welcome to the CTDFjorder documentation! This guide will help you get started with using the CTDFjorder library,
designed to process and analyze CTD (Conductivity, Temperature, Depth) data efficiently.

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none


Introduction
------------

CTDFjorder is a Python package that simplifies the processing and analysis of CTD data. It supports reading data
from various file formats such as RBR (.rsk) and SonTek Castaway (.csv), and offers tools for data cleaning, filtering, and
calculation of derived oceanographic quantities like absolute salinity, density, and potential density. Additionally,
CTDFjorder includes capabilities for visualizing CTD profiles and derived quantities through advanced plotting tools.

Getting Started
---------------

If you're new to CTDFjorder, we recommend starting with the :doc:`Getting Started` guide. It provides a
comprehensive introduction to the library, including installation instructions, basic usage examples,
and an overview of the key features. If you'd prefer to create scripts in Python or R using CTDFjorder then visit the
:doc:`developers` page.

Command-Line Interface (CLI)
----------------------------

The CTDFjorder library includes a command-line interface (CLI) for processing and analyzing CTD data.
The CLI simplifies many common tasks and allows you to process data without needing to write Python code.
To read in depth about the functions used in CTDFjorder read the :doc:`API`.

Below are the key functions from the :class:`~ctdfjorder.CTD.CTD` class used in the default processing CLI:

- :meth:`ctdfjorder.CTD.CTD.remove_upcasts`: Removes upcast data from the dataset.
- :meth:`ctdfjorder.CTD.CTD.remove_non_positive_samples`: Removes samples with non-positive values.
- :meth:`ctdfjorder.CTD.CTD.clean`: Cleans the dataset.
- :meth:`ctdfjorder.CTD.CTD.add_absolute_salinity`: Adds absolute salinity to the dataset.
- :meth:`ctdfjorder.CTD.CTD.add_density`: Adds density calculations to the dataset.
- :meth:`ctdfjorder.CTD.CTD.add_potential_density`: Adds potential density to the dataset.
- :meth:`ctdfjorder.CTD.CTD.add_surface_salinity_temp_meltwater`: Adds surface salinity and temperature meltwater data.
- :meth:`ctdfjorder.CTD.CTD.add_mean_surface_density`: Adds mean surface density data.
- :meth:`ctdfjorder.CTD.CTD.add_mld`: Adds mixed layer depth (MLD) to the dataset.
- :meth:`ctdfjorder.CTD.CTD.add_brunt_vaisala_squared`: Adds Brunt–Väisälä frequency squared (N²) to the dataset.
- :meth:`ctdfjorder.CTD.CTD.save_to_csv`: Saves the processed data to a CSV file.
- :meth:`ctdfjorder.CTD.CTD.get_df`: Returns the dataset as a pandas DataFrame.

Mapping
-------

To enable map plotting you will need a token from `MapBox <https://www.mapbox.com>`_.

License
-------

**CTDFjorder** is released under the MIT License.

Acknowledgments
---------------

CTDFjorder was developed for the Fjord Phyto project. The gsw library was used for certain derived calculations.

References
-----------

[PaVR19]_

[McBa11]_