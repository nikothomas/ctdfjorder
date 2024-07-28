CTDFjorder for Developers
=========================

Welcome to the **CTDFjorder** documentation! This guide will help you get started with using the CTDFjorder library,
designed to process and analyze CTD (Conductivity, Temperature, Depth) data efficiently.

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none


Introduction
------------

**CTDFjorder** is a Python package that simplifies the processing and analysis of CTD data. It supports reading data
from various file formats such as RSK (.rsk) and Castaway (.csv), and offers tools for data cleaning, filtering, and
calculation of derived oceanographic quantities like absolute salinity, density, and potential density. Additionally,
CTDFjorder includes capabilities for visualizing CTD profiles and derived quantities through advanced plotting tools.

Interacting with the API
------------------------

The primary way to interact with **CTDFjorder** is through the :doc:`CTD` class. This class provides a comprehensive
interface for loading, processing, and analyzing CTD data.
Here’s a brief overview of what you can do with the :doc:`CTD` class:

- **Load Data**: Read CTD data from RSK (.rsk) and Castaway (.csv) files.
- **Process Data**: Clean and filter data, remove non-positive samples, and handle various preprocessing tasks.
- **Analyze Data**: Calculate derived quantities such as absolute salinity, density, and potential density.
- **Visualize Data**: Generate plots to visualize CTD profiles and derived quantities.

For detailed information on using the :doc:`CTD` class and other functionalities, refer to the :doc:`API` reference.

Key Methods in the CTD Class
------------------------------------

The :doc:`CTD` class provides a comprehensive set of methods for handling CTD data:

- :meth:`ctdfjorder.CTD.CTD.remove_upcasts`: Removes upcast data from the dataset.
- :meth:`ctdfjorder.CTD.CTD.filter_columns_by_range`: Filters data columns by specified ranges.
- :meth:`ctdfjorder.CTD.CTD.remove_non_positive_samples`: Removes samples with non-positive values.
- :meth:`ctdfjorder.CTD.CTD.remove_invalid_salinity_values`: Removes invalid salinity values.
- :meth:`ctdfjorder.CTD.CTD.clean`: Cleans the dataset.
- :meth:`ctdfjorder.CTD.CTD.add_absolute_salinity`: Adds absolute salinity to the dataset.
- :meth:`ctdfjorder.CTD.CTD.add_density`: Adds density calculations to the dataset.
- :meth:`ctdfjorder.CTD.CTD.add_potential_density`: Adds potential density to the dataset.
- :meth:`ctdfjorder.CTD.CTD.add_surface_salinity_temp_meltwater`: Adds surface salinity and temperature meltwater data.
- :meth:`ctdfjorder.CTD.CTD.add_mean_surface_density`: Adds mean surface density data.
- :meth:`ctdfjorder.CTD.CTD.add_mld`: Adds mixed layer depth (MLD) to the dataset.
- :meth:`ctdfjorder.CTD.CTD.add_bf_squared`: Adds buoyancy frequency squared (N²) to the dataset.
- :meth:`ctdfjorder.CTD.CTD.save_to_csv`: Saves the processed data to a CSV file.
- :meth:`ctdfjorder.CTD.CTD.get_df`: Returns the dataset as a pandas DataFrame.
