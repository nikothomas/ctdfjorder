API Documentation
=================

Welcome to the **CTDFjorder API** documentation! This section provides a detailed overview of the various modules, classes, and functions available in the CTDFjorder library. Use the navigation below to explore the different components of the API.

Introduction
------------

The CTDFjorder API is organized into several modules, each focused on specific functionality. Below is a brief overview of what each module offers:

+----------------------+-----------------------------------------------------------------------------------------+
| Module               | Description                                                                             |
+======================+=========================================================================================+
| :doc:`AI`            | Functions and classes related to artificial intelligence and machine learning           |
|                      | applications within CTDFjorder.                                                         |
+----------------------+-----------------------------------------------------------------------------------------+
| :doc:`CLI`           | Command-line interface for processing and analyzing CTD data.                           |
+----------------------+-----------------------------------------------------------------------------------------+
| :doc:`Constants`     | Definitions of constants used throughout the CTDFjorder library.                        |
+----------------------+-----------------------------------------------------------------------------------------+
| :doc:`CTD`           | Core classes and functions for loading, processing, and analyzing CTD data.             |
+----------------------+-----------------------------------------------------------------------------------------+
| :doc:`Data Classes`  | Data structures used within the CTDFjorder library.                                     |
+----------------------+-----------------------------------------------------------------------------------------+
| :doc:`Exceptions`    | Custom exceptions used within the CTDFjorder library.                                   |
+----------------------+-----------------------------------------------------------------------------------------+
| :doc:`Load CTD`      | Functions for loading CTD data from various file formats.                               |
+----------------------+-----------------------------------------------------------------------------------------+
| :doc:`Metadata`      | Handling of metadata associated with CTD data.                                          |
+----------------------+-----------------------------------------------------------------------------------------+
| :doc:`SCAR Database` | Access and interaction with the SCAR database.                                          |
+----------------------+-----------------------------------------------------------------------------------------+
| :doc:`Utils`         | Utility functions and helpers.                                                          |
+----------------------+-----------------------------------------------------------------------------------------+
| :doc:`Visualize`     | Functions and classes for visualizing CTD data.                                         |
+----------------------+-----------------------------------------------------------------------------------------+

Working with the `CTD` Class
============================

The `CTD` class in `ctdfjorder` is designed to help you load, process, and analyze data from CTD (Conductivity, Temperature, Depth) devices. This tutorial will guide you through the process of using this class, demonstrating key methods and their outputs.

Creating a `CTD` Object
-----------------------

The first step in working with the `CTD` class is to create an object. This object represents a single CTD profile, loaded from a file. The `CTD` class automatically handles different file formats, such as RSK and Castaway.

**Example:**

.. code-block:: python

    >>> from ctdfjorder import CTD

    # Create a CTD object from a Castaway file
    >>> ctd_data = CTD('CC1531002_20181225_114931.csv')

When you instantiate a `CTD` object, it reads and processes the file. If the file is valid, the data is loaded into a Polars DataFrame within the object.

Accessing the Data
------------------

After creating the `CTD` object, you may want to inspect the data. The `get_df` method allows you to retrieve the data as either a Polars or Pandas DataFrame, depending on your preference.

**Example:**

.. code-block:: python

    # Accessing the data as a Polars DataFrame
    >>> output = ctd_data.get_df()
    >>> print(output.head(3))

    shape: (3, 13)
    ┌──────────────┬──────────┬─────────────┬──────────────┬───┬────────────┬───────────────────────────────┬────────────┬────────────┐
    │ sea_pressure ┆ depth    ┆ temperature ┆ conductivity ┆ … ┆ profile_id ┆ filename                      ┆ latitude   ┆ longitude  │
    │ ---          ┆ ---      ┆ ---         ┆ ---          ┆   ┆ ---        ┆ ---                           ┆ ---        ┆ ---        │
    │ f64          ┆ f64      ┆ f64         ┆ f64          ┆   ┆ i32        ┆ str                           ┆ f64        ┆ f64        │
    ╞══════════════╪══════════╪═════════════╪══════════════╪═══╪════════════╪═══════════════════════════════╪════════════╪════════════╡
    │ 0.15         ┆ 0.148676 ┆ 0.32895     ┆ 28413.735648 ┆ … ┆ 0          ┆ CC1531002_20181225_114931.csv ┆ -64.668455 ┆ -62.641775 │
    │ 0.45         ┆ 0.446022 ┆ 0.316492    ┆ 28392.966662 ┆ … ┆ 0          ┆ CC1531002_20181225_114931.csv ┆ -64.668455 ┆ -62.641775 │
    │ 0.75         ┆ 0.743371 ┆ 0.310613    ┆ 28386.78011  ┆ … ┆ 0          ┆ CC1531002_20181225_114931.csv ┆ -64.668455 ┆ -62.641775 │
    └──────────────┴──────────┴─────────────┴──────────────┴───┴────────────┴───────────────────────────────┴────────────┴────────────┘

You can also access the data as a Pandas DataFrame:

.. code-block:: python

    # Accessing the data as a Pandas DataFrame
    >>> output = ctd_data.get_df(pandas=True)
    >>> print(output.head(3))

       sea_pressure     depth  temperature  conductivity  specific_conductivity  ...  pressure  profile_id                       filename   latitude  longitude
    0          0.15  0.148676      0.32895  28413.735648           56089.447456  ...   10.2825           0  CC1531002_20181225_114931.csv -64.668455 -62.641775
    1          0.45  0.446022     0.316492  28392.966662           56076.028991  ...   10.5825           0  CC1531002_20181225_114931.csv -64.668455 -62.641775
    2          0.75  0.743371     0.310613   28386.78011           56076.832208  ...   10.8825           0  CC1531002_20181225_114931.csv -64.668455 -62.641775
    [3 rows x 13 columns]

Cleaning the Data
-----------------

CTD data often requires cleaning to remove invalid or erroneous samples. The `CTD` class provides several methods to clean the data.

Removing Non-Positive Samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can remove rows with non-positive values for key parameters like depth, pressure, or salinity.

**Example:**

.. code-block:: python

    >>> ctd_data.remove_non_positive_samples()
    >>> output = ctd_data.get_df()
    >>> print(output.head(3))

    # Output will now exclude any rows with non-positive values

Removing Upcasts
^^^^^^^^^^^^^^^^

CTD profiles can contain upcasts, where the pressure decreases unexpectedly. These can be removed to ensure data integrity.

**Example:**

.. code-block:: python

    >>> ctd_data.remove_upcasts()
    >>> output = ctd_data.get_df()
    >>> print(output.head(3))

    # Output will now only include downcast data where pressure consistently increases

Filtering Data
--------------

You may want to filter your data based on specific criteria, such as temperature or salinity ranges.

**Example:**

Filtering by Temperature and Salinity Range
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> filters = zip(['temperature', 'salinity'], [20.0, 35.0], [10.0, 30.0])
    >>> ctd_data.filter_columns_by_range(filters=filters)
    >>> output = ctd_data.get_df()
    >>> print(output.head(3))

    # Data is now filtered to include only temperatures between 10.0 and 20.0, and salinity between 30.0 and 35.0

Advanced Analysis
-----------------

The `CTD` class also supports more advanced analysis, such as calculating derived parameters like density or mixed layer depth (MLD).

Adding Absolute Salinity and Density
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, you can calculate and add absolute salinity to the dataset, followed by the density.

**Example:**

.. code-block:: python

    >>> ctd_data.add_absolute_salinity()
    >>> ctd_data.add_density()
    >>> output = ctd_data.get_df()
    >>> print(output.head(3))

    # Output will include new columns for absolute salinity and density

Calculating Mixed Layer Depth (MLD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can calculate the mixed layer depth (MLD) using a specified method and reference depth.

**Example:**

.. code-block:: python

    >>> ctd_data.add_mld(reference=10, method="potential_density_avg", delta=0.05)
    >>> output = ctd_data.get_df()
    >>> print(output.head(3))

    # Output will now include a new column for MLD, calculated based on the given parameters

Saving Your Processed Data
--------------------------

Once you’ve cleaned and analyzed your data, you can save the results to a CSV file.

**Example:**

.. code-block:: python

    >>> ctd_data.save_to_csv('processed_ctd_data.csv', null_value="NA")

    # The processed data is saved to 'processed_ctd_data.csv', with null values represented as 'NA'

Conclusion
----------

This tutorial has walked you through the key functionalities of the `CTD` class in `ctdfjorder`. With these tools, you can load, clean, analyze, and save CTD data efficiently for your oceanographic studies.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   AI
   CLI
   Constants
   CTD
   Data Classes
   Exceptions
   Load CTD
   Metadata
   SCAR Database
   Utils
   Visualize
