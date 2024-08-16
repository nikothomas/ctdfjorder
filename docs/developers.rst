Developers
==========

Welcome to the CTDFjorder developer documentation! This guide will help you get started with using the CTDFjorder library,
designed to process and analyze CTD (Conductivity, Temperature, Depth) data efficiently.
To see everything you can do with CTDFjorder or get more in depth information on the functions used in the tutorial go to the :doc:`./API/index` page.
CTDFjorder is not native to R, but can be used in R scripts with the reticulate package.

Introduction
------------------------

The primary way to interact with CTDFjorder is through the :doc:`/API/CTD/index` class. This class provides a comprehensive
interface for loading, processing, and analyzing CTD data.
Here’s a brief overview of what you can do with the CTD class:

- **Load Data**: Read CTD data from RBR (.rsk) and SonTek Castaway (.csv) files.
- **Process Data**: Clean and filter data, remove non-positive samples, and handle various preprocessing tasks.
- **Analyze Data**: Calculate derived quantities such as absolute salinity, density, and potential density.
- **Visualize Data**: Generate plots to visualize CTD profiles and derived quantities.

.. raw:: html

    <div style="display: flex; justify-content: space-around; margin-top: 80px; margin-bottom: 80px; flex-wrap: wrap; gap: 20px;">

    <a href="#using-ctdfjorder-in-python" style="text-decoration: none; flex: 1 1 45%; max-width: 45%;">
        <button style="width: 100%; font-size: 1.5vw; padding: 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; border-radius: 5px;">
                <i class="fab fa-python" style="font-size: 1.5vw;"></i>
                Python Tutorial
        </button>
    </a>

    <a href="#using-ctdfjorder-in-r" style="text-decoration: none; flex: 1 1 45%; max-width: 45%;">
        <button style="width: 100%; font-size: 1.5vw; padding: 20px; background-color: #2196F3; color: white; border: none; cursor: pointer; border-radius: 5px;">
                <i class="fab fa-r-project" style="font-size: 1.5vw;"></i>
                R Tutorial
        </button>
    </a>

    </div>

Using CTDFjorder in Python
===========================

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

Setting Up Environment
----------------------

If you haven't already installed the CTDFjorder package you can do so with pip.

.. code-block:: console

    pip install ctdfjorder

Creating a CTD Object
----------------------------------------------

The first step in working with the CTD class is to create an object. This object represents a single CTD profile, loaded from a file. The CTD class automatically handles different file formats, such as RSK and Castaway.

**Example:**

.. code-block:: python

    >>> from ctdfjorder import CTD

    # Create a CTD object from a Castaway file
    >>> ctd_data = CTD('CC1531002_20181225_114931.csv')

When you instantiate a CTD object, it reads and processes the file. If the file is valid, the data is loaded into a Polars DataFrame within the object.

Accessing the Data
------------------

After creating the CTD object, you may want to inspect the data. The following method allows you to retrieve the data as either a Polars or Pandas DataFrame, depending on your preference.

**Example:**

.. code-block:: python

    # Accessing the data as a Polars DataFrame
    output = ctd_data.get_df()
    print(output.head(3))

.. code-block:: console

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
    output = ctd_data.get_df(pandas=True)
    print(output.head(3))

.. code-block:: console

       sea_pressure     depth  temperature  conductivity  specific_conductivity  ...  pressure  profile_id                       filename   latitude  longitude
    0          0.15  0.148676      0.32895  28413.735648           56089.447456  ...   10.2825           0  CC1531002_20181225_114931.csv -64.668455 -62.641775
    1          0.45  0.446022     0.316492  28392.966662           56076.028991  ...   10.5825           0  CC1531002_20181225_114931.csv -64.668455 -62.641775
    2          0.75  0.743371     0.310613   28386.78011           56076.832208  ...   10.8825           0  CC1531002_20181225_114931.csv -64.668455 -62.641775
    [3 rows x 13 columns]

Cleaning the Data
-----------------

CTD data often requires cleaning to remove invalid or erroneous samples. The CTD class provides several methods to clean the data.

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

The CTD class also supports more advanced analysis, such as calculating derived parameters like density or mixed layer depth (MLD).

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

This tutorial has walked you through the key functionalities of the CTD class in CTDFjorder. With these tools, you can load, clean, analyze, and save CTD data efficiently for your oceanographic studies.

For detailed information on using the CTD class and other functionalities, refer to the :doc:`./API/index` reference.

Using CTDFjorder in R
===========================

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

Setting Up the Environment
---------------------------

To use the CTDFjorder Python package in R, follow these steps to set up your environment:

1. **Install the `reticulate` package in R**:

   .. code-block:: r

       install.packages("reticulate")

2. **Install the CTDFjorder Python package**:

   You can install the package using pip. From within R, you can do this using `reticulate`:

   .. code-block:: r

       library(reticulate)
       py_install("CTDFjorder")

3. **Optional: Configure `reticulate` to use the correct Python environment**:

   If you have multiple Python environments, ensure `reticulate` is using the right one where CTDFjorder is installed.

   .. code-block:: r

       use_python("/path/to/your/python")

   Replace `"/path/to/your/python"` with the path to the Python executable that has CTDFjorder installed.

Loading the CTD Class
---------------------

Once the environment is set up, you can import the **CTDFjorder** package and start working with the CTD class.

**Example:**

.. code-block:: r

    library(reticulate)
    CTDFjorder <- import("ctdfjorder")

Creating a CTD Object
---------------------

The first step in working with the CTD class is to create an object. This object represents a single CTD profile, loaded from a file. The CTD class automatically handles different file formats, such as RSK and Castaway.

**Example:**

.. code-block:: r

    # Create a CTD object from a Castaway file
    ctd_data <- CTDFjorder$CTD('CC1531002_20181225_114931.csv')

When you instantiate a CTD object, it reads and processes the file. If the file is valid, the data is loaded into a DataFrame within the object.

Accessing the Data
------------------

After creating the CTD object, you may want to inspect the data. The following method allows you to retrieve the data as either a Polars or Pandas DataFrame, depending on your preference.

**Example:**

.. code-block:: r

    # Accessing the data as a Polars DataFrame
    output <- ctd_data$get_df()
    print(output$head(3))

    # Output will be shown as a DataFrame with the first 3 rows displayed

You can also access the data as a Pandas DataFrame:

.. code-block:: r

    # Accessing the data as a Pandas DataFrame
    output <- ctd_data$get_df(pandas = TRUE)
    print(output$head(3))

    # The DataFrame will now be displayed using the Pandas format

Cleaning the Data
-----------------

CTD data often requires cleaning to remove invalid or erroneous samples. The CTD class provides several methods to clean the data.

Removing Non-Positive Samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can remove rows with non-positive values for key parameters like depth, pressure, or salinity.

**Example:**

.. code-block:: r

    ctd_data$remove_non_positive_samples()
    output <- ctd_data$get_df()
    print(output$head(3))

    # Output will now exclude any rows with non-positive values

Removing Upcasts
^^^^^^^^^^^^^^^^

CTD profiles can contain upcasts, where the pressure decreases unexpectedly. These can be removed to ensure data integrity.

**Example:**

.. code-block:: r

    ctd_data$remove_upcasts()
    output <- ctd_data$get_df()
    print(output$head(3))

    # Output will now only include downcast data where pressure consistently increases

Filtering Data
--------------

You may want to filter your data based on specific criteria, such as temperature or salinity ranges.

Filtering by Temperature and Salinity Range
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Example:**

.. code-block:: r

    filters <- list(list('temperature', 20.0, 10.0), list('salinity', 35.0, 30.0))
    ctd_data$filter_columns_by_range(filters)
    output <- ctd_data$get_df()
    print(output$head(3))

    # Data is now filtered to include only temperatures between 10.0 and 20.0, and salinity between 30.0 and 35.0

Advanced Analysis
-----------------

The CTD class also supports more advanced analysis, such as calculating derived parameters like density or mixed layer depth (MLD).

Adding Absolute Salinity and Density
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, you can calculate and add absolute salinity to the dataset, followed by the density.

**Example:**

.. code-block:: r

    ctd_data$add_absolute_salinity()
    ctd_data$add_density()
    output <- ctd_data$get_df()
    print(output$head(3))

    # Output will include new columns for absolute salinity and density

Calculating Mixed Layer Depth (MLD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can calculate the mixed layer depth (MLD) using a specified method and reference depth.

**Example:**

.. code-block:: r

    ctd_data$add_mld(reference = 10, method = "potential_density_avg", delta = 0.05)
    output <- ctd_data$get_df()
    print(output$head(3))

    # Output will now include a new column for MLD, calculated based on the given parameters

Saving Your Processed Data
--------------------------

Once you’ve cleaned and analyzed your data, you can save the results to a CSV file.

**Example:**

.. code-block:: r

    ctd_data$save_to_csv('processed_ctd_data.csv', null_value = "NA")

    # The processed data is saved to 'processed_ctd_data.csv', with null values represented as 'NA'

Conclusion
----------

This tutorial has walked you through the key functionalities of the CTD class in CTDFjorder and how to use it within R using the ``reticulate`` package. With these tools, you can load, clean, analyze, and save CTD data efficiently for your oceanographic studies.

For detailed information on using the CTD class and other functionalities, refer to the :doc:`./API/index` reference.