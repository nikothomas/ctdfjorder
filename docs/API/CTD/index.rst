CTD
====

The CTD class provides a comprehensive interface for loading, processing, and analyzing CTD (Conductivity, Temperature, Depth) data. Below is a detailed breakdown of the key methods and functionalities provided by the CTD class.

.. autoclass:: ctdfjorder.CTD.CTD
    :special-members:

Data Handling
-------------
Methods for loading, accessing, and ensuring data integrity.

* :doc:`get_df <./Data Handling/get_df>`
* :doc:`assert_data_not_empty <./Data Handling/assert_data_not_empty>`
* :doc:`add_metadata <./Data Handling/add_metadata>`
* :doc:`expand_date <./Data Handling/expand_date>`

Data Cleaning
-------------
Methods to clean and prepare CTD data for analysis.

* :method:`filter_columns_by_range <./Data Cleaning/filter_columns_by_range>`
* :doc:`remove_non_positive_samples <./Data Cleaning/remove_non_positive_samples>`
* :doc:`remove_upcasts <./Data Cleaning/remove_upcasts>`
* :doc:`clean <./Data Cleaning/clean>`

Derived Calculations
---------------------
Methods to calculate additional parameters and metrics from the raw CTD data.

* :doc:`add_absolute_salinity <./Derived Calculations/add_absolute_salinity>`
* :doc:`add_density <./Derived Calculations/add_density>`
* :doc:`add_potential_density <./Derived Calculations/add_potential_density>`
* :doc:`add_surface_salinity_temp_meltwater <./Derived Calculations/add_surface_salinity_temp_meltwater>`
* :doc:`add_mean_surface_density <./Derived Calculations/add_mean_surface_density>`
* :doc:`add_mld <./Derived Calculations/add_mld>`
* :doc:`add_brunt_vaisala_squared <./Derived Calculations/add_brunt_vaisala_squared>`
* :doc:`add_potential_temperature <./Derived Calculations/add_potential_temperature>`
* :doc:`add_conservative_temperature <./Derived Calculations/add_conservative_temperature>`
* :doc:`add_dynamic_height <./Derived Calculations/add_dynamic_height>`
* :doc:`add_thermal_expansion_coefficient <./Derived Calculations/add_thermal_expansion_coefficient>`
* :doc:`add_haline_contraction_coefficient <./Derived Calculations/add_haline_contraction_coefficient>`
* :doc:`add_speed_of_sound <./Derived Calculations/add_speed_of_sound>`
* :doc:`add_profile_classification <./Derived Calculations/add_profile_classification>`
* :doc:`calculate_salinity_olf_mld <./Derived Calculations/calculate_salinity_olf_mld>`

Exporting Data
--------------
Methods for exporting CTD data for further analysis or sharing.

* :doc:`save_to_csv <./Exporting Data/save_to_csv>`

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   ./Data Handling/index
   ./Data Cleaning/index
   ./Derived Calculations/index
   ./Exporting Data/index