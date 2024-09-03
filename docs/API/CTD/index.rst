CTD
====

The CTD class provides a comprehensive interface for loading, processing, and analyzing CTD (Conductivity, Temperature, Depth) data. Below is a detailed breakdown of the key methods and functionalities provided by the CTD class.

.. autoclass:: ctdfjorder.CTD.CTD
    :special-members:

Data Handling
-------------
Methods for loading, accessing, and ensuring data integrity.

* :py:meth:`~ctdfjorder.CTD.CTD.get_df`
* :py:meth:`~ctdfjorder.CTD.CTD.assert_data_not_empty`
* :py:meth:`~ctdfjorder.CTD.CTD.add_metadata`
* :py:meth:`~ctdfjorder.CTD.CTD.expand_date`

Data Cleaning
-------------
Methods to clean and prepare CTD data for analysis.

* :py:meth:`~ctdfjorder.CTD.CTD.filter_columns_by_range`
* :py:meth:`~ctdfjorder.CTD.CTD.remove_non_positive_samples`
* :py:meth:`~ctdfjorder.CTD.CTD.remove_upcasts`
* :py:meth:`~ctdfjorder.CTD.CTD.clean`

Derived Calculations
---------------------
Methods to calculate additional parameters and metrics from the raw CTD data.

* :py:meth:`~ctdfjorder.CTD.CTD.add_absolute_salinity`
* :py:meth:`~ctdfjorder.CTD.CTD.add_density`
* :py:meth:`~ctdfjorder.CTD.CTD.add_potential_density`
* :py:meth:`~ctdfjorder.CTD.CTD.add_mean_surface_density`
* :py:meth:`~ctdfjorder.CTD.CTD.add_surface_salinity`
* :py:meth:`~ctdfjorder.CTD.CTD.add_surface_temperature`
* :py:meth:`~ctdfjorder.CTD.CTD.add_meltwater_fraction`
* :py:meth:`~ctdfjorder.CTD.CTD.add_mld`
* :py:meth:`~ctdfjorder.CTD.CTD.add_n_squared`
* :py:meth:`~ctdfjorder.CTD.CTD.add_potential_temperature`
* :py:meth:`~ctdfjorder.CTD.CTD.add_practical_salinity`
* :py:meth:`~ctdfjorder.CTD.CTD.add_conservative_temperature`
* :py:meth:`~ctdfjorder.CTD.CTD.add_dynamic_height`
* :py:meth:`~ctdfjorder.CTD.CTD.add_thermal_expansion_coefficient`
* :py:meth:`~ctdfjorder.CTD.CTD.add_haline_contraction_coefficient`
* :py:meth:`~ctdfjorder.CTD.CTD.add_speed_of_sound`
* :py:meth:`~ctdfjorder.CTD.CTD.add_profile_classification`

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