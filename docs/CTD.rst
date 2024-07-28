CTD
===

The :class:`~ctdfjorder.CTD.CTD` class in the **CTDFjorder** library provides a comprehensive interface for loading, processing, and analyzing CTD data. Below is a detailed breakdown of the key methods and functionalities provided by the :class:`~ctdfjorder.CTD.CTD` class.

Initialization
---------------
.. autoclass:: ctdfjorder.CTD.CTD
    :private-members:

Data Processing
----------------

.. automethod:: ctdfjorder.CTD.CTD.remove_upcasts

.. automethod:: ctdfjorder.CTD.CTD.filter_columns_by_range

.. automethod:: ctdfjorder.CTD.CTD.remove_non_positive_samples

.. automethod:: ctdfjorder.CTD.CTD.remove_invalid_salinity_values

.. automethod:: ctdfjorder.CTD.CTD.clean

Derived Calculations
---------------------

.. automethod:: ctdfjorder.CTD.CTD.add_absolute_salinity

.. automethod:: ctdfjorder.CTD.CTD.add_density

.. automethod:: ctdfjorder.CTD.CTD.add_potential_density

.. automethod:: ctdfjorder.CTD.CTD.add_surface_salinity_temp_meltwater

.. automethod:: ctdfjorder.CTD.CTD.add_mean_surface_density

.. automethod:: ctdfjorder.CTD.CTD.add_mld

.. automethod:: ctdfjorder.CTD.CTD.add_bf_squared

Data Export
------------

.. automethod:: ctdfjorder.CTD.CTD.get_df

.. automethod:: ctdfjorder.CTD.CTD.save_to_csv
