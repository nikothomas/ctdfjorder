CTDFjorder
==========

Welcome to the CTDFjorder documentation! This guide will help you get started with using the CTDFjorder library,
designed to process and analyze CTD (Conductivity, Temperature, Depth) data efficiently.

.. |fa-globe| raw:: html

    <i class="fas fa-globe"></i>

.. |fa-bolt| raw:: html

   <i class="fas fa-bolt"></i>

.. |fa-brain| raw:: html

   <i class="fas fa-brain"></i>

.. |fa-terminal| raw:: html

   <i class="fas fa-terminal"></i>

.. |fa-r| raw:: html

    <i class="fa-brands fa-r-project"></i>

.. |fa-lightbulb| raw:: html

   <i class="fas fa-lightbulb"></i>

.. grid:: 1 1 3 3
    :gutter: 3
    :margin: 0
    :padding: 0

    .. grid-item-card::

       |fa-globe| Agnostic Inputs
       ^^^
       Supports `RBR <https://rbr-global.com>`_ and `SonTek Castaway <https://www.xylem.com/en-us/products--services/analytical-instruments-and-equipment/data-collection-mapping-profiling-survey-systems/ctds/castaway-ctd/>`_ files with broader support coming soon.


    .. grid-item-card::

        |fa-bolt| Rapid Reads
        ^^^
        Uses polars as the data manipulation library for high throughput.


    .. grid-item-card::

        |fa-brain| AI Analytics
        ^^^
        Implements cutting edge small footprint AI models for data cleaning.


    .. grid-item-card::

        |fa-terminal| CLI Interface
        ^^^
        Provides a well-documented command line interface for ease of use and multicore processing when handling numerous files.


    .. grid-item-card::

        |fa-r| Cross-Language Compatibility
        ^^^
        Can be implemented in R scripts with no functionality lost.


    .. grid-item-card::

        |fa-lightbulb| Open Source
        ^^^
        Moving away from proprietary pipelines provides more insight into analysis, and greater assurance in results.


.. raw:: html

    <div style="display: flex; justify-content: space-around; margin-top: 60px; margin-bottom: 80px; flex-wrap: wrap; gap: 20px;">

    <a href="researchers.html" style="text-decoration: none; flex: 1 1 45%; max-width: 45%;">
        <button style="width: 100%; font-size: 1.5vw; padding: 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; border-radius: 5px;">
            Researchers Documentation
        </button>
    </a>

    <a href="developers.html" style="text-decoration: none; flex: 1 1 45%; max-width: 45%;">
        <button style="width: 100%; font-size: 1.5vw; padding: 20px; background-color: #2196F3; color: white; border: none; cursor: pointer; border-radius: 5px;">
            Developers Documentation
        </button>
    </a>

    </div>

Documentation for Researchers
-----------------------------

If you are a researcher looking to use the command-line interface (CLI) for processing and analyzing CTD data, please refer to the :doc:`researchers` page. It covers key functionalities and methods available through the CLI, providing a streamlined approach for data handling without needing extensive programming knowledge.

Documentation for Developers
-----------------------------

If you are a developer interested in interacting with the API, please refer to the :doc:`developers` page. It will help you get started with the library in Python or R.

Issues
-------

If you encounter an issue while running or using CTDFjorder, please submit a report on
the `Issues <https://github.com/nikothomas/ctdfjorder/issues>`__ page. This is the fastest way to get your problem
addressed.

License
-------

CTDFjorder is released under the MIT License.

Citing CTDFjorder
-----------------

Citation information for CTDFjorder can be found here `zenodo.13293665 <https://dx.doi.org10/zenodo.13293665>`__.

Acknowledgments
---------------

CTDFjorder was developed for the Fjord Phyto project. The `GSW-Python <https://github.com/TEOS-10/GSW-Python>` library was used for certain derived calculations.

References
-----------

.. [PaVR19] Pan, B.J.; Vernet, M.; Reynolds, R.A.; Mitchell, B.G.: The optical and biological properties of glacial meltwater in an Antarctic fjord. PLOS ONE 14(2): e0211107 (2019). https://doi.org/10.1371/journal.pone.0211107

.. [McBa11] McDougall, T. J.; Barker, P. M.: Getting started with TEOS-10 and the Gibbs Seawater (GSW) Oceanographic Toolbox. SCOR/IAPSO WG127 (2011).

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   researchers
   developers
   API/index
   cli-tutorial
   changelog