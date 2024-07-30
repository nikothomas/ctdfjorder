CTDFjorder
==========

Welcome to the **CTDFjorder** documentation! This guide will help you get started with using the CTDFjorder library,
designed to process and analyze CTD (Conductivity, Temperature, Depth) data efficiently.

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none


Introduction
------------

**CTDFjorder** is a Python package that simplifies the processing and analysis of CTD data. It supports reading data
from various file formats such as RBR (.rsk) and SonTek Castaway (.csv), and offers tools for data cleaning, filtering, and
calculation of derived oceanographic quantities like absolute salinity, density, and potential density. Additionally,
CTDFjorder includes capabilities for visualizing CTD profiles and derived quantities through advanced plotting tools.

.. raw:: html

    <div style="display: flex; justify-content: space-around; margin-top: 20px;">

    <a href="researchers.html" style="text-decoration: none;">
        <button style="font-size: 20px; padding: 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; border-radius: 5px;">
            Researchers Documentation
        </button>
    </a>

    <a href="developers.html" style="text-decoration: none;">
        <button style="font-size: 20px; padding: 20px; background-color: #2196F3; color: white; border: none; cursor: pointer; border-radius: 5px;">
            Developers Documentation
        </button>
    </a>

    </div>

Documentation for Researchers
-----------------------------

If you are a researcher looking to use the command-line interface (CLI) for processing and analyzing CTD data, please refer to the :doc:`researchers` page. It covers key functionalities and methods available through the CLI, providing a streamlined approach for data handling without needing extensive programming knowledge.

Documentation for Developers
-----------------------------

If you are a developer interested in interacting with the API, please refer to the :doc:`developers` page. It provides detailed information on the :class:`~ctdfjorder.CTD.CTD` class, including its methods and how to utilize them for loading, processing, analyzing, and visualizing CTD data.

Issues
-------
If you encounter an issue while running or using CTDFjorder, please submit a report on
the `Issues <https://github.com/nikothomas/ctdfjorder/issues>` page. This is the fastest way to get your problem
addressed.

License
-------

**CTDFjorder** is released under the MIT License.

Acknowledgments
---------------

**CTDFjorder** was developed for the Fjord Phyto project. The gsw library was used for certain derived calculations.

References
-----------
.. [PaVR19] Pan, B.J.; Vernet, M.; Reynolds, R.A.; Mitchell, B.G.: The optical and biological properties of glacial meltwater in an Antarctic fjord. PLOS ONE 14(2): e0211107 (2019). https://doi.org/10.1371/journal.pone.0211107

.. [McBa11] McDougall, T. J.; Barker, P. M.: Getting started with TEOS-10 and the Gibbs Seawater (GSW) Oceanographic Toolbox. SCOR/IAPSO WG127 (2011).

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   Getting Started
   researchers
   developers
   API