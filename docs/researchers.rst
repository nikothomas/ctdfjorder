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

If you're new to CTDFjorder and want to use the command-line interface, we recommend starting with the :doc:`cli-tutorial`. It provides a
comprehensive introduction to the CLI, including installation instructions, basic usage examples,
and an overview of the key features. The CLI simplifies many common tasks and allows you to process data without needing to write Python code.
To read in depth about the functions used in the CLI go to the :doc:`/API/index` documentation.
If you'd prefer to create scripts in Python or R using CTDFjorder then visit the
:doc:`developers` page.

Command-Line Interface (CLI)
----------------------------


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