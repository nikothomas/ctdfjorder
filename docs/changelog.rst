*************
**Changelog**
*************

.. towncrier release notes start

CTDFjorder 0.8.0 (2024-08-11)
=============================

Features
^^^^^^^^

- Changed MapBox plot to use months instead of seasons for filtering


Bugfixes
^^^^^^^^

- Changed rsk file used for testing to fix warnings about absolute salinity calculation


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Removed grayskull dev dependency
- Removed openpyxl dependency
- Removed poetry2conda in pyproject configuration
- Removed scar database support


Documentation
^^^^^^^^^^^^^

- Removed sphinx workflow, moving to read the docs

Miscellaneous
^^^^^^^^^^^^^

- Changed torch dependency to cpu only on non-MacOS platforms


CTDFjorder 0.7.2 (2024-08-08)
=============================

Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Made optional 'phyto' package for FjordPhyto specific modules
