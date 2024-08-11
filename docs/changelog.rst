*************
**Changelog**
*************

.. towncrier release notes start

CTDFjorder 0.7.4 (2024-08-11)
=============================

Bugfixes
^^^^^^^^

- Changed rsk file used for testing to fix warnings about absolute salinity calculation


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Removed grayskull dev dependency
- Removed openpyxl dependency
- Removed poetry2conda in pyproject configuration
- Removed scar database support


Miscellaneous
^^^^^^^^^^^^^

- Changed torch dependency to cpu only on non-MacOS platforms


CTDFjorder 0.7.2 (2024-08-08)
=============================

Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Made optional 'phyto' package for FjordPhyto specific modules
