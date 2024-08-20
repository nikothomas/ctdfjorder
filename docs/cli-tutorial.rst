CLI Tutorial
=============
.. role:: console(code)
   :language: console
.. role:: python(code)
   :language: python

The following tutorial is for CLI users, if you are using the library to write your own Python or R scripts then you'll want to read
:doc:`developers`.

Installation
------------

We recommend using a conda environment just for CTDFjorder.
To do this, open your terminal (MacOS/Linux) or command prompt (Windows) and run the following commands:

.. code-block:: console

   $ conda deactivate
   $ conda create --name ctdfjorder python=3.12
   $ conda activate ctdfjorder

Then install CTDFjorder using pip:

.. code-block:: console

   (ctdfjorder)$ pip install ctdfjorder

Navigate to your files
----------------------

To process CTD files you must first navigate to the directory with the ctd files you want to process.
Our CTD files are in a folder called **CTD Data** on our desktop:

.. tip::
    For the next step you can type :code:`cd` with a space afterwards (the space is important) and then drag the folder
    into your terminal. Then press :code:`enter` or :code:`return` to execute the command.

.. code-block:: console

   (ctdfjorder) $ cd "/Users/nik/Desktop/CTD Data"

Run ctdcli
----------

Now we will process our files.

.. tip::
    To see what options you have to process the files, type :console:`ctdcli default -h` or view the documentation for the :doc:`/API/CLI/index`.

For the purposes of this demo are assuming that you have the following:

* Files with endings **.rsk** from an RBR instrument or **.csv** from a Castaway device.
* A master sheet which will be used to attach metadata to the CTD tables. This must be named mastersheet.csv and be located in the same folder as your CTD data. Additionally it must have the following fields:
   * UNIQUE ID CODE
   * nominal longitude
   * nominal latitude
   * CTD cast file name
   * location
   * loc id
   * date/time (ISO)
   * sechhi depth

* Access to a public `MapBox <https://docs.mapbox.com/help/getting-started/access-tokens/>`_ token.

If you meet those conditions make your terminal window fullscreen.
Then copy and paste the following into your terminal, and replace :code:`MY_TOKEN` with your public MapBox token.
Members of FjordPhyto can use this token :code:`pk.eyJ1Ijoibmlrb3Rob21hcyIsImEiOiJjbHl2Z2JzbDQxZjEwMmpwd2c1cnJpYmRyIn0.j9l0EXWa2ik51AbAcIe5HQ`

.. tip::

    Add plotting by including :code:`-p` in the command, like so :code:`ctdcli default -r -p --token MY_TOKEN`

.. code-block:: console

   (ctdfjorder) $ ctdcli default -r --token MY_TOKEN

Interpret output
----------------
If you see a spinning globe you did it! Once the files are done processing a table will print with pipeline information
for each file. Green means the file passed a step and red means an error occurred such that the file could not continue to be processed.
Once all files are completed, a map will open as well. The points are individual casts. The map can be filtered.

* Plots are in the **ctdplots** folder next to our original data and were made with functions from the :doc:`./API/Visualize/index`
  module.
* There you will also find a **ctdfjorder_data.csv** with our processed data.
* To investigate files that did not pass the pipeline open the **ctdfjorder.log** file.

Steps
-----
These are the functions we ran through the CLI on each file in this tutorial:

.. code-block:: python

    data = CTD(file)
    data.expand_date(day=False)
    data.remove_upcasts()
    data.remove_non_positive_samples()
    data.filter_columns_by_range(column='salinity', upper_bound=None, lower_bound=10)
    data.add_metadata(master_sheet_path='mastersheet.csv')
    data.clean(method='clean_salinity_ai')
    data.add_surface_salinity()
    data.add_surface_temperature()
    data.add_meltwater_fraction()
    data.add_absolute_salinity()
    data.add_density()
    data.add_potential_density()
    data.add_n_squared()
    data.add_mld_bf()
    data.add_profile_classification()

Congrats! You can now use CTDFjorder to investigate your ctd data. For more in depth information on the processes
executed here, read the :doc:`./API/index`.

CLI Commands
=============
.. argparse::
    :module: ctdfjorder.cli.cli
    :func: build_parser_docs
    :prog: sample

.. toctree::
    :maxdepth: 0
    :hidden: