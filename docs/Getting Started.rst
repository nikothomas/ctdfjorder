Getting Started
===============
.. role:: console(code)
   :language: console

The following tutorial is for CLI users, if you are a developer using the library then you'll want to read
the :doc:`~API` documentation.

Installation
------------

We recommend using a conda environment just for CTDFjorder.
To do this, open your terminal (MacOS/Linux) or command prompt (Windows) and run the following commands:

.. code-block:: console

   $ conda deactivate
   $ conda create --name ctdfjorder python=3.12
   $ conda activate ctdfjorder


To use CTDFjorder, first install it using pip:

.. code-block:: console

   (ctdfjorder) $ pip install ctdfjorder

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
    To see what options you have to process the files, type :console:`ctdcli default -h` or view the documentation for the :doc:`CLI`.

For the purposes of this demo are assuming that you have the following:

* Files with endings **.rsk** from an RBR instrument or **.csv** from a Castaway device.
* A master sheet named **mastersheet.csv** in the same folder as your CTD data, this will be used to attach metadata to the CTD tables.
* Access to a public `MapBox <https://docs.mapbox.com/help/getting-started/access-tokens/>`_ token.

If you meet those conditions make your terminal window fullscreen.
Then copy and paste the following into your terminal, and replace :code:`MY_TOKEN` with your public MapBox token.

.. tip::
    Are you a member of Fjord Phyto? If so run command this instead :console:`ctdcli fjord -t MY_TOKEN`

.. code-block:: console

   (ctdfjorder) $ ctdcli default -r -p -m "mastersheet.csv" -w 4 --show-status --token MY_TOKEN

Interpret output
----------------
If you see a spinning globe you did it! Once the files are done processing a table will print with pipeline information
for each file. Green means the file passed a step, yellow alerts you to unusual data, and red means an error occurred
such that the file could not continue to be processed. Once all files are completed, a map will open as well.
The points are individual casts. The map can be filtered.

* Plots are in the **ctdplots** folder next to our original data.
* There you will also find a **ctdfjorder_data.csv** with our processed data.
* To investigate files that did not pass the pipeline open the **ctdfjorder.log** file.

Congrats! You can now use CTDFjorder to investigate your ctd data.

CLI Commands
=============
.. argparse::
    :filename: ../ctdfjorder/ctdfjorder/cli/cli.py
    :func: build_parser_docs
    :prog: sample

.. toctree::
    :maxdepth: 0
    :hidden: