Getting Started
===============

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


Processing Files in the Command Line
====================================
Files can be processed without needing to write your own script by using the built in command line tool described in
this tutorial.

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
    To see what options you have to process the files, type :code:`ctdcli default -h` or view the documentation for the :doc:`CLI`.

For the purposes of this demo are assuming that you have the following:

* Files with endings **.rsk** from an RBR instrument or **.csv** from a Castaway device.
* A master sheet named **FjordPhyto MASTER SHEET.csv** in the same folder as your CTD data, this will be used to attach metadata to the CTD tables.
* Access to a `MapBox <https://docs.mapbox.com/help/getting-started/access-tokens/>`_ token.

If you meet those conditions make your terminal window fullscreen.
Then copy and paste the following into your terminal, and replace :code:`MY_TOKEN` with your MapBox token.

.. code-block:: console

   (ctdfjorder) $ ctdcli default -r -p -m "FjordPhyto MASTER SHEET.csv" -w 4 --show-table --token MY_TOKEN

Interpret output
----------------
If you see a spinning globe you did it! Once the files are done processing a table will print with pipeline information
for each file. Green means the file passed a step, yellow alerts you to unusual data, and red means an error occurred
such that the file could not continue to be processed. Once all files are completed, a map will open as well.
The points are individual casts. The map can be filtered.

.. tip::
    Plots are in the **ctdplots** folder in the same place as your original CTD data.
    There is also an **output.csv** file with the processed data in the same folder.
    To investigate files that did not pass the pipeline open the **ctdfjorder.log** file.

Congrats! You can now use CTDFjorder to investigate your ctd data.

Using the Library
-----------------
If you'd like to use CTDFjorder to develop in python see the :doc:`API` documentation.