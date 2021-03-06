Processing Video Files
======================

Prepare video processing directory
----------------------------------
Place :download:`trackHead.py<../../trackHead.py>` into a folder that contains the timestamps.csv and .mpg files that you want processed

Execute tracking script from the command line
---------------------------------------------
#. Open a terminal window
#. :ref:`Switch into your trackingEnv<switch>` if you are not already in it
#. cd into your video processing directory
#. Run the trackHead.py script by entering the following:

.. code::

  python trackHead.py

Resulting Output
----------------

Annotated footage
`````````````````

All of the separate .mpg files will be combined into a single .mp4 that contains 3 frames stiched together.

.. image:: mp4_output.png
  :align: center
  :width: 100 %

The left frame is the original video footage. The center frame shows the pixels that remain after filtering
for red and blue. The right frame places circlular marks at the centers of the filtered pixel clusters.

Tracked coordinates data
````````````````````````
A .csv file that combines the timestamp.csv data with x,y coorindinates of the tracked LEDs.

.. image:: csv.png
  :align: center
  :width: 100 %
