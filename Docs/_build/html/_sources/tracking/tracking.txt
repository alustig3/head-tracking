Process video files
===================

1. Place :download:`trackHead.py<../../trackHead.py>` in a folder with .mpg files and timestamps.csv
2. Open terminal session, navigate to the folder with the video files and tracking script, and execute the following:

.. code::

  python trackHead.py

All of the separate .mpg files will be combined into a single .mp4 that contains 3 frames stiched together.

.. image:: mp4_output.png
  :align: center
  :scale: 100 %

The left frame is the original video footage.The center frame shows the pixels that remain after filtering
for red and blue. The right frame places circlular marks at the centers of the filtered pixel clusters.

A .csv file is also created that combines the timestamp.csv data with x,y coorindinates of the tracked LEDs

.. image:: csv.png
  :align: center
  :scale: 100 %
