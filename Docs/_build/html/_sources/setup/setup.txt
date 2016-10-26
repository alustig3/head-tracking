Installation and Setup
======================

Install the Python 3.5 version of Anaconda
------------------------------------------

`Donwload Here <https://www.continuum.io/downloads>`_

Create a new virtual environment
--------------------------------

Open a terminal window and enter the following:

.. code::

  conda create -n trackingEnv python=3.5 numpy pandas

This will create a Python 3.5 environment called "trackingEnv" with the numpy and pandas libraries installed.

.. _switch:

Switch into your new environment
--------------------------------

Mac
+++

.. code::

  source activate trackingEnv

Windows
+++++++

.. code::

  activate trackingEnv

``(trackingEnv)`` should now be shown at the start of the command line.

Install OpenCV 3.1 in your environment
--------------------------------------

.. code::

  conda install -c menpo opencv3=3.1.0
