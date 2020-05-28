Installation and Setup
======================

Install the Python 3.7 version of Anaconda
------------------------------------------

`Download Here <https://www.anaconda.com/distribution/>`_

Create a new virtual environment
--------------------------------

Open a terminal window and enter the following:

.. code::

  conda create -n trackingEnv python=3.8 numpy pandas

This will create a Python 3.8 environment called "trackingEnv" with the numpy and pandas libraries installed.

.. _switch:

Switch into your new environment
--------------------------------

.. code::

  conda activate trackingEnv

``(trackingEnv)`` should now be shown at the start of the command line.

Install OpenCV 4.2 in your environment
--------------------------------------

.. code::

  conda install -c conda-forge opencv=4.2.0
