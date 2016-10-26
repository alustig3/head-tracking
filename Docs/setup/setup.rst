Installation and Setup
======================

Install the Python 3.5 version of Anaconda
------------------------------------------

`Donwload Here <https://www.continuum.io/downloads>`_

Create a new virtual environment
--------------------------------

Open a terminal session and enter the following:

.. code::

  conda create -n trackingEnv python=3.5 numpy pandas

This will create a python 3.5 environment called "trackingEnv" with numpy and pandas installed.

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

Install OpenCV 3.1 in your environment
--------------------------------------

.. code::

  conda install -c menpo opencv3=3.1.0
