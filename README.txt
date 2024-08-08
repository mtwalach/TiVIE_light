Welcome to TiVIE light v 1.0! 

Written by Maria-Theresia Walach, GNU General Public License v3.0, August 2024.

To run TiVIE light, you need several things: 
- You need the TiVIE archive, which you can download (remember to unzip it!) from the following DOI: 10.5281/zenodo.13270754
- You need the TiVIE light code 
    (which is in this folder - congratulations!).
- You need the data provided with TiVIE light, which you can download (remember to unzip it and put it in your tivie_light folder) from the following DOI: 10.5281/zenodo.13271014
- You need a working version of python plus some packages listed below 
    (I recommend using poetry to manage these https://python-poetry.org).


The TiVIE light code takes a timespan as input and then produces plots with TiVIE outputs: 
- It will produce a timeseries (similar to what is published in the original TiVIE paper by Walach & Grocott (submitted, 2024)).
- It will produce individual convection maps of whichever TiVIE mode is chosen to be highlighted.

You can choose the highlighting mode (this also highlights) the mode in the timeseries. 
The timespan and highlighting mode can be set in the config.yaml file. 
The config_example.yaml file is not used anywhere, so you can edit config.yaml and can refer back to config_example.yaml in the future. 

Package requirements (others may work too but this is what TiVIE light was tested with): 
Python v. 3.11 (and its standard libraries e.g. datetime, glob, os etc.)
aacgmv2           2.6.3       A Python wrapper for AACGM-v2 magnetic coordinates
datetime          5.5         This package provides a DateTime data type, as known from Zope. Unless you need to communicate with Zope APIs, you're probably better off using Python...
matplotlib        3.8.4       Python plotting package
numpy             1.26.4      Fundamental package for array computing in Python
pandas            2.2.2       Powerful data structures for data analysis, time series, and statistics
pydarn            4.0         Data visualization library for SuperDARN data
pyyaml            6.0.1       YAML parser and emitter for Python
scipy             1.13.0      Fundamental algorithms for scientific computing in Python
xarray            2024.3.0    N-D labeled arrays and datasets in Python

Be aware that some of these require other prerequisite packages to be installed which you will have to check if you install them manually.
Tip: If you are using Poetry (or something similar), it handles these for you. 

Please remember to reference the work and data accordingly.