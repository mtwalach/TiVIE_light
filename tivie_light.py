# This program processes an interval in time and picks the relevant maps for TS18,
#        mappotential (if available) and TiVIE.
#       It also reads in TiVIE maps and plots them.
#       The plotting can be done in two ways: 1) A timeseries, with an overview of the interval (see TiVIE paper by Walach & Grocott, [submitted to Space Weather, 2024])
#           and the TiVIE model as a series of convection maps.
#
# TiVIE Light v 1.0
# Written by Maria-Theresia Walach, CC-BY, January 2023.
# Packages:
import glob
import datetime
import xarray as xr
import os

from tivie_light_functions import make_date_time
from tivie_light_functions import get_sw_timeseries_mode1
from tivie_light_functions import get_sw_timeseries_mode2
from tivie_light_functions import get_sw_timeseries_mode3
from tivie_light_functions import prep_mode1
from tivie_light_functions import prep_mode2
from tivie_light_functions import prep_mode3
from tivie_light_functions import get_files
from tivie_light_functions import read_files
from tivie_light_functions import calc_potential_for_entire_set
from tivie_light_functions import make_plot
from tivie_light_functions import plot_maps


import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

tivie_archive = config["filepaths"]["tivie_archive"]
sw_archive = config["filepaths"]["sw_archive"]
substorm_archive = config["filepaths"]["substorm_archive"]
storm_archive = config["filepaths"]["storm_archive"]
target_folder = config["filepaths"]["target_folder"]
output_folder = config["filepaths"]["output_folder"]

highlight_mode = config["plotting"]["highlight_mode"]

start_datet = config["date_time_variables"]["start_datet"]
end_datet = config["date_time_variables"]["end_datet"]

###################################
#### Other definitions: ###########
###################################
# define some things for plotting:
deg = "\N{DEGREE SIGN}"

###################################
###################################
######## Get files ready ###################
###################################
cadence = 10
date_time = make_date_time(start_datet, end_datet, cadence)

# # Process TiVIE modes 1, 2 and 3:
omni, clock_angle_bin, tau_bin, imf_bin = get_sw_timeseries_mode1(
    start_datet, end_datet, sw_archive
)
sub_timeseries, mlat_timeseries, onset_timings = get_sw_timeseries_mode2(
    start_datet, end_datet, omni, sw_archive, substorm_archive
)
storm_phases, storm_timeseries = get_sw_timeseries_mode3(
    start_datet, end_datet, omni, storm_archive, sw_archive
)
mode_1 = prep_mode1(clock_angle_bin, tau_bin, imf_bin)
mode_2 = prep_mode2(sub_timeseries, mlat_timeseries)
mode_3 = prep_mode3(storm_phases, storm_timeseries)

missing_files3, missing_n3 = get_files(
    3, mode_3[1:], target_folder, tivie_archive, cadence
)
missing_files2, missing_n2 = get_files(
    2, mode_2[1:], target_folder, tivie_archive, cadence
)
missing_files1, missing_n1 = get_files(
    1, mode_1[1:], target_folder, tivie_archive, cadence
)
#################################
######## Mode 1 #################
#################################
# # Read in mode 1:
file_folder = target_folder + "mode1/"
ds = read_files(file_folder, date_time, missing_n1)
ds_1_err = 0

if isinstance(ds, (xr.Dataset, xr.DataArray)):
    mlat_arr, mlon_arr, potential = calc_potential_for_entire_set(ds)

    # Annex potential to ds:
    ds.update({"potential": (["time", "mlat_arr", "mlon_arr"], potential)})
    ds.coords.update({"mlat_arr": (mlat_arr), "mlon_arr": (mlon_arr)})
    ds["mlat_arr"].attrs["units"] = "degrees"
    ds["mlon_arr"].attrs["units"] = "degrees"

    # Check that the folder exists, if not, we make it:
    if glob.glob(output_folder + "/" + str(start_datet) + "_" + str(end_datet)) == []:
        os.makedirs(output_folder + "/" + str(start_datet) + "_" + str(end_datet) + "/")

    # # Write potential to ncdf file:
    ds.to_netcdf(
        output_folder
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "_mode1.nc"
    )
else:
    ds_1_err += 1

#################################
######## Mode 2 #################
# #################################
# # Read in mode 2:
file_folder = target_folder + "mode2/"
ds = read_files(file_folder, date_time, missing_n2)
ds_2_err = 0

if isinstance(ds, (xr.Dataset, xr.DataArray)):
    mlat_arr, mlon_arr, potential = calc_potential_for_entire_set(ds)

    # Annex potential to ds:
    ds.update({"potential": (["time", "mlat_arr", "mlon_arr"], potential)})
    ds.coords.update({"mlat_arr": (mlat_arr), "mlon_arr": (mlon_arr)})
    ds["mlat_arr"].attrs["units"] = "degrees"
    ds["mlon_arr"].attrs["units"] = "degrees"

    # Check that the folder exists, if not, we make it:
    if glob.glob(output_folder + "/" + str(start_datet) + "_" + str(end_datet)) == []:
        os.makedirs(output_folder + "/" + str(start_datet) + "_" + str(end_datet) + "/")

    # # Write potential to ncdf file:
    ds.to_netcdf(
        output_folder
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "_mode2.nc"
    )
else:
    ds_2_err += 1

#################################
######## Mode 3 #################
#################################
# # Read in mode 3:
file_folder = target_folder + "mode3/"
ds = read_files(file_folder, date_time, missing_n3)
ds_3_err = 0

if isinstance(ds, (xr.Dataset, xr.DataArray)):
    mlat_arr, mlon_arr, potential = calc_potential_for_entire_set(ds)

    # Annex potential to ds:
    ds.update({"potential": (["time", "mlat_arr", "mlon_arr"], potential)})
    ds.coords.update({"mlat_arr": (mlat_arr), "mlon_arr": (mlon_arr)})
    ds["mlat_arr"].attrs["units"] = "degrees"
    ds["mlon_arr"].attrs["units"] = "degrees"

    # Check that the folder exists, if not, we make it:
    if glob.glob(output_folder + "/" + str(start_datet) + "_" + str(end_datet)) == []:
        os.makedirs(output_folder + "/" + str(start_datet) + "_" + str(end_datet) + "/")

    # # Write potential to ncdf file:
    ds.to_netcdf(
        output_folder
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "_mode3.nc"
    )
else:
    ds_3_err += 1

#################################
######## Grab all the data: ####
################################
if ds_1_err == 0:
    ds1 = xr.open_dataset(
        output_folder
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "_mode1.nc"
    )
else:
    ds1 = []
if ds_2_err == 0:
    ds2 = xr.open_dataset(
        output_folder
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "_mode2.nc"
    )
else:
    ds2 = []
if ds_3_err == 0:
    ds3 = xr.open_dataset(
        output_folder
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "_mode3.nc"
    )
else:
    ds3 = []
# #################################
# ######## Plot all data as timeseries: #########
# #################################
status = make_plot(
    output_folder,
    start_datet,
    end_datet,
    onset_timings,
    highlight_mode,  # Choose which mode to highlight here: replace this with 1, 2 or 3, for 1=IMF mode; 2=substorm mode and 3=storms mode.
    omni,
    ds1=ds1,
    ds2=ds2,
    ds3=ds3,
)
# You can choose which models you want to plot here. To miss one out, simply replace with empty array, e.g. during non-substorm times, I will replace ds2=ds2 with ds2=[]
print(status)
#################################
######## Plot maps: ############
################################
status = plot_maps(output_folder, ds1, highlight_mode, start_datet, end_datet)
# put in "highlight_mode" which dataset you want to plot out
#   and which mode (choices are 1 for IMF mode, 2 for substorm mode and 3 for storms)
# Note: mode number (e.g. 3) is for setting annotations
#    and the dataset is what gets plotted so make sure they are the same!
print(status)
#################################
####### When we are done, we'll close the datasets:
#################################
if ds_1_err == 0:
    ds1.close()
if ds_2_err == 0:
    ds2.close()
if ds_3_err == 0:
    ds3.close()
