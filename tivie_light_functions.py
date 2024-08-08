# This collection of functions is needed to run tivie_light
# TiVIE Light v 1.0
# Written by Maria-Theresia Walach, GNU General Public License v3.0, June 2024.

# Packages:
import glob
import numpy as np
import matplotlib.pyplot as plt
import aacgmv2
import shutil
from matplotlib.dates import DateFormatter
from scipy.special import lpmn
from scipy.io import readsav
import datetime
import pydarn
import xarray as xr
import pandas as pd
import os
import subprocess
import math

###################################
#### Other definitions: ###########
###################################
# define some things for plotting:
deg = "\N{DEGREE SIGN}"
###################################


# Define functions we need to calculate Electric field from SD:
###################################
def norm_theta(theta, thetamax):
    # FUNCTION  NORM_THETA
    #
    # Adapted from IDL to python by Maria-Theresia Walach, October 2021
    theta_limit = np.pi
    alpha = theta_limit / thetamax
    theta_prime = alpha * theta
    return theta_prime


###################################
def index_legendre(l, m):
    if m > l:
        k = -1
    if l == 0:
        k = 0
    else:
        if m == 0:
            k = l * l
        else:
            k = l * l + 2 * m - 1
    return k


###################################
def eval_potential(a, plm, phi):
    #   where 'a' is the set of coefficients given as a vector
    #               indexed by k = index_legendre(l,m,/dbl)
    # Note by Maria (October 2019): a are the coefficients from the map fitting
    #     plm is an array (N,Lmax,Lmax) where N = number of points
    #         where the potential is to be evaluated, and the
    #         other two indices give the associated Legendre polynomial
    #         to be evaluated at each position.
    #     phi is the azimuthal coordinate at each evaluation point (also of N length)
    lmax = len(plm)
    # phi needs to be in radians!
    pot = 0.0
    for m in range(0, lmax):
        for l in range(m, lmax):
            k = index_legendre(l, m)
            if m == 0:
                pot = pot + a[k] * plm[0, l]
            else:
                pot = (
                    pot
                    + a[k] * np.cos(m * phi) * plm[m, l]
                    + a[k + 1] * np.sin(m * phi) * plm[m, l]
                )
    return pot


###################################
def calc_pot(phi, theta, latmin, L, M, coeffs):
    # Note: L are the order values and M are the degree values for the coeffs
    # First, make sure every lat is a colat and every degree value is in radians:
    tmax = np.deg2rad(90.0 - latmin)
    cotheta = np.deg2rad(90.0 - theta)
    phi = np.deg2rad(phi)
    if theta >= latmin:
        tprime = norm_theta(cotheta, tmax)
        x = np.cos(tprime)
        plm = lpmn(L.max(), M.max(), x)
        plm = plm[0]  # we only need the values, not their derivatives
        pot = eval_potential(coeffs, plm, phi)
        pot = pot
    else:
        pot = 0.0
    return pot


###################################
def add_grids(rows, cols, n, axis_name, title):
    deg = "\N{DEGREE SIGN}"
    axis_name = plt.subplot(rows, cols, n, projection="polar")
    # polar means plotting has to be in coords [theta, r]
    # Add some gridlines:
    axis_name.set_theta_offset((3 * np.pi) / 2)
    for r in range(0, 24):
        axis_name.plot(
            [(r * np.pi) / 12, (r * np.pi) / 12],
            [50, 0],
            color="silver",
            linestyle=":",
            linewidth=1,
        )
    axis_name.xaxis.grid(True, color="silver", linestyle=":", linewidth=1)
    axis_name.yaxis.grid(True, color="silver", linestyle=":", linewidth=1)
    axis_name.set_rlim(0, 50)
    axis_name.axes.get_xaxis().set_ticks([0, (np.pi) / 2, (np.pi), (3 * np.pi) / 2])
    axis_name.set_xticklabels(["00 MLT", "06", "12 MLT", "18"])
    axis_name.axes.get_yaxis().set_ticks([10, 20, 30, 40, 50])
    axis_name.set_yticklabels(
        ["80" + deg, "70" + deg, "60" + deg, "50" + deg, "40" + deg]
    )
    axis_name.set_title(title)
    return axis_name


###################################
###################################
def make_date_time(start_datet, end_datet, cadence):
    start_year = int(start_datet[0:4])
    start_month = int(start_datet[4:6])
    start_day = int(start_datet[6:8])
    start_hh = int(start_datet[8:10])
    start_mm = int(start_datet[10:12])
    end_year = int(end_datet[0:4])
    end_month = int(end_datet[4:6])
    end_day = int(end_datet[6:8])
    end_hh = int(end_datet[8:10])
    end_mm = int(end_datet[10:12])

    start_ = datetime.datetime(
        year=start_year,
        month=start_month,
        day=start_day,
        hour=start_hh,
        minute=start_mm,
    )
    end_ = datetime.datetime(
        year=end_year, month=end_month, day=end_day, hour=end_hh, minute=end_mm
    )

    timedelta = end_ - start_
    timedelta_cadence = datetime.timedelta(minutes=cadence)

    dt = []
    for i in range(0, int(timedelta / timedelta_cadence)):
        date_time = [start_ + (timedelta_cadence) * i]
        dt += date_time

    return dt


###################################
###################################
# Let's find out which tivie files we need for the timeseries:
def get_sw_timeseries_mode1(start_datet, end_datet, sw_archive):
    start_year = int(start_datet[0:4])
    start_month = int(start_datet[4:6])
    start_day = int(start_datet[6:8])
    start_hh = int(start_datet[8:10])
    start_mm = int(start_datet[10:12])
    end_year = int(end_datet[0:4])
    end_month = int(end_datet[4:6])
    end_day = int(end_datet[6:8])
    end_hh = int(end_datet[8:10])
    end_mm = int(end_datet[10:12])

    years = start_year
    if start_year << end_year:
        years = np.append(years, np.arange(0, end_year - start_year) + start_year)
    year_str = ""
    for i in range(0, len(years)):
        year_str += str(years[i])

    # find SW data for "years":
    sw_files = glob.glob(sw_archive + "omni_" + year_str + "*.sav")
    sw_files.sort()
    print("Found files:", sw_files)

    # read in sw data:
    if len(sw_files) > 1:
        print(
            "There is an error here because Maria did not write any code for what happens in the instance where more than one file is found..."
        )
    else:
        print("Reading in data...")
        sav_data = readsav(sw_files[0])
        print("Finished reading!")

    omni = sav_data["omni"]

    # chop off bits of timeseries that don't need:
    start_idx = (
        (omni.year == start_year)
        & (omni.month == start_month)
        & (omni.day == start_day)
        & (omni.hour == start_hh)
        & (omni.minute == start_mm)
    )
    end_idx = (
        (omni.year == end_year)
        & (omni.month == end_month)
        & (omni.day == end_day)
        & (omni.hour == end_hh)
        & (omni.minute == end_mm)
    )
    omni = omni[
        (np.nonzero(start_idx)[0][0] - 362) : np.nonzero(end_idx)[0][0] + 1
    ]  # Need a bit earlier to work out tau bin: Note max tau length is > 360 mins.

    # Set TiVIE IMF Bins:
    tau_bins = np.zeros([9, 2])
    tau_bins[:, 0] = [20, 30, 40, 60, 90, 120, 180, 240, 360]
    tau_bins[:, 1] = [
        30,
        40,
        60,
        90,
        120,
        180,
        240,
        360,
        10e3,
    ]  # time bins in minutes
    theta_bins = np.zeros([16, 3])  # clock angle bin boundaries in degrees
    # clock angle bin boundaries in degrees, keeping in mind that bin 0 is from theta[-1] to 180 and -180 to theta[0]
    for i in range(1, 17):
        theta_bins[i - 1, :] = [i, 22.5 * (i - 9) - 25, 22.5 * (i - 9) + 25]

    theta_bins[0, 1] = 360.0 + theta_bins[0, 1]
    theta_bins[0, 2] = 360.0 + theta_bins[0, 2]
    middle_theta = 22.5 * (
        np.arange(16) - 8
    )  # centre of bins as per Grocott and Milan, 2014
    imf_bins = np.zeros([4, 2])
    imf_bins[:, 0] = [0, 3, 5, 10]
    imf_bins[:, 1] = [3, 5, 10, 20]  # in nT

    clock_angle_bin = np.zeros([len(omni), 3])
    tau_bin = np.zeros([len(omni), 2])
    imf_bin = np.zeros([len(omni), 2])

    # Calculate which IMF bin we are in:
    for i in range(0, len(omni)):
        # Calculate clock_angle_bin:
        if np.isnan(omni[i].clock_angle) == True:
            if (np.isnan(omni[i].by_gsm) == True) | (np.isnan(omni[i].bz_gsm) == True):
                clock_angle_bin[i] = [0, 0, 0]
                continue
            else:
                omni[i].clock_angle = np.arctan(
                    np.deg2rad(omni[i].by_gsm, omni[i].by_gsm)
                )
                print(omni[i].clock_angle)
        omni[i].clock_angle = (
            omni[i].clock_angle - int(omni[i].clock_angle / 180.0) * 360.0
        )
        angle_diff = np.cos(np.deg2rad(middle_theta)) - np.cos(
            np.deg2rad(omni[i].clock_angle)
        )
        bin_n = (np.abs(angle_diff) == np.min(np.abs(angle_diff))) & (
            np.sign(omni[i].clock_angle) == np.sign(middle_theta)
        )
        if np.sum(bin_n) == 0:
            bin_n = np.abs(angle_diff) == np.min(np.abs(angle_diff))
        clock_angle_bin[i] = theta_bins[np.nonzero(bin_n)[0][0]]
        if (omni[i].clock_angle >= 155.0) | (omni[i].clock_angle < -155.0):
            if np.sum(bin_n) == 1:
                clock_angle_bin[i] = theta_bins[0]
        if (omni[i].clock_angle >= 177.5) | (omni[i].clock_angle < -132.5):
            if clock_angle_bin[i, 0] != 2:
                clock_angle_bin[i] = theta_bins[1]

        # Calculate tau:
        if i < 20:
            continue
        pts = 1.0
        fract = 1.0
        while fract >= 0.9:
            bin_match = clock_angle_bin[int(i - pts) : (i)] == clock_angle_bin[i]
            fract = np.sum(bin_match) / pts
            if (i - pts) == 0:
                break
            pts += 1
        tt = pts
        find_tau_bin = (tt >= tau_bins[:, 0]) & (tt < tau_bins[:, 1])
        if np.sum(np.nonzero(find_tau_bin)) == 0:
            tau_bin[i] = tau_bins[0]
        else:
            tau_bin[i] = tau_bins[np.nonzero(find_tau_bin)[0][0]]

        # Set IMF bin:
        imf_mag = np.sqrt(
            (omni[i].bx_gse ** 2) + (omni[i].by_gsm ** 2) + (omni[i].bz_gsm ** 2)
        )
        find_imf_bin = (imf_bins[:, 0] <= imf_mag) & (imf_bins[:, 1] > imf_mag)
        if np.sum(find_imf_bin) == 0:
            if imf_mag > np.max(imf_bins):
                imf_bin[i] = [20, 10e3]
        else:
            imf_bin[i] = imf_bins[np.nonzero(find_imf_bin)[0][0]]

    # chop off bits of timeseries that don't need:
    start_idx = (
        (omni.year == start_year)
        & (omni.month == start_month)
        & (omni.day == start_day)
        & (omni.hour == start_hh)
        & (omni.minute == start_mm)
    )
    end_idx = (
        (omni.year == end_year)
        & (omni.month == end_month)
        & (omni.day == end_day)
        & (omni.hour == end_hh)
        & (omni.minute == end_mm)
    )
    omni = omni[np.nonzero(start_idx)[0][0] : np.nonzero(end_idx)[0][0]]
    clock_angle_bin = clock_angle_bin[
        np.nonzero(start_idx)[0][0] : np.nonzero(end_idx)[0][0]
    ]
    tau_bin = tau_bin[np.nonzero(start_idx)[0][0] : np.nonzero(end_idx)[0][0]]
    imf_bin = imf_bin[np.nonzero(start_idx)[0][0] : np.nonzero(end_idx)[0][0]]

    return omni, clock_angle_bin, tau_bin, imf_bin


###################################
def get_sw_timeseries_mode2(start_datet, end_datet, omni, sw_archive, substorm_archive):
    start_year = int(start_datet[0:4])
    start_month = int(start_datet[4:6])
    start_day = int(start_datet[6:8])
    start_hh = int(start_datet[8:10])
    start_mm = int(start_datet[10:12])
    end_year = int(end_datet[0:4])
    end_month = int(end_datet[4:6])
    end_day = int(end_datet[6:8])
    end_hh = int(end_datet[8:10])
    end_mm = int(end_datet[10:12])

    years = start_year
    if start_year << end_year:
        years = np.append(years, np.arange(0, end_year - start_year) + start_year)
    year_str = ""
    for i in range(0, len(years)):
        year_str += str(years[i])

    # find SW data for "years":
    sw_files = glob.glob(sw_archive + "omni_" + year_str + "*.sav")
    sw_files.sort()
    print("Found files:", sw_files)

    # read in sw data:
    if len(sw_files) > 1:
        print(
            "There is an error here because Maria did not write any code for what happens in the instance where more than one file is found..."
        )
    else:
        print("Reading in data...")
        sav_data = readsav(sw_files[0])
        print("Finished reading!")

    omni = sav_data["omni"]

    if (start_year >= 2000) & (end_year <= 2005):
        # find substorm data for "years":
        substorm_files = glob.glob(substorm_archive + "IMAGE_SUBSTORMS.txt")
        # read in substorm data:
        if len(substorm_files) > 1:
            print(
                "There is an error here because Maria did not write any code for what happens in the instance where more than one file is found..."
            )
        else:
            print("Reading in data...")
            with open(substorm_files[0]) as f:
                ll = f.readlines()
            print("Finished reading!")
        sub_y = []
        sub_m = []
        sub_d = []
        sub_hh = []
        sub_mm = []
        mlt = []
        mlat = []
        for i in range(2, len(ll)):
            if ll[i][4] == "2":
                sub_y.append(int(ll[i][4:8]))
                sub_m.append(int(ll[i][9:11]))
                sub_d.append(int(ll[i][11:13]))
                sub_hh.append(int(ll[i][14:16]))
                sub_mm.append(int(ll[i][17:19]))
                mlt.append(ll[i][-6:-1])
                mlat.append(ll[i][-20:-14])

        # chop off bits of timeseries that don't need (if not already):
        start_idx = (
            (omni.year == start_year)
            & (omni.month == start_month)
            & (omni.day == start_day)
            & (omni.hour == start_hh)
            & (omni.minute == start_mm)
        )
        end_idx = (
            (omni.year == end_year)
            & (omni.month == end_month)
            & (omni.day == end_day)
            & (omni.hour == end_hh)
            & (omni.minute == end_mm)
        )

        # find_substorms:
        sub_start = (
            (np.asarray(sub_y) <= start_year)
            & (np.asarray(sub_m) <= start_month)
            & (np.asarray(sub_d) <= start_day)
            & (np.asarray(sub_hh) <= start_hh)
            & (np.asarray(sub_mm) <= start_mm)
        )
        if np.sum(sub_start) == 0:
            sub_start = 0
        else:
            sub_start = max(np.nonzero(sub_start)[0])
        sub_end = (
            (np.asarray(sub_y) >= end_year)
            & (np.asarray(sub_m) >= end_month)
            & (np.asarray(sub_d) >= end_day)
            & (np.asarray(sub_hh) >= end_hh)
            & (np.asarray(sub_mm) >= end_mm)
        )
        sub_end = min(np.nonzero(sub_end)[0])
        substorm_timeseries = np.zeros(
            [np.nonzero(end_idx)[0][0] - np.nonzero(start_idx)[0][0]]
        )
        substorm_mlat = np.zeros(
            [np.nonzero(end_idx)[0][0] - np.nonzero(start_idx)[0][0]]
        )
        onset_timings = []

        # Assign all phases:
        # Changed the TiVIE code, so now all substorms are a superposed epoch analysis from -60 mins to +120 minutes from onset
        # cadence is 2 minutes
        for i in range(sub_start, sub_end):
            # check if we are within time range:
            match = (
                (sub_y[i] == omni.year)
                & (sub_m[i] == omni.month)
                & (sub_d[i] == omni.day)
                & (sub_hh[i] == omni.hour)
                & (sub_mm[i] == omni.minute)
            )
            # if not, skip this iteration:
            if np.sum(match) == 0:
                continue
            # otherwise assign:
            else:
                match = np.nonzero(match)[0][0]
                if (match - np.nonzero(start_idx)[0][0]) < 0:
                    # substorm_timeseries[0] = phase_n[i]
                    if (match - np.nonzero(start_idx)[0][0] + 120.0) > 0:
                        substorm_mlat[
                            0 : (match - np.nonzero(start_idx)[0][0] + 120.0)
                        ] = mlat[i]
                        substorm_timeseries[
                            0 : (match - np.nonzero(start_idx)[0][0] + 120.0)
                        ] = np.arange(
                            120 - (match - np.nonzero(start_idx)[0][0]),
                            (match - np.nonzero(start_idx)[0][0]) + 120,
                        )
                    else:
                        continue
                else:
                    if match - np.nonzero(start_idx)[0][0] > len(substorm_timeseries):
                        continue
                    else:
                        onset_timings += [
                            datetime.datetime(
                                year=omni[match].year,
                                month=omni[match].month,
                                day=omni[match].day,
                                hour=omni[match].hour,
                                minute=omni[match].minute,
                            )
                        ]
                        substorm_mlat[
                            match
                            - np.nonzero(start_idx)[0][0]
                            - 60 : match
                            - np.nonzero(start_idx)[0][0]
                            + 121
                        ] = mlat[i]
                        substorm_timeseries[
                            match
                            - np.nonzero(start_idx)[0][0]
                            - 60 : match
                            - np.nonzero(start_idx)[0][0]
                            + 121
                        ] = np.arange(
                            0,
                            len(
                                substorm_timeseries[
                                    match
                                    - np.nonzero(start_idx)[0][0]
                                    - 60 : match
                                    - np.nonzero(start_idx)[0][0]
                                    + 121
                                ]
                            ),
                        )

        new_timeseries = np.asarray(substorm_timeseries, dtype="int")
    else:  # We are outside of the "TiVIE years" - note: TiVIE substorms have only been tested for 2000-2005 thus far.
        ## NOTE: THIS PART OF THE CODE IS EXPERIMENTAL.
        ## It is not recommended to run this for TiVIE mode 2 as it stands because it has not been fully tested.
        ## Please contact Maria to discuss the limitations if you are interested in this.
        ## If you want to play around with it, the following code is to illustrate how you might use another substorm list..
        substorm_files = glob.glob(
            "./walach_2012-2018_EPT75.dat"  # private file path for substorm file.
        )
        substorm_files.sort()
        print("Found files:", substorm_files)

        # read in substorm data:
        if len(substorm_files) != 1:
            if len(substorm_files) > 1:
                print(
                    "There is an error here because Maria did not write any code for what happens in the instance where more than one file is found..."
                )
            else:
                print("NOTE: THIS PART OF THE CODE IS EXPERIMENTAL.")
                print(
                    "We are outside of the 'TiVIE years' - note: TiVIE substorms have only been tested for 2000-2005 thus far."
                )
                print(
                    "There is code for this but because of its experimental nature it is currently not working."
                )
                print(
                    "Please contact Maria to discuss the limitations if you are interested in this."
                )
        else:
            print("Reading in data...")
            print(substorm_files)
            with open(substorm_files[0]) as f:
                l = f.readlines()
            print("Finished reading!")
        sub_y = []
        sub_m = []
        sub_d = []
        sub_hh = []
        sub_mm = []
        phase_n = []
        flag = []
        mlt = []
        mlat = []
        for i in range(0, len(l)):
            if l[i][0] == "2":
                sub_y.append(int(l[i][0:4]))
                sub_m.append(int(l[i][5:7]))
                sub_d.append(int(l[i][8:10]))
                sub_hh.append(int(l[i][11:13]))
                sub_mm.append(int(l[i][14:16]))
                phase_n.append(int(l[i][20:21]))
                flag.append(int(l[i][22:23]))
                mlt.append(l[i][26:29])
                mlat.append(l[i][30:35])

        # chop off bits of timeseries that don't need (if not already):
        start_idx = (
            (omni.year == start_year)
            & (omni.month == start_month)
            & (omni.day == start_day)
            & (omni.hour == start_hh)
            & (omni.minute == start_mm)
        )
        end_idx = (
            (omni.year == end_year)
            & (omni.month == end_month)
            & (omni.day == end_day)
            & (omni.hour == end_hh)
            & (omni.minute == end_mm)
        )

        phase_dur = np.zeros(len(phase_n))
        for i in range(0, len(phase_n) - 1):
            if (sub_y[i] != omni[0].year) | (sub_y[i + 1] != omni[0].year):
                continue
            match1 = (
                (omni.year == sub_y[i])
                & (omni.month == sub_m[i])
                & (omni.day == sub_d[i])
                & (omni.hour == sub_hh[i])
                & (omni.minute == sub_mm[i])
            )
            match2 = (
                (omni.year == sub_y[i + 1])
                & (omni.month == sub_m[i + 1])
                & (omni.day == sub_d[i + 1])
                & (omni.hour == sub_hh[i + 1])
                & (omni.minute == sub_mm[i + 1])
            )
            phase_dur[i] = np.nonzero(match2)[0][0] - np.nonzero(match1)[0][0]

        # find_substorms:
        sub_start = (
            (np.asarray(sub_y) <= start_year)
            & (np.asarray(sub_m) <= start_month)
            & (np.asarray(sub_d) <= start_day)
            & (np.asarray(sub_hh) <= start_hh)
            & (np.asarray(sub_mm) <= start_mm)
        )
        sub_start = max(np.nonzero(sub_start)[0])
        sub_end = (
            (np.asarray(sub_y) >= end_year)
            & (np.asarray(sub_m) >= end_month)
            & (np.asarray(sub_d) >= end_day)
            & (np.asarray(sub_hh) >= end_hh)
            & (np.asarray(sub_mm) >= end_mm)
        )
        sub_end = min(np.nonzero(sub_end)[0])
        substorm_timeseries = np.zeros(
            [np.nonzero(end_idx)[0][0] - np.nonzero(start_idx)[0][0]]
        )
        substorm_mlat = np.zeros(
            [np.nonzero(end_idx)[0][0] - np.nonzero(start_idx)[0][0]]
        )

        # Assign all phases:
        for i in range(sub_start, sub_end):
            # check if we are within time range:
            match = (
                (sub_y[i] == omni.year)
                & (sub_m[i] == omni.month)
                & (sub_d[i] == omni.day)
                & (sub_hh[i] == omni.hour)
                & (sub_mm[i] == omni.minute)
            )
            # if not, skip this iteration:
            if np.sum(match) == 0:
                continue
            # otherwise assign:
            else:
                match = np.nonzero(match)[0][0]
                if (match - np.nonzero(start_idx)[0][0]) < 0:
                    substorm_timeseries[0] = phase_n[i]
                    substorm_mlat[0] = mlat[i]
                    continue
                else:
                    if match - np.nonzero(start_idx)[0][0] > len(substorm_timeseries):
                        continue
                    substorm_timeseries[match - np.nonzero(start_idx)[0][0]] = phase_n[
                        i
                    ]
                    substorm_mlat[match - np.nonzero(start_idx)[0][0]] = mlat[i]
        # Fill in gaps:
        if substorm_timeseries[0] == 0:
            substorm_timeseries[0] = phase_n[sub_start - 1]
            substorm_mlat[0] = mlat[sub_start - 1]
        for i in range(1, len(substorm_timeseries)):
            if substorm_timeseries[i] == 0:
                substorm_timeseries[i] = substorm_timeseries[i - 1]
            if substorm_mlat[i] == 0:
                substorm_mlat[i] = substorm_mlat[i - 1]

        # Now take time warping and magnetic latitude bins of substorm onsets into account:
        # Substorms are SEA of 1 hour before and 2 hours after, with 10 min cadence
        # time_len=3.*60. ; 3 hours: 1 before; 2 after
        # median_dur=time_len ; just a standard Superposed Epoch Analysis
        # dur=time_len/10.
        new_timeseries = np.empty(
            [np.nonzero(end_idx)[0][0] - np.nonzero(start_idx)[0][0]]
        )
        mlat_timeseries = np.empty(
            [np.nonzero(end_idx)[0][0] - np.nonzero(start_idx)[0][0]]
        )
        timeseries_diff = np.zeros(
            [np.nonzero(end_idx)[0][0] - np.nonzero(start_idx)[0][0]]
        )
        timeseries_end_diff = np.zeros(
            [np.nonzero(end_idx)[0][0] - np.nonzero(start_idx)[0][0]]
        )

        for i in range(1, len(new_timeseries) - 1):
            timeseries_diff[i] = substorm_timeseries[i] - substorm_timeseries[i - 1]
            timeseries_end_diff[i] = substorm_timeseries[i + 1] - substorm_timeseries[i]

        onset_start = np.nonzero((substorm_timeseries == 2) & (timeseries_diff == 1))[0]
        growth_start = np.nonzero((substorm_timeseries == 1) & (timeseries_diff < 0))[0]
        recovery_end = np.nonzero(
            (substorm_timeseries == 3) & (timeseries_end_diff < 0)
        )[0]

        onset_timings = []

        for i in range(0, len(onset_start)):
            if ((len(growth_start) < len(onset_start)) & (i == 0)) | (
                (i == 0) & (growth_start[i] > onset_start[i])
            ):
                match = (
                    (sub_y[sub_start] == omni.year)
                    & (sub_m[sub_start] == omni.month)
                    & (sub_d[sub_start] == omni.day)
                    & (sub_hh[sub_start] == omni.hour)
                    & (sub_mm[sub_start] == omni.minute)
                )
                match = np.nonzero(match)[0][0]
                if (match - np.nonzero(start_idx)[0][0]) < 0:
                    a = phase_dur[sub_start]
                else:
                    a = phase_dur[sub_start - 1]
                l = 0
            else:
                if (len(growth_start) > len(onset_start)) & (i == 0):
                    a = onset_start[i] - (growth_start[i + 1])
                    l = a
                else:
                    a = onset_start[i] - (growth_start[i])
                    l = growth_start[i]
            b = recovery_end[i] - (onset_start[i])
            m = onset_start[i]
            n = recovery_end[i]

            growth_len = a
            growth_len_f = growth_len  # / 7.0
            growth_arr = np.arange(growth_len, dtype="int")  # / growth_len_f
            rec_len = b
            rec_len_f = rec_len  # / 10.0
            rec_arr = np.arange(
                rec_len, dtype="int"
            )  # 7 + (np.arange(rec_len, dtype="int") / rec_len_f)

            if ((len(growth_start) < len(onset_start)) & (i == 0)) | (
                (i == 0) & (growth_start[i] > onset_start[i])
            ):
                new_timeseries[l:m] = growth_arr[len(growth_arr) - (m - l) :]
            else:
                new_timeseries[l:m] = growth_arr

            new_timeseries[m:n] = rec_arr

            mlat_timeseries[l:m] = substorm_mlat[onset_start[i]]

            if (onset_start[i] < 0) | (onset_start[i] > len(new_timeseries)):
                continue
            else:
                match = (
                    (sub_y[sub_start + onset_start[i]] == omni.year)
                    & (sub_m[sub_start + onset_start[i]] == omni.month)
                    & (sub_d[sub_start + onset_start[i]] == omni.day)
                    & (sub_hh[sub_start + onset_start[i]] == omni.hour)
                    & (sub_mm[sub_start + onset_start[i]] == omni.minute)
                )
                if np.nonzero(match)[0].size > 0:
                    match = np.nonzero(match)[0][0]
                    onset_timings += [
                        datetime.datetime(
                            year=omni[match].year,
                            month=omni[match].month,
                            day=omni[match].day,
                            hour=omni[match].hour,
                        )
                    ]

        new_timeseries = np.asarray(new_timeseries, dtype="int")
    return new_timeseries, substorm_mlat, onset_timings


###################################
def get_sw_timeseries_mode3(start_datet, end_datet, omni, storm_archive, sw_archive):
    start_year = int(start_datet[0:4])
    start_month = int(start_datet[4:6])
    start_day = int(start_datet[6:8])
    start_hh = int(start_datet[8:10])
    start_mm = int(start_datet[10:12])
    end_year = int(end_datet[0:4])
    end_month = int(end_datet[4:6])
    end_day = int(end_datet[6:8])
    end_hh = int(end_datet[8:10])
    end_mm = int(end_datet[10:12])

    years = start_year
    if start_year << end_year:
        years = np.append(years, np.arange(0, end_year - start_year) + start_year)
    year_str = ""
    for i in range(0, len(years)):
        year_str += str(years[i])

    # find SW data for "years":
    sw_files = glob.glob(sw_archive + "omni_" + year_str + "*.sav")
    sw_files.sort()
    print("Found files:", sw_files)

    # read in sw data:
    if len(sw_files) > 1:
        print(
            "There is an error here because Maria did not write any code for what happens in the instance where more than one file is found..."
        )
    else:
        print("Reading in data...")
        sav_data = readsav(sw_files[0])
        print("Finished reading!")

    omni = sav_data["omni"]

    # find storm data for "years":
    storm_files = glob.glob(storm_archive + "storm_list_1981to2019.sav")
    storm_files.sort()
    print("Found files:", storm_files)

    # read in storm data:
    if len(storm_files) > 1:
        print(
            "There is an error here because Maria did not write any code for what happens in the instance where more than one file is found..."
        )
    else:
        print("Reading in data...")
        sav_data = readsav(storm_files[0])
        print("Finished reading!")
    storm_year = sav_data["storm_year"]
    ti = sav_data["ti"]
    tr = sav_data["tr"]
    t0 = sav_data["t0"]
    symh_min_idx = sav_data["symh_min_idx"]

    # chop off bits of timeseries that don't need (if not already):
    start_idx = (
        (omni.year == start_year)
        & (omni.month == start_month)
        & (omni.day == start_day)
        & (omni.hour == start_hh)
        & (omni.minute == start_mm)
    )
    end_idx = (
        (omni.year == end_year)
        & (omni.month == end_month)
        & (omni.day == end_day)
        & (omni.hour == end_hh)
        & (omni.minute == end_mm)
    )

    # find storms:
    st_before = (
        (np.asarray(storm_year) == start_year)
        & (np.nonzero(start_idx)[0][0] <= symh_min_idx - ti - t0)
        & (np.nonzero(end_idx)[0][0] >= symh_min_idx - ti - t0)
    )
    st_after = (
        (np.asarray(storm_year) == start_year)
        & (np.nonzero(end_idx)[0][0] >= symh_min_idx + tr)
        & (np.nonzero(start_idx)[0][0] <= symh_min_idx + tr)
    )
    st_without = (
        (np.asarray(storm_year) == start_year)
        & (np.nonzero(start_idx)[0][0] >= symh_min_idx - ti - t0)
        & (np.nonzero(end_idx)[0][0] <= symh_min_idx + tr)
    )
    st_within = (
        (np.asarray(storm_year) == start_year)
        & (np.nonzero(start_idx)[0][0] <= symh_min_idx - ti - t0)
        & (np.nonzero(end_idx)[0][0] >= symh_min_idx + tr)
    )

    st_before = np.nonzero(st_before)[0]
    st_after = np.nonzero(st_after)[0]
    st_within = np.nonzero(st_within)[0]
    st_without = np.nonzero(st_without)[0]

    storm_timeseries = np.zeros(
        [len(omni[np.nonzero(start_idx)[0][0] : np.nonzero(end_idx)[0][0]])]
    )

    # Also take time warping  into account:
    # Storms are SEA of initial phase: 650 minutes
    #            main phase: 280 minutes
    #           recovery phase: 1682 minutes
    #                               with 10 min cadence each
    new_storm_timeseries = np.zeros(
        [len(omni[np.nonzero(start_idx)[0][0] : np.nonzero(end_idx)[0][0]])]
    )

    all_st = [st_before, st_after, st_within, st_without]

    for i in range(0, len(all_st)):
        if all_st[i].size == 0:
            continue
        a = symh_min_idx[all_st[i]] - ti[all_st[i]] - t0[all_st[i]]
        a = a[0]
        b = symh_min_idx[all_st[i]] - t0[all_st[i]]
        b = b[0]
        c = symh_min_idx[all_st[i]]
        c = c[0]
        d = symh_min_idx[all_st[i]] + tr[all_st[i]]
        d = int(d[0])

        l = a - (np.nonzero(start_idx)[0][0])
        m = b - (np.nonzero(start_idx)[0][0])
        n = c - (np.nonzero(start_idx)[0][0])
        o = d - (np.nonzero(start_idx)[0][0])
        if l < 0:
            l = 0
        if m < 0:
            m = 0
        if n < 0:
            n = 0
        # Assign timeseries:
        initial_len = ti[all_st[i]]
        initial_len_f = initial_len / 64.0
        initial_arr = np.arange(initial_len, dtype="int") / initial_len_f
        main_len = t0[all_st[i]]
        main_len_f = main_len / 27.0
        main_arr = np.arange(main_len, dtype="int") / main_len_f
        rec_len = tr[all_st[i]]
        rec_len_f = rec_len / 167.1
        rec_arr = np.arange(rec_len, dtype="int") / rec_len_f

        if m > len(storm_timeseries):
            storm_timeseries[l:] = 1
            new_storm_timeseries[l:] = initial_arr[
                (len(initial_arr) - (b - (np.nonzero(start_idx)[0][0]))) : (
                    len(initial_arr) - (b - (np.nonzero(start_idx)[0][0]))
                )
                + len(new_storm_timeseries[l:])
            ]
        else:
            storm_timeseries[l:m] = 1
            new_storm_timeseries[l:m] = initial_arr[
                len(initial_arr)
                - (b - (np.nonzero(start_idx)[0][0])) : len(initial_arr)
                - (b - (np.nonzero(start_idx)[0][0]))
                + len(new_storm_timeseries[l:m])
            ]
        if n > len(storm_timeseries):
            storm_timeseries[m:] = 2
            new_storm_timeseries[m:] = main_arr[
                len(main_arr)
                - (c - (np.nonzero(start_idx)[0][0])) : (
                    len(initial_arr) - (b - (np.nonzero(start_idx)[0][0]))
                )
                + len(new_storm_timeseries[m:])
            ]
        else:
            storm_timeseries[m:n] = 2
            new_storm_timeseries[m:n] = main_arr[
                len(main_arr) - (c - (np.nonzero(start_idx)[0][0])) :
            ]
        if o > len(storm_timeseries):
            storm_timeseries[n:] = 3
            new_storm_timeseries[n:] = rec_arr[: len(new_storm_timeseries[n:])]
        else:
            storm_timeseries[n:o] = 3
            new_storm_timeseries[n:o] = rec_arr[
                len(rec_arr) - (d - (np.nonzero(start_idx)[0][0])) :
            ]

    return storm_timeseries, new_storm_timeseries


###################################
def prep_mode1(clock_angle_bin, tau_bin, imf_bin):
    # Now we need to convert the timeseries data into a string of mapfile names:
    # Work out mode 1:
    mode_1 = [""]

    for i in range(0, len(clock_angle_bin)):
        if np.sum(clock_angle_bin[i, :]) == 0:
            mode_1 += [
                "imf_mode_tau"
                + str(int(tau_bin[i - 1, 0]))
                + "_theta"
                + str(int(clock_angle_bin[i - 1, 1]))
                + "_imf"
                + str(int(imf_bin[i - 1, 0]))
            ]
        else:
            mode_1 += [
                "imf_mode_tau"
                + str(int(tau_bin[i, 0]))
                + "_theta"
                + str(int(clock_angle_bin[i, 1]))
                + "_imf"
                + str(int(imf_bin[i, 0]))
            ]
    return mode_1


###################################
def prep_mode2(timeseries, mlat):
    # Now we need to convert the timeseries data into a string of mapfile names:
    # Work out mode 2:
    mode_2 = [""]
    mlat_bins = [
        [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73],
        [59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74],
    ]
    sel_substorm_mlt = ["20", "03"]

    for i in range(0, len(timeseries)):
        if (mlat[i] < 58) or (mlat[i] >= 74):
            if mlat[i] < 58:
                mlat[i] = 58
            if mlat[i] >= 74:
                mlat[i] = 73.5

        sel_mlat_bins = (mlat_bins[0][:] <= mlat[i]) & (mlat_bins[1][:] > mlat[i])
        sel_mlat_bins = mlat_bins[0][np.nonzero(sel_mlat_bins)[0][0]]
        mode_2 += [
            "substorm_mode_substorm_mlt"
            + str(sel_substorm_mlt[0])
            + "_mlat"
            + str(sel_mlat_bins)
            + "_nil/"
            + str(int(math.floor(timeseries[i])))
        ]

    return mode_2


###################################
def prep_mode3(storm_phases, timeseries):
    # Now we need to convert the timeseries data into a string of mapfile names:
    # Work out mode 3:
    mode_3 = [""]

    for i in range(0, len(storm_phases)):
        if storm_phases[i] > 0:
            mode_3 += [
                "storm_mode_storm_phase"
                + str(int(storm_phases[i] - 1))
                + "/"
                + str(int(math.floor(timeseries[i])))
            ]
        else:
            mode_3 += [""]

    return mode_3


###################################
def get_files(mode_n, mode, target_folder, tivie_archive, cadence):
    missing_files = [""]
    missing_n = [""]
    find_folder = glob.glob(target_folder + "mode" + str(mode_n))
    if find_folder == []:
        print("Target folder does not exist for mode " + str(mode_n))
        print("Looking for..: " + target_folder + "mode" + str(mode_n))
        return
    else:
        # wipe target folder:
        shutil.rmtree(target_folder + "mode" + str(mode_n) + "/")
        # remake clean:
        os.makedirs(target_folder + "mode" + str(mode_n) + "/")
    # Now that we have found the correct folder, we will copy files across:
    for i in range(0, len(mode), cadence):
        if mode_n == 1:
            find_file = glob.glob(
                tivie_archive
                + "mode_"
                + str(mode_n)
                + "/"
                + mode[i]
                + "/"
                + "0.fit.map"
            )
            if find_file == []:
                print("Skipped :" + str(i))
                missing_files += [mode[i]]
                missing_n += [i]
            else:
                source = (
                    tivie_archive
                    + "mode_"
                    + str(mode_n)
                    + "/"
                    + mode[i]
                    + "/"
                    + "0.fit.map"
                )
                destination = (
                    target_folder
                    + "mode"
                    + str(mode_n)
                    + "/"
                    + "{0:05d}".format(i)
                    + ".fit.map"
                )
                shutil.copy(source, destination)
                try:
                    subprocess.check_output(
                        "maptocnvmap -vb "
                        + destination
                        + " > "
                        + target_folder
                        + "mode"
                        + str(mode_n)
                        + "/"
                        + "{0:05d}".format(i)
                        + ".fit.cnvmap",
                        shell=True,
                        stderr=subprocess.STDOUT,
                    )
                except subprocess.CalledProcessError as e:
                    print(e)
                    missing_files += [mode[i]]
                    missing_n += [i]
                subprocess.run(["rm", destination], capture_output=True)
        if mode_n == 2:
            find_file = glob.glob(
                tivie_archive
                + "/mode_"
                + str(mode_n)
                + "/FREY_range_lim_2min_HEC/"  # "/FREY_substorms_no_mlt_split/"
                + mode[i]
                + ".fit.map"
            )
            if find_file == []:
                print("Skipped :" + str(i))
                missing_files += [mode[i]]
                missing_n += [i]
            else:
                source = (
                    tivie_archive
                    + "/mode_"
                    + str(mode_n)
                    + "/FREY_range_lim_2min_HEC/"  # "/FREY_substorms_no_mlt_split/"
                    + mode[i]
                    + ".fit.map"
                )
                destination = (
                    target_folder
                    + "mode"
                    + str(mode_n)
                    + "/"
                    + "{0:05d}".format(i)
                    + ".fit.map"
                )
                shutil.copy(source, destination)
                print(source, destination)
                try:
                    subprocess.check_output(
                        "maptocnvmap -vb "
                        + destination
                        + " > "
                        + target_folder
                        + "mode"
                        + str(mode_n)
                        + "/"
                        + "{0:05d}".format(i)
                        + ".fit.cnvmap",
                        shell=True,
                        stderr=subprocess.STDOUT,
                    )
                except subprocess.CalledProcessError as e:
                    print(e)
                    missing_files += [mode[i]]
                    missing_n += [i]
                subprocess.run(["rm", destination], capture_output=True)
        if mode_n == 3:
            find_file = glob.glob(
                tivie_archive + "mode_" + str(mode_n) + "/" + mode[i] + ".fit.map"
            )
            if find_file == []:
                print("Skipped :" + str(i))
                missing_files += [mode[i]]
                missing_n += [i]
            else:
                source = (
                    tivie_archive + "mode_" + str(mode_n) + "/" + mode[i] + ".fit.map"
                )
                destination = (
                    target_folder
                    + "mode"
                    + str(mode_n)
                    + "/"
                    + "{0:05d}".format(i)
                    + ".fit.map"
                )
                shutil.copy(source, destination)
                try:
                    subprocess.check_output(
                        "maptocnvmap -vb "
                        + destination
                        + " > "
                        + target_folder
                        + "mode"
                        + str(mode_n)
                        + "/"
                        + "{0:05d}".format(i)
                        + ".fit.cnvmap",
                        shell=True,
                        stderr=subprocess.STDOUT,
                    )
                except subprocess.CalledProcessError as e:
                    print(e)
                    missing_files += [mode[i]]
                    missing_n += [i]
                subprocess.run(["rm", destination], capture_output=True)
    print("Your files are now available in " + target_folder + "mode" + str(mode_n))
    return missing_files, missing_n


###################################
def read_files(file_folder, correct_dt, missing_n):
    map_files = glob.glob(file_folder + "/*.cnvmap")
    map_files.sort()
    if map_files != []:
        print(map_files)
        print("Reading in map files")
        map_data = []
        for file in map_files:
            if os.stat(file).st_size == 0:
                continue
            records = pydarn.SuperDARNRead(file).read_map()
            map_data += records
        print("Reading complete...")

        # Assign SuperDARN data:
        syr = [i["start.year"] for i in map_data]
        smo = [i["start.month"] for i in map_data]
        sdy = [i["start.day"] for i in map_data]
        shr = [i["start.hour"] for i in map_data]
        smt = [i["start.minute"] for i in map_data]
        dt_from_file = []
        dt = []
        file_n = []
        for i in range(0, len(syr)):
            temp_datetime = [
                datetime.datetime(
                    year=syr[i],
                    month=smo[i],
                    day=sdy[i],
                    hour=shr[i],
                    minute=smt[i],
                )
            ]
            dt_from_file += temp_datetime
            file_n += [
                int(map_files[i][len(map_files[i]) - 16 : len(map_files[i]) - 11])
            ]
        for i in range(0, len(file_n)):
            if len(dt) == 0:
                delta = datetime.timedelta(minutes=file_n[i])
                dt += [correct_dt[0] + delta]
            else:
                delta = datetime.timedelta(minutes=(file_n[i] - file_n[i - 1]))
                dt += [dt[-1] + delta]

        # Now get all the other parameters:
        cpcp = [i["pot.drop"] for i in map_data]
        pot_min = [i["pot.min"] for i in map_data]
        pot_max = [i["pot.max"] for i in map_data]
        hmb = [i["latmin"] for i in map_data]
        n = [i["nvec"] for i in map_data]
        doping_level = [i["doping.level"] for i in map_data]
        imf_flag = [i["IMF.flag"] for i in map_data]
        imf_delay = [i["IMF.delay"] for i in map_data]
        imf_bx = [i["IMF.Bx"] for i in map_data]
        imf_by = [i["IMF.By"] for i in map_data]
        imf_bz = [i["IMF.Bz"] for i in map_data]
        imf_vx = [i["IMF.Vx"] for i in map_data]
        tilt = [i["model.tilt"] for i in map_data]
        kp = [i["IMF.Kp"] for i in map_data]
        imf_angle = [i["model.angle"] for i in map_data]
        imf_esw = [i["model.level"] for i in map_data]
        model_name = [i["model.name"] for i in map_data]
        hemisphere = [i["hemisphere"] for i in map_data]
        chi_sqr = [i["chi.sqr"] for i in map_data]

        # Read in vector data:
        mlat = []
        mlon = []
        v_dir = []
        v_mag = []
        n_tot = []
        for i in range(0, len(n)):
            n_tot = np.append(n_tot, np.sum(n[i]))
            if np.sum(n[i]) != 0:
                mlat += [map_data[i]["vector.mlat"]]
                mlon += [map_data[i]["vector.mlon"]]
                v_dir += [map_data[i]["vector.kvect"]]
                v_mag += [map_data[i]["vector.vel.median"]]
            else:
                mlat += [[np.nan]]
                mlon += [[np.nan]]
                v_dir += [[np.nan]]
                v_mag += [[np.nan]]

        # Make data array:
        if np.sum(n_tot) > 0:
            mag_meas = np.full((len(v_mag), int(max(n_tot))), 99999, dtype="float64")
            dir_meas = np.full((len(v_dir), int(max(n_tot))), 99999, dtype="float64")
            mlat_meas = np.full((len(mlat), int(max(n_tot))), 99999, dtype="float64")
            mlon_meas = np.full((len(mlon), int(max(n_tot))), 99999, dtype="float64")

            # Assign data:
            for i in range(0, len(mag_meas)):
                mag_meas[i, 0 : len(v_mag[i])] = v_mag[i]
                dir_meas[i, 0 : len(v_dir[i])] = v_dir[i]
                mlat_meas[i, 0 : len(mlat[i])] = mlat[i]
                mlon_meas[i, 0 : len(mlon[i])] = mlon[i]

        # Read in model data:
        model_mlat = [i["model.mlat"] for i in map_data]
        model_mlon = [i["model.mlon"] for i in map_data]
        model_dir = [i["model.kvect"] for i in map_data]
        model_mag = [i["model.vel.median"] for i in map_data]

        N = [i["N"] for i in map_data]
        N1 = [i["N+1"] for i in map_data]
        N2 = [i["N+2"] for i in map_data]
        N3 = [i["N+3"] for i in map_data]
        order = [i["fit.order"] for i in map_data]
        boundary_mlat = [i["boundary.mlat"] for i in map_data]
        boundary_mlon = [i["boundary.mlon"] for i in map_data]
        latmin = [i["latmin"] for i in map_data]

        # Put the SuperDARN model data into an Xarray:
        # Find maximum number of model vectors:
        n_max = 0
        for i in range(0, len(model_mag)):
            if len(model_mag[i]) > n_max:
                n_max = len(model_mag[i])

        # Make data array:
        mag_values = np.full((len(model_mag), n_max), 99999, dtype="float64")
        dir_values = np.full((len(model_dir), n_max), 99999, dtype="float64")
        mlat_values = np.full((len(model_mlat), n_max), 99999, dtype="float64")
        mlon_values = np.full((len(model_mlon), n_max), 99999, dtype="float64")

        # Assign data:
        for i in range(0, len(model_mag)):
            mag_values[i, 0 : len(model_mag[i])] = model_mag[i]
            dir_values[i, 0 : len(model_dir[i])] = model_dir[i]
            mlat_values[i, 0 : len(model_mlat[i])] = model_mlat[i]
            mlon_values[i, 0 : len(model_mlon[i])] = model_mlon[i]

        if (np.sum(n_tot) > 0) & (len(dt) > 0):
            # Assign it to an xarray:
            ds = xr.Dataset(
                {
                    "cpcp": (["time"], cpcp),
                    "pot_min": (["time"], pot_min),
                    "pot_max": (["time"], pot_max),
                    "hmb": (["time"], hmb),
                    "n": (["time"], n_tot),
                    "doping_level": (["time"], doping_level),
                    "imf_flag": (["time"], imf_flag),
                    "imf_delay": (["time"], imf_delay),
                    "imf_bx": (["time"], imf_bx),
                    "imf_by": (["time"], imf_by),
                    "imf_bz": (["time"], imf_bz),
                    "imf_vx": (["time"], imf_vx),
                    "imf_angle": (["time"], imf_angle),
                    "imf_esw": (["time"], imf_esw),
                    "tilt": (["time"], tilt),
                    "kp": (["time"], kp),
                    "model_name": (["time"], model_name),
                    "hemisphere": (["time"], hemisphere),
                    "chi_sqr": (["time"], chi_sqr),
                    "model_magnitude": (["time", "model_gridpoints"], mag_values),
                    "model_k_vector": (["time", "model_gridpoints"], dir_values),
                    "model_mlat": (["time", "model_gridpoints"], mlat_values),
                    "model_mlon": (["time", "model_gridpoints"], mlon_values),
                    "magnitude": (["time", "measured_gridpoints"], mag_meas),
                    "k_vector": (["time", "measured_gridpoints"], dir_meas),
                    "mlat_meas": (["time", "measured_gridpoints"], mlat_meas),
                    "mlon_meas": (["time", "measured_gridpoints"], mlon_meas),
                    "N": (["time", "coefficients"], N),
                    "N1": (["time", "coefficients"], N1),
                    "N2": (["time", "coefficients"], N2),
                    "N3": (["time", "coefficients"], N3),
                    "fitting_order": (["time"], order),
                    "latmin": (["time"], latmin),
                    "boundary_mlat": (["time", "HMB_boundary"], boundary_mlat),
                    "boundary_mlon": (["time", "HMB_boundary"], boundary_mlon),
                    "file_time": (dt_from_file),
                    "missing_n": (missing_n),
                    "time_records_without_gaps": (correct_dt),
                },
                coords={"time": (dt)},
            )
            # Set units:
            ds["magnitude"].attrs["units"] = "m/s"
            ds["k_vector"].attrs["units"] = "degrees from magnetic meridian"
            ds["mlat_meas"].attrs["units"] = "degrees"
            ds["mlon_meas"].attrs["units"] = "degrees"
            ds["cpcp"].attrs["units"] = "kV"
            ds["pot_min"].attrs["units"] = "kV"
            ds["pot_max"].attrs["units"] = "kV"
            ds["hmb"].attrs["units"] = "degrees"
            ds["imf_delay"].attrs["units"] = "minutes"
            ds["imf_bx"].attrs["units"] = "nT"
            ds["imf_by"].attrs["units"] = "nT"
            ds["imf_bz"].attrs["units"] = "nT"
            ds["imf_vx"].attrs["units"] = "km/s"
            ds["model_magnitude"].attrs["units"] = "m/s"
            ds["model_k_vector"].attrs["units"] = "degrees from magnetic meridian"
            ds["model_mlat"].attrs["units"] = "degrees"
            ds["model_mlon"].attrs["units"] = "degrees"
            ds["latmin"].attrs["units"] = "degrees"
            ds["boundary_mlat"].attrs["units"] = "degrees"
            ds["boundary_mlon"].attrs["units"] = "degrees"
        else:
            # Assign it to an xarray:
            ds = xr.Dataset(
                {
                    "cpcp": (["time"], cpcp),
                    "pot_min": (["time"], pot_min),
                    "pot_max": (["time"], pot_max),
                    "hmb": (["time"], hmb),  # "n": (["time"], n_tot),
                    "doping_level": (["time"], doping_level),
                    "imf_flag": (["time"], imf_flag),
                    "imf_delay": (["time"], imf_delay),
                    "imf_bx": (["time"], imf_bx),
                    "imf_by": (["time"], imf_by),
                    "imf_bz": (["time"], imf_bz),
                    "imf_vx": (["time"], imf_vx),
                    "imf_angle": (["time"], imf_angle),
                    "imf_esw": (["time"], imf_esw),
                    "tilt": (["time"], tilt),
                    "kp": (["time"], kp),
                    "model_name": (["time"], model_name),
                    "hemisphere": (["time"], hemisphere),
                    "chi_sqr": (["time"], chi_sqr),
                    "model_magnitude": (["time", "model_gridpoints"], mag_values),
                    "model_k_vector": (["time", "model_gridpoints"], dir_values),
                    "model_mlat": (["time", "model_gridpoints"], mlat_values),
                    "model_mlon": (
                        ["time", "model_gridpoints"],
                        mlon_values,
                    ),  # "N": (["time", "coefficients"],N), # "N1": (["time", "coefficients"], N1), # "N2": (["time", "coefficients"], N2), # "N3": (["time", "coefficients"], N3),
                    "fitting_order": (["time"], order),
                    "latmin": (["time"], latmin),
                    "boundary_mlat": (["time", "HMB_boundary"], boundary_mlat),
                    "boundary_mlon": (["time", "HMB_boundary"], boundary_mlon),
                    "file_time": (dt_from_file),
                    "missing_n": (missing_n),
                    "time_records_without_gaps": (correct_dt),
                },
                coords={"time": (dt)},
            )
            # Set units:
            ds["cpcp"].attrs["units"] = "kV"
            ds["pot_min"].attrs["units"] = "kV"
            ds["pot_max"].attrs["units"] = "kV"
            ds["hmb"].attrs["units"] = "degrees"
            ds["imf_delay"].attrs["units"] = "minutes"
            ds["imf_bx"].attrs["units"] = "nT"
            ds["imf_by"].attrs["units"] = "nT"
            ds["imf_bz"].attrs["units"] = "nT"
            ds["imf_vx"].attrs["units"] = "km/s"
            ds["model_magnitude"].attrs["units"] = "m/s"
            ds["model_k_vector"].attrs["units"] = "degrees from magnetic meridian"
            ds["model_mlat"].attrs["units"] = "degrees"
            ds["model_mlon"].attrs["units"] = "degrees"
            ds["latmin"].attrs["units"] = "degrees"
            ds["boundary_mlat"].attrs["units"] = "degrees"
            ds["boundary_mlon"].attrs["units"] = "degrees"
            print("Finished reading files!")
    else:
        ds = []
    return ds


###################################
def calc_potential_for_entire_set(ds):
    # Calculate potential:
    mlat_arr = np.asarray(np.arange(40.5, 90.5))
    mlon_arr = np.asarray(np.arange(0, 360, 2))
    potential = np.zeros([len(ds["time"].values), len(mlat_arr), len(mlon_arr)])
    print("Calculating potential..This may take some time...")
    for t in range(0, len(ds["time"].values)):
        for i in range(0, len(mlat_arr) * len(mlon_arr)):
            if (
                np.any(
                    (
                        np.asarray(
                            int(
                                np.round(
                                    ((mlon_arr[int(i / len(mlat_arr))] + 360.0) % 360)
                                    / 5
                                )
                                * 5
                            )
                        )
                        == ds["boundary_mlon"]
                    )
                    & (ds["boundary_mlat"].isel(time=t) > mlat_arr[(i % len(mlat_arr))])
                )
                is True
            ):
                potential[(i % len(mlat_arr)), int(i / len(mlat_arr))] = 0
            try:
                potential[t, (i % len(mlat_arr)), int(i / len(mlat_arr))] = (
                    calc_pot(
                        mlon_arr[int(i / len(mlat_arr))],
                        mlat_arr[(i % len(mlat_arr))],
                        ds["latmin"].isel(time=t).values,
                        ds["N"].isel(time=t).values,
                        ds["N1"].isel(time=t).values,
                        ds["N2"].isel(time=t).values,
                    )
                    / 1e3
                )
            except KeyError:
                potential[t, (i % len(mlat_arr)), int(i / len(mlat_arr))] = 0
    print("Finished calculating potential.")

    return mlat_arr, mlon_arr, potential


###################################
def make_plot(
    output_folder,
    start_datet,
    end_datet,
    onset_timings,
    mode,
    omni,
    ds1=[],
    ds2=[],
    ds3=[],
):
    # Check that the output folder exists:
    if (
        glob.glob((output_folder + "/" + str(start_datet) + "_" + str(end_datet) + "/"))
        == []
    ):
        os.makedirs(output_folder + "/" + str(start_datet) + "_" + str(end_datet) + "/")

    # Make datetime from omni:
    omni_time = [
        datetime.datetime(
            year=omni[i].year,
            month=omni[i].month,
            day=omni[i].day,
            hour=omni[i].hour,
            minute=omni[i].minute,
        )
        for i in range(len(omni))
    ]

    # Make the timeseries plot and save it!
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(
        7, figsize=(15, 10), sharex=True
    )
    # Set title:
    syear = start_datet[0:4]
    smonth = start_datet[4:6]
    sday = start_datet[6:8]
    shour = start_datet[8:10]
    sminute = start_datet[10:12]
    eyear = end_datet[0:4]
    emonth = end_datet[4:6]
    eday = end_datet[6:8]
    ehour = end_datet[8:10]
    eminute = end_datet[10:12]
    title = (
        sday
        + "/"
        + smonth
        + "/"
        + syear
        + " "
        + shour
        + ":"
        + sminute
        + " UT to "
        + eday
        + "/"
        + emonth
        + "/"
        + eyear
        + " "
        + ehour
        + ":"
        + eminute
        + " UT"
    )
    fig.suptitle(title, y=0.9)
    # Do all the plotting based on which datasets are defined:

    # Make sure that gaps in SW are reflected in plotting:
    if len(ds1) >= 1:
        a_idx = np.where(ds1["time"] == ds1["time_records_without_gaps"])[1]
        cpcp_ds1 = np.full(len(ds1["time_records_without_gaps"]), fill_value=np.nan)
        hmb_ds1 = np.full(len(ds1["time_records_without_gaps"]), fill_value=np.nan)
        for i in range(0, len(ds1["time_records_without_gaps"])):
            cpcp_ds1[i] = ds1["cpcp"].sel(
                time=ds1["time_records_without_gaps"].values[i], method="pad"
            )
            hmb_ds1[i] = ds1["hmb"].sel(
                time=ds1["time_records_without_gaps"].values[i], method="pad"
            )
        ax5.plot(
            ds1["time_records_without_gaps"],
            cpcp_ds1 / 1e3,
            color="aquamarine",
            label="TiVIE mode 1",
            alpha=0.8,
        )
        ax6.plot(
            ds1["time_records_without_gaps"],
            hmb_ds1,
            color="aquamarine",
            alpha=0.8,
        )
        ax7.semilogy(ds1["time"], ds1["n"], color="aquamarine", alpha=0.8)

    if len(ds2) >= 1:
        b_idx = np.where(ds2["time"] == ds2["time_records_without_gaps"])[1]
        ax5.plot(
            ds2["time_records_without_gaps"][b_idx],
            ds2["cpcp"] / 1e3,
            color="goldenrod",
            label="TiVIE mode 2",
            alpha=0.8,
        )
        ax6.plot(
            ds2["time_records_without_gaps"][b_idx],
            ds2["hmb"],
            color="goldenrod",
            alpha=0.8,
        )
        ax7.semilogy(
            ds2["time_records_without_gaps"][b_idx],
            ds2["n"],
            color="goldenrod",
            alpha=0.8,
        )

    if len(ds3) >= 1:
        c_idx = np.where(ds3["time"] == ds3["time_records_without_gaps"])[1]
        ax5.plot(
            ds3["time_records_without_gaps"][c_idx],
            ds3["cpcp"] / 1e3,
            color="orange",
            label="TiVIE mode 3",
            alpha=0.8,
        )
        ax6.plot(
            ds3["time_records_without_gaps"][c_idx],
            ds3["hmb"],
            color="orange",
            alpha=0.8,
        )
        ax7.semilogy(
            ds3["time_records_without_gaps"][c_idx],
            ds3["n"],
            color="orange",
            alpha=0.8,
        )

    if len(omni) >= 1:
        ax1.plot(
            omni_time,
            omni["bx_gse"],
            color="coral",
            label="$B_{x}$",
            alpha=0.8,
        )
        ax1.plot(
            omni_time,
            omni["by_gsm"],
            color="firebrick",
            label="$B_{y}$",
            alpha=0.8,
        )
        ax1.plot(
            omni_time,
            omni["bz_gsm"],
            color="magenta",
            label="$B_{z}$",
            alpha=0.8,
        )
        ax2.plot(omni_time, omni["vsw"], color="mediumorchid", alpha=0.8)
        ax3.plot(omni_time, omni["al"], color="olivedrab", label="AL", alpha=0.8)
        ax3.plot(omni_time, omni["au"], color="teal", label="AU", alpha=0.8)
        ax4.plot(omni_time, omni["symh"], color="pink", alpha=0.8)

    # # Highlight the chosen mode:
    if mode == 1:
        ax5.fill_between(
            ds1["time_records_without_gaps"],
            0,
            cpcp_ds1 / 1e3,
            color="aquamarine",
            alpha=0.3,
        )
        ax6.fill_between(
            ds1["time_records_without_gaps"][a_idx],
            0,
            ds1["hmb"],
            color="aquamarine",
            alpha=0.3,
        )
        ax7.fill_between(
            ds1["time_records_without_gaps"][a_idx],
            0,
            ds1["n"],
            color="aquamarine",
            alpha=0.3,
        )
    if mode == 2:
        ax5.fill_between(
            ds2["time_records_without_gaps"][b_idx],
            0,
            ds2["cpcp"] / 1e3,
            color="goldenrod",
            alpha=0.3,
        )
        ax6.fill_between(
            ds2["time_records_without_gaps"][b_idx],
            0,
            ds2["hmb"],
            color="goldenrod",
            alpha=0.3,
        )
        ax7.fill_between(
            ds2["time_records_without_gaps"][b_idx],
            0,
            ds2["n"],
            color="goldenrod",
            alpha=0.3,
        )
    if mode == 3:
        ax5.fill_between(
            ds3["time_records_without_gaps"][c_idx],
            0,
            ds3["cpcp"] / 1e3,
            color="orange",
            alpha=0.3,
        )
        ax6.fill_between(
            ds3["time_records_without_gaps"][c_idx],
            0,
            ds3["hmb"],
            color="orange",
            alpha=0.3,
        )
        ax7.fill_between(
            ds3["time_records_without_gaps"][c_idx],
            0,
            ds3["n"],
            color="orange",
            alpha=0.3,
        )

    # Add annotations and lines etc, irrespective of chosen mode:
    ax1.set_ylabel("IMF B [nT]")
    ax1.axhline(y=0, color="silver", linestyle="-", linewidth=2)
    for i in range(0, len(onset_timings)):
        ax1.axvline(x=onset_timings[i], color="black", linestyle="--")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2.set_ylabel("$V_{SW}$ [km/s]")
    for i in range(0, len(onset_timings)):
        ax2.axvline(x=onset_timings[i], color="black", linestyle="--")
    ax2.grid(True)
    plt.subplots_adjust(hspace=0)

    ax3.axhline(y=0, color="silver", linestyle="-", linewidth=2)
    ax3.set_ylabel("AL & AU [nT]")
    for i in range(0, len(onset_timings)):
        ax3.axvline(x=onset_timings[i], color="black", linestyle="--")
    ax3.legend(loc="upper left")
    ax3.grid(True)

    ax4.set_ylabel("Sym-H [nT]")
    for i in range(0, len(onset_timings)):
        ax4.axvline(x=onset_timings[i], color="black", linestyle="--")
    ax4.grid(True)

    ax5.set_ylabel("$\Phi_{CPCP}$ [kV]")
    ax5.grid(True)
    for i in range(0, len(onset_timings)):
        ax5.axvline(x=onset_timings[i], color="black", linestyle="--")
    ax5.legend(loc="upper left")

    ax6.set_ylabel("$\lambda_{HMB}$ [" + deg + "]")
    ax6.set_ylim([40, 80])
    for i in range(0, len(onset_timings)):
        ax6.axvline(x=onset_timings[i], color="black", linestyle="--")
    ax6.grid(True)

    ax7.set_ylabel("n")
    for i in range(0, len(onset_timings)):
        ax7.axvline(x=onset_timings[i], color="black", linestyle="--")
    ax7.grid(True)

    # Format x-tick labels:
    ax7.tick_params(labelrotation=45)
    ax7.xaxis.set_major_formatter(DateFormatter("%d/%m %H:%M"))

    # Force start and end:
    ax1.set_xlim([ds1["time"][0], ds1["time"][-1]])
    ax2.set_xlim([ds1["time"][0], ds1["time"][-1]])
    ax3.set_xlim([ds1["time"][0], ds1["time"][-1]])
    ax4.set_xlim([ds1["time"][0], ds1["time"][-1]])
    ax5.set_xlim([ds1["time"][0], ds1["time"][-1]])
    ax6.set_xlim([ds1["time"][0], ds1["time"][-1]])
    ax7.set_xlim([ds1["time"][0], ds1["time"][-1]])

    # Add number of ticks on xaxis:
    ax7.set_xticks(ds1["time_records_without_gaps"][0::216])

    # Save plot:
    plt.savefig(
        output_folder
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "/"
        + "event_overview"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + ".png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()
    status = (
        "Figures saved at "
        + output_folder
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "/"
        + "event_overview"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + ".png"
    )

    return status


###################################
def plot_maps(
    output_folder, sd, mode_n, start_datet, end_datet
):  # sd is the dataset can be any of the TiVIE runs or datasets (e.g. use ds3 for storms and set mode = 3)
    # Check that the output folder exists:
    if (
        glob.glob(
            (output_folder + "/" + str(start_datet) + "_" + str(end_datet) + "/maps/")
        )
        == []
    ):
        os.makedirs(
            output_folder + "/" + str(start_datet) + "_" + str(end_datet) + "/maps/"
        )
    if mode_n == 1:
        title_addendum = "IMF mode"
    if mode_n == 2:
        title_addendum = "Substorm mode"
    if mode_n == 3:
        title_addendum = "Geomagnetic storm mode"
    if (mode_n != 1) & (mode_n != 2) & (mode_n != 3):
        print(
            "You did not select a valid TiVIE mode! Please select a valid TiVIE mode (1,2 or 3)."
        )
        status = "Error in mode selection."
        return status
    for i in range(0, len(sd["time"])):
        fig = plt.figure(figsize=(11, 10))
        tmp_dt = datetime.datetime(
            year=int(sd["time"].isel(time=i).dt.year),
            month=int(sd["time"].isel(time=i).dt.month),
            day=int(sd["time"].isel(time=i).dt.day),
            hour=int(sd["time"].isel(time=i).dt.hour),
            minute=int(sd["time"].isel(time=i).dt.minute),
        )
        # Add title:
        ts = pd.to_datetime(np.datetime_as_string(sd["time"].isel(time=i)))
        d_title = ts.strftime("%d %B %Y %H:%M") + " UT"
        fig.suptitle(d_title, fontsize=16, fontweight="bold")

        # Add grids:
        ax1 = add_grids(1, 1, 1, "ax1", "TiVIE Model v1.0: " + title_addendum)

        # Add data:
        ft_tmp_dt = datetime.datetime(
            year=int(sd["file_time"].isel(file_time=i).dt.year),
            month=int(sd["file_time"].isel(file_time=i).dt.month),
            day=int(sd["file_time"].isel(file_time=i).dt.day),
            hour=int(sd["file_time"].isel(file_time=i).dt.hour),
            minute=int(sd["file_time"].isel(file_time=i).dt.minute),
        )
        levels = np.arange(int(-33), int(34), 6)
        shft = aacgmv2.convert_mlt([0], ft_tmp_dt, m2a=False) * 15.0
        lons_rads = np.deg2rad(np.array(sd["mlon_arr"]) + shft)
        colats = sd["mlat_arr"]
        colats = colats[colats >= 0]
        pot = sd["potential"].sel(time=tmp_dt, mlat_arr=colats, method="nearest")
        colats = 90.0 - colats

        # Fix the missing wedge issue in matplotlib:
        dlon = lons_rads[1] - lons_rads[0]
        wrp_lons_rads = np.concatenate((lons_rads, lons_rads[-1:] + dlon))
        wrp_pot = np.concatenate((pot, pot[:, 0:1]), axis=1)

        # Plot potentials:
        contours = ax1.contour(
            wrp_lons_rads,
            colats,
            wrp_pot,
            colors="k",
            levels=levels,
            extend="both",
            linewidths=0.5,
            alpha=0.6,
        )
        pot_map = ax1.contourf(
            wrp_lons_rads,
            colats,
            wrp_pot,
            cmap="RdBu_r",
            levels=levels,
            extend="both",
        )

        # Plot HMB:
        lons_rads_hmb = np.deg2rad(np.array(sd["boundary_mlon"].isel(time=i)) + shft)
        colats_hmb = 90.0 - sd["boundary_mlat"].isel(time=i)
        ax1.plot(lons_rads_hmb, colats_hmb, color="aquamarine")

        # Add locations of measurements:
        lons_rads_meas = np.deg2rad(np.array(sd["mlon_meas"].isel(time=i)) + shft)
        colats_meas = 90.0 - sd["mlat_meas"].isel(time=i)
        ax1.scatter(
            lons_rads_meas,
            colats_meas,
            marker=".",
            color="black",
            s=0.1,
            alpha=0.6,
        )

        # Add anotations:
        ax1.annotate(
            "n:" + str(np.int64(sd["n"].isel(time=i).values)),
            xy=(np.deg2rad(-45.0), 80),
            xycoords="data",
            annotation_clip=False,
        )
        ax1.annotate(
            "$\Phi_{CPCP}$:"
            + str(np.int64((np.abs(wrp_pot.min()) + wrp_pot.max()) * 100.0) / 100.0)
            + "kV",
            xy=(np.deg2rad(-45.0), 90),
            xycoords="data",
            annotation_clip=False,
        )

        # Add colourbar:
        fig = plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax = plt.axes([0.85, 0.1, 0.025, 0.8])
        fig = plt.colorbar(pot_map, cax=cax, label="Potential [kV]", ticks=levels)

        # Save figure:
        plt.savefig(
            output_folder
            + "/"
            + str(start_datet)
            + "_"
            + str(end_datet)
            + "/maps/TiVIE_mode_"
            + str(mode_n)
            + "_"
            + "{:05.0f}".format(i)
            + ".png",
            dpi=300,
            edgecolor="white",
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
        )
        print(i, sd["time"][i].values)
    status = (
        "Figures saved at "
        + output_folder
        + "/"
        + str(start_datet)
        + "_"
        + str(end_datet)
        + "/maps/TiVIE_mode_"
        + str(mode_n)
        + "_***.png"
    )
    plt.close("all")
    return status
