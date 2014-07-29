#!/usr/bin/env python

# THIS IS VERY ROUGH AND READY
# Importantly it doesn't take into account
# any longitudinal data
# Created on 22nd July 2014
# by Kirstie Whitaker
# kw401@cam.ac.uk

# IMPORTS
import pandas as pd
import numpy as np
from glob import glob
import os
import sys

# Get the fs_rois_dir from the command line
fs_rois_dir=sys.argv[1]
# Get the full path to the behaviour csv file
behav_file=sys.argv[2]

file_list_mean = glob(os.path.join(fs_rois_dir, '*mean.csv'))
file_list_volume = glob(os.path.join(fs_rois_dir, '*volume.csv'))
file_list = file_list_mean + file_list_volume

df_behav = pd.read_csv(behav_file)

for f in file_list:
    df_meas = pd.read_csv(f)
    if not df_meas.columns[1] == 'occ':
        df = df_behav.merge(df_meas, on='nspn_id')
        df.sort('nspn_id', inplace=True)
        c_drop = [ x for x in df.columns if 'Measure' in x ][0]
        df.drop(c_drop, inplace=True, axis=1)
        df.to_csv(f.replace('.csv', '_behavmerge.csv'))
