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

#=============================================================================
# FUNCTIONS
#=============================================================================

def usage():
    print "USAGE freesurfer_rois_behavmerge.py <FS_ROIS_DIR> <BEHAV_FILE>"
    sys.exit()
    
#=============================================================================
# READ IN COMMAND LINE ARGUMENTS
#=============================================================================
# Get the fs_rois_dir from the command line
fs_rois_dir=sys.argv[1]
# Get the full path to the behaviour csv file
behav_file=sys.argv[2]

if not os.path.isfile(behav_file):
    print "Can't find behav_file"
    usage()

#=============================================================================
# READ IN BEHAVIOURAL FILE
#=============================================================================
df_behav = pd.read_csv(behav_file)
    
#=============================================================================
# MERGE SEGMENTATION VALUES
#=============================================================================
file_list_mean = glob(os.path.join(fs_rois_dir, '*mean.csv'))
file_list_volume = glob(os.path.join(fs_rois_dir, '*volume.csv'))
file_list = file_list_mean + file_list_volume

for f in file_list:
    df_meas = pd.read_csv(f)
    df = df_behav.merge(df_meas, on=['nspn_id', 'occ'])
    df.sort('nspn_id', inplace=True)
    c_drop = [ x for x in df.columns if 'Measure' in x ][0]
    df.drop(c_drop, inplace=True, axis=1)
    df.to_csv(f.replace('.csv', '_behavmerge.csv'))

#=============================================================================
# MERGE SURFACE PARCELLATION VALUES
#=============================================================================
measure_list = [ 'area', 'volume', 'thickness', 
                    'meancurv', 'gauscurv', 
                    'foldind', 'curvind' ]

file_list = []
for measure in meaure_list:
    file_list += glob(os.path.join(fs_rois_dir, '*{}.csv'.format(measure)))

for f in file_list:
    df_meas = pd.read_csv(f)
    df = df_behav.merge(df_meas, on=['nspn_id', 'occ'])
    df.sort('nspn_id', inplace=True)
    c_drop = [ x for x in df.columns if '.' in x ][0]
    df.drop(c_drop, inplace=True, axis=1)
    df.to_csv(f.replace('.csv', '_behavmerge.csv'))
