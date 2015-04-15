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
# MERGE MEASURES WITH BEHAV VALUES
#=============================================================================
measure_list = [ 'mean', 'area', 
                    'volume', 'thickness',
                    'thicknessstd',
                    'meancurv', 'gauscurv', 
                    'foldind', 'curvind' ]

file_list = []
for measure in measure_list:
    file_list += glob(os.path.join(fs_rois_dir, '*{}.csv'.format(measure)))

file_list += glob(os.path.join(fs_rois_dir, '*mean.csv'))
file_list += glob(os.path.join(fs_rois_dir, '*std.csv'))

for f in file_list:
    print f
    df_meas = pd.read_csv(f)
    if 'nspn_id' in df_meas.columns:
        df = df_behav.merge(df_meas, on=['nspn_id', 'occ'])
        df.sort('nspn_id', inplace=True)
        c_drop = [ x for x in df.columns if 'Measure' in x ]
        #c_drop += [ x for x in df.columns if '.' in x ]
        if c_drop:
            df.drop(c_drop, inplace=True, axis=1)
        f_out = f.replace('.', '')
        f_out = f_out.replace('csv', '_behavmerge.csv')
        df.to_csv(f_out,float_format='%.5f')
        
