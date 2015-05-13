#!/usr/bin/env python

'''
Freesurfer_rois_behavmerge.py
Created on 22nd July 2014
by Kirstie Whitaker
kw401@cam.ac.uk

This code merges the output of freesurfer_combine_CT_stats.sh
with a csv file containing nspn_id and occ to create a
"behav_merge" file which is saved in a folder called FS_BEHAV
which is created in the same folder as the behavioural file
'''

#=============================================================================
# IMPORTS
#=============================================================================
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
# Create a list of the freesurfer measures
measure_list = [ 'mean', 'area', 
                    'volume', 'thickness',
                    'thicknessstd',
                    'meancurv', 'gauscurv', 
                    'foldind', 'curvind',
                     'std' ]

# Create an empty file list
file_list = []

# Loop through all the measures, find all the files that end with 
# those words and add them to the file list
for measure in measure_list:
    
    file_list += glob(os.path.join(fs_rois_dir, '*{}.csv'.format(measure)))

# Loop through the files
for f in file_list:
    print f
    # Check the number of lines that are in the file
    with open(f) as fid:
        num_lines = len(fid.readlines())
        
    # And only try to merge files that have content
    if num_lines > 0:
        
        # Read the csv roi file into a data frame
        df_meas = pd.read_csv(f)
        
        if 'nspn_id' in df_meas.columns:
            
            # Merge on 'nspn_id' and 'occ'
            df = df_behav.merge(df_meas, on=['nspn_id', 'occ'])
            
            # Sort into ascending nspn_id
            df.sort('nspn_id', inplace=True)
            
            # Drop columns containing the word 'Measure' 
            # if they exist
            c_drop = [ x for x in df.columns if 'Measure' in x ]
            if c_drop:
                df.drop(c_drop, inplace=True, axis=1)
            
            # Create an output file name that removes any '.' symbols
            # in the file name
            f_name = os.path.basename(f)
            f_out = f_name.replace('.', '')
            # and appends '_behavmerge.csv'
            f_out = f_out.replace('csv', '_behavmerge.csv')
            
            # Put this file in the same folder as the behavioural file
            behav_dir = os.path.dirname(behav_file)
            f_out = os.path.join(behav_dir, f_out)
            df.to_csv(f_out,float_format='%.5f')
            
