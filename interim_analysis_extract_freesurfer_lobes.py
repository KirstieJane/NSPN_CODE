#!/usr/bin/env python

'''
Extract some measures from the stats files created by freesurfer.

Note that this is designed to run within Kirstie's DTI processing pipeline
That is, the subject "name" is SURF and it will be found in the HIGHRES
directory.

Kirstie Whitaker
kw401@cam.ac.uk
Started: 12th January 2014
'''

#==============================================================================
# IMPORTS
#------------------------------------------------------------------------------
from glob import glob
import os
import re
import pandas as pd

#==============================================================================
# DEFINE SOME VARIABLES
#------------------------------------------------------------------------------
# First define the original directory locations
nspn_dir = glob(os.path.join('/*','*','NSPN'))[0]
#preprocessed_dir = os.path.join(nspn_dir, 'data_ready', '1.0-preprocessed')
#freesurferR1_dir=(os.path.join(preprocessed_dir, 'uchange.interim.2013', 'dicom', 'FREESURFER_R1'))

# Now define the directory that you're moving everyone to:
data_dir=os.path.join(nspn_dir, 'workspaces', 'kw401', 'UCHANGE', 'INTERIM_ANALYSIS')

# And create a list of subject IDs
sublist_file = os.path.join(data_dir, 'sublist')
with open(sublist_file, 'r') as sublist_fid:
    sublist = sublist_fid.read().split('\n')

# Name the output file you're going to write to:
output_file = os.path.join(data_dir, 'FS_SUMMARY', 'KW_freesurfer_volume_data.csv')
if not os.path.isdir(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))
 
# Define the measures we want to extract
measures = [ 'ICV', 'CorticalWhiteMatterVol', 'CortexVol', 'SubCortGrayVol', 'TotalGrayVol', 'frontal', 'occipital', 'temporal', 'parietal' ]

#==============================================================================
# RUN THE ACTUAL CODE!
#------------------------------------------------------------------------------
# For each person we're going to just concatenate the 

# Loop through the subjects
# Create a list of the aseg stats files in the freesurfer directory
aseg_files = sorted(glob(os.path.join(data_dir, 'SUB_DATA', '*', 'HIGHRES',  'MRI0', 'SURF_R1', 'stats', 'aseg.stats')))
lobes_files_right = sorted(glob(os.path.join(data_dir, 'SUB_DATA', '*', 'HIGHRES',  'MRI0', 'SURF_R1', 'stats', 'rh.lobesStrict.stats')))
lobes_files_left = sorted(glob(os.path.join(data_dir, 'SUB_DATA', '*', 'HIGHRES',  'MRI0', 'SURF_R1', 'stats', 'lh.lobesStrict.stats')))

# Open the output_file
with open(output_file, 'w') as f:

    # Start by writing out the header
    # First column is the nspn_id
    f.write('nspn_id')
    with open(aseg_files[0], 'r') as names_file:
        for line in names_file:
            for measure in measures:
                if re.search(measure, line):
                    line_split = line.split(',')
                    f.write(',{}'.format(line_split[1].strip()))
    with open(lobes_files_left[0], 'r') as names_file:
        for line in names_file:
            for measure in measures:
                if re.search(measure, line):
                    line_split = line.split()
                    f.write(',{}_lh'.format(line_split[0].strip()))
    with open(lobes_files_right[0], 'r') as names_file:
        for line in names_file:
            for measure in measures:
                if re.search(measure, line):
                    line_split = line.split()
                    f.write(',{}_rh'.format(line_split[0].strip()))
    f.write('\n')

    # Loop through all the nspn_ids and write out the important information you need
    for sub, aseg_file, lobe_right, lobe_left in zip(sublist, aseg_files, lobes_files_right, lobes_files_left):
        f.write('{}'.format(sub))
        with open(aseg_file, 'r') as stats_f:
            for line in stats_f:
                for measure in measures:
                    if re.search(measure, line):
                        line_split = line.split(',')
                        f.write(',{}'.format(line_split[3].strip()))

        with open(lobe_left, 'r') as stats_f:
            for line in stats_f:
                for measure in measures:
                    if re.search(measure, line):
                        line_split = line.split()
                        f.write(',{}'.format(line_split[3].strip()))
        with open(lobe_right, 'r') as stats_f:
            for line in stats_f:
                for measure in measures:
                    if re.search(measure, line):
                        line_split = line.split()
                        f.write(',{}'.format(line_split[3].strip()))
        f.write('\n')
# That's the end
