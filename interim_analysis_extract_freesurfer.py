#!/usr/bin/env python

# Extract some freesurfer measures from the aseg.stats files
# that are in the NSPN interim analysis directory, created by
# Roger.

# Kirstie Whitaker
# kw401@cam.ac.uk
# 12th January 2014

#==============================================================================
# IMPORTS
#------------------------------------------------------------------------------
from glob import glob
import os
import re
import pandas as pd

#==============================================================================
# DEFINE SOME FUNCTIONS
#------------------------------------------------------------------------------
# file_list = glob('*.stats')

# for file in file_list:

    # print file
    # with open(file, 'r') as f:
        # for i, line in enumerate(f.readlines()):
            # if 'ColHeaders' in line:
                # headers_line=i
                # headers_list=line.split()[2:]

    # data = pd.read_csv(file, skiprows=headers_line+1, header=None, names=headers_list, delim_whitespace=True, index_col=1)

    # print data.describe()

# mri_annotation2label --sd `pwd` --subject SURF_R1 --hemi rh --lobesStrict SURF_R1/label/rh.lobesStrict

# mris_anatomical_stats -a SURF_R1/label/rh.lobesStrict.annot -f SURF_R1/stats/rh.lobesStrict.stats SURF_R1 rh


#==============================================================================
# DEFINE SOME VARIABLES
#------------------------------------------------------------------------------
# First define the original directory locations
nspn_dir = glob(os.path.join('/*','*','NSPN'))[0]
preprocessed_dir = os.path.join(nspn_dir, 'data_ready', '1.0-preprocessed')
freesurferR1_dir=(os.path.join(preprocessed_dir, 'uchange.interim.2013', 'dicom', 'FREESURFER_R1'))

# Now define the directory that you're moving everyone to:
data_dir=os.path.join(nspn_dir, 'workspaces', 'kw401', 'UCHANGE', 'INTERIM_ANALYSIS')

# And create a list of subject IDs
sublist_file = os.path.join(data_dir, 'sublist')
with open(sublist_file, 'r') as sublist_fid:
    sublist = sublist_fid.read().split('\n')

# Name the output file you're going to write to:
output_file = os.path.join(data_dir, 'FS_SUMMARY', 'KW_freesurfer_volume_data_' + suffix + '.csv')
if not os.path.isdir(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))
 
# Define the measures we want to extract
measures = [ 'ICV', 'CorticalWhiteMatterVol', 'CortexVol', 'SubCortGrayVol', 'TotalGrayVol' ]

#==============================================================================
# RUN THE ACTUAL CODE!
#------------------------------------------------------------------------------
# For each person we're going to just concatenate the 

# Loop through the subjects
for sub in sublist:
    sub_dir=os.path.join(data_dir, 'SUB_DATA', sub, 'HIGHRES', 'MRI0')
    

for freesurfer_dir, suffix in zip(freesurferR1_dir, 'R1'):
    # Create a list of the aseg stats files in the freesurfer directory
    aseg_files = sorted(glob(os.path.join(freesurfer_dir, '*struc_brain', 'stats', 'aseg.stats')))

    # The NSPN ids are the first part of the filename
    nspn_ids = [ name.split(os.path.sep)[-3].split('_')[0] for name in aseg_files ]

    # Finally, we can start to write out the information we need
    
    # and open it
    with open(output_file, 'w') as f:
    
        # Start by writing out the header
        # First column is the nspn_id
        f.write('nspn_id')
        with open(aseg_files[0], 'r') as names_file:
            for line in names_file:
                for measure in measures:
                    if re.search(measure, line):
                        line_split = line.split(',')
                        f.write(',{}_{}'.format(line_split[1].strip(), suffix))
        f.write('\n')

        # Loop through all the nspn_ids and write out the important information you need
        for sub, aseg_file in zip(nspn_ids, aseg_files):
            f.write('{}'.format(sub))
            with open(aseg_file, 'r') as stats_f:
                for line in stats_f:
                    for measure in measures:
                        if re.search(measure, line):
                            line_split = line.split(',')
                            f.write(',{}'.format(line_split[3].strip()))
            f.write('\n')
    # That's the end