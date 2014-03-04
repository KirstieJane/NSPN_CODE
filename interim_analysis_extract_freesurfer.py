#!/usr/bin/env python

# Extract some freesurfer measures from the aseg.stats files
# that are in the NSPN interim analysis directory, created by
# Roger.

# Kirstie Whitaker
# kw401@cam.ac.uk
# 12th January 2014

# IMPORTS
from glob import glob
import os
import re

# First define the directory locations
nspn_dir = glob(os.path.join('/*','*','NSPN'))[0]
preprocessed_dir = os.path.join(nspn_dir, 'data_ready', '1.0-preprocessed')
freesurferR1_dir=(os.path.join(preprocessed_dir, 'uchange.interim.2013', 'dicom', 'FREESURFER_R1'))
mpm_dir=(os.path.join(preprocessed_dir, 'uchange.interim.2013', 'dicom', 'MPM'))

# Define the measures we want to extract
measures = [ 'ICV', 'CorticalWhiteMatterVol', 'CortexVol', 'SubCortGrayVol', 'TotalGrayVol' ]

for freesurfer_dir, suffix in zip(freesurferR1_dir, 'R1'):
    # Create a list of the aseg stats files in the freesurfer directory
    aseg_files = sorted(glob(os.path.join(freesurfer_dir, '*struc_brain', 'stats', 'aseg.stats')))

    # The NSPN ids are the first part of the filename
    nspn_ids = [ name.split(os.path.sep)[-3].split('_')[0] for name in aseg_files ]

    # Finally, we can start to write out the information we need
    # Name the output file you're going to write to:
    output_file = os.path.join(preprocessed_dir, 'freesurfer_volume_data_' + suffix + '.csv')
    
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