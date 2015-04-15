#!/usr/bin/env python

'''
Making synthetic data to test the cortical projection accuracy
'''

import nibabel as nib
import numpy as np
import os
import sys

if len(sys.argv[1]) == 1:
    print 'No surfer directory provided!'
    print 'USAGE: making_synthetic_data.py <surfer_dir>'
    print '    eg: making_synthetic_data.py /home/kw401/UCHANGE/INTERIM_ANALYSIS/SUB_DATA/10736/SURFER/MRI0/'
    sys.exit()

# Get the surfer directory from the arguments
surfer_dir = sys.argv[1]

# Define the input and output filenames
aparc_filename = os.path.join(surfer_dir, 'parcellation', '500.aparc.nii.gz')
MT_filename = os.path.join(surfer_dir, 'mri', 'MT.mgz')
synth_filename = os.path.join(surfer_dir, 'mri', 'synthetic.mgz')

# Check that the input file name exists
if not os.path.isfile(aparc_filename):
    print "ERROR: 500.aparc.nii.gz doesn't exist in 'parcellation' folder"
    print 'USAGE: making_synthetic_data.py <surfer_dir>'
    print '    eg: making_synthetic_data.py /home/kw401/UCHANGE/INTERIM_ANALYSIS/SUB_DATA/10736/SURFER/MRI0/'
    sys.exit()

# Load the parcellation and MT files
parc_img = nib.load(aparc_filename)
MT_img = nib.load(MT_filename)

# Get the data
parc_data = parc_img.get_data()
parc_data=parc_data[...,0]
MT_data = MT_img.get_data()

# Create a copy of the MT_data as the synth_data
synth_data = np.zeros_like(MT_data)

# Figure out the grey and white matter means from the real data
gm_mean = MT_data[parc_data>1000].mean()
gm_std = MT_data[parc_data>1000].std()
wm_mean = MT_data[(parc_data==41) + (parc_data==2)].mean()
wm_std = MT_data[(parc_data==41) + (parc_data==2)].std()

# Assign grey matter voxels to have a random value from the distribution
# defined by their grey matter mean and standard deviations
synth_data[parc_data>1000] = np.random.normal(gm_mean, gm_std, size=parc_data[parc_data>1000].shape)

# Assign white matter voxels to have a random value from the distribution
# defined by their white matter mean and standard deviations
synth_data[parc_data==41] = np.random.normal(wm_mean, wm_std, size=parc_data[parc_data==41].shape)
synth_data[parc_data==2] = np.random.normal(wm_mean, wm_std, size=parc_data[parc_data==2].shape)
synth_data[(parc_data>200) & (parc_data<250)] = np.random.normal(wm_mean, 
                                                                    wm_std, 
                                                                    size=parc_data[(parc_data>200)
                                                                                    & (parc_data<250)].shape)

# Now save the file
synth_img = nib.MGHImage(synth_data, MT_img.get_affine())
synth_img.to_filename(synth_filename)
