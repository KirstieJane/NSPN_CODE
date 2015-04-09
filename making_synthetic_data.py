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
synth_filename = os.path.join(surfer_dir, 'synthetic', 'synthetic.nii.gz')

# Check that the input file name exists
if not os.path.isfile(aparc_filename):
    print "ERROR: 500.aparc.nii.gz doesn't exist in 'parcellation' folder"
    print 'USAGE: making_synthetic_data.py <surfer_dir>'
    print '    eg: making_synthetic_data.py /home/kw401/UCHANGE/INTERIM_ANALYSIS/SUB_DATA/10736/SURFER/MRI0/'
    sys.exit()

# Make the output directory if it doesn't yet exist
if not os.path.isdir(os.path.join(surfer_dir, 'synthetic')):
    os.mkdir(os.path.join(surfer_dir, 'synthetic'))

# Load the parcellation file    
img = nib.load(aparc_filename)

# Get the data
data = img.get_data()

# Assign grey matter voxels to have a mean of 900
# and a standard deviation of 50
data[data>1000] = np.random.normal(900, 50, size=data[data>1000].shape)

# Assign white matter voxels to have a mean of 1500
# and a standard deviation of 75
data[data==41] = np.random.normal(1500, 75, size=data[data==41].shape)
data[data==2] = np.random.normal(1500, 75, size=data[data==2].shape)
data[(data>200) & (data<250)] = np.random.normal(1500, 75, size=data[(data>200) & (data<250)].shape)

# Set all other values to 0
data[data<300] = 0

# Now save the file
img_synth = nib.Nifti1Image(data, img.get_affine())
img_synth.to_filename(synth_filename)
