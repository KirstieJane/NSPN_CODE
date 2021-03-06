#!/usr/bin/env python

'''
Making synthetic data to test the cortical projection accuracy
'''

import nibabel as nib
import numpy as np
import os
import sys

if len(sys.argv) == 1:
    print 'No surfer directory provided!'
    print 'USAGE: making_synthetic_data.py <surfer_dir>'
    print '    eg: making_synthetic_data.py /home/kw401/UCHANGE/INTERIM_ANALYSIS/SUB_DATA/10736/SURFER/MRI0/'
    sys.exit()
elif not os.path.isdir(sys.argv[1])
    print 'Surfer directory does not exist!'
    print 'USAGE: making_synthetic_data.py <surfer_dir>'
    print '    eg: making_synthetic_data.py /home/kw401/UCHANGE/INTERIM_ANALYSIS/SUB_DATA/10736/SURFER/MRI0/'
    sys.exit()

# Get the surfer directory from the arguments
surfer_dir = sys.argv[1]

# Define the input and output filenames
aparc_cort_filename = os.path.join(surfer_dir, 'parcellation', '500.aparc_cortical_consecutive.nii.gz')
aparc_white_filename = os.path.join(surfer_dir, 'parcellation', '500.aparc_cortical_expanded_consecutive_WMoverlap.nii.gz')
MT_filename = os.path.join(surfer_dir, 'mri', 'MT.mgz')
synth_filename = os.path.join(surfer_dir, 'mri', 'synthetic.mgz')

# Check that the input file names all exist
for f_name in [ aparc_cort_filename, aparc_white_filename, MT_filename ]:
    if not os.path.isfile(f_name):
        print "ERROR: {} doesn't exist in 'parcellation' folder".format(f_name)
        print 'USAGE: making_synthetic_data.py <surfer_dir>'
        print '    eg: making_synthetic_data.py /home/kw401/UCHANGE/INTERIM_ANALYSIS/SUB_DATA/10736/SURFER/MRI0/'
        sys.exit()

# If the output file already exists then you don't have to overwrite it
if os.path.isfile(synth_filename):
    print 'Data already exists - exiting. Delete {} if you want to overwrite it'.format(synth_filename)
    sys.exit()
    
# Load the parcellation and MT files
parc_cort_img = nib.load(aparc_cort_filename)
parc_white_img = nib.load(aparc_white_filename)
MT_img = nib.load(MT_filename)

# Get the data
parc_cort_data = parc_cort_img.get_data()
parc_cort_data = parc_cort_data[...,0]
parc_white_data = parc_white_img.get_data()
MT_data = MT_img.get_data()

# Create a copy of the MT_data as the synth_data
synth_data = np.zeros_like(MT_data)

# Loop through the different parcellations, assign the 
# appropriate voxels from the MT map to the synth_data
# and shuffle the voxel values within each region
print 'Calculating synthetic data for each parcel'
for i in range(1,309):
    # cort
    synth_data[parc_cort_data==i] = MT_data[parc_cort_data==i]
    x = synth_data[parc_cort_data==i]
    np.random.shuffle(x)
    synth_data[parc_cort_data==i] = x
    
    # white
    synth_data[parc_white_data == i] = MT_data[parc_white_data == i]
    x = synth_data[parc_white_data==i]
    np.random.shuffle(x)
    synth_data[parc_white_data==i] = x

# Now save the file
synth_img = nib.MGHImage(synth_data, MT_img.get_affine())
synth_img.to_filename(synth_filename)
