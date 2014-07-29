#!/usr/bin/env python

# NEEDS SOME TIDYING BUT IT WORKS!

import os
import numpy as np
import scipy.io as sio
from glob import glob
import pandas as pd

import nibabel as nib
from surfer import Brain

import itertools as it
from scipy.stats.stats import linregress

import matplotlib.pylab as plt
import sys

data_file = sys.argv[1]

if not os.path.isfile(data_file):
    print "NO CSV FILE"
    sys.exit()
    
data_dir = os.path.join('/work','imagingG', 'NSPN', 'workspaces', 'kw401', 'UCHANGE','INTERIM_ANALYSIS')
subjects_dir = os.path.join(data_dir ,'SUB_DATA')
fs_rois_dir = os.path.join(data_dir, 'FS_ROIS')

subject_id = "fsaverageSubP"
hemi_list = [ "lh", "rh" ]
surface_list = [ "inflated", "pial" ]

for hemi, surface in it.product(hemi_list, surface_list):

    print hemi, surface
    
    """
    Read in the standard parcellation
    Of course you'll have to change this when you're looping through different files!
    """
    aparc_file = os.path.join(subjects_dir,
                              subject_id, "label",
                              hemi + ".500.aparc.annot")
                              
    aparc_names_file =  os.path.join(subjects_dir,
                              subject_id, "parcellation",
                              "500.names.txt")
                              
    parc_names = [line.strip() for line in open(aparc_names_file)]
    parc_names = parc_names[41::]
    labels, ctab, names = nib.freesurfer.read_annot(aparc_file)
    
    """
    Read in the data
    """
    
    df = pd.read_csv(data_file, header=None, names=parc_names, dtype=np.float)

    roi_data = np.zeros(len(names))*-99
    
    for i, name in enumerate(names):
        parc_name = '{}_{}'.format(hemi, name)

        if parc_name in df.columns:
            roi_data[i] = df[parc_name].mean()
            
    
    """
    Figure out the thresholds
    """
    u = df.max(axis=1).values[0]
    l = df.min(axis=1).values[0]
    u = np.ceil(u*20.0)/20.0
    l = np.floor(l*20.0)/20.0
    
    if l < 0:
        cmap = "RdBu_r"
        if l**2 > u**2:
            u = l*-1
        else:
            l = u*-1
    else:
        cmap = "jet"
        
    """
    Make a vector containing the data point at each vertex.
    """
    vtx_data = roi_data[labels]
    
    """
    Display these values on the brain.
    """
    ### MEAN
    brain = Brain(subject_id, hemi, surface,
                  subjects_dir = subjects_dir,
                  config_opts=dict(background="white"))

    
    brain.add_data(vtx_data,
                    l, 
                    u,
                    thresh = -98,
                    colormap=cmap,
                    alpha=.8)
    
    views_list = [ 'medial', 'lateral' ]
    prefix = '_'.join([data_file.strip('.csv'), surface, hemi])
    brain.save_imageset(prefix = os.path.join(fs_rois_dir, prefix),
                        views = views_list, 
                        colorbar = range(len(views_list)) )
                        