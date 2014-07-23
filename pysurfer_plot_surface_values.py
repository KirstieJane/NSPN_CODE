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


data_dir = os.path.join('/work','imagingG', 'NSPN', 'workspaces', 'kw401', 'UCHANGE','INTERIM_ANALYSIS')
subjects_dir = os.path.join(data_dir ,'SUB_DATA')
fs_rois_dir = os.path.join(data_dir, 'FS_ROIS')

subject_id = "fsaverageSubP"
hemi_list = [ "lh", "rh" ]
surface_list = [ "inflated", "pial" ]

measure_list = [ 'FA', 'MD' ]

for hemi, surface, measure in it.product(hemi_list, surface_list, measure_list):

    print hemi, surface, measure
    
    """
    Bring up the visualization.
    """
    brain = Brain(subject_id, hemi, surface,
                  subjects_dir = subjects_dir,
                  config_opts=dict(background="white"))

    """
    Read in the standard parcellation
    Of course you'll have to change this when you're looping through different files!
    """
    aparc_file = os.path.join(subjects_dir,
                              subject_id, "label",
                              hemi + ".aparc.annot")
                              
    labels, ctab, names = nib.freesurfer.read_annot(aparc_file)

    """
    Read in the data
    """
    data_file = os.path.join(fs_rois_dir, 
                                measure + '_wmparc_mean_behavmerge.csv')
    df = pd.read_csv(data_file)
    
    """
    Fill in the data for each region on the surface
    """
    roi_data_mean = np.zeros(len(names))
    roi_data_std = np.zeros(len(names))
    roi_data_r = np.zeros(len(names))
    roi_data_p = np.ones(len(names))
    roi_data_m = np.ones(len(names))
    
    for i, name in enumerate(names):
        wm_name = 'wm-' + hemi + '-' + name
        if wm_name in df.columns:
            roi_data_mean[i] = df[wm_name].mean()
            roi_data_std[i] = df[wm_name].std()
            m, c, r, p, sterr = linregress(df.age_scan, df[wm_name])
            roi_data_m[i] = m
            roi_data_r[i] = r
            roi_data_p[i] = 1 - p


    """
    Make a vector containing the data point at each vertex.
    """
    vtx_data_mean = roi_data_mean[labels]
    vtx_data_std = roi_data_std[labels]
    vtx_data_r = roi_data_r[labels]
    vtx_data_p = roi_data_p[labels]
    vtx_data_m = roi_data_m[labels]

    """
    Display these values on the brain.
    """
    ### MEAN
    min = roi_data_mean[roi_data_mean<>0].min()
    max = roi_data_mean[roi_data_mean<>0].max()
    print min, max
    brain.add_data(vtx_data_mean,
                    roi_data_mean[roi_data_mean<>0].min(), 
                    roi_data_mean[roi_data_mean<>0].max(),
                    thresh = 0,
                    colormap="GnBu",
                    alpha=.8)
    
    views_list = [ 'medial', 'lateral' ]
    prefix = '_'.join([measure, hemi, surface, 'mean'])
    brain.save_imageset(prefix = os.path.join(fs_rois_dir, prefix),
                        views = views_list, 
                        colorbar = range(len(views_list)) )
                        
    