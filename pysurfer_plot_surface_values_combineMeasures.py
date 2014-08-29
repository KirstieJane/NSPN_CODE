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
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


def combine_pngs(measure1, measure2, surface, stat, fs_rois_dir):

    figsize = (4.5,4)
    fig = plt.figure(figsize = figsize, facecolor='white')

    grid = gridspec.GridSpec(2, 2)
    grid.update(left=0, right=1, top=1, bottom = 0.08, wspace=0, hspace=0)

    f_list = [ '_'.join([os.path.join(fs_rois_dir, measure1), 'lh', surface, stat,       measure2, 'lateral.png']),
                '_'.join([os.path.join(fs_rois_dir, measure1), 'rh', surface, stat, measure2, 'lateral.png']),
                '_'.join([os.path.join(fs_rois_dir, measure1), 'lh', surface, stat, measure2, 'medial.png']),
                '_'.join([os.path.join(fs_rois_dir, measure1), 'rh', surface, stat, measure2, 'medial.png']) ]

                           
    for g_loc, f in zip(grid, f_list):
        ax = plt.Subplot(fig, g_loc)
        fig.add_subplot(ax)
        img = mpimg.imread(f)
        if 'lateral' in f:
            img_cropped = img[58:598,60:740,:]
        else:
            img_cropped = img[28:618,60:740,:]
        ax.imshow(img_cropped, interpolation='none')
        ax.set_axis_off()

    grid_cbar = gridspec.GridSpec(1,1)
    grid_cbar.update(left=0, right=1, top=0.08, bottom=0, wspace=0, hspace=0)
    ax = plt.Subplot(fig, grid_cbar[0])
    fig.add_subplot(ax)
    img = mpimg.imread(f)
    img_cbar = img[605:,:]
    ax.imshow(img_cbar, interpolation='none')
    ax.set_axis_off()
    
    filename = os.path.join(fs_rois_dir, '{}_{}_{}_{}_combined.png'.format(measure1,
    surface,stat, measure2))
    print filename
    fig.savefig(filename, bbox_inches=0, dpi=300)




data_dir = os.path.join('/work','imagingG', 'NSPN', 'workspaces', 'kw401', 'UCHANGE','INTERIM_ANALYSIS')
subjects_dir = os.path.join(data_dir ,'SUB_DATA')
fs_rois_dir = os.path.join(data_dir, 'FS_ROIS')

subject_id = "fsaverageSubP"
hemi_list = [ "lh", "rh" ]
surface_list = [ "inflated", "pial" ]

measure_list = [ 'FA' ]
measure2 = 'CT'

for measure1 in measure_list:
    for hemi, surface in it.product(hemi_list, surface_list):

        print hemi, surface, measure1
    
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
                                  
        wm_names = [line.strip() for line in open(aparc_names_file)]
        wm_names = wm_names[41::]
        labels, ctab, names = nib.freesurfer.read_annot(aparc_file)
        
        """
        Read in the data
        """
        data_file1 = os.path.join(fs_rois_dir, 
                                    measure1 + '_500cortExpConsecWMoverlap_mean_behavmerge.csv')
        df1 = pd.read_csv(data_file1)
        
        data_file2 = os.path.join(fs_rois_dir, 
                                    measure2 + '_500cortExpConsecWMoverlap_mean_behavmerge.csv')
        df2 = pd.read_csv(data_file2)
        
        # RENAME THE COLUMNS SO THEY MATCH THE NAMES IN THE APARC FILE
        # NOTE: this really should be put into the behav merge code
        # rather than here!
        colnames = df1.columns
        seg_cols = [ x for x in colnames if 'Seg' in x ]
        
        for i, wm_name in enumerate(wm_names):
            df1[wm_name] = df1[seg_cols[i+1]]
            df2[wm_name] = df2[seg_cols[i+1]]

        """
        Fill in the data for each region on the surface
        """
        roi_data_mean = np.ones(len(names))*-99
        roi_data_std = np.ones(len(names))*-99
        roi_data_r = np.ones(len(names))*-99
        roi_data_p = np.ones(len(names))*-99
        roi_data_m = np.ones(len(names))*-99
            
        for i, name in enumerate(names):
            #wm_name = 'wm-' + hemi + '-' + name
            wm_name = '{}_{}'.format(hemi, name)
     
            if wm_name in df1.columns:
                df_merge = df1.merge(df2, on='nspn_id')
                roi_data_mean[i] = df1[wm_name].mean()
                roi_data_std[i] = df1[wm_name].std()
                m, c, r, p, sterr = linregress(df_merge[wm_name + '_x'], df_merge[wm_name + '_y'])
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
        
        ### PEARSON CORR w MEASURE 2
        brain = Brain(subject_id, hemi, surface,
                      subjects_dir = subjects_dir,
                      config_opts=dict(background="white"))

        l = roi_data_r[roi_data_mean>-99].min()
        u = roi_data_r[roi_data_mean>-99].max()
        l = np.floor(l*20)/20.0
        u = np.ceil(u*20)/20.0
        
        # Make sure the colorbar is centered
        if l**2 < u **2:
            l = u*-1
        else:
            u = l*-1
        
        brain.add_data(vtx_data_r,
                        -0.35, 
                        0.35,
                        thresh = -98,
                        colormap="RdBu_r",
                        alpha=.8)
        
        views_list = [ 'medial', 'lateral' ]
        prefix = '_'.join([measure1, hemi, surface, 'r', measure2])
        brain.save_imageset(prefix = os.path.join(fs_rois_dir, prefix),
                            views = views_list, 
                            colorbar = range(len(views_list)) )
                            

        ### SIGNIFICANCE w AGE
        brain = Brain(subject_id, hemi, surface,
                      subjects_dir = subjects_dir,
                      config_opts=dict(background="white"))

        l = roi_data_p[roi_data_mean>-99].min()
        u = roi_data_p[roi_data_mean>-99].max()
        l = np.floor(l*20)/20.0
        u = np.ceil(u*20)/20.0
        
        brain.add_data(vtx_data_p,
                        0.95, 
                        1.0,
                        thresh = 0.95,
                        colormap="autumn",
                        alpha=.8)
        
        views_list = [ 'medial', 'lateral' ]
        prefix = '_'.join([measure1, hemi, surface, 'p', measure2])
        brain.save_imageset(prefix = os.path.join(fs_rois_dir, prefix),
                            views = views_list, 
                            colorbar = range(len(views_list)) )
                           
    combine_pngs(measure1, measure2, surface, 'r', fs_rois_dir)
    combine_pngs(measure1, measure2, surface, 'p', fs_rois_dir)
