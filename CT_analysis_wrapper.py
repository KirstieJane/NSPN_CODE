#!/usr/bin/env python

'''
Wrapper code to create list of values to plot on freesurfer surface
'''
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from glob import glob 

data_dir = os.path.join('/work','imagingG', 'NSPN', 'workspaces', 'kw401', 'UCHANGE','INTERIM_ANALYSIS')
subjects_dir = os.path.join(data_dir ,'SUB_DATA')
fs_rois_dir = os.path.join(data_dir, 'FS_ROIS')
subject_id = "fsaverageSubP"

parc = '500aparc'
measure_list = [ 'area', 'volume', 'thickness', 'gauscurv' ]

output_dir = os.path.join(fs_rois_dir, parc)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

aparc_names_file =  os.path.join(subjects_dir,
                          subject_id, "parcellation",
                          "500.names.txt")
                          
# Read in the names from the aparc names file 
# dropping the first 41 ################# BUUUUUG - needs to be fixed
aparc_names = [line.strip() for line in open(aparc_names_file)]
aparc_names = aparc_names[41::]

##### MEAN #####
for measure in measure_list:
    # Read in the data
    data_file = os.path.join(fs_rois_dir, '_'.join(['PARC', parc, measure, 'behavmerge.csv']))
    df = pd.read_csv(data_file)
    
    data_cols = [ x.rstrip('_{}'.format(measure)) for x in df.columns ]
    df.columns = data_cols
    
    roi_data_mean = np.ones(len(aparc_names))*-99
    roi_data_std = np.ones(len(aparc_names))*-99
    roi_data_r = np.ones(len(aparc_names))*-99
    roi_data_p = np.ones(len(aparc_names))*-99
    roi_data_m = np.ones(len(names))*-99

    for i, name in enumerate(aparc_names):
    
        roi_data_mean[i] = df[name].mean()
        roi_data_std[i] = df[name].std()
        
        formula = '{} ~ age_scan'.format(name)
        mod = sm.ols(formula=formula, data=df)

        roi_data_m[i] = m
        roi_data_r[i] = r
        roi_data_p[i] = 1 - p

    roi_data_mean = np.ones(len(aparc_names))*-99
    roi_data_std = np.ones(len(aparc_names))*-99
    roi_data_r = np.ones(len(aparc_names))*-99
    roi_data_p = np.ones(len(aparc_names))*-99
    roi_data_m = np.ones(len(names))*-99

    roi_values = df[aparc_names].mean()
    roi_values.to_csv(os.path.join(fs_rois_dir, parc, '_'.join([measure, 'mean.txt'])), 
                        index=False)

##### CORR WITH AGE #####                        
    r_values = 