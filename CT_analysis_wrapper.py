#!/usr/bin/env python

'''
Wrapper code to create list of values to plot on freesurfer surface
'''
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from glob import glob 
from scipy.stats.stats import linregress
import itertools as it

data_dir = os.path.join('/work','imagingG', 'NSPN', 'workspaces', 'kw401', 'UCHANGE','INTERIM_ANALYSIS')
subjects_dir = os.path.join(data_dir ,'SUB_DATA')
fs_rois_dir = os.path.join(data_dir, 'FS_ROIS')
subject_id = "fsaverageSubP"

parc = '500aparc'
seg = '500cortConsec'
parc_measure_list = [ 'area', 'volume', 'thickness', 'gauscurv' ]
seg_measure_list = [ 'MT', 'R1', 'R2s' ]

output_dir = os.path.join(fs_rois_dir, parc)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

aparc_names_file =  os.path.join(subjects_dir,
                          subject_id, "parcellation",
                          "500.names.txt")
                          
# Read in the names from the aparc names file 
# dropping the first 41 ################# BUUUUUG - needs to be fixed
aparc_names = [ line.strip() for line in open(aparc_names_file) ]
aparc_names = aparc_names[41::]

##================
# PARC FIRST
for measure in parc_measure_list:
    # Read in the data
    data_file = os.path.join(fs_rois_dir, '_'.join(['PARC', parc, measure, 'behavmerge.csv']))
    df = pd.read_csv(data_file)
    
    data_cols = [ x.replace('_{}'.format(measure), '') for x in df.columns ]
    df.columns = data_cols
        
    roi_data_mean = np.ones(len(aparc_names))*-99
    roi_data_std = np.ones(len(aparc_names))*-99
    roi_data_r = np.ones(len(aparc_names))*-99
    roi_data_p = np.ones(len(aparc_names))*-99
    roi_data_m = np.ones(len(aparc_names))*-99

    for i, name in enumerate(aparc_names):
    
        ##### MEAN #####
        roi_data_mean[i] = df[name].mean()
        roi_data_std[i] = df[name].std()
        
        ##### CORR WITH AGE #####                        
        m, c, r, p, sterr = linregress(df['age_scan'], df[name])

        roi_data_m[i] = m
        roi_data_r[i] = r
        roi_data_p[i] = 1 - p
        
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['PARC', measure, 'mean.txt'])), 
                        roi_data_mean,
                        fmt='%.6f')
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['PARC', measure, 'std.txt'])), 
                        roi_data_std,
                        fmt='%.6f')
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['PARC', measure, 'age', 'm.txt'])), 
                        roi_data_m,
                        fmt='%.6f')
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['PARC', measure, 'age', 'r.txt'])), 
                        roi_data_r,
                        fmt='%.6f')
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['PARC', measure, 'age', 'p.txt'])), 
                        roi_data_p,
                        fmt='%.6f')
                        
    df[aparc_names].corr().to_csv(os.path.join(fs_rois_dir, parc, '_'.join(['PARC', measure, 'corrMat.txt'])),
                                    header=False, index=False)


##### CORR PARCS WITH EACH OTHER #####
for measure1, measure2 in it.combinations(parc_measure_list,2):
    # Read in the data
    data_file1 = os.path.join(fs_rois_dir, '_'.join(['PARC', parc, measure1, 'behavmerge.csv']))
    df1 = pd.read_csv(data_file1)
    
    data_file2 = os.path.join(fs_rois_dir, '_'.join(['PARC', parc, measure2, 'behavmerge.csv']))
    df2 = pd.read_csv(data_file2)
    
    data_cols = [ x.replace('_{}'.format(measure1), '') for x in df1.columns ]
    df1.columns = data_cols
    
    data_cols = [ x.replace('_{}'.format(measure2), '') for x in df2.columns ]
    df2.columns = data_cols

    df_merge = df1.merge(df2, on='nspn_id')
    
    roi_data_r = np.ones(len(aparc_names))*-99
    roi_data_p = np.ones(len(aparc_names))*-99
    roi_data_m = np.ones(len(aparc_names))*-99

    for i, name in enumerate(aparc_names):
    
        ##### CORR WITH EACH OTHER #####
        m, c, r, p, sterr = linregress(df_merge[name+'_x'], df_merge[name+'_y'])

        roi_data_m[i] = m
        roi_data_r[i] = r
        roi_data_p[i] = 1 - p
        
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['PARC', measure1, measure2, 'm.txt'])), 
                        roi_data_m,
                        fmt='%.6f')
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['PARC', measure1, measure2, 'r.txt'])), 
                        roi_data_r,
                        fmt='%.6f')
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['PARC', measure1, measure2, 'p.txt'])), 
                        roi_data_p,
                        fmt='%.6f')

##===============
# SEGMENTATIONS NEXT
for measure in seg_measure_list:
    # Read in the data
    if not measure == 'volume':
        data_file = os.path.join(fs_rois_dir, '_'.join(['SEG', measure, seg, 'mean_behavmerge.csv']))
    else:
        data_file = os.path.join(fs_rois_dir, '_'.join(['SEG', seg, 'volume_behavmerge.csv']))

    df = pd.read_csv(data_file)
    
    data_cols = [ x.replace('_{}'.format(measure), '') for x in df.columns ]
    df.columns = data_cols
    
    roi_data_mean = np.ones(len(aparc_names))*-99
    roi_data_std = np.ones(len(aparc_names))*-99
    roi_data_r = np.ones(len(aparc_names))*-99
    roi_data_p = np.ones(len(aparc_names))*-99
    roi_data_m = np.ones(len(aparc_names))*-99

    for i, name in enumerate(aparc_names):
    
        ##### MEAN #####
        roi_data_mean[i] = df[name].mean()
        roi_data_std[i] = df[name].std()
        
        ##### CORR WITH AGE #####                        
        m, c, r, p, sterr = linregress(df['age_scan'], df[name])

        roi_data_m[i] = m
        roi_data_r[i] = r
        roi_data_p[i] = 1 - p
        
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['SEG', measure, 'mean.txt'])), 
                        roi_data_mean,
                        fmt='%.6f')
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['SEG', measure, 'std.txt'])), 
                        roi_data_std,
                        fmt='%.6f')
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['SEG', measure, 'age', 'm.txt'])), 
                        roi_data_m,
                        fmt='%.6f')
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['SEG', measure, 'age', 'r.txt'])), 
                        roi_data_r,
                        fmt='%.6f')
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['SEG', measure, 'age', 'p.txt'])), 
                        roi_data_p,
                        fmt='%.6f')
    df[aparc_names].corr().to_csv(os.path.join(fs_rois_dir, parc, '_'.join(['SEG', measure, 'corrMat.txt'])),
                                    header=False, index=False)

##### CORR SEGS WITH EACH OTHER #####
for measure1, measure2 in it.combinations(seg_measure_list,2):
    # Read in the data
    if not measure1 == 'volume':
        data_file1 = os.path.join(fs_rois_dir, '_'.join(['SEG', measure1, seg, 'mean_behavmerge.csv']))
    else:
        data_file1 = os.path.join(fs_rois_dir, '_'.join(['SEG', seg, 'volume_behavmerge.csv']))
    df1 = pd.read_csv(data_file1)

    if not measure2 == 'volume':
        data_file2 = os.path.join(fs_rois_dir, '_'.join(['SEG', measure2, seg, 'mean_behavmerge.csv']))
    else:
        data_file2 = os.path.join(fs_rois_dir, '_'.join(['SEG', seg, 'volume_behavmerge.csv']))
    df2 = pd.read_csv(data_file2)

    data_cols = [ x.replace('_{}'.format(measure1), '') for x in df1.columns ]
    df1.columns = data_cols
    
    data_cols = [ x.replace('_{}'.format(measure2), '') for x in df2.columns ]
    df2.columns = data_cols

    df_merge = df1.merge(df2, on='nspn_id')
    
    roi_data_r = np.ones(len(aparc_names))*-99
    roi_data_p = np.ones(len(aparc_names))*-99
    roi_data_m = np.ones(len(aparc_names))*-99

    for i, name in enumerate(aparc_names):
    
        ##### CORR WITH EACH OTHER #####
        m, c, r, p, sterr = linregress(df_merge[name+'_y'], df_merge[name+'_x'])

        roi_data_m[i] = m
        roi_data_r[i] = r
        roi_data_p[i] = 1 - p
        
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['SEG', measure1, measure2, 'm.txt'])), 
                        roi_data_m,
                        fmt='%.6f')
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['SEG', measure1, measure2, 'r.txt'])),
                        roi_data_r,
                        fmt='%.6f')
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['SEG', measure1, measure2, 'p.txt'])), 
                        roi_data_p,
                        fmt='%.6f')

#====================================
# NOW CORRELATE SEGS WITH PARCS                        
for measure1, measure2 in it.product(parc_measure_list, seg_measure_list):
    # Read in the data
    data_file1 = os.path.join(fs_rois_dir, '_'.join(['PARC', parc, measure1, 'behavmerge.csv']))
    df1 = pd.read_csv(data_file1)
    
    if not measure2 == 'volume':
        data_file2 = os.path.join(fs_rois_dir, '_'.join(['SEG', measure2, seg, 'mean_behavmerge.csv']))
    else:
        data_file2 = os.path.join(fs_rois_dir, '_'.join(['SEG', seg, 'volume_behavmerge.csv']))
    df2 = pd.read_csv(data_file2)

    data_cols = [ x.replace('_{}'.format(measure1), '') for x in df1.columns ]
    df1.columns = data_cols
    
    data_cols = [ x.replace('_{}'.format(measure2), '') for x in df2.columns ]
    df2.columns = data_cols

    df_merge = df1.merge(df2, on='nspn_id')
    
    roi_data_r = np.ones(len(aparc_names))*-99
    roi_data_p = np.ones(len(aparc_names))*-99
    roi_data_m = np.ones(len(aparc_names))*-99

    for i, name in enumerate(aparc_names):
    
        ##### CORR WITH EACH OTHER #####
        m, c, r, p, sterr = linregress(df_merge[name+'_y'], df_merge[name+'_x'])

        roi_data_m[i] = m
        roi_data_r[i] = r
        roi_data_p[i] = 1 - p
        
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['PSEG', measure1, measure2, 'm.txt'])), 
                        roi_data_m,
                        fmt='%.6f')
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['PSEG', measure1, measure2, 'r.txt'])), 
                        roi_data_r,
                        fmt='%.6f')
    np.savetxt(os.path.join(fs_rois_dir, parc, '_'.join(['PSEG', measure1, measure2, 'p.txt'])), 
                        roi_data_p,
                        fmt='%.6f')

# lalaland

