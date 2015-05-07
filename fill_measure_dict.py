#!/usr/bin/env python

'''
measure_dict is a dictionary containing values for all 308 regions
in the NSPN parcellation. The keys explain which measure exactly.
'''

import numpy as np
import pandas as pd
import os 
import sys

def read_in_df(data_file):
    df = pd.read_csv(data_file, sep=',')
    
    # Only keep the first scan!
    df = df[df.occ==0]
    
    data_cols = [ x.replace('_{}'.format('thicknessstd'), '') for x in df.columns ]
    df.columns = data_cols
    data_cols = [ x.replace('_{}'.format('thickness'), '') for x in df.columns ]
    df.columns = data_cols
    
    # Define a few variables you want
    df['young'] = 0
    df['young'][df['age_scan']<np.percentile(df.age_scan, 50)] = 1
    
    df['ones'] = df['age_scan'] * 0 + 1
    df['age'] = df['age_scan']
    
    df['Global'] = df[aparc_names].mean(axis=1)
    '''
    # If there is a corresponding standard deviation
    # file then read in the standard deviation!
    if 'mean' in data_file:
        std_data_file = data_file.replace('mean', 'std')
    else:
        std_data_file = data_file.replace('thickness', 'thicknessstd')
    
    if os.path.isfile(std_data_file):
        df_std = pd.read_csv(std_data_file, sep=',')
        df_std = df_std[df_std.occ==0]
        
        data_cols = [ x.replace('_{}'.format('thicknessstd'), '') for x in df_std.columns ]
        df_std.columns = data_cols
        data_cols = [ x.replace('_{}'.format('thickness'), '') for x in df_std.columns ]
        df_std.columns = data_cols
        
        df['Global_std'] = np.sqrt(np.average(df_std[aparc_names]**2, axis=1))
    '''
    return df

    
#=============================================================================
# Import some useful functions
#=============================================================================
study_dir = os.path.join('/home', 'kw401', 'UCHANGE', 'INTERIM_ANALYSES')
if not os.path.isdir(study_dir):
    study_dir = os.path.join('/scratch', 'kw401', 'UCHANGE_INTERIM')
scripts_dir = os.path.join(study_dir, 'SCRIPTS')
sys.path.append(os.path.join(scripts_dir, 'NSPN_CODE'))
from regional_correlation_functions import *

#=============================================================================
# Read in the behavioural files you might want
#=============================================================================
paper_dir = os.path.join(study_dir, 'CT_MT_ANALYSES')
data_dir = os.path.join(study_dir, 'FS_ROIS')
fsaverage_dir = os.path.join(study_dir, 'SUB_DATA', 'fsaverageSubP')

behav_data_file = os.path.join(study_dir, 'uchange_interim_cohort.csv')
ct_data_file = os.path.join(data_dir, 'PARC_500aparc_thickness_behavmerge.csv')
aparc_names_file = os.path.join(fsaverage_dir, 'parcellation', '500.names.txt' )

# Read in the names from the aparc names file 
# dropping the first 41 ################# BUUUUUG - needs to be fixed
aparc_names = [ line.strip() for line in open(aparc_names_file) ]
aparc_names = aparc_names[41::]

#=============================================================================
# Create an empty dictionary
# and read in the ct file because you'll
# need that for correlations later
#=============================================================================
measure_dict = {}

df_ct = read_in_df(os.path.join(data_dir, 'PARC_500aparc_thickness_behavmerge.csv'))

#=============================================================================
# Create a list of filenames and measure names
#=============================================================================
filename_list = []
measure_name_list = []

filename_list += [ os.path.join(data_dir, 'PARC_500aparc_thickness_behavmerge.csv') ]
measure_name_list += [ 'CT' ]

filename_list += [ os.path.join(data_dir, 'SEG_MT_500cortConsec_mean_behavmerge.csv') ]
measure_name_list += [ 'MTall' ]

### FRACTIONAL DEPTHS
for i in np.arange(0.0,110,10):
    
    # MT
    filename_list += [ os.path.join(data_dir, 
                       'PARC_500aparc_MT_projfrac{:+04.0f}_mean_behavmerge.csv'.format(i)) ]
    measure_name_list += [ 'MT_projfrac{:+04.0f}'.format(i) ]
    
    # Synthetic
    filename_list += [ os.path.join(data_dir, 
                       'PARC_500aparc_synthetic_projfrac{:+04.0f}_mean_behavmerge.csv'.format(i)) ]
    measure_name_list += [ 'synthetic_projfrac{:+04.0f}'.format(i) ]
    
### ABSOLUTE DEPTHS FROM BOUNDARY
for i in np.arange(-20,-201,-20):
    
    # MT
    filename_list += [ os.path.join(data_dir, 
                       'PARC_500aparc_MT_projdist{:+04.0f}_fromBoundary_mean_behavmerge.csv'.format(i)) ]
    measure_name_list += [ 'MT_projdist{:+04.0f}'.format(i) ]
    
    # Synthetic
    filename_list += [ os.path.join(data_dir, 
                       'PARC_500aparc_synthetic_projdist{:+04.0f}_fromBoundary_mean_behavmerge.csv'.format(i)) ]
    measure_name_list += [ 'synthetic_projdist{:+04.0f}'.format(i) ]

#=============================================================================
# Then fill in the dictionary with:
#     * MEAN
#     * STANDARD DEVIATION
#     * CORR W AGE
#     * CORR W CT
#=============================================================================

for filename, measure_name in zip(filename_list, measure_name_list):
    
    df = read_in_df(filename)
    
    #-------------------------------------------------------------------------
    # 308 regions
    #-------------------------------------------------------------------------
    # MEAN
    measure_dict['{}_all_mean'.format(measure_name)] = df[aparc_names].mean(axis=0).values

    # STD
    measure_dict['{}_all_std'.format(measure_name)] = df[aparc_names].std(axis=0).values

    # CORR W AGE
    (m_array, r_array, 
        p_array, p_fdr_array,
        m_mask_array, m_fdr_mask_array) = regional_linregress(df, 'age_scan', aparc_names)

    measure_dict['{}_all_slope_age'.format(measure_name)] = m_array
    measure_dict['{}_all_slope_age_r'.format(measure_name)] = r_array
    measure_dict['{}_all_slope_age_p'.format(measure_name)] = p_array
    measure_dict['{}_all_slope_age_p_fdr'.format(measure_name)] = p_fdr_array
    measure_dict['{}_all_slope_age_m_mask'.format(measure_name)] = m_mask_array
    measure_dict['{}_all_slope_age_m_fdr_mask'.format(measure_name)] = m_fdr_mask_array

    # CORR W CT
    if not measure_name == 'CT':
        (m_array, r_array, 
            p_array, p_fdr_array,
            m_mask_array, m_fdr_mask_array) = regional_linregress_byregion(df, df_ct, aparc_names)

        measure_dict['{}_all_slope_ct'.format(measure_name)] = m_array
        measure_dict['{}_all_slope_ct_r'.format(measure_name)] = r_array
        measure_dict['{}_all_slope_ct_p'.format(measure_name)] = p_array
        measure_dict['{}_all_slope_ct_p_fdr'.format(measure_name)] = p_fdr_array
        measure_dict['{}_all_slope_ct_m_mask'.format(measure_name)] = m_mask_array
        measure_dict['{}_all_slope_ct_m_fdr_mask'.format(measure_name)] = m_fdr_mask_array

    #-------------------------------------------------------------------------
    # N subjects
    #-------------------------------------------------------------------------
    # MEAN
    measure_dict['{}_global_mean'.format(measure_name)] = df['Global'].values

    # STD
    measure_dict['{}_global_std'.format(measure_name)] = df['Global_std'].values
