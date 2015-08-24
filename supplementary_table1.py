#!/usr/bin/env python

#=============================================================================
# Created by Kirstie Whitaker
# 27th July 2015
# Contact: kw401@cam.ac.uk
#
# This code creates supplementary table 1 for the NSPN Cortical Myelination
# manuscript
#=============================================================================

#=============================================================================
# IMPORTS
#-----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
import requests

#=============================================================================
# INPUT FILES
#-----------------------------------------------------------------------------
# Name the data directory and input files
dropbox_dir = glob(os.path.join('C:/','Users', 'Kirstie', 'Dropbox'))[0]
external_scripts_dir = os.path.join(dropbox_dir, 'GitHub', 'DESCRIBING_DATA')

data_dir = os.path.join(dropbox_dir,'KW_NSPN', 'DEMOGRAPHICS')

uchange_file = os.path.join(data_dir, 'UCHANGE_CompleteCohort_BaselineOnly_20150722.csv')

ses_file = os.path.join(data_dir, 'NSPNSociodemographic_DATA_2015-07-27_1713.csv')

wasi_ehi_ethnicity_file = os.path.join(data_dir, 'NSPNNeuroscienceInPs_DATA_2015-07-30_1734.csv')

age_gender_file = os.path.join(data_dir, '2K_Ethnicity_info_20150731.csv')

if not os.path.isfile(uchange_file):
    print "BEHAVIOURAL FILE DOESN'T EXIST"
    sys.exit()
    
    
#=============================================================================
# READ IN THE FILES TO DATA FRAMES
#-----------------------------------------------------------------------------
#====================
# UCHANGE
#====================
df_uchange = pd.read_csv(uchange_file)
keep_cols = ['nspn_id', 
             'age_scan', 'age_bin', 
             'validation', 'discovery', 
             'mri_centre', 'wbic', 'ucl' ]
df_uchange = df_uchange.loc[:, keep_cols]

#====================
# SES
#====================
df_ses = pd.read_csv(ses_file)
keep_cols = ['nspn_id', 'imd_2007']
df_ses = df_ses.loc[:, keep_cols]

#====================
# WASI, EHI
#====================
df_wasi_ehi = pd.read_csv(wasi_ehi_ethnicity_file)
df_wasi_ehi['nspn_id'] = df_wasi_ehi['id_nspn']
# Fix known typo value
df_wasi_ehi.loc[df_wasi_ehi['ehi_handedness_score']=='76,47', 'ehi_handedness_score'] = 76.47
df_wasi_ehi['ehi_handedness_score'] = df_wasi_ehi['ehi_handedness_score'].astype(float)
# Fix crazy IQ scores
df_wasi_ehi.loc[df_wasi_ehi['wasi_zz_iq_full2_iq'] > 200, 'wasi_zz_iq_full2_iq'] = np.nan
keep_cols = ['nspn_id', 'wasi_zz_iq_full2_iq', 'ehi_handedness_score']
df_wasi_ehi = df_wasi_ehi.loc[df_wasi_ehi['redcap_event_name']=='iua_baseline_arm_1', keep_cols]

#====================
# ETHNICITY
#====================
df_ethnicity = pd.read_csv(wasi_ehi_ethnicity_file)
df_ethnicity['nspn_id'] = df_ethnicity['id_nspn']
keep_cols = df_ethnicity.columns[4:]
df_ethnicity = df_ethnicity.loc[df_ethnicity['redcap_event_name']=='home_q_pack_arm_1', keep_cols]

#===============================================================
# AGE, GENDER
df_age_gender = pd.read_csv(age_gender_file)
df_age_gender['nspn_id'] = df_age_gender['subject_id']
df_age_gender['age_hqp'] = df_age_gender['age at hqp1']
df_age_gender.loc[df_age_gender['study_primary'] == '2K_cohort', 'study_primary'] = '2K_Cohort'
keep_cols = [ 'nspn_id', 'sex', 'ethnicity', 'centre', 'age_hqp']
df_age_gender = df_age_gender.loc[df_age_gender['study_primary']=='2K_Cohort', keep_cols]
data_dir = dropbox_dir = glob(os.path.join('C:/','Users', 'Kirstie', 'Dropbox', 'KW_NSPN', 'DEMOGRAPHICS'))[0]

uchange_file = os.path.join(data_dir, 'UCHANGE_CompleteCohort_BaselineOnly_20150722.csv')
ses_file = os.path.join(data_dir, 'NSPNSociodemographic_DATA_2015-07-27_1713.csv')
wasi_ehi_file = os.path.join(data_dir, 'NSPNPSRQNSPNProcesse_DATA_2015-07-27_2335.csv')

df_uchange = pd.read_csv(uchange_file)
df_ses = pd.read_csv(ses_file)
df_wasi_ehi = pd.read_csv(wasi_ehi_file)
df_wasi_ehi['nspn_id'] = df_wasi_ehi['id_nspn']
df_wasi_ehi = df_wasi_ehi.loc[df_wasi_ehi['redcap_event_name']=='baseline_assessmen_arm_1', :]

df_uchange_ses = df_uchange.merge(df_ses, on='nspn_id', how='left')
df = df_uchange_ses.merge(df_wasi_ehi, on='nspn_id', how='left')

df.describe()

for measure in [ 'age_scan', 'wasi_zz_iq_full2_iq', 'imd_2007', 'ehi_handedness_score']:
    print 'MEASURE: {}'.format(measure)
    for validation, group in df.groupby(['validation']):
        iqr = (np.percentile(group.loc[:, measure], 75) 
                   - np.percentile(group.loc[:, measure], 25))
        print '  {} N = {}, Mean: {:2.2f} (Std: {:2.2f}), Median: {:2.2f} (IQR: {:2.2f})'.format(validation,
                                                group.loc[:, measure].count(),
                                                group.loc[:, measure].mean(), 
                                                group.loc[:, measure].std(),
                                                group.loc[:, measure].median(),
                                                iqr)
    
    

