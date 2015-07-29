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
    
    

