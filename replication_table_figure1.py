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
# FUNCTIONS
#=============================================================================

#------------------------------------------------------------------------------
# Define your latex add in caption function
#------------------------------------------------------------------------------
def add_caption(latex_table, caption):
    '''
    Just add in a row on the second line down to include the caption (title)
    for this table
    '''
    latex_table_list = latex_table.split('\n')
    latex_table_list[0] = latex_table_list[0] + '\n\\caption{{{}}} \\\\'.format(caption)         
    latex_table = '\n'.join(latex_table_list)

    return latex_table


#=============================================================================
# IMPORTS
#-----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
import requests
from glob import glob

#=============================================================================
# INPUT FILES
#-----------------------------------------------------------------------------
# Name the data directory and input files
dropbox_dir = glob(os.path.join('C:/','Users', 'Kirstie', 'Dropbox'))[0]
external_scripts_dir = os.path.join(dropbox_dir, 'GitHub', 'DESCRIBING_DATA')

data_dir = os.path.join(dropbox_dir,'KW_NSPN', 'CT_MT_ANALYSES', 'DEMOGRAPHICS')

uchange_file = os.path.join(data_dir, 'UCHANGE_CompleteCohort_BaselineOnly_20150722.csv')

ses_file = os.path.join(data_dir, 'NSPNSociodemographic_DATA_2015-07-27_1713.csv')

wasi_ehi_ethnicity_file = os.path.join(data_dir, 'NSPNNeuroscienceInPs_DATA_2015-07-30_1734.csv')

age_gender_file = os.path.join(data_dir, '2K_Ethnicity_info_20150731.csv')


if not os.path.isfile(uchange_file):
    print "BEHAVIOURAL FILE DOESN'T EXIST"
    sys.exit()
    

#=============================================================================
# GET STARTED BY SETTING THE CAPTION AND NAME YOUR OUTPUT FILE
#-----------------------------------------------------------------------------
caption = 'Participant Demographics'

table_file = os.path.join(data_dir, 'SupplementaryTable1.tex')

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

#=============================================================================
# MERGE THE DATA FRAMES TOGETHER
#-----------------------------------------------------------------------------
df_temp1 = df_age_gender.merge(df_uchange, how='outer', on='nspn_id')
df_temp2 = df_temp1.merge(df_wasi_ehi, how='outer', on='nspn_id')
df_temp3 = df_temp2.merge(df_ethnicity, how='outer', on='nspn_id')
df = df_temp3.merge(df_ses, how='outer', on='nspn_id')

df.loc[df['discovery'].isnull(), 'discovery'] = 0
df.loc[df['validation'].isnull(), 'validation'] = 0


#=============================================================================
# CREATE A DICT FOR THE THREE GROUPS
#-----------------------------------------------------------------------------
name_dict = { (0.0, 0.0) : 'Remaining 2K',
              (1.0, 0.0) : 'Discovery',
              (0.0, 1.0) : 'Validation'}

table_dict = {}

for name, data in df.groupby(['discovery', 'validation']):
    print '==={}==='.format(name_dict[name])
    
    data_list = []
    col_list = []
    
    #=======================================================
    # NUMBER OF PARTICIPANTS
    #=======================================================
    n = data['nspn_id'].count()
    print '    Number of participants: {}'.format(n)
    data_list += [n]
    col_list += ['\\textbf{Number of participants}']
    
    #=======================================================
    # PERCENTAGE MALE
    #=======================================================
    n_male = data.loc[data['sex'] == 'Male', 'sex'].count()
    n_missing = data.loc[data['sex'].isnull(), 'sex'].shape[0]
    print '    Gender: {:2.1f}% male, n missing = {:2.0f}'.format(n_male * 100.0 / n, n_missing)
    data_list += ['{:2.1f}\% male'.format(n_male * 100.0/n)]
    col_list += ['\\textbf{Gender}']
    
    #=======================================================
    # AGES (MEDIAN, IQR)
    #=======================================================
    if name[0] + name[1] == 0:
        age_var = 'age_hqp'
    else:
        age_var = 'age_scan'
    
    # Means and Stds - not used here but just in case
    mean_age = data.loc[data[age_var].notnull(), age_var].mean()
    std_age = data.loc[data[age_var].notnull(), age_var].std()

    # Missing values
    n_missing = data.loc[data[age_var].isnull(), age_var].shape[0]
        
    # Median and IQRs
    med_age = np.percentile(data.loc[data[age_var].notnull(), age_var], 50)
    upper_age = np.percentile(data.loc[data[age_var].notnull(), age_var], 75)
    lower_age = np.percentile(data.loc[data[age_var].notnull(), age_var], 25)
    print '    Age: {:2.1f} (IQR: {:2.1f}-{:2.1f}), n missing = {:2.0f}'.format(med_age,
                                                                                lower_age, 
                                                                                upper_age,
                                                                                n_missing)
    data_list += ['{:2.1f}'.format(med_age) ]
    data_list += ['(IQR: {:2.1f}-{:2.1f})'.format(lower_age, upper_age)]
    
    col_list += ['\\multirow{2}{*}{\\textbf{Age (years)}}']
    col_list += ['']
    
    #=======================================================
    # FULL IQ (MEDIAN, IQR)
    #=======================================================
    # Means and Stds - not used here but just in case
    mean_iq = data.loc[data['wasi_zz_iq_full2_iq'].notnull(), 'wasi_zz_iq_full2_iq'].mean()
    std_iq = data.loc[data['wasi_zz_iq_full2_iq'].notnull(), 'wasi_zz_iq_full2_iq'].std()

    # Missing values
    n_missing = data.loc[data['wasi_zz_iq_full2_iq'].isnull(), 'wasi_zz_iq_full2_iq'].shape[0]
        
    # Median and IQRs
    med_iq = np.percentile(data.loc[data['wasi_zz_iq_full2_iq'].notnull(), 
                                                 'wasi_zz_iq_full2_iq'], 50)
    upper_iq = np.percentile(data.loc[data['wasi_zz_iq_full2_iq'].notnull(), 
                                                   'wasi_zz_iq_full2_iq'], 75)
    lower_iq = np.percentile(data.loc[data['wasi_zz_iq_full2_iq'].notnull(), 
                                                   'wasi_zz_iq_full2_iq'], 25)
    print '    IQ: {:2.1f} (IQR: {:2.1f}-{:2.1f}), n missing = {:2.0f}'.format(med_iq,
                                                                                lower_iq, 
                                                                                upper_iq,
                                                                                n_missing)
    if n > 2000:
        data_list += ['\\multirow{2}{*}{NA}']
        data_list += ['']
    else:
        data_list += ['{:2.1f}'.format(med_iq) ]
        data_list += ['(IQR: {:2.1f}-{:2.1f})'.format(lower_iq, upper_iq)]
    
    col_list += ['\\multirow{2}{*}{\\textbf{IQ}}']
    col_list += ['']

    #=======================================================
    # HANDEDNESS (MEDIAN, IQR)
    #=======================================================
    # Means and Stds - not used here but just in case
    mean_ehi = data.loc[data['ehi_handedness_score'].notnull(), 'ehi_handedness_score'].mean()
    std_ehi = data.loc[data['ehi_handedness_score'].notnull(), 'ehi_handedness_score'].std()

    # Missing values
    n_missing = data.loc[data['ehi_handedness_score'].isnull(), 'ehi_handedness_score'].shape[0]
        
    # Median and IQRs
    med_ehi = np.percentile(data.loc[data['ehi_handedness_score'].notnull(), 
                                                 'ehi_handedness_score'], 50)
    upper_ehi = np.percentile(data.loc[data['ehi_handedness_score'].notnull(), 
                                                 'ehi_handedness_score'], 75)
    lower_ehi = np.percentile(data.loc[data['wasi_zz_iq_full2_iq'].notnull(), 
                                                 'ehi_handedness_score'], 25)
    print '    EHI: {:2.1f} (IQR: {:2.1f}-{:2.1f}), n missing = {:2.0f}'.format(med_ehi,
                                                                                lower_ehi, 
                                                                                upper_ehi,
                                                                                n_missing)
    if n > 2000:
        data_list += ['\\multirow{2}{*}{NA}']
        data_list += ['']
    else:
        data_list += ['{:2.1f}'.format(med_ehi) ]
        data_list += ['(IQR: {:2.1f}-{:2.1f})'.format(lower_ehi, upper_ehi)]
    
    col_list += ['\\multirow{2}{*}{\\textbf{Handedness}}']
    col_list += ['']
        
    #=======================================================
    # INDEX MULTIPLE DEPRIVATION (MEDIAN, IQR)
    #=======================================================
    # Means and Stds - not used here but just in case
    mean_imd = data.loc[data['imd_2007'].notnull(), 'imd_2007'].mean()
    std_imd = data.loc[data['imd_2007'].notnull(), 'imd_2007'].std()

    # Missing values
    n_missing = data.loc[data['imd_2007'].isnull(), 'imd_2007'].shape[0]
        
    # Median and IQRs
    med_imd = np.percentile(data.loc[data['imd_2007'].notnull(), 
                                                'imd_2007'], 50)
    upper_imd = np.percentile(data.loc[data['imd_2007'].notnull(), 
                                                'imd_2007'], 75)
    lower_imd = np.percentile(data.loc[data['imd_2007'].notnull(), 
                                                'imd_2007'], 25)
    print '    IMD: {:2.1f} (IQR: {:2.1f}-{:2.1f}), n missing = {:2.0f}'.format(med_iq,
                                                                                lower_imd, 
                                                                                upper_imd,
                                                                                n_missing)
    data_list += ['{:2.1f}'.format(med_imd) ]
    data_list += ['(IQR: {:2.1f}-{:2.1f})'.format(lower_imd, upper_imd)]
    
    col_list += ['\\multirow{2}{*}{\\textbf{IMD}}']
    col_list += ['']

    #=======================================================
    # ETHNICITY (% WHITE)
    n_white = data.loc[data['psq_a4_ethnic_group']==1, 'psq_a4_ethnic_group'].count()
    n_mixed = data.loc[data['psq_a4_ethnic_group']==2, 'psq_a4_ethnic_group'].count()
    n_asian = data.loc[data['psq_a4_ethnic_group']==3, 'psq_a4_ethnic_group'].count()
    n_black = data.loc[data['psq_a4_ethnic_group']==4, 'psq_a4_ethnic_group'].count()
    n_other = data.loc[data['psq_a4_ethnic_group']==5, 'psq_a4_ethnic_group'].count()
    n_declined = data.loc[data['psq_a4_ethnic_group']==6, 'psq_a4_ethnic_group'].count()
    n_known_missing = data.loc[data['psq_a4_ethnic_group']==999, 'psq_a4_ethnic_group'].count()
    n_missing = data.loc[data['psq_a4_ethnic_group'].isnull(), 'psq_a4_ethnic_group'].shape[0]
    print '    Ethnicity: {:2.1f}% White, {:2.1f}% Asian, {:2.1f}% Black'.format(n_white*100.0/n, 
                                                                                     n_asian *100.0/n, 
                                                                                     n_black*100.0/n)
    print '               {:2.1f}% mixed, {:2.1f}% other, {:2.1f}% declined to state'.format(n_mixed*100.0/n,
                                                                                                 n_other*100.0/n,
                                                                                                 n_declined*100.0/n)
    print '               {:2.1f}% known missing, {:2.1f}% unknown missing'.format(n_known_missing*100.0/n,
                                                                                       n_missing*100.0/n)
    
    data_list += [ '{:2.1f}\% White'.format(n_white*100.0/n) ]
    data_list += [ '{:2.1f}\% Asian'.format(n_asian*100.0/n) ]
    data_list += [ '{:2.1f}\% Black'.format(n_black*100.0/n) ]
    data_list += [ '{:2.1f}\% Mixed'.format(n_mixed*100.0/n) ]
    data_list += [ '{:2.1f}\% Other'.format(n_other*100.0/n) ]
    data_list += [ '{:2.1f}\% Declined to state'.format(n_declined*100.0/n) ]
        
    col_list += ['\\multirow{6}{*}{\\textbf{Ethnicity}}']
    col_list += [ '' ] *5

    table_dict['\\textbf{{{}}}'.format(name_dict[name])] = data_list

#=============================================================================
# MAKE THIS DICT INTO A PANDAS DATA FRAME
#-----------------------------------------------------------------------------
table_df = pd.DataFrame(table_dict, index=col_list)
table_df = table_df.loc[:, ['\\textbf{Discovery}', '\\textbf{Validation}', '\\textbf{Remaining 2K}']]

#=============================================================================
# NOW WRITE THIS DATA FRAME TO A LATEX FILE
#-----------------------------------------------------------------------------
latex_table = table_df.to_latex(longtable=True, 
                                    index=True, 
                                    escape=False)

latex_table = latex_table.replace('llll', 'lccc')
latex_table = add_caption(latex_table, caption)

latex_header = '\n'.join([ '\\documentclass{article}', 
                            '\\usepackage{booktabs}', 
                            '\\usepackage[a4paper, left={1cm}, right={1cm}, top={1.5cm}, bottom={1.5cm}, portrait]{geometry}',
                            '\\usepackage{longtable}',
                            '\\usepackage{array}',
                            '\\usepackage{multirow}'
                            '\\begin{document}',
                            '' ])

latex_footer = '\n\\end{document}\n'

# Write the file to the table_filename
with open(table_file, 'w') as f:
    f.write(latex_header)
    f.write(latex_table)
    f.write(latex_footer)
    
# Write the file to the table_filename without the latex header and footer
with open(table_file.replace('.tex', '.txt'), 'w') as f:
    f.write(latex_table)

