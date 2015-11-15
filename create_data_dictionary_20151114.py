#!/usr/bin/env python

'''
This is the code that creates a data dictionary
that makes uploading to redcap happen

It was heavily edited on 14th November 2015
by Kirstie Whitaker
kw401@cam.ac.uk
'''

#===================================================================
# IMPORT WHAT YOU NEED
#-------------------------------------------------------------------
import sys
import os
import numpy as np
from glob import glob
import pandas as pd

#===================================================================
# SET YOUR FS_BEHAV DATA DIR
#fs_behav_dir = sys.argv[1]
fs_behav_dir = '.'
#-------------------------------------------------------------------

#===================================================================
# DEFINE THE MEASURES YOU CARE ABOUT
#-------------------------------------------------------------------
seg_measure_list = [ 'A', 'R1', 'MT', 'R2s', 'volume' ]

parc_measure_list = [ 'area', 'volume', 'thickness', 
                    'meancurv', 'gauscurv', 
                    'foldind', 'curvind' ]

#===================================================================
# SETUP OUTPUT FILE
#-------------------------------------------------------------------
# Define the output name
data_dict_fname = os.path.join(fs_behav_dir, 'data_dict.csv')

# Write the header into the output file
header='Variable / Field Name,Form Name,Section Header,Field Type,Field Label,"Choices, Calculations, OR Slider Labels",Field Note,Text Validation Type OR Show Slider Number,Text Validation Min,Text Validation Max,Identifier?,Branching Logic (Show field only if...),Required Field?,Custom Alignment,Question Number (surveys only),Matrix Group Name,Matrix Ranking?\nid_nspn,nspn_id_form,,text,NSPN ID,,,,,,,,,,,,\n'

with open(data_dict_fname, 'w') as f:
    f.write(header)

#===================================================================
# CREATE DATA UPLOAD FILE AND ADD TO DATA DICTIONARY
#-------------------------------------------------------------------
# Find all the combination files before they were merged with
# any behavioural values
measure_list = [ 'mean', 'area', 
                    'volume', 'thickness', 
                    'meancurv', 'gauscurv', 
                    'foldind', 'curvind' ]

file_list = []
for measure in measure_list:
    file_list += glob(os.path.join(fs_behav_dir, '*{}_behavmerge.csv'.format(measure)))

# Now loop through all those files
for fname in file_list:
    #-----------------------------------------------------------------
    # Read in the file to a pandas data frame
    df = pd.read_csv(fname)
    
    #-----------------------------------------------------------------
    # Get the names of the columns containing the regionsal data
    cols = [ name for name in df.columns[23:] if not 'unknown' in name ]
    
    #-----------------------------------------------------------------
    # Make all words lower case and replace - for _
    names = [ x.rsplit('_', 1)[0] for x in cols ]

    #-----------------------------------------------------------------
    # Now add the appropriate prefix to the name
    # so that there aren't any duplicates in the data dictionary
    fbasename = os.path.basename(fname).split('_behavmerge.csv')[0]

    if 'PARC' in fname:
        prefix = 'PARC'
        measure = fbasename.split('_', 2)[2].split('_mean')[0].split('_fromBoundary')[0]
        seg = fbasename.split('_',2)[1]
        
        names = [ '{}_{}_{}_{}'.format(prefix.lower(), seg.lower(), measure.lower(), name) for name in names ]
        names = [ x.replace('-', '') for x in names ]
        names = [ x.replace('+', '') for x in names ]
    
    #-----------------------------------------------------------------
    # Append these names to the data dictionary file
    form_name = '{}_{}_{}_stats'.format(prefix.lower(), seg.lower(), measure.lower())
    form_name = form_name.replace('-', '')
    form_name = form_name.replace('+', '')
    print form_name
    data_dict_list = [ '{},{},,text,{},,,number,,,,,,,,,'.format(x, form_name, y) for (x,y) in zip(names, cols) ] 

    with open(data_dict_fname, 'a') as f:
        for x in data_dict_list:
            f.write('{}\n'.format(x))
    
    #-----------------------------------------------------------------
    # Append the data to the data upload file
    # First the nspn_id and recap_event_name
    upload_df = df.loc[:, ['nspn_id'] + cols ]
    # Replace the column names with the names in the data_dict
    upload_df.columns = ['id_nspn'] + names
    
    # Then the redcap event name and complete marker
    upload_df['redcap_event_name'] = 'baseline_assessmen_arm_1'

    # Add a column at the end
    upload_df['{}_complete'.format(form_name)] = 1
    
    data_upload_file = os.path.join(fs_behav_dir, 'data_upload_{}_{}_{}.csv'.format(prefix, measure, seg))
    print data_upload_file
    
    upload_df.to_csv(data_upload_file, index=False)

#=============================================================================
# THE END
#=============================================================================
