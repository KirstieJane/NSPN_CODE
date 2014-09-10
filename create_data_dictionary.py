#!/usr/bin/env python

import sys
import os
import numpy as np
from glob import glob

fs_rois_dir = sys.argv[1]

seg_measure_list = [ 'A', 'R1', 'MT', 'R2s', 
                    'PDw', 'FA', 'MD', 
                    'MO', 'L1', 'L23', 
                    'sse', 'volume' ]

parc_measure_list = [ 'area', 'volume', 'thickness', 
                    'meancurv', 'gauscurv', 
                    'foldind', 'curvind' ]

#=============================================================================
# SETUP OUTPUT FILE
#=============================================================================
# Define the output name
output_name = os.path.join(fs_rois_dir, 'data_dict.csv')

# Write the header into the output file
header='Variable / Field Name,Form Name,Section Header,Field Type,Field Label,"Choices, Calculations, OR Slider Labels",Field Note,Text Validation Type OR Show Slider Number,Text Validation Min,Text Validation Max,Identifier?,Branching Logic (Show field only if...),Required Field?,Custom Alignment,Question Number (surveys only),Matrix Group Name,Matrix Ranking?\nid_nspn,nspn_id_form,,text,NSPN ID,,,,,,,,,,,,\n'

with open(output_name, 'w') as f:
    f.write(header)

'''
#=============================================================================
# CREATE LISTS OF ALL SEGMENTATIONS AND ALL PARCELLATIONS
#=============================================================================
seg_list = glob(os.path.join(fs_rois_dir, 'SEG*mean.csv'))
seg_list = [ x.rsplit('_',2)[1] for x in seg_list ]
seg_list = list(set(seg_list))

parc_list = glob(os.path.join(fs_rois_dir, 'PARC_*area.csv'))
parc_list = [ x.rsplit('_',2)[1] for x in parc_list ]
parc_list = list(set(parc_list))
'''
#=============================================================================
# CREATE DATA UPLOAD FILE AND ADD TO DATA DICTIONARY
#=============================================================================

'''
for measure in seg_measure_list:

    for seg in seg_list:

        fname = os.path.join(fs_rois_dir, 'SEG_{}_{}_mean.csv'.format(measure, seg) )
        
        data_upload_file = os.path.join(fs_rois_dir, 'data_upload_SEG_{}_{}.csv'.format(measure, seg))
                
        if measure == 'volume':
            fname = os.path.join(fs_rois_dir, 'SEG_{}_volume.csv'.format(seg) )

        if os.path.isfile(fname):
            with open(fname) as f:
                # Get the header row from the csv data file
                names_list = f.readline().strip('\n').split(',')
            
            # Strip the new lines, make all words lower case
            # replace - for _ and add the measure_seg prefix
            names = [ x.lower() for x in names_list ]
            names = [ x.replace('-', '_') for x in names ]
            names = [ '{}_{}_{}'.format(measure.lower(), seg.lower(), x) for x in names ]

            #-----------------------------------------------------------------
            # Append these names to the data dictionary file
            form_name = '{}_{}_stats'.format(measure.lower(), seg.lower())
            data_dict_list = [ '{},{},,text,{},,,number,,,,,,,,,'.format(x, form_name, y) for (x,y) in zip(names, names_list) ] 

            with open(output_name, 'a') as f:
                for x in data_dict_list:
                    f.write('{}\n'.format(x))
                
            #-----------------------------------------------------------------
            # Append the data to the data upload file
            with open(fname) as f:
                data = f.readlines()
            data = [ x.strip('\n') for x in data ]
            np_data = np.array([ x.split(',') for x in data[1:]])
            np_data = np.hstack([np.array(names)[:, None], np_data.T])

            upload_data = np.copy(np_data)

            np.savetxt(data_upload_file, upload_data, fmt='%s', delimiter=",")

            #-----------------------------------------------------------------
            # Append a header and footer to this data so that it can be
            # uploaded correctly
            n_subs = upload_data.shape[1]-1
            data_header_list_1 = [ 'id_nspn' ] + list(upload_data[0,1:])
            data_header_list_2 = [ 'redcap_event_name' ] + ['baseline_assessmen_arm_1'] * n_subs

            data_header_1 = ','.join(data_header_list_1)
            data_header_2 = ','.join(data_header_list_2)

            data_footer_list = [ '{}_complete'.format(form_name) ] + ['1'] * n_subs
            data_footer = ','.join(data_footer_list)

            with open(data_upload_file, 'r+') as f:
                data = f.read()
                f.seek(0)
                f.write(data_header_1 + '\n')
                f.write(data_header_2 + '\n')
                f.write(data)
                f.write(data_footer)
# The end

'''

# Find all the combination files before they were merged with
# any behavioural values
measure_list = [ 'mean', 'area', 
                    'volume', 'thickness', 
                    'meancurv', 'gauscurv', 
                    'foldind', 'curvind' ]

file_list = []
for measure in measure_list:
    file_list = glob(os.path.join(fs_rois_dir, '*{}.csv'.format(measure)))

# Now loop through all those files
for fname in file_list:
    #-----------------------------------------------------------------
    # Get the header row
    with open(fname) as f:
        names_list = f.readline().strip('\n').split(',')
        
    #-----------------------------------------------------------------
    # Strip the new lines, make all words lower case
    # and replace - for _
    names = [ x.lower() for x in names_list ]
    names = [ x.replace('-', '_') for x in names ]

    #-----------------------------------------------------------------
    # Now add the appropriate prefix to the name
    # so that there aren't any duplicates in the data dictionary
    if 'SEG' in fname:
        prefix = 'SEG'
        if 'volume' in fname:
            measure = 'volume'
            seg = os.path.basename(fname).rsplit('_')[1]
        else:
            measure = os.path.basename(fname).split('_')[1]
            seg = os.path.basename(fname).split('_')[2]
    elif 'PARC' in fname:
        prefix = 'PARC'
        measure = os.path.basename(fname).split('_')[2]
        seg = os.path.basename(fname).split('_')[1]

    else:
        continue
        
    names = [ '{}_{}_{}_{}'.format(prefix.lower(), measure.lower(), seg.lower(), x) for x in names ]
    
    #-----------------------------------------------------------------
    # Append these names to the data dictionary file
    form_name = '{}_{}_{}_stats'.format(prefix.lower(), measure.lower(), seg.lower())
    print form_name
    data_dict_list = [ '{},{},,text,{},,,number,,,,,,,,,'.format(x, form_name, y) for (x,y) in zip(names, names_list) ] 

    with open(output_name, 'a') as f:
        for x in data_dict_list:
            f.write('{}\n'.format(x))
    
    #-----------------------------------------------------------------
    # Append the data to the data upload file
    with open(fname) as f:
        data = f.readlines()
    data = [ x.strip('\n') for x in data ]
    np_data = np.array([ x.split(',') for x in data[1:]])
    np_data = np.hstack([np.array(names)[:, None], np_data.T])

    upload_data = np.copy(np_data)

    data_upload_file = os.path.join(fs_rois_dir, 'data_upload_{}_{}_{}.csv'.format(prefix, measure, seg))
    print data_upload_file
    np.savetxt(data_upload_file, upload_data, fmt='%s', delimiter=",")
    
    #-----------------------------------------------------------------
    # Append a header and footer to this data so that it can be
    # uploaded correctly
    n_subs = upload_data.shape[1]-1
    data_header_list_1 = [ 'id_nspn' ] + list(upload_data[0,1:])
    data_header_list_2 = [ 'redcap_event_name' ] + ['baseline_assessmen_arm_1'] * n_subs

    data_header_1 = ','.join(data_header_list_1)
    data_header_2 = ','.join(data_header_list_2)

    data_footer_list = [ '{}_complete'.format(form_name) ] + ['1'] * n_subs
    data_footer = ','.join(data_footer_list)

    with open(data_upload_file, 'r+') as f:
        data = f.read()
        f.seek(0)
        f.write(data_header_1 + '\n')
        f.write(data_header_2 + '\n')
        f.write(data)
        f.write(data_footer)

#=============================================================================
# THE END
#=============================================================================
