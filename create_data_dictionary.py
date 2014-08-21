#!/usr/bin/env python

import sys
import os
import numpy as np


data_dir = sys.argv[1]

measure_list = [ 'R1', 'MT', 'R2s', 'PDw', 'FA', 'MD', 'MO', 'L1', 'L23', 'sse', 'volume' ]

seg_list = [ 'aseg', 'wmparc', 'lobesStrict', '500cortExpConsecWMoverlap', '500cortConsec' ]

output_name = os.path.join(data_dir, 'data_dict.csv')
header='Variable / Field Name,Form Name,Section Header,Field Type,Field Label,"Choices, Calculations, OR Slider Labels",Field Note,Text Validation Type OR Show Slider Number,Text Validation Min,Text Validation Max,Identifier?,Branching Logic (Show field only if...),Required Field?,Custom Alignment,Question Number (surveys only),Matrix Group Name,Matrix Ranking?\nid_nspn,nspn_id_form,,text,NSPN ID,,,,,,,,,,,,\n'

with open(output_name, 'w') as f:
    f.write(header)

for measure in measure_list:

    for seg in seg_list:

        data_upload_file = os.path.join(data_dir, 'data_upload_{}_{}.csv'.format(measure, seg))
                
        fname = os.path.join(data_dir, '{}_{}_mean.csv'.format(measure, seg) )
        if measure == 'volume':
            fname = os.path.join(data_dir, '{}_volume.csv'.format(seg) )

        if os.path.isfile(fname):
            with open(fname) as f:
                names_list = f.readline().strip('\n').split(',')
            
            # Strip the new lines, make all words lower case
            # replace - for _ and add the measure_seg prefix
            names = [ x.lower() for x in names_list ]
            names = [ x.replace('-', '_') for x in names ]
            names = [ '{}_{}_{}'.format(measure.lower(), seg.lower(), x) for x in names ]

            # Append these names to the data dictionary file
            form_name = '{}_{}_stats'.format(measure.lower(), seg.lower())
            data_dict_list = [ '{},{},,text,{},,,number,,,,,,,,,'.format(x, form_name, y) for (x,y) in zip(names, names_list) ] 

            with open(output_name, 'a') as f:
                for x in data_dict_list:
                    f.write('{}\n'.format(x))
                
            # Append the data to the data upload file
            with open(fname) as f:
                data = f.readlines()
            data = [ x.strip('\n') for x in data ]
            np_data = np.array([ x.split(',') for x in data[1:]])
            np_data = np.hstack([np.array(names)[:, None], np_data.T])

            upload_data = np.copy(np_data)

            np.savetxt(data_upload_file, upload_data, fmt='%s', delimiter=",")

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