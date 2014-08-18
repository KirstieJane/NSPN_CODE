#!/usr/bin/env python

import sys
import os

data_dir = sys.argv[1]

measure_list = [ 'R1', 'MT', 'R2s', 'PDw', 'FA', 'MD', 'MO', 'L1', 'L23', 'sse' ]

seg_list = [ 'aseg', 'wmparc', 'lobesStrict', '500cortExpConsecWMoverlap', '500cortConsec']

output_name = os.path.join(data_dir, 'data_dict.csv')
header='Variable / Field Name,Form Name,Section Header,Field Type,Field Label,"Choices, Calculations, OR Slider Labels",Field Note,Text Validation Type OR Show Slider Number,Text Validation Min,Text Validation Max,Identifier?,Branching Logic (Show field only if...),Required Field?,Custom Alignment,Question Number (surveys only),Matrix Group Name,Matrix Ranking?\nid_nspn,nspn_id_form,,text,NSPN ID,,,,,,,,,,,,\n'

with open(output_name, 'w') as f:
    f.write(header)

for seg in seg_list:
    for measure in measure_list:
                
        fname = os.path.join(data_dir, '{}_{}_mean.csv'.format(measure, seg) )

        if os.path.isfile(fname):
            with open(fname) as f:
                names_list = f.readline().strip('\n').split(',')
            
            # Strip the new lines, make all words lower case
            # replace - for _ and add the measure_seg prefix
            names = [ x.lower() for x in names_list ]
            names = [ x.replace('-', '_') for x in names ]
            names = [ '{}_{}_{}'.format(measure.lower(), seg.lower(), x) for x in names ]

            form_name = '{}_{}_stats'.format(measure.lower(), seg.lower())
            data_dict_list = [ '{},{},,text,{},,,number,,,,,,,,,'.format(x, form_name, y) for (x,y) in zip(names, names_list) ] 

            with open(output_name, 'a') as f:
                for x in data_dict_list:
                    f.write('{}\n'.format(x))
                
# The end
