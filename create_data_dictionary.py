#!/usr/bin/env python

import sys
import os

data_dir = sys.argv[1]

measure_list = [ 'R1', 'MT', 'R2s', 'PDw', 'FA', 'MD', 'MO', 'L1', 'L23', 'sse' ]

seg_list = [ 'aseg', 'wmparc', 'lobesStrict', '500cortExpConsecWMoverlap', '500cortConsec']

for seg in seg_list:
    for measure in measure_list:
        print seg
        print measure
        print '{}_{}_mean.csv'.format(measure, seg)
                
        fname = os.path.join(data_dir, '{}_{}_mean.csv'.format(measure, seg) )
        print fname
        with open(fname) as f:
            names_list = f.readline()
        
        # Strip the new lines, make all words lower case
        # replace - for _ and add the measure_seg prefix
        names = [ x.strip('\n') for x in names_list ]
        names = [ x.lower() for x in names ]
        names = [ x.replace('-', '_') for x in names ]
        names = [ '{}_{}_{}'.format(measure, seg, x) for x in names ]

        form_name = '{}_{}_stats'.format(measure, seg)
        data_dict_list = [ '{},{},,text,{},,,,,,,,,,,,'.format(x, form_name, y) for (x,y) in zip(names, names_list) ] 

        output_name = os.path.join(data_dir, 'data_dict.csv')
        with open(output_name, 'w') as f:
            for x in data_dict_list:
                f.write('{}\n'.format(x))
                
# The end
