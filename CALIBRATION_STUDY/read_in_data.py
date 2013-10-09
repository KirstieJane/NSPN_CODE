#!/usr/bin/env python

def read_in_data(results_file):
    
    """
    read_in_data reads in the data from the results_file, and includes the loc_IDs
    
    Inputs:     results_file
    
    Output:     data array
    
    """
    
    #==========================================================================
    import numpy as np
    import numpy.lib.recfunctions as rfn
    #==========================================================================
    
    print '  Reading in data'
    
    data = np.genfromtxt(results_file, dtype=None, delimiter=' ', names=True)

    # Define the loc_ids variable
    # 
    # 1: WBIC
    # 2: CBSU
    # 3: UCL
    
    # Append this information to the data rec_array. And sort.

    loc_ids = np.ones_like(data['sub'])
    loc_ids[data['loc']=='WBIC'] = 1
    loc_ids[data['loc']=='CBSU'] = 2
    loc_ids[data['loc']=='UCL'] = 3
    
    data = rfn.append_fields(data, 'loc_id', data=loc_ids, usemask=False, asrecarray=True)
    
    data.sort(order='loc_id')
    
    return data


