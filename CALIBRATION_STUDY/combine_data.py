#!/usr/bin/env python

def combine_data(data_all, data_new, dict):
    
    """
    combine_data reads in the data from a series of results_files,
    and includes the loc_IDs
    
    Inputs:     results_file
    
    Output:     data array
    
    """
    
    #==========================================================================
    import numpy as np
    import numpy.lib.recfunctions as rfn
    #==========================================================================
    
    print '  Combining data'
    
    for key, value in dict.items():
        # Define a new column that is the same value for everyone in the
        # new data array
        
        col = np.array(value)
        col = np.repeat(col, len(data_new))
        
        # Add this column to the data_new recarray
        
        data_new = rfn.append_fields(data_new, key, col, usemask=False, asrecarray=True)

    # If data_all exists, then join data_new to the end of it
    if not data_all == None:
        
        data_all = rfn.stack_arrays((data_all, data_new), usemask=False, asrecarray=True)
    
    # If data_all doesn't yet exist then data_new becomes data_all
    else:
        data_all = data_new
        
    data_all.sort(order='loc_id')
    
    return data_all


