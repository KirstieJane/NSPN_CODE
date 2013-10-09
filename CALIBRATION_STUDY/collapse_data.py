#!/usr/bin/env python

def collapse_data(data_dir, incl_excl_list, n_b0s_list, sep_av_list, transform_list, b0_orders):
    
    """
    collapse_data reads in files from a series of results_files and collapses
    across all of them so they can be plotted together
    
    Inputs:     data_dir
                incl_excl_list
                n_b0s_list
                sep_av_list
                transform_list
                b0_orders
        
    Output:     data array
    
    """
    
    #==========================================================================
    import os
    import numpy as np
    import numpy.lib.recfunctions as rfn
    from glob import glob
    import itertools as it
    #------------------------------------------------------------------------------
    from combine_data import combine_data
    #==========================================================================
    
    print '  Collapsing data: B0 orders'

    # Find all the results files in all the b0_order folders
    for incl_excl, n_b0s, sep_av, transform, roi_name in it.product(incl_excl_list, n_b0s_list, sep_av_list, transform_list, roi_list):

        # Start off with an empty data array
        data_allorders = None
        
        for b0_order in b0_orders:
            glob_string = os.path.join(data_dir, 'RESULTS', incl_excl, 'B0S_{}'.format(n_b0s),
                                    'B0_ORDER_{}'.format(b0_order), sep_av, transform, '{}_FA_MD_vol.txt'.format(roi_name))

            files = glob(glob_string)

            dict = {'b0_order': b0_order}
            
            # Read in every file and combine them
            for file in files:
                data = np.genfromtxt(file, dtype=None, delimiter=' ', names=True)
                data_allorders = combine_data(data_allorders, data, dict)
            
            # Name the results dir that this is going into:
            results_allorders_dir = os.path.join(data_dir, 'RESULTS', incl_excl, 'B0S_{}'.format(n_b0s),
                                    'ALL_ORDERS', sep_av, transform)
                                    
    return data_allorders, results_allorders_dir
    