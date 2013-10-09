#!/usr/bin/env python

def Q_ec_vol_n6(data_dir, incl_excl_list, sep_av_list, transform_list, roi_list, colors, shapes):
    """
    Q_ec_vol_n6 asks the question:
        "How does the volume that you register to affect the measurement
        when you use all the data"
        
    It reads in all the necessary files from a series of results_files and collapses
    across all of them so they can be plotted together
    
    Inputs:     data_dir
                incl_excl_list
                sep_av_list
                transform_list
                roi_list
        
    Output:     data array
    
    """
    
    #==========================================================================
    import os
    import numpy as np
    import numpy.lib.recfunctions as rfn
    from glob import glob
    import itertools as it
    #---------------------------------------------------------------------------
    from combine_data import combine_data
    from get_b0_orders import get_b0_orders
    from plot_data import plot_data
    #==========================================================================
    
    print '  Question: How does the choice of eddy correct volume affect the measurements?'

    # Find all the results files in all the b0_order folders
    for incl_excl, sep_av, transform, roi_name in it.product(incl_excl_list, sep_av_list, transform_list, roi_list):

        # Start off with an empty data array
        data_allorders = None
        
        b0_orders = get_b0_orders(np.int(6))

        for b0_order in b0_orders:
            glob_string = os.path.join(data_dir, 'RESULTS', incl_excl, 'B0S_6',
                                    'B0_ORDER_{}'.format(b0_order), sep_av, transform, '{}_FA_MD_vol.txt'.format(roi_name))

            files = glob(glob_string)

            dict = {'b0_order': b0_order}
            
            # Read in every file and combine them
            for file in files:
                data = np.genfromtxt(file, dtype=None, delimiter=' ', names=True)
                data_allorders = combine_data(data_allorders, data, dict)
            
        # Name the results dir that this is going into:
        results_allorders_dir = os.path.join(data_dir, 'RESULTS', incl_excl, 'B0S_6',
                                'ALL_ORDERS', sep_av, transform)
        
        # Now plot the data
        plot_data(data_allorders, results_allorders_dir, roi_name, colors, shapes)
            