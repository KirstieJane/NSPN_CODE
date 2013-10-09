#!/usr/bin/env python

def Q_n_b0s(data_dir, incl_excl_list, sep_av_list, transform_list, roi_list, ec_b0_list, colors, shapes):
    """
    Q_ec_vol_n6 asks the question:
        "How does the number of B0s you include change your measurement?"
        
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
    from read_in_data import read_in_data
    #==========================================================================
    
    print '  Question: How does the number of B0s change your measurement'

    # Find all the results files in all the b0_order folders
    for incl_excl, sep_av, transform, roi_name, ec_b0 in it.product(incl_excl_list, sep_av_list, transform_list, roi_list, ec_b0_list):

        # Start off with an empty data array
        data_allorders_allb0s = None
        
        for n_b0s in range(1,7):
            
            b0_orders = get_b0_orders(np.int(n_b0s))

            b0_orders = [ order for order in b0_orders if order[:2] == ec_b0 ]
            
            for b0_order in b0_orders:
            
                glob_string = os.path.join(data_dir, 'RESULTS', incl_excl, 'B0S_{}'.format(n_b0s),
                                        'B0_ORDER_{}'.format(b0_order), sep_av, transform, '{}_FA_MD_vol.txt'.format(roi_name))

                files = glob(glob_string)

                dict = { 'b0_order': b0_order, 'n_b0s' : n_b0s }
                
                # Read in every file and combine them
                for file in files:
                    data = read_in_data(file)
                    data_allorders_allb0s = combine_data(data_allorders_allb0s, data, dict)
                
        # Name the results dir that this is going into:
        results_allorders_allb0s_dir = os.path.join(data_dir, 'RESULTS', incl_excl, 'ALL_B0S',
                                'B0_{}'.format(ec_b0), sep_av, transform)
        
        # Now plot the data
        plot_data(data_allorders_allb0s, results_allorders_allb0s_dir, roi_name, colors, shapes)