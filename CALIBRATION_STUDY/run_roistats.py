#!/usr/bin/env python

def run_roistats(data_dir, incl_excl, n_b0s, b0_order, sep_av, subs, locs, scans):

    """
    Created by Kirstie Whitaker
    Email: kw401@cam.ac.uk
    
    runs a bash script for all the various roi extractions
    
    Inputs:     data_dir
                incl_excl
                n_b0s
                b0_order
                sep_av
                subs
                locs
                scans
                rois_dir
                
    Usage:      run_roistats(data_dir, incl_excl, n_b0s,
                                    b0_order, sep_av, subs,
                                    locs, scans, rois_dir)
    """
    #==========================================================================
    import itertools as it
    import subprocess
    import os
    #==========================================================================

    # You're going to loop over the subs, locations and scans
    for sub, loc, scan in it.product(subs, locs, scans):

        # Define the various components needed by the registration script
        registration_script = os.path.join(data_dir, 'SCRIPTS',
                                            'DTI_PROCESSING-master',
                                            'roistats.sh')

        rois_dir = os.path.join(data_dir, 'ROIS')
        
        dti_dir = os.path.join(data_dir, sub, loc, scan,
                                    incl_excl, 'B0S_{}'.format(n_b0s),
                                    'B0_ORDER_{}'.format(b0_order), 
                                    sep_av)
                                    
        reg_dir = os.path.join(data_dir, sub, loc, 'REG')
               
        ec_b0 = b0_order.split('_')[0]
        
        # Setup the command
        command = '{} {} {} {} {} {}'.format(registration_script, 
                                        rois_dir,
                                        dti_dir,
                                        reg_dir,
                                        scan,
                                        ec_b0)
    
        # And if the dti directory exists, then run it!
        if os.path.exists(dti_dir):
            subprocess.call(command, shell=True)
        
