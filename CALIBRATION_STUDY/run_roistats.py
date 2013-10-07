#!/usr/bin/env python

def run_roistats(data_dir, incl_excl, n_b0s, b0_order, sep_av, subs, locs, scans, transform):

    """
    Created by Kirstie Whitaker
    Email: kw401@cam.ac.uk
    
    run_registrations runs all the necessary registrations
    
    Inputs:     data_dir
                incl_excl
                n_b0s
                b0_order
                sep_av
                subs
                locs
                scans
                
    Usage:      run_registrations(data_dir, incl_excl, n_b0s,
                                    b0_order, sep_av, transform, subs,
                                    locs, scans)
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
                                            'registrations.sh')

        dti_dir = os.path.join(data_dir, sub, loc, scan,
                                    incl_excl, 'B0S_{}'.format(n_b0s),
                                    'B0_ORDER_{}'.format(b0_order), 
                                    sep_av)
                                    
        highres_dir = os.path.join(data_dir, sub, loc, 'MPRAGE')
               
        ec_b0 = b0_order.split('_')[0]
        
        # Setup the command
        command = '{} {} {} {} {}'.format(registration_script, 
                                        dti_dir,
                                        highres_dir,
                                        scan,
                                        ec_b0)
    
        # And if the dti directory exists, then run it!
        if os.path.exists(dti_dir):
            subprocess.call(command, shell=True)
        
