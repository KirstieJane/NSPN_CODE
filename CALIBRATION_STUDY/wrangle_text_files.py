#!/usr/bin/env python

def wrangle_text_files(data_dir, incl_excl, n_b0s, b0_order, sep_av, transform, roi_name, subs, locs, scans):
    """
    Created by Kirstie Whitaker
    Email: kw401@cam.ac.uk
    
    wrangle_text_files finds all the FA, MD and volume data in files that are deep in the
    subject's individual directories and puts the appropriate information together in one file
    in the RESULTS directory
    
    Inputs:     data_dir
                incl_excl
                n_b0s
                b0_order
                sep_av
                transform
                roi_name
                subs
                locs
                scans
                
    Returns:    results_dir
                results_file
        
    Usage:      results_dir, results_file = wrangle_text_files(data_dir, incl_excl, n_b0s,
                                                b0_order, sep_av, transform, roi_name, subs,
                                                locs, scans)
    """
    #==========================================================================
    import itertools as it
    import os
    #==========================================================================

    # Define your results_dir. This folder replicates a lot of the directory structure in the subject dirs in a directory called "RESULTS" in the data_dir.
    
    # Example results_dir: RESULTS/EXCL_VOL31/B0S_6/B0_ORDER_00_14_27_41_54_68/SEPARATE/MNI_FNIRT_MPRAGE_BBR_B0
    results_dir = os.path.join(data_dir, 'RESULTS', incl_excl, 'B0S_{}'.format(n_b0s),
                                'B0_ORDER_{}'.format(b0_order), sep_av, transform)
    
    # Make the directory if it doesn't yet exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Define the results file
    # This is specific to the ROI name
    # Example results_file: bodycc_FA_MD_vol.txt
    results_file = os.path.join(results_dir, '{}_FA_MD_vol.txt'.format(roi_name))
    
    # Write all the necessary data into this results_file
    with open(results_file, 'w') as outfile:
        # Start with the header
        outfile.write('sub loc scan fa md vol_vox vol_mm3\n')

        # Next loop over the subs, locations and scans
        for sub, loc, scan in it.product(subs, locs, scans):
            
            # Create path to the correct mask dir
            mask_dir = os.path.join( data_dir, sub, loc, scan,
                                    incl_excl, 'B0S_{}'.format(n_b0s),
                                    'B0_ORDER_{}'.format(b0_order),
                                    sep_av, 'MASK_MNI_TO_DTI', transform )
        
            fa_filename = os.path.join(mask_dir, '{}_FA.txt'.format(roi_name))
            md_filename = os.path.join(mask_dir, '{}_MD.txt'.format(roi_name))
            vol_filename = os.path.join(mask_dir, '{}_vol.txt'.format(roi_name))
    
            # If the <roi>_fa.txt file exists then write all the data to the results file
            if os.path.exists(fa_filename):
                
                # First read the individual files into a variable
                # Note that the end of lines are replaced for the first two (fa and md)
                with open(fa_filename, 'r') as infile:
                    fa = infile.read().replace(' \n', '')
                with open(md_filename, 'r') as infile:
                    md = infile.read().replace(' \n', '')
                with open(vol_filename, 'r') as infile:
                    vol = infile.read()
                
                # Then write everything out to the results_file
                outfile.write('{} {} {} {} {} {}'.format(sub, loc, scan, fa, md, vol))
    
    return results_file, results_dir
                
