#!/usr/bin/env python

def plot_data(data, results_dir, roi_name, colors, shapes):
    """
    plot_data does exactly what it says on the tin
    
    It calls plot_by_subs and plot_by_locs and saves these pictures
    as png files in the results_dir
    
    Inputs:     data (numpy rec array)
                results_dir
                roi_name
                colors
                shapes
    """
    #==========================================================================
    import numpy as np
    import os
    #--------------------------------------------------------------------------
    from plot_by_subs import plot_by_subs
    from plot_by_locs import plot_by_locs
    from plot_by_b0s import plot_by_b0s
    #==========================================================================
    
    # Make sure the results_dir exists
    if not os.path.exists(results_dir):
       os.makedirs(results_dir)

    output_name = os.path.join(results_dir, '{}_plot_by_locs.png'.format(roi_name))

    if not os.path.exists(output_name):
        print '  Plotting data'
    
        # Get a list of the subjects:
        subs = [ sub for sub in data['sub'] ]
        subs = list(set(subs))
        sub_ids = [ np.int(sub) for sub in subs ]

        # List of location_ids
        loc_ids = [ loc_id for loc_id in data['loc_id'] ]
        loc_ids = list(set(loc_ids))
        loc_ids = [ np.int(loc_id) for loc_id in loc_ids ]

        # List of locations
        locs = [ data['loc'][data['loc_id']==loc_id][0] for loc_id in loc_ids ]
        
        # List of scans
        scans = [ scan for scan in data['scan'] ]
        scans = list(set(scans))
        
        output_name = os.path.join(results_dir, '{}_plot_by_subs.png'.format(roi_name))
        
        plot_by_subs(data=data, output_name=output_name,
                                colors=colors, shapes=shapes,
                                sub_ids=sub_ids, loc_ids=loc_ids, scans=scans,
                                roi_name=roi_name, figsize=(15,5))
        
        output_name = os.path.join(results_dir, '{}_plot_by_locs.png'.format(roi_name))
        
        plot_by_locs(data=data, output_name=output_name,
                                colors=colors, shapes=shapes,
                                sub_ids= sub_ids, loc_ids=loc_ids, locs=locs, scans=scans,
                                roi_name=roi_name, figsize=(15,5))
                                
        if 'n_b0s' in data.dtype.names:
            output_name = os.path.join(results_dir, '{}_plot_by_b0s.png'.format(roi_name))
            
            plot_by_b0s(data=data, output_name=output_name,
                                    colors=colors, shapes=shapes,
                                    sub_ids= sub_ids, loc_ids=loc_ids, locs=locs, scans=scans,
                                    roi_name=roi_name, figsize=(15,5))
                                
    else:
        print '  Data already plotted'
