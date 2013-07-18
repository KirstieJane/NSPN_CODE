#!/usr/bin/env python

def read_in_data(results_file, results_dir):
    
    """
    read_in_data reads in the data from the results_file, and includes the loc_IDs
    """
    data = np.genfromtxt(results_file, dtype=None, delimiter=' ', names=True)

    # Define the loc_ids variable
    # 
    # * 1: WBIC
    # * 2: CBSU
    # * 3: UCL
    # 
    # Append this information to the data rec_array. And sort.

    loc_ids = np.ones_like(data['sub'])
    loc_ids[data['loc']=='WBIC'] = 1
    loc_ids[data['loc']=='CBSU'] = 2
    loc_ids[data['loc']=='UCL'] = 3
    
    data = rfn.append_fields(data, 'loc_id', data=loc_ids, usemask=False, asrecarray=True)
    
    data.sort(order='loc_id')
    

    # **Now we're ready to make our plots**
    # 
    sub_ids = [ np.int(sub) for sub in subs ]
    loc_ids = list(set(loc_ids))
    
    output_name = os.path.join(results_dir, '{}_plot_by_subs.png'.format(roi_name))
    
    plot_by_subs(data=data, output_name=output_name,
                            colors=colors, shapes=shapes,
                            sub_ids= sub_ids, loc_ids=loc_ids, figsize=(15,5))
    
    output_name = os.path.join(results_dir, '{}_plot_by_locs.png'.format(roi_name))
    
    plot_by_locs(data=data, output_name=output_name,
                            colors=colors, shapes=shapes,
                            sub_ids= sub_ids, loc_ids=loc_ids, locs=locs, figsize=(15,5))

