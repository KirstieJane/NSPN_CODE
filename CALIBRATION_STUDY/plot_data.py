#!/usr/bin/env python

def plot_data(results_file, results_dir):
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
    
    # Define your colors and shapes
    # 
    # Colors:
    # 
    # * Sub 6: pink
    # * Sub 7: green
    # * Sub 8: yellow

    colors = np.array([['#810051','#138900','#a6a300'],
                        ['#95256c','#389e28','#bfbc30'],
                        ['#c7007d', '#1dd300', '#fffa00'],
                        ['#e366b5','#7ae969','#fffc73']])

    # Shapes:
    # 
    # * Loc 1 (WBIC): circle
    # * Loc 2 (CBSU): diamond
    # * Loc 3 (UCL): triangle

    shapes = np.array(['o','D','^'])

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

