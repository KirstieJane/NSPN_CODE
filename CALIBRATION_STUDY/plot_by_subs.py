#!/usr/bin/env python

def plot_by_subs(data, output_name, colors, shapes, sub_ids, loc_ids, roi_name, figsize=(15,5)):
    """
    Plot_by_subs takes a data rec_array and loops through three measures:
        fa, md, and vol_vox and plots each on separate plots with the
        x-axis representing the individual subjects
    
    Required:   data rec_array (eg: data)
                output_name (eg: results_dir/'plot_by_subs.png')
                colors (eg: colors)
                shapes (eg: shapes)
                sub_ids (eg: sub_ids)
                loc_ids (eg: loc_ids)

    Optional:   figsize (default 15 x 5)

    Example usage:
        plot_by_subs(data=data, output_name=results_dir/'plot_by_subs.png',
                        colors=colors, shapes=shapes, sub_ids=sub_ids,
                        loc_ids=loc_ids, figsize=(15,5))

    """
    #==========================================================================
    import numpy as np
    import matplotlib.pylab as plt
    from matplotlib.ticker import MaxNLocator
    #==========================================================================
    
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    ax = [ax1, ax2, ax3]
    
    # Loop through the three measures (FA, MD, VOL_VOX)
    for count, measure in enumerate(['fa', 'md', 'vol_vox']):

        # Now loop through the subjects and the locations
        for i, sub in enumerate(sub_ids):

            for j, loc in enumerate(loc_ids):

                # Assign the correct color and shape for the marker
                c=colors[j,i]
                m=shapes[j]
                
                # Mask the data so you only have this sub at this loc's numbers
                mask = ( data['sub'] == sub ) & ( data['loc_id']== loc )

                # Find the number of data points you have for this sub at this location
                n = np.sum(mask)

                if n > 1:
                    # If you have more than one data point then we're going to plot
                    # the individual points (kinda small and a little transparent)
                    ax[count].scatter(np.ones(n)*sub, data[measure][mask],
                                            c=c, edgecolor=c,
                                            marker=m, s=20, alpha=0.5 )
                    
                    # ... connect them with a line ...
                    ax[count].plot(np.ones(n)*sub, data[measure][mask], c=c)
                
                # And for everyone we'll plot the average
                # (which is just the data if you only have one point)
                mean = np.average(data[measure][mask])
                ax[count].scatter(sub, mean,
                                    c=c, edgecolor=c,
                                    marker=m, s=50 )
                
                # Set the y limits
                # This is to deal with very small numbers (the MaxNLocator gets all turned around!)
                buffer = ( np.max(data[measure]) - np.min(data[measure]) ) / 10
                upper = np.max(data[measure]) + buffer
                lower = np.min(data[measure]) - buffer
                ax[count].set_ybound(upper, lower)    

        # Set the axis labels
        ax[count].set_ylabel('{}'.format(measure.upper()))
        ax[count].set_xlabel('Subject ID')

    # Adjust the power limits so that you use scientific notation on the y axis
    plt.ticklabel_format(style='sci', axis='y')
    [ a.yaxis.major.formatter.set_powerlimits((-3,3)) for a in ax ] 
    
    # Adjust the y axis ticks so that there are 6 at sensible places
    [ a.yaxis.set_major_locator(MaxNLocator(6)) for a in ax ]
    
    # Set the xaxis ticks to be sensible
    [ a.xaxis.set_major_locator(MaxNLocator(4)) for a in ax ]
    
    # Set the x axis ticks to be at the sub_ids
    [ a.set_xticks(sub_ids) for a in ax ]
    
    # Set the overall title
    fig.suptitle('Region of interest: {}'.format(roi_name), fontsize=20)
    plt.subplots_adjust(top=0.85)
    
    # And now save it
    plt.savefig(output_name, bbox_inches=0, facecolor='w', edgecolor='w', transparent=True)


