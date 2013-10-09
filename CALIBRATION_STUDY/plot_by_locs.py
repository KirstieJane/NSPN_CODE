#!/usr/bin/env python

def plot_by_locs(data, output_name, colors, shapes, sub_ids, loc_ids, locs, roi_name, figsize=(15,5)):
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
                locs (eg: locs)

    Optional:   figsize (default 15 x 5)

    Example usage:
        plot_by_locs(data=data, output_name=results_dir/'plot_by_subs.png',
                        colors=colors, shapes=shapes, sub_ids=sub_ids,
                        loc_ids=loc_ids, locs=locs, figsize=(15,5))

    """
    #==========================================================================
    import numpy as np
    import matplotlib.pylab as plt
    from matplotlib.ticker import MaxNLocator
    #==========================================================================
    
    # Set up the figure
    fig = plt.figure(figsize=(15,5))
    ax1 = plt.subplot(131)  # FA axis
    ax2 = plt.subplot(132)  # MD axis
    ax3 = plt.subplot(133)  # volume axis
    ax = [ax1, ax2, ax3]
    
    # Loop through the three measures (FA, MD, VOL_VOX)
    for count, measure in enumerate(['fa', 'md', 'vol_vox']):

        # Now loop through the subjects
        for i, sub in enumerate(sub_ids):

            # First things first, we want to set up a new numpy array
            # that will hold the mean values for each location so we
            # can plot a line through them at the end
            
            mean_array = np.zeros(3)
            
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
                    ax[count].scatter(np.ones(n)*loc, data[measure][mask],
                                            c=c, edgecolor=c,
                                            marker=m, s=20, alpha=0.5 )
                    
                    # ... connect them with a line ...
                    #ax[count].plot(np.ones(n)*loc, data[measure][mask], c=c)
                
                # And for everyone we'll plot the average
                # (which is just the data if you only have one point)
                mean = np.average(data[measure][mask])
                mean_array[j] = mean # Update the mean_array for plotting later!
                ax[count].scatter(loc, mean,
                                    c=c, edgecolor=c,
                                    marker=m, s=50 )
                
            # Now that we've filled up the mean_array let's plot it :)
            c=colors[3,i]
            ax[count].plot(loc_ids, mean_array, c=c, zorder=0)
            
            # Set the y limits
            # This is to deal with very small numbers (the MaxNLocator gets all turned around!)
            buffer = ( np.max(data[measure]) - np.min(data[measure]) ) / 10
            upper = np.max(data[measure]) + buffer
            lower = np.min(data[measure]) - buffer
            ax[count].set_ybound(upper, lower)  
            
        # Set the axis labels    
        ax[count].set_ylabel('{}'.format(measure.upper()))
        ax[count].set_xlabel('Scanner Location')
        
        # And label the x axis with the scanner locations
        ax[count].set_xticklabels(locs)

    # Adjust the power limits so that you use scientific notation on the y axis
    plt.ticklabel_format(style='sci', axis='y')
    [ a.yaxis.major.formatter.set_powerlimits((-3,3)) for a in ax ] 
    
    # Adjust the y axis ticks so that there are 6 at sensible places
    [ a.yaxis.set_major_locator(MaxNLocator(6)) for a in ax ]
    
    # Set the x axis ticks to be at the loc_ids
    [ a.set_xticks(loc_ids) for a in ax ]

    # Set the overall title
    fig.suptitle('Region of interest: {}'.format(roi_name), fontsize=20)
    plt.subplots_adjust(top=0.85)
    
    # And now save it!
    plt.savefig(output_name, bbox_inches=0, facecolor='w', edgecolor='w', transparent=True)
