# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# NSPN Calibration Study: DTI analyses

# <headingcell level=3>

# Overview

# <markdowncell>

# As a pilot for the first study involving the NSPN cohort (UCHANGE) we ran a "travelling heads" calibration study.
# 
# Three adults were scanned using the same protocol at three idential scanners (in different locations).

# <headingcell level=3>

# Most basic analysis

# <markdowncell>

# **Lets start by looking at a very standard way of analysing the data**
# 
# 1. Eddy-correct the data to the first B0 acquisition
# 2. Run the tensor fit and create FA and MD maps
# 3. Register the first B0 volume to the MPRAGE
# 4. Register the MPRAGE to standard space via FNIRT
# 5. Invert the warp and then back project the body of the corpus callosum ROI to DTI space
# 6. Extract the mean FA, mean MD and the volume of this ROI
# 7. Plot
# 
# Steps 1 - 6 are done in a bash script, but here we plot the data

# <markdowncell>

# **First of all we have to wrangle the data we need**

# <markdowncell>

# Import whatever you need

# <codecell>

import itertools as it
import os
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator, AutoLocator

# Define the data directory

data_dir = '/work/imagingD/NSPN/workspaces/kw401/BOOTSTRAPPING_ANALYSES'

# Define some variables
# 
# *  subs: subject ids
# *  locs: scanner locations
# *  scans: which of the DTI scans you're using (This only applies to the UCL data where there are two. At all other sites there is only one.)
# *  incl_excl: either the original data or data excluding volume 31 which is not the same at CBSU as it is at the other sites
# *  n_b0s: number of b0s used to calculate tensor
# *  ec_b0: the b0 that was used as the eddy_correct target volume
# *  b0_order: the order that the b0s were in the 4D file
# *  sep_av: whether the b0s were averaged before the tensor was fit or kept separate
# *  transform: the registration path from MNI to DTI space
# *  roi_name: the name of the roi

subs = [ '06', '07', '08' ]
locs = [ 'WBIC', 'CBSU', 'UCL' ]
scans = [ 'DTI_2A', 'DTI_2B' ]
incl_excl = [ 'EXCL_VOL31']
n_b0s = [ '6' ]
ec_b0 = [ '00' ]
b0_order = [ '00_14_27_41_54_68' ]
sep_av = [ 'SEPARATE' ] 
transform = [ 'MNI_FNIRT_MPRAGE_BBR_B0' ]
roi_name = [ 'lcing', 'rcing', 'wholebrain', 'bodycc' ]

# The FA, MD and volume data are all in files created in the appropriate directories deep in the data_dir.
# So now we're going to put them all together.

# Define your results_dir. This folder replicates a lot of the directory structure in the subject dirs in a directory called "RESULTS" in the data_dir.
# Make the results_dir if it doesn't already exist.
# Make a list of all the data files that you need to concatenate. Then concatenate them!

def wrangle_text_files(incl_excl, n_b0s, ec_b0, b0_order, sep_av, transform, roi_name):
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
                

def plot_by_subs(data, output_name, colors, shapes, sub_ids, loc_ids, figsize=(15,5)):
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


def plot_by_locs(data, output_name, colors, shapes, sub_ids, loc_ids, locs, figsize=(15,5)):
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
    
    fig = plt.figure(figsize=(15,5))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
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
                    ax[count].plot(np.ones(n)*loc, data[measure][mask], c=c)
                
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



# **We have our data! Let's load it into numpy and start plotting**

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

for incl_excl, n_b0s, ec_b0, b0_order, sep_av, transform, roi_name in it.product(incl_excl, n_b0s, ec_b0, b0_order, sep_av, transform, roi_name):
    
    results_file, results_dir = wrangle_text_files(incl_excl, n_b0s, ec_b0, b0_order, sep_av, transform, roi_name)
    
    plot_data(results_file, results_dir)


# The end