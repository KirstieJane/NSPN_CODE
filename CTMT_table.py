#!/usr/bin/env python

'''
Code to make a table for the supplementary analyses
for the NSPN interim cohort
'''

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
import itertools as it
from scipy.stats import linregress 
import networkx as nx

def read_in_df(data_file, aparc_names):
    df = pd.read_csv(data_file, sep=',')
    data_cols = [ x.replace('_{}'.format('thickness'), '') for x in df.columns ]
    df.columns = data_cols
    df['young'] = 0
    df['young'][df['age_scan']<np.percentile(df.age_scan, 50)] = 1
    
    df['ones'] = df['age_scan'] * 0 + 1
    df['age'] = df['age_scan']
    
    df['Global'] = df[aparc_names].mean(axis=1)
    
    df = df[df.occ==0]
    
    if df['Global'].mean() > 100:
        df[aparc_names] = df[aparc_names]/1000.0
        
    return df
    

# Define the data directory
#data_dir=os.path.join('/scratch', 'kw401', 'UCHANGE_INTERIM')
data_dir=os.path.join('/home', 'kw401', 'UCHANGE', 'INTERIM_ANALYSES')

# Define the data files we want to report on
ct_data_file = os.path.join(data_dir, 'FS_ROIS', 'PARC_500aparc_thickness_behavmerge.csv')
mt_data_file = os.path.join(data_dir, 'FS_ROIS', 'SEG_MT_500cortConsec_mean_behavmerge.csv')
mt70_data_file = os.path.join(data_dir, 'FS_ROIS', 'PARC_500aparc_MT_projfrac+030_mean_behavmerge.csv')
ct_graph_file = os.path.join(data_dir, 'FS_ROIS', 'GRAPHS', 'Graph_CT_covar_ones_all_COST_10.gpickle')

# Load the names of each region
aparc_names_file = os.path.join(data_dir, 'SUB_DATA', 'fsaverageSubP', 'parcellation', '500.names.txt' )
aparc_names = [ line.strip() for line in open(aparc_names_file) ]
aparc_names = aparc_names[41::]

# Load the lobes that correspond to each region
lobes_file = os.path.join(data_dir, 'SUB_DATA', 'fsaverageSubP', 'parcellation', '500.lobes.txt' )
lobes = [ line.strip() for line in open(lobes_file) ]
lobes = lobes[41::]

# Split up the aparc_names list into just the region names
region_list = [ x.split('_')[1] for x in aparc_names ]
region_list = list(set(region_list))

# Define the hemispheres we care about
hemi_list = [ 'lh', 'rh' ]

# Load the data
ct_df = read_in_df(ct_data_file, aparc_names)
mt_df = read_in_df(mt_data_file, aparc_names)
mt70_df = read_in_df(mt70_data_file, aparc_names)
G = nx.read_gpickle(ct_graph_file)

# Get the graph degrees
deg = G.degree()

# Define the output files
table_file_308 = os.path.join(data_dir, 'CT_MT_ANALYSES', 'TABLES', '308_regions.csv')
table_file_68 = os.path.join(data_dir, 'CT_MT_ANALYSES', 'TABLES', '68_regions.csv')
table_file_34 = os.path.join(data_dir, 'CT_MT_ANALYSES', 'TABLES', '34_regions.csv')

#=============================================================================
# We're going to start by writing out the values for every region
# separately for each hemisphere
#=============================================================================

# Create a header for the table
header = [ 'Lobe', 'Region', 'Hemisphere', 
            'Mean CT (SD)', 'Slope CT with age (x10-3)', 'p',
            'Mean MTall (SD)', 'Slope MTall with age (x10-3)', 'p',
            'Mean MT70 (SD)', 'Slope MT70 with age (x10-3)', 'p',
            'Slope CT with MTall', 'p',
            'Slope CT with MT70', 'p',
            'N Sub Regions', 'Degree' ]

with open(table_file_308, 'w') as f:
    f.write(','.join(header))
    f.write('\n')

for roi in aparc_names:
   
    # Get the lobe that this region is in
    lobe = lobes[aparc_names.index(roi)]
    
    # And the associated hemisphere
    hemi = roi.split('_')[0]
    
    # Create a list that will hold all the text you'd
    # like to print to the table
    table_list = [ lobe, roi.split('_',1)[1] , hemi]
    
    # Loop through the three data frames
    for df in [ ct_df, mt_df, mt70_df ]:
        
        # Get the mean value
        mean = df[roi].mean()
        
        # And the standard deviation
        sd = df[roi].std() 
        
        # Append these values to the table list
        # Adjust the number of decimal places according to
        # the mean value
        if mean < 1:
            table_list += [ '{:2.3f} ({:2.3f})'.format(mean, sd) ]
        else:
            table_list += [ '{:2.2f} ({:2.2f})'.format(mean, sd) ]
        
        # Find the correlation with age the average within each region
        m, c, r, p, sterr = linregress(df['age_scan'], df[roi])
        if p < 0.001:
            p = '<0.001'
        else:
            p = '{:2.3f}'.format(p)
        
        # Multiply all the values by 1000 so that they print out sensibly
        m = m * 1000.0
            
        # Change the number of decimal places according to
        # the mean measure
        if mean < 1:
            table_list += [ '{:2.2f}'.format(m), '{}'.format(p) ]
        else:
            table_list += [ '{:2.1f}'.format(m), '{}'.format(p) ]
        
    # Correlation between mean CT and mean MTall
    m, c, r, p, sterr = linregress(ct_df[roi], mt_df[roi])
    if p < 0.001:
        p = '<0.001'
    else:
        p = '{:2.3f}'.format(p)
    
    table_list += [ '{:2.3f}'.format(m), '{}'.format(p) ]
    
    # Correlation between mean CT and mean MT70
    m, c, r, p, sterr = linregress(ct_df[roi], mt70_df[roi])
    if p < 0.001:
        p = '<0.001'
    else:
        p = '{:2.3f}'.format(p)
    
    table_list += [ '{:2.3f}'.format(m), '{}'.format(p) ]
    
    # This isn't a combination across any regions
    # so the number of sub regions is always 1
    n = 1
    
    table_list += [ '{}'.format(n)]
    
    # Get the average degree for this region
    roi_deg = deg[aparc_names.index(roi)]

    table_list += [ '{:2.0f}'.format(roi_deg) ]
    
    with open(table_file_308, 'a') as f:
        f.write(','.join(table_list))
        f.write('\n')

    
#=============================================================================
# Next we're going to collapse the sub regions into their larger parcellations
# separately for each hemisphere
#=============================================================================

# Create a header for the table
header = [ 'Lobe', 'Region', 'Hemisphere', 
            'Mean CT (SD)', 'Slope CT with age (x10-3)', 'p',
            'Mean MTall (SD)', 'Slope MTall with age (x10-3)', 'p',
            'Mean MT70 (SD)', 'Slope MT70 with age (x10-3)', 'p',
            'Slope CT with MTall', 'p',
            'Slope CT with MT70', 'p',
            'N Sub Regions', 'Degree' ]

with open(table_file_68, 'w') as f:
    f.write(','.join(header))
    f.write('\n')
    
for roi, hemi in it.product(region_list, hemi_list):
   
    # Get the lobe that this region is in
    lobe = lobes[aparc_names.index('{}_{}_part1'.format(hemi, roi))]
    
    # Create a list that will hold all the text you'd
    # like to print to the table
    table_list = [ lobe, roi , hemi]
    
    # Loop through the three data frames
    for df in [ ct_df, mt_df, mt70_df ]:
        
        # Make a list of regions in the 500 parcellation
        # that correspond to each region in the D-K parcellation
        roi_list = [ x for x in aparc_names if roi in x and hemi in x ]
        
        # Get the mean of the mean value for each sub parcellation
        # of the aparc region
        # (This is fine because you have the same number of values
        # going into each regional calculation. It isn't *exactly*
        # the same as adding up all the vertices/voxels in the whole
        # region because they aren't exactly the same...but they are
        # close enough!)
        mean = df[roi_list].mean().mean()
        
        # Get the standard devation across the different sub parcellations
        # by averaging the variances and then taking the square root
        # http://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
        sd = np.sqrt(np.var(df[roi_list]).mean()) 
        
        # Append these values to the table list
        # Adjust the number of decimal places according to
        # the mean value
        if mean < 1:
            table_list += [ '{:2.3f} ({:2.3f})'.format(mean, sd) ]
        else:
            table_list += [ '{:2.2f} ({:2.2f})'.format(mean, sd) ]
        
        # Find the correlation with age the average within each region
        m, c, r, p, sterr = linregress(df['age_scan'], df[roi_list].mean(axis=1))
        if p < 0.001:
            p = '<0.001'
        else:
            p = '{:2.3f}'.format(p)
        
        # Multiply all the values by 1000 so that they print out sensibly
        m = m * 1000.0
            
        # Change the number of decimal places according to
        # the mean measure
        if mean < 1:
            table_list += [ '{:2.2f}'.format(m), '{}'.format(p) ]
        else:
            table_list += [ '{:2.1f}'.format(m), '{}'.format(p) ]
        
    # Correlation between mean CT and mean MTall
    m, c, r, p, sterr = linregress(ct_df[roi_list].mean(axis=1), mt_df[roi_list].mean(axis=1))
    if p < 0.001:
        p = '<0.001'
    else:
        p = '{:2.3f}'.format(p)
    
    table_list += [ '{:2.3f}'.format(m), '{}'.format(p) ]
    
    # Correlation between mean CT and mean MT70
    m, c, r, p, sterr = linregress(ct_df[roi_list].mean(axis=1), mt70_df[roi_list].mean(axis=1))
    if p < 0.001:
        p = '<0.001'
    else:
        p = '{:2.3f}'.format(p)
    
    table_list += [ '{:2.3f}'.format(m), '{}'.format(p) ]
    
    # Write out the number of sub parcellations this region has
    n = len(roi_list)
    
    table_list += [ '{}'.format(n)]
    
    # Get the average degree for this region
    deg_list = []
    for sub_r in roi_list:
        deg_list += [ deg[aparc_names.index(sub_r)] ]
    
    mean_deg = np.mean(deg_list)
    table_list += [ '{:2.1f}'.format(mean_deg) ]
    
    with open(table_file_68, 'a') as f:
        f.write(','.join(table_list))
        f.write('\n')        
        
#=============================================================================
# And follow that by collapsing across hemispheres
#=============================================================================

# Create a header for the table
header = [ 'Lobe', 'Region', 
            'Mean CT (SD)', 'Slope CT with age (x10-3)', 'p',
            'Mean MTall (SD)', 'Slope MTall with age (x10-3)', 'p',
            'Mean MT70 (SD)', 'Slope MT70 with age (x10-3)', 'p',
            'Slope CT with MTall', 'p',
            'Slope CT with MT70', 'p',
            'N Sub Regions', 'Degree' ]

with open(table_file_34, 'w') as f:
    f.write(','.join(header))
    f.write('\n')
    
for roi in region_list:

    # Get the lobe that this region is in
    lobe = lobes[aparc_names.index('lh_{}_part1'.format(roi))]

    # Create a list that will hold all the text you'd
    # like to print to the table
    table_list = [ lobe, roi ]

    # Loop through the three data frames
    for df in [ ct_df, mt_df, mt70_df ]:
        
        # Make a list of regions in the 500 parcellation
        # that correspond to each region in the D-K parcellation
        roi_list = [ x for x in aparc_names if roi in x ]
        
        # Get the mean of the mean value for each sub parcellation
        # of the aparc region
        # (This is fine because you have the same number of values
        # going into each regional calculation. It isn't *exactly*
        # the same as adding up all the vertices/voxels in the whole
        # region because they aren't exactly the same...but they are
        # close enough!)
        mean = df[roi_list].mean().mean()
        
        # Get the standard devation across the different sub parcellations
        # by averaging the variances and then taking the square root
        # http://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
        sd = np.sqrt(np.var(df[roi_list]).mean()) 
        
        # Append these values to the table list
        # Adjust the number of decimal places according to
        # the mean value
        if mean < 1:
            table_list += [ '{:2.3f} ({:2.3f})'.format(mean, sd) ]
        else:
            table_list += [ '{:2.2f} ({:2.2f})'.format(mean, sd) ]
        
        # Find the correlation with age the average within each region
        m, c, r, p, sterr = linregress(df['age_scan'], df[roi_list].mean(axis=1))
        if p < 0.001:
            p = '<0.001'
        else:
            p = '{:2.3f}'.format(p)

        # Multiply all the values by 1000 so that they print out sensibly
        m = m * 1000.0
            
        # Change the number of decimal places according to
        # the mean measure
        if mean < 1:
            table_list += [ '{:2.2f}'.format(m), '{}'.format(p) ]
        else:
            table_list += [ '{:2.1f}'.format(m), '{}'.format(p) ]
        
    # Correlation between mean CT and mean MTall
    m, c, r, p, sterr = linregress(ct_df[roi_list].mean(axis=1), mt_df[roi_list].mean(axis=1))
    if p < 0.001:
        p = '<0.001'
    else:
        p = '{:2.3f}'.format(p)
    
    table_list += [ '{:2.3f}'.format(m), '{}'.format(p) ]
    
    # Correlation between mean CT and mean MT70
    m, c, r, p, sterr = linregress(ct_df[roi_list].mean(axis=1), mt70_df[roi_list].mean(axis=1))
    if p < 0.001:
        p = '<0.001'
    else:
        p = '{:2.3f}'.format(p)
        
    table_list += [ '{:2.3f}'.format(m), '{}'.format(p) ]
    
    # Write out the number of sub parcellations this region has
    n = len(roi_list)
    
    table_list += [ '{}'.format(n)]
    
    # Get the average degree for this region
    deg_list = []
    for sub_r in roi_list:
        deg_list += [ deg[aparc_names.index(sub_r)] ]
    
    mean_deg = np.mean(deg_list)
    table_list += [ '{:2.1f}'.format(mean_deg) ]
    
    with open(table_file_34, 'a') as f:
        f.write(','.join(table_list))
        f.write('\n')
            


