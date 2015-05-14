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
from NSPN_functions import *

#==============================================================================
def create_3tables(data_dir, graph_dir, table_dir, fsaverage_dir):
    '''
    This is the main script! It makes your three tables!
    '''
    print '    Setting up table data'
    table_var_dict = setup_table_data(data_dir, graph_dir, table_dir, fsaverage_dir)
    print '    Making table n=308'
    write_table(table_var_dict, n=308)
    print '    Making table n=68'
    write_table(table_var_dict, n=68)
    print '    Making table n=34'
    write_table(table_var_dict, n=34)


def setup_table_data(data_dir, graph_dir, table_dir, fsaverage_dir):
    '''
    Read in the files you need, and define the output files
    '''
    
    # Create a table_var dictionary to contain all the important
    # information you're going to create here
    table_var_dict = {}
    
    # Define the data files we want to report from
    # (from which we want to report :P)
    table_var_dict['ct_data_file'] = os.path.join(data_dir, 
                                                        'PARC_500aparc_thickness_behavmerge.csv')
    table_var_dict['mt_data_file'] = os.path.join(data_dir, 
                                                        'SEG_MT_500cortConsec_mean_behavmerge.csv')
    table_var_dict['mt70_data_file'] = os.path.join(data_dir,
                                                        'PARC_500aparc_MT_projfrac+030_mean_behavmerge.csv')
                                                        
    table_var_dict['ct_graph_file'] = os.path.join(graph_dir,
                                                         'Graph_CT_covar_ones_all_COST_10.gpickle')

    # Load the names of each region
    aparc_names_file = os.path.join(fsaverage_dir, 'parcellation', '500.names.txt' )
    table_var_dict['aparc_names'] = [ line.strip() for line in open(aparc_names_file) ]
    table_var_dict['aparc_names'] = table_var_dict['aparc_names'][41::]

    # Load the lobes that correspond to each region
    lobes_file = os.path.join(fsaverage_dir, 'parcellation', '500.lobes.txt' )
    table_var_dict['lobes'] = [ line.strip() for line in open(lobes_file) ]
    table_var_dict['lobes'] = table_var_dict['lobes'][41::]

    # Split up the aparc_names list into just the region names
    table_var_dict['region_list'] = [ x.split('_')[1] for x in table_var_dict['aparc_names'] ]
    table_var_dict['region_list'] = list(set(table_var_dict['region_list']))

    # Define the hemispheres we care about
    table_var_dict['hemi_list'] = [ 'lh', 'rh' ]

    # Load the data
    table_var_dict['ct_df'] = read_in_df(table_var_dict['ct_data_file'], 
                                            table_var_dict['aparc_names'])
    table_var_dict['mt_df'] = read_in_df(table_var_dict['mt_data_file'], 
                                            table_var_dict['aparc_names'])
    table_var_dict['mt70_df'] = read_in_df(table_var_dict['mt70_data_file'], 
                                            table_var_dict['aparc_names'])
    table_var_dict['G'] = nx.read_gpickle(table_var_dict['ct_graph_file'])

    # Get the graph degrees
    table_var_dict['deg'] = table_var_dict['G'].degree()

    # Define the output files
    table_var_dict['table_file_308'] = os.path.join(table_dir, '308_regions.csv')
    table_var_dict['table_file_68'] = os.path.join(table_dir, '68_regions.csv')
    table_var_dict['table_file_34'] = os.path.join(table_dir, '34_regions.csv')
        
    # Create a header for the table
    table_var_dict['header'] = [ 'Lobe', 'Region', 'Hemisphere', 
                                    'Mean CT (SD)', 'Slope CT with age (x10-3)', 'perm_p',
                                    'Mean MTall (SD)', 'Slope MTall with age (x10-3)', 'perm_p',
                                    'Mean MT70 (SD)', 'Slope MT70 with age (x10-3)', 'perm_p',
                                    'Slope CT with MTall', 'perm_p',
                                    'Slope CT with MT70', 'perm_p',
                                    'N Sub Regions', 'Degree' ]

    return table_var_dict



#==============================================================================
def write_table(table_var_dict, n=308):
    '''
    Write out the values into a table calling a buch of 
    little functions along the way
    '''
    
    #------------------------------------------------------
    # Put the header at the top of the file
    write_header(table_var_dict, n)

    #------------------------------------------------------
    # Define the roi list you're going to loop over
    roi_list = get_roi_list(table_var_dict, n)
    
    #------------------------------------------------------
    # Loop through the regions in the roi_list
    for roi in roi_list:
        
        #------------------------------------------------------------
        # Create the table_list with the appropriate
        # lobe, region and if necessary hemisphere names
        # and the sub_roi_list with all the rois that should be 
        # considered in the calculations
        table_list, sub_roi_list = create_lists(table_var_dict, roi, n)
        
        #------------------------------------------------------------
        # Loop through the three data frames
        for df in [ table_var_dict['ct_df'], 
                    table_var_dict['mt_df'],
                    table_var_dict['mt70_df'] ]:
            
            #------------------------------------------------------------
            # Write the mean and standard deviation into the table_list
            table_list = append_mean_std(table_list, df, sub_roi_list)
            
            #------------------------------------------------------------
            # Find the correlation with age the average within each region
            table_list = append_correlation(table_list, df, ['age_scan'], sub_roi_list)
            
        #---------------------------------------------------------------------
        # Look at correlations across pairs of data frames
        
        # Adjust the roi_names because we're going to merge the data frames
        ct_sub_roi_list = [ '{}_ct'.format(x) for x in sub_roi_list ]
        mt_sub_roi_list = [ '{}_mt'.format(x) for x in sub_roi_list ]
        
        # Correlation between mean CT and mean MTall
        ct_mt_df = table_var_dict['ct_df'].merge(table_var_dict['mt_df'], 
                                                    on='nspn_id', 
                                                    how='inner', 
                                                    suffixes=['_ct', '_mt'])
                                                    
        table_list = append_correlation(table_list, ct_mt_df, ct_sub_roi_list, mt_sub_roi_list)
        
        # Correlation between mean CT and mean MT70
        ct_mt70_df = table_var_dict['ct_df'].merge(table_var_dict['mt70_df'], 
                                                    on='nspn_id', 
                                                    how='inner', 
                                                    suffixes=['_ct', '_mt'])
                                                    
        table_list = append_correlation(table_list, ct_mt70_df, ct_sub_roi_list, mt_sub_roi_list)
        
        #---------------------------------------------------------------------
        # Write the number of sub regions for this roi into the table_list
        n_sub = len(sub_roi_list)
        
        table_list += [ '{}'.format(n_sub)]
        
        #---------------------------------------------------------------------        
        # Get the average degree for this region
        table_list = append_degree(table_var_dict, table_list, sub_roi_list)
        
        #---------------------------------------------------------------------        
        # And then write out the table :)
        write_table_list(table_var_dict, table_list, n=n)
        
#==============================================================================        
def write_header(table_var_dict, n=308):
    '''
    Write the header into the text file
    n can be 308, 68 or 34
    '''
    # Get the appropriate variables from the dictionary
    table_file_name = table_var_dict['table_file_{}'.format(n)]
    header = table_var_dict['header']
    
    # Remove the hemisphere column if n=34
    if n == 34:
        header.remove('Hemisphere')

    # Put the header at the top of the file
    with open(table_file_name, 'w') as f:
        f.write(','.join(header))
        f.write('\n')

        
def get_roi_list(table_var_dict, n=308):
    '''
    Figure out which of the aparc_names regions you're going to loop over
    '''
    # If n = 308 then it's really easy, you want all the aparc_names regions
    if n == 308:
        roi_list = table_var_dict['aparc_names']
        
    # If n = 68 you want the DK regions so we'll pick the first aparc region
    # for each DK atlas region
    if n == 68:
        roi_list = [ x for x in table_var_dict['aparc_names'] if  x.endswith('part1') ]

    # If n = 34 you want only the left hemisphere DK regions
    if n == 34:
        roi_list = [ x for x in table_var_dict['aparc_names'] if  x.endswith('part1') and x.startswith('lh') ]
        
    return roi_list
    
#==============================================================================
def create_lists(table_var_dict, roi, n=308):
    '''
    Creates the table_list which contains the text that will be written out
    at the end, and also the sub_roi_list that contains all the regions in
    aparc_names that should be considered for this row in the table
    '''
    # Get the lobe that this region is in
    lobe = table_var_dict['lobes'][table_var_dict['aparc_names'].index(roi)]
    
    # And the associated hemisphere
    hemi = roi.split('_')[0]
    
    # Everything's easy if you're going to loop through every single one
    if n == 308:
        roi_name = roi.split('_',1)[1]
        table_list = [ lobe, roi_name, hemi ]
        sub_roi_list = [ roi ]

    # If you're collapsing across DK region but not hemisphere
    # you'll need to strip the "partX" section of the roi_name
    if n == 68:
        roi_name = roi.split('_')[1]
        table_list = [ lobe, roi_name, hemi ]
        sub_roi_list = [ x for x in table_var_dict['aparc_names'] if roi_name in x and hemi in x ]
        
    # If you're collapsing across hemisphere then drop the
    # hemi part
    if n == 34:
        roi_name = roi.split('_')[1]
        table_list = [ lobe, roi_name ]
        sub_roi_list = [ x for x in table_var_dict['aparc_names'] if roi_name in x ]
    
    return table_list, sub_roi_list
    
#==============================================================================
def append_mean_std(table_list, df, sub_roi_list):
    '''
    This is super easy if you only have one roi in the
    sub_roi_list (n=308), but a little more complicated
    when you have a few that you're going to average across
    '''
    # Get the mean of the mean value for each sub parcellation
    # of the aparc region
    # (This is fine because you have the same number of values
    # going into each regional calculation. It isn't *exactly*
    # the same as adding up all the vertices/voxels in the whole
    # region because they aren't exactly the same...but they are
    # close enough!)
    mean = df[sub_roi_list].mean().mean()
    
    # Get the standard devation across the different sub parcellations
    # by averaging the variances and then taking the square root
    # http://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
    sd = np.sqrt(np.var(df[sub_roi_list]).mean()) 
        
    # Append these values to the table list
    # Adjust the number of decimal places according to
    # the mean value
    if mean < 1:
        table_list += [ '{:2.3f} ({:2.3f})'.format(mean, sd) ]
    else:
        table_list += [ '{:2.2f} ({:2.2f})'.format(mean, sd) ]
        
    return table_list

#==============================================================================
def append_correlation(table_list, df, x_col, y_col):
    '''
    Figure out the correlation between x_col(s) and y_col(s)
    and append those values to the table_list
    '''
    if len(x_col) == 1 and len(y_col) == 1:
        # Find the correlation between x_col and y_col
        m, c, r, p, sterr, perm_p = permutation_correlation(df[x_col[0]].values, df[y_col[0]].values)

    elif len(x_col) > 1 and len(y_col) == 1:
        # Find the correlation between the average of x_col(s) and y_col
        m, c, r, p, sterr, perm_p = permutation_correlation(df[x_col].mean(axis=1).values, df[y_col[0]].values)
    
    elif len(x_col) == 1 and len(y_col) > 1:
        # Find the correlation between the average of x_col and y_col(s)
        m, c, r, p, sterr, perm_p = permutation_correlation(df[x_col[0]].values, df[y_col].mean(axis=1).values)
        
    else:
        # Find the correlation between the average of x_col(s) and y_col(s)
        m, c, r, p, sterr, perm_p = permutation_correlation(df[x_col].mean(axis=1).values, df[y_col].mean(axis=1).values)
    
    # Adjust very small p values to a readable format
    if perm_p < 0.001:
        perm_p = '<0.001'
    else:
        perm_p = '{:2.3f}'.format(perm_p)
    
    # Multiply all the slope values by 1000 so that they print out sensibly
    m = m * 1000.0
        
    # Change the number of decimal places according to
    # the intercept measure
    if c < 1:
        table_list += [ '{:2.2f}'.format(m), '{}'.format(perm_p) ]
    else:
        table_list += [ '{:2.1f}'.format(m), '{}'.format(perm_p) ]

    return table_list

#==============================================================================
def append_degree(table_var_dict, table_list, sub_roi_list):
    '''
    Write the average degree across each sub region to the table list
    '''
    deg_list = []
    for sub_r in sub_roi_list:
        deg_list += [ table_var_dict['deg'][table_var_dict['aparc_names'].index(sub_r)] ]
    
    mean_deg = np.mean(deg_list)
    
    table_list += [ '{:2.1f}'.format(mean_deg) ]
    
    return table_list
    
#==============================================================================
def write_table_list(table_var_dict, table_list, n=308):

    f_name = table_var_dict['table_file_{}'.format(n)]
    
    with open(f_name, 'a') as f:
            f.write(','.join(table_list))
            f.write('\n')
