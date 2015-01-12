#!/usr/bin/env python

'''
Translating the NSPN_CorticalThickness analyses from an ipython notebook
into "proper" code.

In particular, this is in order to scale up the MT analyses from a segmentation
approach to a parcellation sampling at different depths.
'''

#=============================================================================
# A bunch o imports
#=============================================================================
import numpy as np
import statsmodels as sm
from statsmodels.sandbox.stats.multicomp import multipletests
import os
from glob import glob
import sys
import pandas as pd
import matplotlib.pylab as plt
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, spearmanr, normaltest, wilcoxon, linregress
from scipy.stats.kde import gaussian_kde
from statsmodels.formula.api import ols
import brewer2mpl
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from tabulate import tabulate
import itertools as it
import networkx as nx
import operator
import community as community
import scipy.io as sio
#import seaborn as sns
import matplotlib as mpl



#=============================================================================
# Import some additional scripts
#=============================================================================

study_dir = os.path.join('/home', 'kw401', 'UCHANGE', 'INTERIM_ANALYSES')
scripts_dir = os.path.join(study_dir, 'SCRIPTS')
#describing_data_scripts_dir = os.path.join(scripts_dir, 'DESCRIBING_DATA')

sys.path.append(os.path.join(scripts_dir, 'DESCRIBING_DATA', 'PLOTTING_SCRIPTS'))
from create_violin_plots import *
from create_stats_dict import create_stats_dict
from make_table_group_compare import make_table_group_compare
from create_scatter_plots import plot_scatter_dtimeasures, scatter_interaction
from create_correlation_matrix import *

sys.path.append(os.path.join(scripts_dir, 'NSPN_CODE'))
from networkx_functions import *

#=============================================================================
# Define a few fun functions
#=============================================================================
def average_vs_age(df, y_label, colors, figure_name, title, aparc_names):
    
    '''
    Plot the average of all regions within a data frame across age
    '''
    df['Global'] = df[aparc_names].mean(axis=1)
    
    # Define the measures of interest
    measures = ['age_scan', 'Global']

    # Define the data frame(s)
    df_list = [ df ]

    # Define group names
    group_names = [ 'all' ]

    # X and Y var names
    x = 'age_scan'
    y = [ 'Global' ]

    # X and Y labels
    labels = ['Age (years)', y_label ]


    # Pick your model
    formula= 'y ~ x'

    # Figure height
    height=5

    fig, ax = plot_scatter_dtimeasures(df_list, x, y, group_names, height, labels, title, colors,
                                grid_layout='just_one', legend=False, legend_loc='lower right', formula=formula)

    fig.savefig(figure_name, bbox_inches=0, dpi=100)
    
    plt.close()


def average_vs_ct(df, df_ct, y_label, colors, figure_name, title, aparc_names):
    
    '''
    Plot the average of all regions within a data frame across age
    '''
    df['Global'] = df[aparc_names].mean(axis=1)
    
    df['Global_CT'] = df_ct['Global']
    
    # Define the measures of interest
    measures = ['Global_CT', 'Global']

    # Define the data frame(s)
    df_list = [ df ]

    # Define group names
    group_names = [ 'all' ]

    # X and Y var names
    x = 'Global_CT'
    y = [ 'Global' ]

    # X and Y labels
    labels = ['Cortical Thickness (mm)', y_label ]

    # Pick your model
    formula= 'y ~ x'

    # Figure height
    height=5

    fig, ax = plot_scatter_dtimeasures(df_list, x, y, group_names, height, labels, title, colors,
                                grid_layout='just_one', legend=False, legend_loc='lower right', formula=formula)

    fig.savefig(figure_name, bbox_inches=0, dpi=100)
    
    plt.close()

def get_degrees(G, cost):
    
    '''
    This can be used to plot either a histogram or a KDE function
    by changing the option from either 'hist' or 'kde'
    '''
    # Create the figure
    fig, ax = plt.subplots(figsize=(6,4))
    
    # Degree only has meaning if you don't have a full graph!
    # So while we'll *call* those values "degrees" it actually
    # represents strength...but only for the cost=100 graph
    if cost < 100:
        # Binarize the graph
        for u,v,d in G.edges(data=True):
            d['weight']=1

    # Get the degrees of the graph
    degrees = G.degree(weight='weight').values()
    
    return degrees
        
def degree_distn(G_list, cost, group_list, title, figure_name, measure, option='hist'):
    
    '''
    This can be used to plot either a histogram or a KDE function
    by changing the option from either 'hist' or 'kde'
    '''
    # Create the figure
    fig, ax = plt.subplots(figsize=(6,4))
    
    degrees_list = []
    
    for G in G_list:
        # Degree only has meaning if you don't have a full graph!
        # So while we'll *call* those values "degrees" it actually
        # represents strength...but only for the cost=100 graph
        if cost < 100:
            # Binarize the graph
            for u,v,d in G.edges(data=True):
                d['weight']=1

        # Get the degrees of the graph
        degrees = G.degree(weight='weight').values()
        degrees_list += [degrees]
        
    if option == 'hist':
        # The ranges are different for the different costs
        # They're hardwired here
        if cost > 15:
            x = np.arange(0, 180, 10)
        if cost == 10:
            x = np.arange(0, 100, 10)
        if cost == 02:
            x = np.arange(0, 50, 5)
         
        color_list = [ color_dict[group] for group in group_list ]
        
        # Plot the histogram 
        ax.hist(degrees_list,
                bins=x,
                color=color_list,
                normed=1,
                label=group_list)
        
    elif option == 'kde':
        for degrees, group in zip(degrees_list, group_list):
            # Calculate and plot the kde function
            pdf = gaussian_kde(degrees)
            
            # The ranges are different for the different costs
            # They're hardwired here
            if cost > 15:
                x = np.arange(0, 180, 1)
            if cost == 10:
                x = np.arange(0, 100, 1)
            if cost == 02:
                x = np.arange(0, 50, 1)

            ax.plot(x, pdf(x), color=color_dict[group], label=group)

    # Set the appropriate x and y limits
    if cost == 100:
        ax.set_xlim((0, 180))
        ax.set_ylim((0,0.02))
    if cost == 20:
        ax.set_xlim((0, 180))
        ax.set_ylim((0,0.015))
    if cost == 10:
        ax.set_xlim((0, 100))
        ax.set_ylim((0,0.025))
    if cost == 2:
        ax.set_xlim((0, 50))
        ax.set_ylim((0,0.08))
        
    if len(G_list) > 1:
        ax.legend(loc='upper left', framealpha=0.0, title = measure.upper())

    fig.savefig(figure_name, bbox_inches=0, dpi=300)
    plt.close(fig)
    
    
def read_in_df(data_file):
    df = pd.read_csv(data_file, sep=',')
    data_cols = [ x.replace('_{}'.format('thickness'), '') for x in df.columns ]
    df.columns = data_cols
    df['young'] = 0
    df['young'][df['age_scan']<np.percentile(df.age_scan, 50)] = 1
    
    df['ones'] = df['age_scan'] * 0 + 1
    df['age'] = df['age_scan']
    
    df['Global'] = df[aparc_names].mean(axis=1)
    
    return df


#=============================================================================
# Define some useful locations that you're going to need
#=============================================================================

paper_dir = os.path.join(study_dir, 'CT_MT_ANALYSES')
data_dir = os.path.join(study_dir, 'FS_ROIS')
results_dir = os.path.join(paper_dir, 'RESULTS')
figures_dir = os.path.join(paper_dir, 'FIGS')
if not os.path.isdir(figures_dir):
    os.makedirs(figures_dir)

fsaverage_dir = os.path.join(study_dir, 'SUB_DATA', 'fsaverageSubP')

#=============================================================================
# Define some useful files that you're going to need
#=============================================================================
behav_data_file = os.path.join(study_dir, 'uchange_interim_cohort.csv')
ct_data_file = os.path.join(data_dir, 'PARC_500aparc_thickness_behavmerge.csv')
#wm_mt_data_file = os.path.join(data_dir, 'SEG_MT_500cortExpConsecWMoverlap_mean_behavmerge.csv')
aparc_names_file = os.path.join(fsaverage_dir, 'parcellation', '500.names.txt' )
centroids_file = os.path.join(fsaverage_dir, 'parcellation', '500.centroids.txt' )


#=============================================================================
# Read in some useful data frames
#=============================================================================
# Read in the names from the aparc names file 
# dropping the first 41 ################# BUUUUUG - needs to be fixed
aparc_names = [ line.strip() for line in open(aparc_names_file) ]
aparc_names = aparc_names[41::]

# Read in the locations of each region from the centroids file 
# dropping the first 41 ################# BUUUUUG - needs to be fixed
centroids = np.loadtxt(centroids_file) # [ line.strip() for line in open(centroids_file) ]
centroids = centroids[41:,:]

# Use the centroids to create a dictionary for each of the
# three orthogonal directions
axial_pos = {key: value for (key, value) in zip(range(len(centroids)),centroids[:,:2])}
sagittal_pos = {key: value for (key, value) in zip(range(len(centroids)),centroids[:,1:])}
coronal_pos = {key: value for (key, value) in zip(range(len(centroids)),centroids[:,0::2])}

# Create a useful color list :)
color_list = brewer2mpl.get_map('Set1', 'Qualitative', 8).mpl_colors

'''

#=============================================================================
# Make plots of each Global value against age & for MT against CT
#=============================================================================
print "=================================================="
print "Making plots of global values vs age and each other"

# CT

y_label = 'Cortical Thickness (mm)'

cm = plt.get_cmap('PuOr_r') 
cNorm  = mpl.colors.Normalize(vmin=-1, vmax=1)
scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)
colorVal = scalarMap.to_rgba(-0.75)

colors = [ colorVal ]

figure_name = os.path.join(figures_dir, 'Global_CT_vs_Age.png')

title = 'Global cortical thickness decreases with age'

if not os.path.isfile(figure_name):
    df_ct = read_in_df(ct_data_file)
    average_vs_age(df_ct, y_label, colors, figure_name, title, aparc_names)

# MT

for i in np.arange(0.0,110,10):
    
    cort_mt_data_file = os.path.join(data_dir, 'PARC_500aparc_MT_projfrac{:+04.0f}_behavmerge.csv'.format(i))

    y_label = 'Magnetisation Transfer'
    
    colors = [ color_list[0] ]
    figure_name = os.path.join(figures_dir, 'Global_MT_frac{:+04.0f}_vs_Age.png'.format(i))
    title = 'Global magnetisation transfer increases with age'
    
    if not os.path.isfile(figure_name):
        df_mt_cort = read_in_df(cort_mt_data_file)
        average_vs_age(df_mt_cort, y_label, colors, figure_name, title, aparc_names)
    
    colors = [ color_list[2] ]
    figure_name = os.path.join(figures_dir, 'Global_MT_frac{:+04.0f}_vs_CT.png'.format(i))
    title = 'Global MT decreases with CT'
    
    if not os.path.isfile(figure_name):
        df_mt_cort = read_in_df(cort_mt_data_file)
        average_vs_ct(df_mt_cort, df_ct, y_label, colors, figure_name, title, aparc_names)


#=============================================================================
# Create the correlation matrices
#=============================================================================
print "=================================================="
print "Making or loading correlation matrices"

mat_dict = {}

#for covars in [ ['ones'], ['age'], ['male'], ['age', 'male'] ]:
for covars, demean in it.product([ ['ones'] ], [ True , False ]):
    
    #=======
    # CT 
    #=======

    # ALL
    if demean:
        key = 'CT_demean_covar_{}_all'.format('_'.join(covars))        
        mat_name = os.path.join(data_dir, 'CORR_MATS', 'Mat_CT_demean_Corr_covar_{}_ALL.txt'.format('_'.join(covars)))
    else:
        key = 'CT_covar_{}_all'.format('_'.join(covars))
        mat_name = os.path.join(data_dir, 'CORR_MATS', 'Mat_CT_Corr_covar_{}_ALL.txt'.format('_'.join(covars)))

    print key

    # If it doesn't already exist, then make it
    if not os.path.isfile(mat_name):
        df_ct = read_in_df(ct_data_file)
        mat_dict[key] = create_mat(df_ct, aparc_names, covars, demean=demean)[1]
        save_mat(mat_dict[key], mat_name)
    # Otherwise just load it into the dictionary
    else:
        mat_dict[key] = np.loadtxt(mat_name)

    # YOUNG
    if demean:
        key = 'CT_demean_covar_{}_young'.format('_'.join(covars))        
        mat_name = os.path.join(data_dir, 'CORR_MATS', 'Mat_CT_demean_Corr_covar_{}_YOUNG.txt'.format('_'.join(covars)))
    else:
        key = 'CT_covar_{}_young'.format('_'.join(covars))
        mat_name = os.path.join(data_dir, 'CORR_MATS', 'Mat_CT_Corr_covar_{}_YOUNG.txt'.format('_'.join(covars)))

    print key

    # If it doesn't already exist, then make it
    if not os.path.isfile(mat_name):
        df_ct = read_in_df(ct_data_file)
        mat_dict[key] = create_mat(df_ct[df_ct['young']==1], aparc_names, covars, demean=demean)[1]
        save_mat(mat_dict[key], mat_name)
    # Otherwise just load it into the dictionary
    else:
        mat_dict[key] = np.loadtxt(mat_name)

    # OLD
    if demean:
        key = 'CT_demean_covar_{}_old'.format('_'.join(covars))        
        mat_name = os.path.join(data_dir, 'CORR_MATS', 'Mat_CT_demean_Corr_covar_{}_OLD.txt'.format('_'.join(covars)))
    else:
        key = 'CT_covar_{}_old'.format('_'.join(covars))
        mat_name = os.path.join(data_dir, 'CORR_MATS', 'Mat_CT_Corr_covar_{}_OLD.txt'.format('_'.join(covars)))

    print key

    # If it doesn't already exist, then make it
    if not os.path.isfile(mat_name):
        df_ct = read_in_df(ct_data_file)
        mat_dict[key] = create_mat(df_ct[df_ct['young']==0], aparc_names, covars, demean=demean)[1]
        save_mat(mat_dict[key], mat_name)
    # Otherwise just load it into the dictionary
    else:
        mat_dict[key] = np.loadtxt(mat_name)
    
    # Loop through the MT fractional depths
    for i in np.arange(0.0,110,10):
    
        cort_mt_data_file = os.path.join(data_dir, 'PARC_500aparc_MT_projfrac{:+04.0f}_behavmerge.csv'.format(i))

        # ALL
        if demean:
            key = 'MT_projfrac{:+04.0f}_demean_covar_{}_all'.format(i, '_'.join(covars))        
            mat_name = os.path.join(data_dir, 'CORR_MATS', 'Mat_MT_projfrac{:+04.0f}_demean_Corr_covar_{}_ALL.txt'.format(i, '_'.join(covars)))
        else:
            key = 'MT_projfrac{:+04.0f}_covar_{}_all'.format(i, '_'.join(covars))
            mat_name = os.path.join(data_dir, 'CORR_MATS', 'Mat_MT_projfrac{:+04.0f}_Corr_covar_{}_ALL.txt'.format(i, '_'.join(covars)))
    
        print key

        # If it doesn't already exist, then make it
        if not os.path.isfile(mat_name):
            df_mt_cort = read_in_df(cort_mt_data_file)
            mat_dict[key] = create_mat(df_mt_cort, aparc_names, covars, demean=demean)[1]
            save_mat(mat_dict[key], mat_name)
        # Otherwise just load it into the dictionary
        else:
            mat_dict[key] = np.loadtxt(mat_name)
    
        # YOUNG
        if demean:
            key = 'MT_projfrac{:+04.0f}_demean_covar_{}_young'.format(i, '_'.join(covars))        
            mat_name = os.path.join(data_dir, 'CORR_MATS', 'Mat_MT_projfrac{:+04.0f}_demean_Corr_covar_{}_YOUNG.txt'.format(i, '_'.join(covars)))
        else:
            key = 'MT_projfrac{:+04.0f}_covar_{}_young'.format(i, '_'.join(covars))
            mat_name = os.path.join(data_dir, 'CORR_MATS', 'Mat_MT_projfrac{:+04.0f}_Corr_covar_{}_YOUNG.txt'.format(i, '_'.join(covars)))
        
        print key
        
        # If it doesn't already exist, then make it
        if not os.path.isfile(mat_name):
            df_mt_cort = read_in_df(cort_mt_data_file)
            mat_dict[key] = create_mat(df_mt_cort[df_mt_cort['young']==1], aparc_names, covars, demean=demean)[1]
            save_mat(mat_dict[key], mat_name)
        # Otherwise just load it into the dictionary
        else:
            mat_dict[key] = np.loadtxt(mat_name)
    

        # OLD
        if demean:
            key = 'MT_projfrac{:+04.0f}_demean_covar_{}_old'.format(i, '_'.join(covars))        
            mat_name = os.path.join(data_dir, 'CORR_MATS', 'Mat_MT_projfrac{:+04.0f}_demean_Corr_covar_{}_OLD.txt'.format(i, '_'.join(covars)))
        else:
            key = 'MT_projfrac{:+04.0f}_covar_{}_old'.format(i, '_'.join(covars))
            mat_name = os.path.join(data_dir, 'CORR_MATS', 'Mat_MT_projfrac{:+04.0f}_Corr_covar_{}_OLD.txt'.format(i, '_'.join(covars)))
        
        print key

        # If it doesn't already exist, then make it
        if not os.path.isfile(mat_name):
            df_mt_cort = read_in_df(cort_mt_data_file)
            mat_dict[key] = create_mat(df_mt_cort[df_mt_cort['young']==0], aparc_names, covars, demean=True)[1]
            save_mat(mat_dict[key], mat_name)
        # Otherwise just load it into the dictionary
        else:
            mat_dict[key] = np.loadtxt(mat_name)
    
#=============================================================================
# Lets do some graaaaaphs!
#=============================================================================
print "=================================================="
print "Making or loading graphs"

graph_dict = {}

# Make the graphs data directory if it doesn't already exist
if not os.path.isdir(os.path.join(data_dir, 'GRAPHS')):

    os.mkdir(os.path.join(data_dir, 'GRAPHS'))

# Loop through all the matrices in mat_dict
for k in mat_dict.keys():
    print k
    # Read in the matrix
    M = mat_dict[k]
    
    #=======
    # Read in the full graph first
    #=======
    
    # Define the graph's file name
    g_filename = os.path.join(data_dir, 'GRAPHS', 'Graph_{}_COST_100.gpickle'.format(k))
    
    # If it already exists just read it in from the pickled file
    if os.path.isfile(g_filename):
        graph_dict['{}_COST_100'.format(k)] = nx.read_gpickle(g_filename)
    
    # Otherwise you'll have to create it using the graph_at_cost function above
    else:
        graph_dict['{}_COST_100'.format(k)] = full_graph(M)
        
        # Save it as a gpickle file so you don't have to do this next time!
        nx.write_gpickle(graph_dict['{}_COST_100'.format(k)], g_filename)
    
    # For three different costs...
    for cost in [2, 10, 20]:
        
        # Define the graph's file name
        g_filename = os.path.join(data_dir, 'GRAPHS', 'Graph_{}_COST_{:02.0f}.gpickle'.format(k, cost))
        
        # If it already exists just read it in from the pickled file
        if os.path.isfile(g_filename):
            graph_dict['{}_COST_{:02.0f}'.format(k, cost)] = nx.read_gpickle(g_filename)
        
        # Otherwise you'll have to create it using the graph_at_cost function above
        else:
            graph_dict['{}_COST_{:02.0f}'.format(k, cost)] = graph_at_cost(M, cost)
            
            # Save it as a gpickle file so you don't have to do this next time!
            nx.write_gpickle(graph_dict['{}_COST_{:02.0f}'.format(k, cost)], g_filename)
    
#=============================================================================
# Make some pictures
#=============================================================================
print "=================================================="
print "Making some figures"

for key in graph_dict.keys():

    measure = key.split('_covar')[0]
    cost =  key.split('_covar_')[1].split('_COST_')[1]
    cost = np.float(cost)
    covars =  key.split('_covar_')[1].split('_COST_')[0].rsplit('_', 1)[0]
    group = key.split('_covar_')[1].split('_COST_')[0].rsplit('_', 1)[1]
    
    M = mat_dict['{}_covar_{}_{}'.format(measure, covars, group)]
    
    G = graph_dict['{}_covar_{}_{}_COST_{:02.0f}'.format(measure, covars, group, cost)]
    
    G_edge = graph_dict['{}_covar_{}_{}_COST_{:02.0f}'.format(measure, covars, group, 2)]
    
    title = '{} SUBS - {} - covar {}'.format(group.upper(), measure, covars)

    print title, cost
    filename = os.path.join(figures_dir, '{}_covar_{}_{}_network_COST_{:02.0f}.png'.format(measure, covars, group.upper(), cost))

    if not os.path.isfile(filename):
        fig = plot_modules(G, G_edge, cost_mod=cost, title=title, axial_pos=axial_pos, coronal_pos=coronal_pos, sagittal_pos=sagittal_pos,)
        fig.savefig(filename, bbox_inches=0, dpi=300)
        plt.close(fig)

#=============================================================================
# Save a picture of each matrix
#=============================================================================
print "=================================================="
print "Making some pictures of matrices"

for key in mat_dict.keys():

    M = mat_dict[key]
    
    measure = key.split('_covar')[0]
    covars =  key.split('_covar_')[1].rsplit('_', 1)[0]
    group = key.split('_covar_')[1].rsplit('_', 1)[1]
    
    title = '{} SUBS - {} - covar {}'.format(group.upper(), measure, covars)
    
    print title
    filename = os.path.join(figures_dir, '{}_covar_{}_{}_matrix.png'.format(measure, covars, group.upper()))

    if not os.path.isfile(filename):
        
        fig, ax = plt.subplots(figsize=(6,5))
        M_ax = ax.imshow(M, interpolation='none', vmin=-0.5, vmax=0.5)
        ax.set_axis_off()

        cbar = fig.colorbar(M_ax, ticks=[-0.5, 0, 0.5])
        plt.tight_layout()
        fig.savefig(filename, bbox_inches=0, dpi=300)
        plt.close(fig)
        
        
#=============================================================================
# Make some histogrms of the distributions
#=============================================================================
print "=================================================="
print "Making some histograms of the correlation distributions"

color_dict = { 'all': 'green', 'young' : 'red', 'old' : 'blue' }

for key in mat_dict.keys():

    M = mat_dict[key]
    
    measure = key.split('_covar')[0]
    covars =  key.split('_covar_')[1].rsplit('_', 1)[0]
    group = key.split('_covar_')[1].rsplit('_', 1)[1]

    title = '{} SUBS - {} - covar {}'.format(group.upper(), measure, covars)

    print title
    # Make the kde smoothed image
    filename = os.path.join(figures_dir, '{}_covar_{}_{}_corrdistn_KDE.png'.format(measure, covars, group.upper()))
    
    if not os.path.isfile(filename):

        fig, ax = plt.subplots(figsize=(6,4))
        
        x = np.arange(-0.4, 1, 0.01)
    
        data = M[np.triu_indices_from(M, k=1)]
    
        pdf = gaussian_kde(data)
    
        ax.plot(x, pdf(x), color=color_dict[group], label=group)
    
        ax.set_ylim((0, 4.0))
    
        fig.savefig(filename, bbox_inches=0, dpi=300)
        plt.close(fig)
    
    # Make the more standard histogram
    filename = os.path.join(figures_dir, '{}_covar_{}_{}_corrdistn_HIST.png'.format(measure, covars, group.upper()))

    if not os.path.isfile(filename):
        fig, ax = plt.subplots(figsize=(6,4))
    
        ax.hist(M[np.triu_indices_from(M, k=1)],
                bins=np.arange(-0.4, 1.05, 0.05),
                color=color_dict[group],
                normed=1)
        
        ax.set_xlim((-0.4, 1.0))
        ax.set_ylim((0, 4.0))
        
        fig.savefig(filename, bbox_inches=0, dpi=300)
        plt.close(fig)
    
# Make the comparison between young and old histograms

old_keys = [ x for x in mat_dict.keys() if 'old' in x ]

for key in old_keys:
    M = mat_dict[key]
    
    measure = key.split('_covar')[0]
    covars =  key.split('_covar_')[1].rsplit('_', 1)[0]

    M_young = mat_dict['{}_covar_{}_{}'.format(measure, covars, 'young')]
    M_old = mat_dict['{}_covar_{}_{}'.format(measure, covars, 'old')]
    title = 'Young vs Old - {} - covar {}'.format(measure, covars)

    print title
    
    # Make the KDE image
    filename = os.path.join(figures_dir, '{}_covar_{}_{}_corrdistn_KDE.png'.format(measure, covars, 'OLDvsYOUNG'))

    if not os.path.isfile(filename):

        fig, ax = plt.subplots(figsize=(6,4))
    
        x = np.arange(-0.4, 1, 0.01)
    
        data_young = M_young[np.triu_indices_from(M_young, k=1)]
        data_old = M_old[np.triu_indices_from(M_old, k=1)]
    
        pdf_young = gaussian_kde(data_young)
        pdf_old = gaussian_kde(data_old)
    
        ax.plot(x, pdf_young(x), color=color_dict['young'], label='young')
        ax.plot(x, pdf_old(x), color=color_dict['old'], label='old')
    
        ax.set_ylim((0,4.0))
    
        ax.legend(loc='upper left', title = measure.upper())
        fig.savefig(filename, bbox_inches=0, dpi=300)
        plt.close(fig)
    
    
    # Make the more standard histogram
    filename = os.path.join(figures_dir, '{}_covar_{}_{}_corrdistn_HIST.png'.format(measure, covars, 'OLDvsYOUNG'))

    if not os.path.isfile(filename):
        fig, ax = plt.subplots(figsize=(6,4))
    
        ax.hist([data_young, data_old],
                bins=np.arange(-0.4, 1.05, 0.05),
                color=[color_dict['young'], color_dict['old']],
                label=['young', 'old'],
                normed=1)
        
        ax.set_xlim((-0.4, 1.0))
        ax.set_ylim((0, 4.0))
        
        ax.legend(loc='upper left', title = measure.upper())
        fig.savefig(filename, bbox_inches=0, dpi=300)
        plt.close(fig)
        
#====
# DEGREE DISTRIBUTION
#====
for key in graph_dict.keys():
        
    measure = key.split('_covar')[0]
    cost =  key.split('_covar_')[1].split('_COST_')[1]
    cost = np.float(cost)
    covars =  key.split('_covar_')[1].split('_COST_')[0].rsplit('_', 1)[0]
    group = key.split('_covar_')[1].split('_COST_')[0].rsplit('_', 1)[1]
    
    G = graph_dict['{}_covar_{}_{}_COST_{:02.0f}'.format(measure, covars, group, cost)]
    
    color_dict = { 'all': 'green', 'young' : 'red', 'old' : 'blue' }

    title = '{} SUBS - {} - covar {} - cost {:02.0f}%'.format(group.upper(), measure, covars, cost)
    
    print title
    
    # Make the KDE figure if it doesn't yet exist
    filename = os.path.join(figures_dir, 
                                '{}_covar_{}_{}_degreedistn_COST_{:02.0f}_KDE.png'.format(measure,
                                                                                            covars,
                                                                                            group.upper(),
                                                                                            cost))
    if not os.path.isfile(filename):
        degrees = degree_distn([G], cost, [group], title, filename, measure, option='kde')
                
    # Make the regular histogram figure if it doesn't yet exist
    filename = os.path.join(figures_dir, 
                                '{}_covar_{}_{}_degreedistn_COST_{:02.0f}_HIST.png'.format(measure,
                                                                                            covars,
                                                                                            group.upper(),
                                                                                            cost))
    if not os.path.isfile(filename):
        degree_distn([G], cost, [group], title, filename, measure, option='hist') 

    # Write out the degrees to a text file for figure making later
    filename = os.path.join(data_dir, '{}_covar_{}_{}_degrees_COST_{:02.0f}.txt'.format(measure,
                                                                                        covars,
                                                                                        group.upper(),
                                                                                        cost))
    if not os.path.isfile(filename):
        degrees = get_degrees(G, cost)
        np.savetxt(filename, degrees)

    # Now do the same, but for the comparison between the old and young
    if 'old' in key:

        # Create a module appropriate graph (at cost: cost_mod)
        G_young = graph_dict[key.replace('old', 'young')]
        G_old = graph_dict[key]
        
        title = 'Young vs Old - {} - covar {} - cost {:02.0f}%'.format(measure, covars, cost)

        print title
        
        G_list = [ G_young, G_old ]
        group_list = [ 'young', 'old' ]
        
        # Make the KDE figure if it doesn't yet exist
        filename = os.path.join(figures_dir, 
                                    '{}_covar_{}_{}_degreedistn_COST_{:02.0f}_KDE.png'.format(measure,
                                                                                                covars,
                                                                                                'OLDvsYOUNG',
                                                                                                cost))
        if not os.path.isfile(filename):
            degree_distn(G_list, cost, group_list, title, filename, measure, option='kde')
            
        # Make the regular histogram figure if it doesn't yet exist
        filename = os.path.join(figures_dir, 
                                    '{}_covar_{}_{}_degreedistn_COST_{:02.0f}_HIST.png'.format(measure,
                                                                                                covars,
                                                                                                'OLDvsYOUNG',
                                                                                                cost))
        if not os.path.isfile(filename):
            degree_distn(G_list, cost, group_list, title, filename, measure, option='hist')
                
        
#=============================================================================
# Corr between degree and mean & slope of a variable
#=============================================================================
color_dict = { 'all': 'green', 'young' : 'red', 'old' : 'blue' }

# Define some measures that are going to go on the y axes...
measure_dict = {}

# Calculate the mean value for each group for CT
measure_dict['CT_all_mean'] = df_ct[aparc_names].mean(axis=0).values
measure_dict['CT_young_mean'] = df_ct[aparc_names][df_ct.young==1].mean(axis=0).values
measure_dict['CT_old_mean'] = df_ct[aparc_names][df_ct.young==0].mean(axis=0).values

# Calculate the difference between the old and young groups for CT
measure_dict['CT_diff'] = ( df_ct[aparc_names][df_ct.young==0].mean(axis=0).values 
                                - df_ct[aparc_names][df_ct.young==1].mean(axis=0).values )

m_array = np.ones(len(aparc_names))
for i, roi in enumerate(aparc_names):
    m, c, r, p, std_err = linregress(df_ct['age_scan'].values,
                                        df_ct[roi].values)
    m_array[i] = m
    
measure_dict['CT_all_slope'.format(i)] = m_array

m_array = np.ones(len(aparc_names))
for i, roi in enumerate(aparc_names):
    m, c, r, p, std_err = linregress(df_ct['age_scan'][df_ct.young==1].values,
                                        df_ct[roi][df_ct.young==1].values)
    m_array[i] = m
    
measure_dict['CT_young_slope'.format(i)] = m_array

m_array = np.ones(len(aparc_names))
for i, roi in enumerate(aparc_names):
    m, c, r, p, std_err = linregress(df_ct['age_scan'][df_ct.young==0].values,
                                        df_ct[roi][df_ct.young==0].values)
    m_array[i] = m
    
measure_dict['CT_old_slope'.format(i)] = m_array


for i in np.arange(0.0,110,10):
    
    cort_mt_data_file = os.path.join(data_dir, 'PARC_500aparc_MT_projfrac{:+04.0f}_behavmerge.csv'.format(i))

    df_mt_cort = pd.read_csv(cort_mt_data_file)
    data_cols = [ x.replace('_{}'.format('thickness'), '') for x in df_mt_cort.columns ]
    df_mt_cort.columns = data_cols
    df_mt_cort['young'] = 0
    df_mt_cort['young'][df_mt_cort['age_scan']<np.percentile(df_mt_cort['age_scan'], 50)] = 1

    measure_dict['MT_projfrac{:+04.0f}_all_mean'.format(i)] = df_mt_cort[aparc_names].mean(axis=0).values
    measure_dict['MT_projfrac{:+04.0f}_young_mean'.format(i)] = df_mt_cort[aparc_names][df_mt_cort.young==1].mean(axis=0).values
    measure_dict['MT_projfrac{:+04.0f}_old_mean'.format(i)] = df_mt_cort[aparc_names][df_mt_cort.young==0].mean(axis=0).values
    
    m_array = np.ones(len(aparc_names))
    for j, roi in enumerate(aparc_names):
        m, c, r, p, std_err = linregress(df_mt_cort['age_scan'].values,
                                            df_mt_cort[roi].values)
        m_array[j] = m
    
    measure_dict['MT_projfrac{:+04.0f}_all_slope'.format(i)] = m_array
    
    m_array = np.ones(len(aparc_names))
    for j, roi in enumerate(aparc_names):
        m, c, r, p, std_err = linregress(df_mt_cort['age_scan'][df_mt_cort.young==1].values,
                                            df_mt_cort[roi][df_mt_cort.young==1].values)
        m_array[j] = m
    
    measure_dict['MT_projfrac{:+04.0f}_young_slope'.format(i)] = m_array
    
    m_array = np.ones(len(aparc_names))
    for j, roi in enumerate(aparc_names):
        m, c, r, p, std_err = linregress(df_mt_cort['age_scan'][df_mt_cort.young==0].values,
                                            df_mt_cort[roi][df_mt_cort.young==0].values)
        m_array[j] = m
        
    measure_dict['MT_projfrac{:+04.0f}_old_slope'.format(i)] = m_array

print measure_dict.keys()


# MAKE A LIST OF MEASURES THAT YOU'LL LOOP THROUGH
measures_list = [ 'CT' ]
for i in np.arange(0.0,110,10):
    measures_list += [ 'MT_projfrac{:+04.0f}'.format(i) ]

measures_list = [ 'CT', 'MT_projfrac+030' ]
print measures_list

# NOW CORRELATE ALL OF THESE WITH THE DIFFERENT GRAPHS

for key in graph_dict.keys():
    
#for key in [ x for x in graph_dict.keys() if 'MT_projfrac+030_covar_ones' in x and 'COST_20' in x]:
    measure_graph = key.split('_covar')[0]
    cost =  key.split('_covar_')[1].split('_COST_')[1]
    cost = np.float(cost)
    covars =  key.split('_covar_')[1].split('_COST_')[0].rsplit('_', 1)[0]
    group = key.split('_covar_')[1].split('_COST_')[0].rsplit('_', 1)[1]
    
    M = mat_dict['{}_covar_{}_{}'.format(measure_graph, covars, group)]
    
    G = graph_dict['{}_covar_{}_{}_COST_{:02.0f}'.format(measure_graph, covars, group, cost)]
    
    color_dict = { 'all': 'green', 'young' : 'red', 'old' : 'blue' }
    
    title = '{} SUBS - {} - covar {} - cost {:02.0f}%'.format(group.upper(), measure_graph, covars, cost)
    
    print title

    for measure_value in measures_list:
    
        # Binarize the graph
        for u,v,d in G.edges(data=True):
            d['weight']=1
    
        # Get the degrees of the graph
        degrees = G.degree().values()
    
        # MEANS
        means = measure_dict['{}_all_mean'.format(measure_value)]
        
        filename = os.path.join(figures_dir, 
                                '{}_covar_{}_{}_corr_Mean{}_COST_{:02.0f}.png'.format(measure_graph, covars, group.upper(), measure_value, cost))
        #if os.path.isfile(filename):
        # Define the data frame(s)
        df =  pd.DataFrame({'x' : degrees,
                            'y' : means})
                            
        df_list = [ df ]
    
        # Define group names
        group_names = [ measure_value.upper() ]
    
        # X and Y var names
        x = 'x'
        y = [ 'y' ]
    
        # X and Y labels
        labels = ['Degree', 'Mean {}'.format(measure_value)]
    
        # Group colors
        color_list = [ color_dict[group] ]
    
        # Pick your model
        formula= 'y ~ x'
        #formula= 'y ~ I(x**2) + x'
    
        # Figure height
        height=5
    
        fig, ax = plot_scatter_dtimeasures(df_list, x, y, group_names, height, labels, title, color_list,
                                    grid_layout='just_one', legend=False, legend_loc='center right', formula=formula)
    
    
        fig.savefig(filename, bbox_inches=0, dpi=300)
        plt.close(fig)
    
    
        # SLOPES
        slopes = measure_dict['{}_all_slope'.format(measure_value)]
        
        filename = os.path.join(figures_dir, 
                                '{}_covar_{}_{}_corr_Slope{}_COST_{:02.0f}.png'.format(measure_graph, covars, group.upper(), measure_value, cost))
        #if os.path.isfile(filename):
        # Define the data frame(s)
        df =  pd.DataFrame({'x' : degrees,
                            'y' : slopes})
                
        df_list = [ df ]
    
        # Define group names
        group_names = [ measure_value.upper() ]
    
        # X and Y var names
        x = 'x'
        y = [ 'y' ]
    
        # X and Y labels
        labels = ['Degree', 'Slope {} with Age'.format(measure_value)]
    
        # Group colors
        colors = [ color_dict[group] ]
    
        # Pick your model
        formula= 'y ~ x'
        #formula= 'y ~ I(x**2) + x'
    
        # Figure height
        height=5
    
        fig, ax = plot_scatter_dtimeasures(df_list, x, y, group_names, height, labels, title, colors,
                                    grid_layout='just_one', legend=False, legend_loc='center right', formula=formula)
    
    
        fig.savefig(filename, bbox_inches=0, dpi=300)
        plt.close(fig)
    
'''
    
#=============================================================================
# Create some lovely violin plots
#=============================================================================

#=============================================================================
###### MEAN #####
# Start with the mean.
# You're going to have 10 fractional values from 
#=============================================================================
cm = plt.get_cmap('jet') 
cNorm  = mpl.colors.Normalize(vmin=400, vmax=1450)
scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)

df_list = []
labels_list = []
color_list = []

# Start by reading in the 10 fractional values from the cortex
for i in np.arange(100,-1,-10):
    cort_mt_data_file = os.path.join(data_dir, 'PARC_500aparc_MT_projfrac{:+04.0f}_behavmerge.csv'.format(i))
    df = read_in_df(cort_mt_data_file)
    df_list += [df]
    if i == 100:
        labels_list += ["Pial"]
    elif i == 0:
        labels_list += ["Grey/white"]
    else:
        labels_list += ['{:2.0f}%'.format(i)]

    color_list += [scalarMap.to_rgba(df['Global'].mean())]

# Then append the 9 steps into white matter (each a distance of 0.2mm)
for i in np.arange(-20,-201,-20):
    wm_mt_data_file = os.path.join(data_dir, 'PARC_500aparc_MT_projdist{:+04.0f}_fromBoundary_behavmerge.csv'.format(i))
    df = read_in_df(wm_mt_data_file)
    df_list += [df]
    labels_list += ['{:2.2f}mm'.format(i/100*-1)]

    color_list += [scalarMap.to_rgba(df['Global'].mean())]
    
#    
for n, i in enumerate(np.hstack([np.arange(100,-1,-10), np.arange(-20,-201,-20)])):
    if i > -1:
        figure_name = os.path.join(figures_dir, 'Global_MT_DifferentDepths_Pialto{:+04.0f}percent.png'.format(i))
    else:
        figure_name = os.path.join(figures_dir, 'Global_MT_DifferentDepths_Pialto{:+04.0f}mm.png'.format(i*-1))
        
    # Start by plotting all of the violins
    # even though you're going to delete some of them!
    violin_mt_mean = create_violin_plots(df_list, 
                                            ['Global'], 
                                            ['Magnetisation Transfer'], 
                                            7, 
                                            labels_list)

    # We're going to mess around with the axes a bit here
    # this could potentially go into the create_violin_plots
    # code at some point....
    ax = violin_mt_mean.get_axes()[0]
    
    # Find all the parts of the figure that have a fill color
    # - you're going to change those!
    artist_list = ax.get_default_bbox_extra_artists()
    violin_fill_list = violin_mt_mean.findobj(match=mpl.collections.PolyCollection)

    # Specifically change them to the appropriate mean value
    for a, color in zip(violin_fill_list, color_list):
        a.set_facecolor(color)
    
    # However, now that you've done that, you're actually
    # going to get rid of some of these, so that you 
    # can animate your plots
    print len(violin_fill_list[-1:n:-1])
    for a in violin_fill_list[-1:n:-1]:
        a.remove()
        
    box_plot_list = violin_mt_mean.findobj(match=mpl.lines.Line2D)

    end = 8*(len(labels_list)-n-1)
    if end > 0:
        for b in box_plot_list[-(8*(len(labels_list)-n-1))::]:
            #b.set_linewidth(0)
            b.set_color('w')
        
    # Re-do the tick labels so that they're *all* there
    # and rotate them
    ax.set_xticklabels(labels_list, rotation=90)

    # Set the y-limits so they're fixed for all plots
    ax.set_ylim(450,1450)

    # Tighten up the plot layout because the labels will
    # have messed it all up!
    plt.tight_layout()

    violin_mt_mean.savefig(figure_name, bbox_inches=0,dpi=100)

    plt.close()
    
###### SLOPE #####
cm = plt.get_cmap('PRGn')
cNorm  = mpl.colors.Normalize(vmin=-7, vmax=7)
scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)

df_list = []
labels_list = []
color_list = []

for i in np.arange(100,-1,-10):
    cort_mt_data_file = os.path.join(data_dir, 'PARC_500aparc_MT_projfrac{:+04.0f}_behavmerge.csv'.format(i))
    df = read_in_df(cort_mt_data_file)
    if i == 100:
        labels_list += ["Pial"]
    elif i == 0:
        labels_list += ["Grey/white"]
    else:
        labels_list += ['{:2.0f}%'.format(i)]

    m_array = np.ones(len(aparc_names))
    for i, roi in enumerate(aparc_names):
        m, c, r, p, std_err = linregress(df['age_scan'].values,
                                            df[roi].values)
        m_array[i] = m
    
    df_list += [pd.DataFrame(m_array, columns=['Slope'])]

    color_list += [scalarMap.to_rgba(m_array.mean())]

for n, i in enumerate(np.arange(100,-1,-10)):
    figure_name = os.path.join(figures_dir, 'Slope_w_Age_MT_DifferentDepths_100to{:+04.0f}.png'.format(i))

    # Start by plotting all of the violins
    # even though you're going to delete some of them!
    violin_mt_slope = create_violin_plots(df_list, 
                                            ['Slope'], 
                                            ['Change in MT per year'], 
                                            7, 
                                            labels_list)

    # We're going to mess around with the axes a bit here
    # this could potentially go into the create_violin_plots
    # code at some point....
    ax = violin_mt_slope.get_axes()[0]
    
    # Find all the parts of the figure that have a fill color
    # - you're going to change those!
    artist_list = ax.get_default_bbox_extra_artists()
    violin_fill_list = violin_mt_slope.findobj(match=mpl.collections.PolyCollection)

    # Specifically change them to the appropriate mean value
    for a, color in zip(violin_fill_list, color_list):
        a.set_facecolor(color)
    
    # However, now that you've done that, you're actually
    # going to get rid of some of these, so that you 
    # can animate your plots
    for a in violin_fill_list[-1:n:-1]:
        a.remove()
        
    box_plot_list = violin_mt_slope.findobj(match=mpl.lines.Line2D)

    end = 8*(len(labels_list)-n-1)
    if end > 0:
        for b in box_plot_list[-(8*(len(labels_list)-n-1))::]:
            #b.set_linewidth(0)
            b.set_color('w')
        
    # Re-do the tick labels so that they're *all* there
    # and rotate them
    ax.set_xticklabels(labels_list, rotation=90)

    # Set the y-limits so they're fixed for all plots
    ax.set_ylim(-10,15)

    # Tighten up the plot layout because the labels will
    # have messed it all up!
    plt.tight_layout()

    violin_mt_slope.savefig(figure_name, bbox_inches=0,dpi=100)

    plt.close()
    
    
###### SIGNIFICANCE #####
cm = plt.get_cmap('Reds_r')
cNorm  = mpl.colors.Normalize(vmin=0, vmax=0.5)
scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)

df_list = []
labels_list = []
color_list = []

for i in np.arange(100,-1,-10):
    cort_mt_data_file = os.path.join(data_dir, 'PARC_500aparc_MT_projfrac{:+04.0f}_behavmerge.csv'.format(i))
    df = read_in_df(cort_mt_data_file)
    if i == 100:
        labels_list += ["Pial"]
    elif i == 0:
        labels_list += ["Grey/white"]
    else:
        labels_list += ['{:2.0f}%'.format(i)]

    p_array = np.ones(len(aparc_names))
    for i, roi in enumerate(aparc_names):
        m, c, r, p, std_err = linregress(df['age_scan'].values,
                                            df[roi].values)
        p_array[i] = p
    
    df_list += [pd.DataFrame(p_array, columns=['sig'])]

    color_list += [scalarMap.to_rgba(p_array.mean())]

for n, i in enumerate(np.arange(100,-1,-10)):
    figure_name = os.path.join(figures_dir, 'Sig_w_Age_MT_DifferentDepths_100to{:+04.0f}.png'.format(i))

    # Start by plotting all of the violins
    # even though you're going to delete some of them!
    violin_mt_slope = create_violin_plots(df_list, 
                                            ['sig'], 
                                            ['Significance of change in MT per year'], 
                                            7, 
                                            labels_list)

    # We're going to mess around with the axes a bit here
    # this could potentially go into the create_violin_plots
    # code at some point....
    ax = violin_mt_slope.get_axes()[0]
    
    # Find all the parts of the figure that have a fill color
    # - you're going to change those!
    artist_list = ax.get_default_bbox_extra_artists()
    violin_fill_list = violin_mt_slope.findobj(match=mpl.collections.PolyCollection)

    # Specifically change them to the appropriate mean value
    for a, color in zip(violin_fill_list, color_list):
        a.set_facecolor(color)
    
    # However, now that you've done that, you're actually
    # going to get rid of some of these, so that you 
    # can animate your plots
    for a in violin_fill_list[-1:n:-1]:
        a.remove()
        
    box_plot_list = violin_mt_slope.findobj(match=mpl.lines.Line2D)

    end = 8*(len(labels_list)-n-1)
    if end > 0:
        for b in box_plot_list[-(8*(len(labels_list)-n-1))::]:
            #b.set_linewidth(0)
            b.set_color('w')
        
    # Re-do the tick labels so that they're *all* there
    # and rotate them
    ax.set_xticklabels(labels_list, rotation=90)

    # Set the y-limits so they're fixed for all plots
    ax.set_ylim(0,1)

    # Tighten up the plot layout because the labels will
    # have messed it all up!
    plt.tight_layout()

    violin_mt_slope.savefig(figure_name, bbox_inches=0,dpi=100)

    plt.close()