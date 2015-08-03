#!/usr/bin/env python

'''
This code makes the figures for the manuscript "
'''
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib as mpl
import os
import sys
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from glob import glob
import itertools as it
import matplotlib.patches as mpatches

# Read in some of the other NSPN_CODE functions too
#this_scripts_dir=os.path.dirname(os.path.abspath(__file__))
#sys.path.append(this_scripts_dir)

from networkx_functions import *
from regional_correlation_functions import *

def plot_rich_club(rc, rc_rand, ax=None, figure_name=None, x_max=200, y_max=1.2, color=sns.color_palette()[0], norm=False):
    '''
    Make a pretty plot of the rich club values per degree
    along with the rich club values you'd expect by chance
    from a random network with preserved degree distribution
    
    rc and rc_rand are calculated by the rich_club function
    that is saved within the networkx_functions.py file
    ''' 
    # Import what you need
    import matplotlib.pylab as plt
    import seaborn as sns
    
    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)
    
    if not ax:
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig=None
    
    if not norm:
        # Plot the real rich club data
        sns.tsplot(rc, color=color, ax=ax)
        
        # Plot the random rich club data with confidence intervals error bars
        sns.tsplot(rc_rand.T, err_style='ci_bars', color='grey', ci=95, ax=ax)
    
    else:
        # Divide the real rich club by the averge of the
        # randomised rich club to get a normalised curve
        rc_norm = rc / rc_rand.T
        sns.tsplot(rc_norm, err_style='ci_bars', color=color, ax=ax, ci=95)
        
    # Fix the x and y axis limits
    ax.set_xlim((0, x_max))
    ax.set_ylim((0, y_max))
    
    # Make sure there aren't too many bins!
    plt.locator_params(nbins=5)
    
    # Set the x and y axis labels
    ax.set_xlabel("Degree")
    if not norm:
        ax.set_ylabel("Rich Club")
    else:
        ax.set_ylabel("Normalised Rich Club")
    
    # Despine because we all agree it looks better that way
    sns.despine()
    
    if figure_name:
        # Do the tight layout because, again, it looks better!
        fig.tight_layout()
    
        # And save the figure
        fig.savefig(figure_name, bbox_inches=0, dpi=100)
        plt.close(fig)
    
    else:
        return ax

def plot_degree_dist(G, ax=None, figure_name=None, x_max=200, y_max=0.1, color=sns.color_palette()[0]):
    '''
    Make a pretty plot of the degree distribution
    along with the degree distibution of an Erdos Renyi random
    graph that has the same number of nodes and edges
    '''
    # Import what you need
    import matplotlib.pylab as plt
    import seaborn as sns
    
    # Calculate the degrees from the graph
    degrees = np.array(G.degree().values())
    degrees = degrees.astype('float')
    
    # Calculate the Erdos Renyi graph from the main graph
    # it has to match the number of nodes and edges
    nodes = len(G.nodes())
    cost =  G.number_of_edges() * 2.0 / (nodes*(nodes-1))
    G_ER = nx.erdos_renyi_graph(nodes, cost)
    
    # Now calculate the degrees for the ER graph
    degrees_ER = np.array(G_ER.degree().values())
    degrees_ER = degrees_ER.astype('float')
        
    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)
    
    if not ax:
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig=None
    
    # Plot the read degrees and the ER degrees
    sns.kdeplot(degrees, ax=ax)
    sns.kdeplot(degrees_ER, ax=ax, color='grey')
    
    # Fix the x and y axis limits
    ax.set_xlim((0, x_max))
    ax.set_ylim((0, y_max))
    # Make sure there aren't too many bins!
    plt.locator_params(nbins=4)
    
    # Set the x and y axis labels
    ax.set_xlabel("Degree")
    ax.set_ylabel("Probability")
    
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    
    # Despine because we all agree it looks better that way
    sns.despine()
    
    if figure_name:
        # Do the tight layout because, again, it looks better!
        fig.tight_layout()
    
        # And save the figure
        fig.savefig(figure_name, bbox_inches=0, dpi=100)
        plt.close(fig)
    
    else:
        return ax
    
def plot_network_measures(measure_dict, ax=None, figure_name=None, y_max=2.5, y_min=-0.5, color=sns.color_palette()[0]):
    '''
    Create a plot of the network measures
    along with their random counterparts
    '''
    import seaborn as sns
    import matplotlib.pylab as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    # Set the seaborn context and whotnot
    sns.set_style('white')
    sns.set_context("poster", font_scale=2)
    
    # Read the measures dictionary into an array
    df = pd.DataFrame(measure_dict)
    
    # And re-order the columns in the data frame so that
    # the graph will look nice
    df = df[['a', 'a_rand', 
                'M', 'M_rand', 
                'E', 'E_rand', 
                'C', 'C_rand', 
                'L', 'L_rand',
                'sigma', 'sigma_rand']] 

    if not ax:
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig=None
    
    # Add a bar plot for each measure
    for i in range(len(df.columns)/2):
        # Show the actual measure with error bars
        # (Note that the error will be 0 for all measures
        # except the small world coefficient)
        if df[df.columns[i*2]].std() > 0.0000001:
            ci = stats.norm.ppf((1+0.95)/2, scale=np.std(df[df.columns[i*2]]))
        else:
            ci = 0
      
        ax.bar(i-0.12, 
                    df[df.columns[i*2]].mean(),
                    yerr=ci,
                    width=0.2, 
                    align='center', 
                    color=color,
                    ecolor=color,
                    edgecolor='black')
                    
        # Show the random networks with error bars
        if df[df.columns[i*2+1]].std() > 0.0000001:
            ci = stats.norm.ppf((1+0.95)/2, scale=np.std(df[df.columns[i*2+1]]))
        else:
            ci = 0
                
        ax.bar(i+0.12, 
                    df[df.columns[i*2+1]].mean(),
                    yerr=ci,
                    width=0.2,
                    align='center',
                    color='grey',
                    ecolor='grey',
                    edgecolor='black')
    
    # Sort out the xtick labels
    ax.set_xticks(range(len(df.columns)/2))
    ax.set_xticklabels(df.columns[::2])

    # Put in a bar at y=0
    ax.axhline(0, linewidth=0.5, color='black')
    
    # Fix the y axis limits
    ax.set_ylim((y_min, y_max))
    
    # Make sure there aren't too many bins!
    plt.locator_params(axis='y', nbins=5)
    
    # Set the y axis label
    ax.set_ylabel("Network measures")
    
    # Despine because we all agree it looks better that way
    sns.despine()
    
    if figure_name:
        # Do the tight layout because, again, it looks better!
        fig.tight_layout()
    
        # And save the figure
        fig.savefig(figure_name, bbox_inches=0, dpi=100)
        plt.close(fig)
    
    else:
        return ax

def plot_sagittal_network(G, 
                         G_edge,
                         sagittal_pos,
                         axial_pos,
                         integer_adjust=3,
                         fractional_adjust=2.5,
                         cmap_name='jet',
                         ax=None, 
                         figure_name=None):
    
    import matplotlib.pylab as plt
    import numpy as np
    import networkx as nx
    import community
    import seaborn as sns
    
    # Save the colormap
    cmap = plt.get_cmap(cmap_name)
    
    # Binarize both of these graphs
    for u,v,d in G.edges(data=True):
        d['weight']=1
    
    for u,v,d in G_edge.edges(data=True):
        d['weight']=1
        
    # Compute the best partition based on the threshold you've specified in cost
    partition = community.best_partition(G)

    # Create a sorted list of communitites (modules) according to their average
    # Y coordinate (front to back)
    module_list = sort_partition(partition, axial_pos)
    
    # Display the number of modules
    size = np.float(len(module_list))

    if not ax:
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig=None
    
    # Loop through all the nodes, sorted acording to their x value
    # meaning that we're going to plot nodes on the LEFT side of the
    # brain first so they appear behind the nodes on the RIGHT side of
    # the brain
    x_values = []
    for node in G.nodes():
        x_values.append(axial_pos[node][0])
        
    node_list = [ node for (x_coord, node) in sorted(zip(x_values, G.nodes())) ]
    
    # Start the node loop
    for node in node_list:
    
        # Look up which module the node is in
        mod = partition[node]
        
        # Get the correct color acording to the sorted partition list
        color = cmap( module_list.index(mod) / np.float(size) )
        
        # Now draw on the node
        nx.draw_networkx_nodes(G, sagittal_pos,
                            [node], 
                            node_size = integer_adjust + fractional_adjust * np.array(G.degree(node)),
                            node_color = color,
                            ax = ax)
                                    
    # Add in all the edges
    nx.draw_networkx_edges(G_edge, sagittal_pos, alpha=0.2, ax = ax)

    # Change the x and y limits to make the images look a bit better
    ax.set_xlim(-120, 80)
    ax.set_ylim(-45, 75)

    # Turn the axis labels off
    ax.set_axis_off()
    
    if figure_name:
        # Do the tight layout because, again, it looks better!
        fig.tight_layout()
    
        # And save the figure
        fig.savefig(figure_name, bbox_inches=0, dpi=100)
        plt.close(fig)
    
    else:
        return ax

def pretty_scatter(x, y, x_label='x', y_label='y', x_max=None, x_min=None, y_max=None, y_min=None, figure_name=None, ax=None, figure=None, color='k', marker_colors=None, marker_shapes=None, marker_size=100, marker='o'):
    '''
    This function creates a scatter plot with a regression line
    for the y variable against the degrees of graph G
    '''
    # Import what you need
    import matplotlib.pylab as plt
    import seaborn as sns
    
    # Load the data into a data frame
    df =  pd.DataFrame({x_label : x,
                        y_label : y})
        
    # Create the figure if you need to
    if not ax:
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 10))
        # Set the seaborn context and style
        sns.set(style="white")
        sns.set_context("poster", font_scale=2)
    else:
        fig = figure
        
    # Create a marker colors list if not given
    if not marker_colors:
        marker_colors = [color] * len(df[x_label])
        
    # Create a marker colors list if not given
    if not marker_shapes:
        marker_shapes = [ marker ] * len(df[x_label])
    df['marker_shapes'] = marker_shapes
    df.sort(columns='marker_shapes', inplace=True)
        
    # Create the linear regression plot
    ax = sns.regplot(x_label, y_label, 
                        df, ci=95, 
                        ax=ax, 
                        color=color, 
                        scatter_kws={'marker' : 'none'})

    # Add in each of the different points so they have
    # the right color and shape
    for _x, _y, _s, _c in zip(df[x_label], df[y_label], marker_shapes, marker_colors):
        ax.scatter(_x, _y, marker=_s, c=_c, lw=0.25, s=marker_size)

    # Fix the x and y axis limits
    if np.isscalar(x_max) and np.isscalar(x_min):
        ax.set_xlim((x_min, x_max))
    if np.isscalar(y_max) and np.isscalar(y_min):
        ax.set_ylim((y_min, y_max))
    
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    # Make sure there aren't too many bins!
    ax.locator_params(axis='y', nbins=4)
    ax.locator_params(axis='x', nbins=5)
    
    # Put a line at y = 0
    ax.axhline(0, linewidth=1, color='black', linestyle='--')

    # Despine because we all agree it looks better that way
    sns.despine()
    
    if figure_name:
        # Do the tight layout because, again, it looks better!
        fig.tight_layout()
    
        # And save the figure
        fig.savefig(figure_name, bbox_inches=0, dpi=100)
        plt.close(fig)
    
    else:
        return ax
        

def degree_r_values(graph_dict, y, covars_list=['ones'], measure='CT', group='all'):
    
    r_array = np.ones([30])
    p_array = np.ones([30])
    
    cost_list = range(1,31)
    
    for i, cost in enumerate(cost_list):
    
        cost = np.float(cost)
        covars = '_'.join(covars_list)
        
        key = '{}_covar_{}_{}_COST_{:02.0f}'.format(measure, covars, group, cost)
        
        G = graph_dict[key]
        
        degrees = np.array(G.degree().values())
        (r_array[i], p_array[i]) = pearsonr(degrees, y)
    
    return r_array, p_array
        
def create_violin_labels():
    '''
    A little function to create a labels list for the MT depth
    violin plots
    '''
    # Create an empty list for the names
    labels_list = []
    
    # Create a list of all the depths you care about
    depth_list = np.hstack([np.arange(100,-1,-10), np.arange(-40, -81, -40)])
    #depth_list = np.hstack([np.arange(100,-1,-10), np.arange(-20, -101, -20)])

    # Loop through all the depths
    for i in depth_list:
        
        # Fill in the appropriate label
        if i == 100:
            labels_list += ["Pial"]
        elif i == 0:
            labels_list += ["GM/WM"]
        elif i > 0: 
            labels_list += ['{:2.0f}%'.format(100.0 - i)]
        else: 
            labels_list += ['{:2.1f}mm'.format(i/-100.0)]

    return labels_list

def create_violin_data(measure_dict, mpm='MT', measure='all_slope_age', cmap='RdBu_r', cmap_min=-7, cmap_max=7):
    '''
    A little function to create a the data frame list
    for the MT depth violin plots
    
    INPUTS:
        measure_dict --- dictionary containing measure values
        measure -------- one of 'mean'
                                'std'
                                'all_slope_age'
                                'all_slope_ct'
                             default = 'all_slope_age'
        colormap ------- matplotlib colormap
                             default = 'RdBu_r'
    '''
    import matplotlib as mpl
    
    # Create an empty data frame for the data 
    # and an empty list for the associated colors
    n_values = len(measure_dict['{}_projfrac+000_{}'.format(mpm, measure)])
    df =  pd.DataFrame({'index' : range(n_values)})
    color_list = []
    color_dict = {}
    
    # Set up the color mapping
    cm = plt.get_cmap(cmap)
    cNorm  = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max)            #### THIS PROBABLY NEEDS ADJUSTING!
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)

    # Create a list of all the depths you care about
    depth_list = np.hstack([np.arange(100,-1,-10), np.arange(-40, -81, -40)])
    
    # (Note that originally there were 0.2mm steps into white matter
    # using the following command)
    #depth_list = np.hstack([np.arange(100,-1,-10), np.arange(-20, -101, -20)])
    
    # Loop through all the depths
    for i in depth_list:
        
        # Fill in the appropriate data
        if i >= 0:
            m_array = measure_dict['{}_projfrac{:+04.0f}_{}'.format(mpm, i, measure)]
        else:
            m_array = measure_dict['{}_projdist{:+04.0f}_{}'.format(mpm, i, measure)]

        df['{}'.format(i)] = m_array
            
        color_list += [scalarMap.to_rgba(np.mean(df['{}'.format(i)]))]
        
        color_dict['{}'.format(i)] = scalarMap.to_rgba(np.percentile(df['{}'.format(i)], 50))
        
    return df, color_list, color_dict


def violin_mt_depths(measure_dict, mpm='MT', measure='all_slope_age', cmap='PRGn', cmap_min=-7, cmap_max=7, y_max=None, y_min=None, figure_name=None, ax=None, figure=None, y_label=None, vert=True, lam_labels=True, cbar=False):
    '''
    INPUTS:
        data_dir --------- where the PARC_*_behavmerge.csv files are saved
        measure_dict
        vert ------------- create vertical box plots (rather than horizontal)
    '''
    
    # Import what you need
    import matplotlib.pylab as plt
    import seaborn as sns
    
    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)
    
    # Get the data, colors and labels
    df, color_list, color_dict = create_violin_data(measure_dict, 
                                                        mpm=mpm, 
                                                        measure=measure, 
                                                        cmap=cmap, 
                                                        cmap_min=cmap_min, 
                                                        cmap_max=cmap_max)
    
    labels_list = create_violin_labels()
        
    # Create the figure if you need to
    if not ax:
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = figure
        
    # Create the box plot
    ##### You could change this here to a violin plot if you wanted to...
    ax = sns.boxplot(df[df.columns[1:]], palette=color_dict, ax=ax, vert=vert)
    
    if vert:
        # Fix the y axis limits
        if np.isscalar(y_max) and np.isscalar(y_min):
            ax.set_ylim((y_min, y_max))
        # Set tick labels to be in scientific format if they're larger than 100
        # or smaller than 0.001
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        # Make sure there aren't too many bins!
        ax.locator_params(axis='y', nbins=4)        
        # Add in the tick labels and rotate them
        ax.set_xticklabels(labels_list, rotation=90)
        # Put a line at the grey white matter boundary
        # and another at y=0
        ax.axvline(10, linewidth=1, color='black', linestyle='--', zorder=-1)
        ax.axhline(0, linewidth=1, color='black', linestyle='-', zorder=-1)
        # Set the y label if it's been given
        if y_label:
            ax.set_ylabel(y_label)

    else:
        # Fix the y axis limits
        if np.isscalar(y_max) and np.isscalar(y_min):
            ax.set_xlim((y_min, y_max))
        ax.set_yticklabels(labels_list)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-5,5))    
        # Make sure there aren't too many bins!
        ax.locator_params(axis='x', nbins=4)
        # Put a line at the grey white matter boundary
        # and another at x=0
        ax.axhline(10, linewidth=1, color='black', linestyle='--', zorder=-1)
        ax.axvline(0, linewidth=1, color='black', linestyle='-', zorder=-1)
        # Set the y label if it's been given
        if y_label:
            ax.set_xlabel(y_label)

    # Despine because we all agree it looks better that way
    sns.despine()
    
    # Add in the laminae
    ax = violin_add_laminae(ax, vert=vert, labels=lam_labels)

    # Add a colorbar if necessary:
    if cbar:
        
        cb_grid = gridspec.GridSpec(1,1)
        pos = ax.get_position()

        if vert:
            cb_grid.update(left=pos.x1+0.01, right=pos.x1+0.02, bottom=pos.y0, top=pos.y1, wspace=0, hspace=0)
        else:
            cb_grid.update(left=pos.x0, right=pos.x1, bottom=pos.y0-0.04, top=pos.y0-0.03, wspace=0, hspace=0)    
            
        fig = add_colorbar(cb_grid[0], fig, 
                                cmap_name=cmap, 
                                y_min = y_min,
                                y_max = y_max,
                                cbar_min=cmap_min, 
                                cbar_max=cmap_max,
                                show_ticks=False,
                                vert=vert)
    
        if not vert:
            # If you add in a colorbar then you need to move the x axis label
            # down just a smidge
            ax.set_xlabel(y_label, labelpad=30)
        
    if figure_name:
        # Do the tight layout because, again, it looks better!
        fig.tight_layout()
    
        # And save the figure
        fig.savefig(figure_name, bbox_inches=0, dpi=100)
        plt.close(fig)
        
    else:
        return ax

def violin_add_laminae(ax, vert=True, labels=True):
    '''
    Great big thank yous to Konrad Wagstyl for journeying
    to the actual library and reading an actual book to pull
    out these values from von Economo's original work.
    I took these values from Konrad, averaged across regions to
    get an average thickness per region, added these together 
    to get an average total thickness and divided each value by 
    this total number to get the percentages.
    
    I then scaled the percentages so they lay ontop of a scale
    from 0 - 10 corresponding to the 11 sample depths for the 
    freesurfer analyses.
    
    The variance around each value was reasonably small.
    Means:
        0.9	1.6	4.6	5.7	7.6	11.0
    Standard deviations:
        0.17 0.21 0.25 0.12	0.10 0.12

    Mean + 1 standard devation:
        1.6	2.2	5.0	6.0	7.8	10.9
    Mean - 1 standard deviation:
        2.0	2.6	5.5	6.3	8.0	11.1
    '''
    boundary_values = [0.0, 0.8, 1.4, 4.2, 5.1, 6.9, 10.0]
    numerals = [ 'I', 'II', 'III', 'IV', 'V', 'VI', 'WM' ]

    # Figure out where the bottom of the plot lies
    # (this changes according to the number of samples into
    # white matter that you've plotted)
    if vert:
        bottom = ax.get_xlim()[0]
    else:
        bottom = ax.get_ylim()[0]

    boundary_values += [ bottom ]

    # Put in the mean boundaries
    for top, bottom in zip(boundary_values[1::2], boundary_values[2::2]):
        
        if vert:
            ax.axvspan(top, bottom, facecolor=(226/255.0, 226/255.0, 226/255.0), alpha=1.0, edgecolor='none', zorder=-1)

        else:
            ax.axhspan(top, bottom, facecolor=(226/255.0, 226/255.0, 226/255.0), alpha=1.0, edgecolor='none', zorder=-1)
    
    if labels:
    
        for top, bottom, numeral in zip(boundary_values[0::1], boundary_values[1::1], numerals):

            if vert:
                x_pos = np.mean([top, bottom])
                y_pos = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                ax.text(x_pos, y_pos, numeral,
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=25)
            else:
                x_pos = ax.get_xlim()[1] - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05
                y_pos = np.mean([top, bottom])
                ax.text(x_pos, y_pos, numeral,
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=25)
                            
    return ax

    
def old_figure_1(graph_dict, 
                    figures_dir, 
                    sagittal_pos, 
                    axial_pos, 
                    measure_dict, 
                    n=10, 
                    measure='CT', 
                    covars_list=['ones'], 
                    group='all'):
    
    big_fig, ax_list = plt.subplots(6, 5, figsize=(40, 35), facecolor='white', sharey='row')
    
    cost_list = [ 5, 10, 15, 20, 30 ]
    
    for i, cost in enumerate(cost_list):
        cost = np.float(cost)
        covars = '_'.join(covars_list)        
        
        key = '{}_covar_{}_{}_COST_{:02.0f}'.format(measure, covars, group, cost)
        print key
        
        G = graph_dict['{}_covar_{}_{}_COST_{:02.0f}'.format(measure, covars, group, cost)]
        G_edge = graph_dict['{}_covar_{}_{}_COST_{:02.0f}'.format(measure, covars, group, 2)]
        
        #==== SHOW THE AXIAL VIEW =====-=======================
        figure_name = os.path.join(figures_dir, 
                                        '{}_covar_{}_{}_sagittalnetwork_COST_{:02.0f}.png'.format(measure,
                                                                                                    covars, 
                                                                                                    group.upper(), 
                                                                                                    cost))
        plot_sagittal_network(G, G_edge, sagittal_pos, axial_pos, 
                                integer_adjust=0.1, fractional_adjust=100.0/cost, cmap_name='jet',
                                figure_name=figure_name)
                                
        ax_list[0, i] = plot_sagittal_network(G, G_edge, sagittal_pos, axial_pos, 
                                                integer_adjust=0.1, fractional_adjust=100.0/cost, cmap_name='jet',
                                                ax=ax_list[0, i])
        
        #==== SET UP RANDOM GRAPH =====-=======================
        # Start by creating n random graphs
        R_list = []
        for _ in range(n):
            R_list += [ random_graph(G) ]
        
        #============= DEGREE DISTRIBUTION ====================
        figure_name = os.path.join(figures_dir, 
                                        '{}_covar_{}_{}_degreesKDE_COST_{:02.0f}.png'.format(measure,
                                                                                                    covars, 
                                                                                                    group.upper(), 
                                                                                                    cost))
        plot_degree_dist(G, figure_name=figure_name, x_max=100, y_max=0.1, color=sns.color_palette()[0])
        
        ax_list[1, i] = plot_degree_dist(G, ax=ax_list[1, i], x_max=200, y_max=0.1, color=sns.color_palette()[0])
        
        #============= RICH CLUB ==============================
        figure_name = os.path.join(figures_dir, 
                                        '{}_covar_{}_{}_richclub_COST_{:02.0f}.png'.format(measure,
                                                                                                    covars, 
                                                                                                    group.upper(), 
                                                                                                    cost))
        deg, rc, rc_rand = rich_club(G, R_list, n=n)
        plot_rich_club(rc, rc_rand, figure_name=figure_name, x_max=100, y_max=1.2, color=sns.color_palette()[0])    
        ax_list[2, i] = plot_rich_club(rc, rc_rand, ax=ax_list[2, i], x_max=200, y_max=1.2, color=sns.color_palette()[0])
        
        #============= NETWORK MEASURES =======================
        figure_name = os.path.join(figures_dir, 
                                        '{}_covar_{}_{}_networkmeasures_COST_{:02.0f}.png'.format(measure,
                                                                                                    covars, 
                                                                                                    group.upper(), 
                                                                                                    cost))
        network_measure_dict = calculate_network_measures(G, R_list, n=n)
        plot_network_measures(network_measure_dict, 
                                figure_name=figure_name, 
                                y_max=2.5, y_min=-0.5, 
                                color=sns.color_palette()[0])
        ax_list[3, i] = plot_network_measures(network_measure_dict, 
                                                    ax=ax_list[3, i], 
                                                    y_max=2.5, y_min=-0.5, 
                                                    color=sns.color_palette()[0])
        
        #============= CORR DEGREE W/slope CT age =======================
        ax_list[4, i] = pretty_scatter(G.degree().values(), measure_dict['CT_all_slope_age'], 
                                                x_label='Degree', y_label='Slope CT with age', 
                                                x_max=100, x_min=0, 
                                                y_max=0.05, y_min=-0.1, 
                                                color='k',
                                                ax=ax_list[4, i],
                                                figure=big_fig)
                                                
        #============= CORR DEGREE W/slope MT age =======================
        ax_list[5, i] = pretty_scatter(G.degree().values(), measure_dict['MT_projfrac+030_all_slope_age'], 
                                                x_label='Degree', y_label='Slope MT(70%) with age', 
                                                x_max=100, x_min=0, 
                                                y_max=0.020, y_min=-0.010, 
                                                color='k',
                                                ax=ax_list[5, i],
                                                figure=big_fig)
    
    # Get rid of y axis labels for columns that aren't on the left side
    [ a.set_ylabel('') for a in ax_list[:,1:].reshape(-1) ]
    
    # RAAAANDOMLY - and I don't know why this is happening
    # set the x limits for the very last plot to those of the one
    # next to it - HMMMMMM
    ax_list[5,i].set_xlim( ax_list[5,i-1].get_xlim() )
    
    # Nice tight layout
    big_fig.tight_layout()
    
    big_fig.subplots_adjust(top=0.95)
    
    for i, cost in enumerate(cost_list):
        big_fig.text((2*i+1)/(len(cost_list)*2.0), 0.99, 
                        'density: {:.0f}%'.format(np.float(cost)),
                        horizontalalignment='center',
                        verticalalignment='top',
                        fontsize=60,
                        weight='bold')
                        
    # Save the figure
    filename = os.path.join(figures_dir, 
                            'SuppFigure1_{}_covar_{}.png'.format(measure, 
                                                                    covars))

    big_fig.savefig(filename, bbox_inches=0, dpi=100)
    
    plt.close()

def old_figure_2(df_ct, df_mpm, measure_dict, figures_dir, results_dir, aparc_names, mpm='MT'):
    
    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)

    big_fig, ax_list = plt.subplots(3,3, figsize=(30, 18), facecolor='white')
    
    #==== CORRELATE GLOBAL CT WITH AGE =============================
    figure_name = os.path.join(figures_dir, 'Global_CT_corr_Age.png')
        
    color=sns.color_palette('RdBu_r', 10)[1]
    
    pretty_scatter(df_ct['age_scan'], df_ct['Global'], 
                    x_label='Age (years)', y_label='Cortical Thickness\n(mm)', 
                    x_max=25, x_min=14, 
                    y_max=3.0, y_min=2.4, 
                    figure_name=figure_name,
                    color=color)
                            
    ax_list[0, 0] = pretty_scatter(df_ct['age_scan'], df_ct['Global'], 
                    x_label='Age (years)', y_label='Cortical Thickness\n(mm)', 
                    x_max=25, x_min=14, 
                    y_max=3.0, y_min=2.4, 
                    color=color,
                    ax=ax_list[0, 0],
                    figure=big_fig)
                        
    #==== CORRELATE GLOBAL MT(70) WITH AGE =============================
    figure_name = os.path.join(figures_dir, 
                                    'Global_{}_projfrac+030_corr_Age.png'.format(mpm))
        
    color=sns.color_palette('PRGn_r', 10)[1]
    
    pretty_scatter(df_mpm['age_scan'], df_mpm['Global'], 
                    x_label='Age (years)', y_label='Magnetisation Transfer\nat 70% cortical depth', 
                    x_max=25, x_min=14, 
                    y_max=1.05, y_min=0.8, 
                    figure_name=figure_name,
                    color=color)
                            
    ax_list[1, 0] = pretty_scatter(df_mpm['age_scan'], df_mpm['Global'], 
                    x_label='Age (years)', y_label='Magnetisation Transfer\nat 70% cortical depth', 
                    x_max=25, x_min=14, 
                    y_max=1.05, y_min=0.8, 
                    color=color,
                    ax=ax_list[1, 0],
                    figure=big_fig)
    
    #==== CORRELATE GLOBAL MT(70) WITH CT =============================
    figure_name = os.path.join(figures_dir, 
                                    'Global_{}_projfrac+030_corr_CT.png'.format(mpm))
        
    color=sns.color_palette('PRGn', 10)[1]
    
    pretty_scatter(df_ct['Global'], df_mpm['Global'], 
                    x_label='Cortical Thickness (mm)', y_label='Magnetisation Transfer\nat 70% cortical depth', 
                    x_max=3.0, x_min=2.4, 
                    y_max=1.05, y_min=0.8, 
                    figure_name=figure_name,
                    color=color)
                            
    ax_list[2, 0] = pretty_scatter(df_ct['Global'], df_mpm['Global'], 
                    x_label='Cortical Thickness (mm)', y_label='Magnetisation Transfer\nat 70% cortical depth', 
                    x_max=3.0, x_min=2.4, 
                    y_max=1.05, y_min=0.8, 
                    color=color,
                    ax=ax_list[2, 0],
                    figure=big_fig)
    
    #==== SHOW PYSURFER CT CORR AGE =============================
    #figure_name = os.path.join(results_dir, 
    #                                'Global_MT_projfrac+030_corr_CT.png')
    #img = mpimg.imread(f)
    #ax_list[0,1].imshow(img)
    # EASY - but needs fiddling with - TBD 
    
    #==== CORRELATE GLOBAL CT WITH DeltaCT =============================
    figure_name = os.path.join(figures_dir, 
                                    'Mean_CT_corr_slope_CT_age.png')
        
    color=sns.color_palette('RdBu_r', 10)[1]
    
    pretty_scatter(measure_dict['CT_all_mean'], measure_dict['CT_all_slope_age'],
                    x_label='Cortical Thickness (mm)', y_label='Slope CT with age', 
                    x_max=4.0, x_min=1.8, 
                    y_max=0.04, y_min=-0.04, 
                    figure_name=figure_name,
                    color=color)
                            
    ax_list[0, 2] = pretty_scatter(measure_dict['CT_all_mean'], measure_dict['CT_all_slope_age'],
                    x_label='Cortical Thickness (mm)', y_label='Slope CT with age\n', 
                    x_max=4.0, x_min=1.8, 
                    y_max=0.04, y_min=-0.04, 
                    color=color,
                    ax=ax_list[0, 2],
                    figure=big_fig)
    
    
    #==== SHOW CORR WITH AGE AT DIFFERENT DEPTHS ======================
    figure_name = os.path.join(figures_dir, 
                                    '{}_projfrac+030_corr_Age_DifferentDepths.png'.format(mpm))
    
    violin_mt_depths(measure_dict,
                        measure='all_slope_age',
                        cmap='PRGn',
                        y_max=0.015, y_min=-0.010, 
                        cmap_min=-0.007, cmap_max=0.007,
                        figure_name=figure_name,
                        mpm=mpm,
                        vert=False)
                        
    ax_list[1, 2] = violin_mt_depths(measure_dict,
                                        y_label='Slope MT(70%)\nwith age',
                                        measure='all_slope_age',
                                        y_max=0.015, y_min=-0.010, 
                                        cmap_min=-0.007, cmap_max=0.007,
                                        ax=ax_list[1, 2],
                                        figure=big_fig,
                                        mpm=mpm)
    
    #==== SHOW CORR WITH CT AT DIFFERENT DEPTHS ======================
    figure_name = os.path.join(figures_dir, 
                                    '{}_projfrac+030_corr_CT_DifferentDepths.png'.format(mpm))
    
    violin_mt_depths(measure_dict,
                        measure='all_slope_ct',
                        cmap='PRGn',
                        y_min=-7.0,
                        y_max=3.0,
                        cmap_min=-3.0,
                        cmap_max=3.0,
                        figure_name=figure_name,
                        mpm=mpm,
                        vert=False)
    
    ax_list[2, 2] = violin_mt_depths(measure_dict,
                                        ylabel='Slope MT(70%)\nwith CT',
                                        measure='all_slope_ct',
                                        cmap='PRGn',
                                        y_min=-7.0,
                                        y_max=3.0,
                                        cmap_min=-3.0,
                                        cmap_max=3.0,
                                        ax=ax_list[2, 2],
                                        figure=big_fig,
                                        mpm=mpm)
    
    # Allign the y labels for each column    
    for ax in ax_list.reshape(-1):
        ax.yaxis.set_label_coords(-0.12, 0.5)
    
    # Turn off the axes for the middle column
    for ax in ax_list[:,1]:
        ax.axis('off')
        
    # Nice tight layout
    big_fig.tight_layout()
    
    # Save the figure
    filename = os.path.join(figures_dir, 'Figure2.png')
    big_fig.savefig(filename, bbox_inches=0, dpi=100)
    
    plt.close()

    
def old_figure_3(graph_dict, measure_dict, figures_dir, covars_list=['ones'], group='all', measure='CT'):

    import matplotlib.pylab as plt
    import numpy as np
    import networkx as nx

    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)

    big_fig, ax_list = plt.subplots(2,3, figsize=(30, 12), facecolor='white')
    
    cost = 10    
    cost = np.float(cost)
    covars = '_'.join(covars_list)

    key = '{}_covar_{}_{}_COST_{:02.0f}'.format(measure, covars, group, cost)

    G = graph_dict[key]
    pc_dict = participation_coefficient(G)    
    pc = np.array(pc_dict.values())
    degrees = np.array(G.degree().values())
    
    #==== CORRELATE DEGREES WITH CHANGE IN CT WITH AGE =============================
    figure_name = os.path.join(figures_dir, 
                                    '{}_covar_{}_{}_corrDegreesSlopeCTAge_COST_{:02.0f}.png'.format(measure,
                                                                                                covars, 
                                                                                                group.upper(), 
                                                                                                cost))
    pretty_scatter(degrees, measure_dict['CT_all_slope_age'], 
                    x_label='Degree', y_label='Slope CT with age', 
                    x_max=100, x_min=0, 
                    y_max=0.05, y_min=-0.1, 
                    figure_name=figure_name,
                    color='k')
                            
    ax_list[0, 0] = pretty_scatter(degrees, measure_dict['CT_all_slope_age'], 
                    x_label='Degree', y_label='Slope CT with age', 
                    x_max=100, x_min=0, 
                    y_max=0.05, y_min=-0.1, 
                    color='k',
                    ax=ax_list[0, 0],
                    figure=big_fig)
    
    #==== CORRELATE PARTICIPATION COEFFS WITH CHANGE IN CT WITH AGE =============================
    figure_name = os.path.join(figures_dir, 
                                    '{}_covar_{}_{}_corrPCSlopeCTAge_COST_{:02.0f}.png'.format(measure,
                                                                                                covars, 
                                                                                                group.upper(), 
                                                                                                cost))
    pretty_scatter(pc[pc>0], measure_dict['CT_all_slope_age'][pc>0], 
                    x_label='Participation Coefficient', y_label='Slope CT with age', 
                    x_max=1, x_min=0, 
                    y_max=0.05, y_min=-0.1, 
                    figure_name=figure_name,
                    color='k')
                            
    ax_list[1, 0] = pretty_scatter(pc[pc>0], measure_dict['CT_all_slope_age'][pc>0], 
                    x_label='Participation Coefficient', y_label='Slope CT with age', 
                    x_max=1, x_min=0, 
                    y_max=0.05, y_min=-0.1, 
                    color='k',
                    ax=ax_list[1, 0],
                    figure=big_fig)
                    
    #==== CORRELATE DEGREES WITH CHANGE IN MT30 WITH AGE =============================
    figure_name = os.path.join(figures_dir, 
                                    '{}_covar_{}_{}_corrDegreesSlopeMT+030Age_COST_{:02.0f}.png'.format(measure,
                                                                                                covars, 
                                                                                                group.upper(), 
                                                                                                cost))
    pretty_scatter(degrees, measure_dict['MT_projfrac+030_all_slope_age'], 
                    x_label='Degree', y_label='Slope MT(70%) with age', 
                    x_max=100, x_min=0, 
                    y_max=20, y_min=-10, 
                    figure_name=figure_name,
                    color='k')
                            
    ax_list[0, 1] = pretty_scatter(degrees, measure_dict['MT_projfrac+030_all_slope_age'], 
                    x_label='Degree', y_label='Slope MT(70%) with age', 
                    x_max=100, x_min=0, 
                    y_max=0.020, y_min=-0.010, 
                    color='k',
                    ax=ax_list[0, 1],
                    figure=big_fig)
        
        
    #==== CORRELATE PARTICIPATION COEFFS WITH CHANGE IN MT30 WITH AGE =============================
    figure_name = os.path.join(figures_dir, 
                                    '{}_covar_{}_{}_corrPCSlopeMT+030Age_COST_{:02.0f}.png'.format(measure,
                                                                                                covars, 
                                                                                                group.upper(), 
                                                                                                cost))
    pretty_scatter(pc[pc>0], measure_dict['MT_projfrac+030_all_slope_age'][pc>0], 
                    x_label='Participation Coefficient', y_label='Slope MT(70%) with age', 
                    x_max=1, x_min=0, 
                    y_max=20, y_min=-10, 
                    figure_name=figure_name,
                    color='k')
                            
    ax_list[1, 1] = pretty_scatter(pc[pc>0], measure_dict['MT_projfrac+030_all_slope_age'][pc>0], 
                    x_label='Participation Coefficient', y_label='Slope MT(70%) with age', 
                    x_max=1, x_min=0, 
                    y_max=20, y_min=-10, 
                    color='k',
                    ax=ax_list[1, 1],
                    figure=big_fig)
                    
    #==== CORRELATE DEGREES WITH CHANGE IN MT30 WITH CT =============================
    figure_name = os.path.join(figures_dir, 
                                    '{}_covar_{}_{}_corrDegreesSlopeMT+030CT_COST_{:02.0f}.png'.format(measure,
                                                                                                covars, 
                                                                                                group.upper(), 
                                                                                                cost))
    pretty_scatter(degrees, measure_dict['MT_projfrac+030_all_slope_ct'], 
                    x_label='Degree', y_label='Slope MT(70%) with CT', 
                    x_max=100, x_min=0, 
                    y_max=0.005, y_min=-0.005, 
                    figure_name=figure_name,
                    color='k')
                            
    ax_list[0, 2] = pretty_scatter(degrees, measure_dict['MT_projfrac+030_all_slope_ct'], 
                    x_label='Degree', y_label='Slope MT(70%) with CT', 
                    x_max=100, x_min=0, 
                    y_max=0.005, y_min=-0.005, 
                    color='k',
                    ax=ax_list[0, 2],
                    figure=big_fig)
        
    #==== CORRELATE PARTICIPATION COEFFS WITH CHANGE IN MT30 WITH AGE =============================
    figure_name = os.path.join(figures_dir, 
                                    '{}_covar_{}_{}_corrPCSlopeMT+030Age_COST_{:02.0f}.png'.format(measure,
                                                                                                covars, 
                                                                                                group.upper(), 
                                                                                                cost))
    pretty_scatter(pc[pc>0], measure_dict['MT_projfrac+030_all_slope_ct'][pc>0], 
                    x_label='Participation Coefficient', y_label='Slope MT(70%) with ct', 
                    x_max=1, x_min=0, 
                    y_max=0.005, y_min=-0.005, 
                    figure_name=figure_name,
                    color='k')
                            
    ax_list[1, 2] = pretty_scatter(pc[pc>0], measure_dict['MT_projfrac+030_all_slope_ct'][pc>0], 
                    x_label='Participation Coefficient', y_label='Slope MT(70%) with CT', 
                    x_max=1, x_min=0, 
                    y_max=0.005, y_min=-0.005, 
                    color='k',
                    ax=ax_list[1, 2],
                    figure=big_fig)

    # RAAAANDOMLY - and I don't know why this is happening
    # set the x limits for the very last plot to those of the one
    # next to it - HMMMMMM
    #ax_list[3,i].set_xlim( ax_list[3,i-1].get_xlim() )
    
    # Nice tight layout
    big_fig.tight_layout()
    
    # Save the figure
    filename = os.path.join(figures_dir, 
                                'Figure3_{}_covar_{}_{}_COST_{:02.0f}.png'.format(measure, 
                                                                                    covars,
                                                                                    group,
                                                                                    cost))
    big_fig.savefig(filename, bbox_inches=0, dpi=100)
    
    plt.close()
    
    
def partial_volume_fig(measure_dict, figures_dir):

    big_fig, ax_list = plt.subplots(2, 4, figsize=(40, 20), facecolor='white')
        
    #==== SHOW MEAN MT AT DIFFERENT DEPTHS ======================                            
    ax_list[0, 0] = violin_mt_depths(measure_dict,
                                        map='MT',
                                        measure='global_mean',
                                        y_min=0,
                                        y_max=2.0,
                                        cmap='jet',
                                        cmap_min=0,
                                        cmap_max=2.0,
                                        ax=ax_list[0, 0],
                                        figure=big_fig)
                                        
    ax_list[1, 0] = violin_mt_depths(measure_dict,
                                        map='synthetic',
                                        measure='global_mean',
                                        y_min=0,
                                        y_max=2.0,
                                        cmap='jet',
                                        cmap_min=0,
                                        cmap_max=2.0,
                                        ax=ax_list[1, 0],
                                        figure=big_fig)
    
    #==== SHOW STD AT DIFFERENT DEPTHS ======================                            
    ax_list[0, 1] = violin_mt_depths(measure_dict,
                                        map='MT',
                                        measure='global_std',
                                        y_min=0,
                                        y_max=0.6,
                                        cmap='jet',
                                        cmap_min=0.0,
                                        cmap_max=0.6,
                                        ax=ax_list[0, 1],
                                        figure=big_fig)
                                        
    ax_list[1, 1] = violin_mt_depths(measure_dict,
                                        map='synthetic',
                                        measure='global_std',
                                        y_min=0,
                                        y_max=0.6,
                                        cmap='jet',
                                        cmap_min=0,
                                        cmap_max=0.6,
                                        ax=ax_list[1, 1],
                                        figure=big_fig)
                        
    #==== SHOW CORR W AGE AT DIFFERENT DEPTHS ======================                            
    ax_list[0, 2] = violin_mt_depths(measure_dict,
                                        map='MT',
                                        measure='all_slope_age',
                                        y_min=-10,
                                        y_max=15,
                                        cmap='PRGn',
                                        cmap_min=-15,
                                        cmap_max=15,
                                        ax=ax_list[0, 2],
                                        figure=big_fig)
                                        
    ax_list[1, 2] = violin_mt_depths(measure_dict,
                                        map='synthetic',
                                        measure='all_slope_age',
                                        y_min=-10,
                                        y_max=15,
                                        cmap='PRGn',
                                        cmap_min=-15,
                                        cmap_max=15,
                                        ax=ax_list[1, 2],
                                        figure=big_fig)
                                        
    #==== SHOW CORR W CT AT DIFFERENT DEPTHS ======================                            
    ax_list[0, 3] = violin_mt_depths(measure_dict,
                                        map='MT',
                                        measure='all_slope_ct',
                                        y_min=-0.01,
                                        y_max=0.005,
                                        cmap='PRGn',
                                        cmap_min=-0.01,
                                        cmap_max=0.01,
                                        ax=ax_list[0, 3],
                                        figure=big_fig)
                                        
    ax_list[1, 3] = violin_mt_depths(measure_dict,
                                        map='synthetic',
                                        measure='all_slope_ct',
                                        y_min=-0.01,
                                        y_max=0.005,
                                        cmap='PRGn',
                                        cmap_min=-0.01,
                                        cmap_max=0.01,
                                        ax=ax_list[1, 3],
                                        figure=big_fig)

    # Nice tight layout
    big_fig.tight_layout()
    
    # Save the figure
    filename = os.path.join(figures_dir, 'PartialVolumeFig_AcrossParticipants.png')
    big_fig.savefig(filename, bbox_inches=0, dpi=100)
    
    plt.close()
    
# MEAN MAGNETISATION TRANSFER ACROSS ALL PARTICIPANTS
def all_mean_mt(measure_dict, figures_dir, mpm='MT'):

    figure_name = os.path.join(figures_dir, 
                                    '{}_all_mean_DifferentDepths.png'.format(mpm))
                                    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    ax = violin_mt_depths(measure_dict,
                        measure='all_mean',
                        ylabel='Magnetisation Transfer',
                        y_min=0.0,
                        y_max=2.0,
                        cmap='jet',
                        cmap_min=0.2,
                        cmap_max=1.8,
                        figure=fig,
                        ax=ax,
                        mpm=mpm)
    
    # Nice tight layout
    big_fig.tight_layout()    
    fig.subplots_adjust(right=0.9)
    
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=0.2, vmax=1.8)

    cax = fig.add_axes([0.93, 0.3, 0.02, 0.6])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                   norm=norm,
                                   orientation='vertical',
                                   ticks=np.arange(0.2, 1.81, 0.8))
                                   
    cax.tick_params(labelsize=20)
    
    # Save the figure
    fig.savefig(figure_name, bbox_inches=0, dpi=100)
    
    plt.close()

                        
def nodal_ct_mt(measure_dict, figures_dir, mpm='MT'):

    figure_name = os.path.join(figures_dir, 
                                    'Nodal_CT_corr_{}_segCort.png'.format(mpm))
                                    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    ax = pretty_scatter(measure_dict['CT_all_mean'], measure_dict['{}all_all_mean'.format(mpm)], 
                    x_label='Average Cortical Thickness (mm)', y_label='Average Magnetisation Transfer', 
                    x_max=3.8, x_min=1.9, 
                    y_max=1.00, y_min=0.750, 
                    color='k',
                    ax=ax,
                    figure=fig)
    
def get_von_economo_color_dict(von_economo):
    '''
    Create a color dictionary for the von economo values you pass
    The color_list is hard coded at the moment... might change one day
    '''
    color_list = [ 'purple', 'blue', 'green', 'orange', 'yellow' ]
    #color_list = [ '0.5', '0.6', '0.7', '0.8', '0.9' ]
    # You need to make it into a color dictionary
    color_dict={}
    for i, color in enumerate(color_list):
        color_dict[i+1] = color
            
    return color_dict

def get_von_economo_shapes_dict(von_economo):
    '''
    Create a dictionary containing a different marker shape for 
    each of the the von economo values you pass
    The shape_list is hard coded at the moment... might change one day
    '''
    shape_list = [ 'o', '^', 's', 'v', 'd' ]
    # You need to make it into a color dictionary
    shape_dict={}
    for i, shape in enumerate(shape_list):
        shape_dict[i+1] = shape
            
    return shape_dict
    
def von_economo_boxes(measure_dict, figures_dir, von_economo, measure='CT_all_mean', group_label='Cortical Laminar Pattern', y_label=None, y_min=1.5, y_max=4.0, figure_name=None, figure=None, ax=None, von_economo_colors=True, color_dict="muted", cmap_name=None, max_color=False, min_color=False, alpha=1.0):

    # Read the data into a data frame
    df = pd.DataFrame( { 'x' : measure_dict[measure],
                         group_label : von_economo } )
                        
    # If you've turned on the von_economo_colors flag
    # then you'll always used the set color scheme
    if von_economo_colors:
        color_dict = get_von_economo_color_dict(von_economo)
        
    else:
        color_dict = color_dict

    # If you've passed a colormap then you're going to make a
    # color dict from that colormap
    if cmap_name:
        cmap = plt.get_cmap(cmap_name)
        color_dict = {}
        n = len(set(von_economo))
        for i, value in enumerate(set(von_economo)):
            color_dict[value] = cmap(np.float(i + 0.5)/n)
    
    # Order the box plots from max to min
    order = range(np.floor(np.min(von_economo)).astype('int'),
                np.floor(np.max(von_economo)).astype('int')+1)

    # Create the figure if you need to
    if not ax:
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        # Set the seaborn style
        sns.set(style="white")
        sns.set_context("poster", font_scale=2)
    else:
        fig = figure
        
    # Make the box plot
    bp = sns.boxplot(df.x[df.x>-99], 
                        groupby=df[group_label], 
                        order=order,
                        palette=color_dict, 
                        ax=ax)
    
    # Set the y label if it's been given
    if y_label:
        ax.set_ylabel(y_label)

    # Set the y limits
    ax.set_ylim((y_min, y_max))
    
    # Make the max median line red if requested
    if max_color:
        medians = [ line.get_ydata()[0] for line in bp.get_lines()[4::6] ]
        max_median = np.max(medians)
        for line in bp.get_lines()[4::6]:
            if line.get_ydata()[0] == max_median:
                line.set_color(max_color)
                
    # Make the minimum median line red if requested
    if min_color:
        medians = [ line.get_ydata()[0] for line in bp.get_lines()[4::6] ]
        min_median = np.min(medians)
        for line in bp.get_lines()[4::6]:
            if line.get_ydata()[0] == min_median:
                line.set_color(min_color)
    
    # Change the alpha value for the fill color if requested
    start_i = len(set(von_economo))*6 + 2
    stop_i = len(set(von_economo))*7 + 2
    for patch in bp.get_default_bbox_extra_artists()[start_i:stop_i]:
        fc = patch.get_facecolor()
        patch.set_facecolor((fc[0], fc[1], fc[2], alpha))
    
    # Make sure there aren't too many bins!
    ax.locator_params(axis='y', nbins=4)
    
    # Put a line at y = 0
    ax.axhline(0, linewidth=1, color='black', linestyle='--')

    if figure_name:
        # Do the tight layout because, again, it looks better!
        fig.tight_layout()
    
        # And save the figure
        fig.savefig(figure_name, bbox_inches=0, dpi=100)
        plt.close(fig)
    
    else:
        return ax

def von_economo_scatter(measure_dict, figures_dir, von_economo, measure='CT_all_mean', x_label='x', y_label='y', x_min=1.5, x_max=4.0, y_min=0.8, y_max=1.2, figure_name=None, figure=None, ax=None):

    # Set the seaborn style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)

    # Read the data into a data frame
    df = pd.DataFrame( { x_label : measure_dict[x_label],
                         y_label : measure_dict[y_label],
                         'Cortical Laminar Pattern' : von_economo } )
                        
    # You'll always use this color_list
    color_list = [ 'purple', 'blue', 'green', 'orange', 'yellow' ]
    
    # You need to make it into a color dictionary
    color_dict={}
    for i, color in enumerate(color_list):
        color_dict[i+1] = color
        
    # Create the figure if you need to
    if not ax:
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    else:
        fig = figure
    
    for i in range(1,6):
        df_i = df[df['Cortical Laminar Pattern']==i]
        # Create the linear regression plot
        ax = sns.regplot(x_label, y_label, df_i, ci=95, ax=ax, color=color_dict[i], scatter_kws={'s': 60})
    
    # Fix the x and y axis limits
    if np.isscalar(x_max) and np.isscalar(x_min):
        ax.set_xlim((x_min, x_max))
    if np.isscalar(y_max) and np.isscalar(y_min):
        ax.set_ylim((y_min, y_max))
    
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))

    # Make sure there aren't too many bins!
    ax.locator_params(axis='y', nbins=4)
    
    # Put a line at y = 0
    ax.axhline(0, linewidth=1, color='black', linestyle='--')

    # Despine because we all agree it looks better that way
    sns.despine()
    
    if figure_name:
        # Do the tight layout because, again, it looks better!
        fig.tight_layout()
    
        # And save the figure
        fig.savefig(figure_name, bbox_inches=0, dpi=100)
        plt.close(fig)
    
    else:
        return ax

def add_four_hor_brains(grid, f_list, big_fig):
    '''
    Take the four pysurfer views (left lateral, left medial,
    right medial and right lateral) and arrange them in a row
    according to the grid positions given by grid
    
    grid    :  the gridspec list of grid placements
    f_list  :  list of four file pysurfer image files
    big_fig :  the figure to which you're adding the images
    '''
    for g_loc, f in zip(grid, f_list):
        img = mpimg.imread(f)
        # Crop the figures appropriately
        # NOTE: this can change depending on which system you've made the 
        # images on originally - it's a bug that needs to be sorted out!
        if 'lateral' in f:
            img_cropped = img[115:564, 105:(-100),:]
        else:
            img_cropped = img[90:560, 60:(-55),:]

        # Add an axis to the big_fig
        ax_brain = plt.Subplot(big_fig, g_loc)
        big_fig.add_subplot(ax_brain)
        
        # Show the brain on this axis
        ax_brain.imshow(img_cropped, interpolation='none')
        ax_brain.set_axis_off()
    
    return big_fig

def add_colorbar(grid, big_fig, cmap_name, y_min=0, y_max=1, cbar_min=0, cbar_max=1, vert=False, label=None, show_ticks=True):
    '''
    Add a colorbar to the big_fig in the location defined by grid 
    
    grid       :  grid spec location to add colormap
    big_fig    :  figure to which colorbar will be added
    cmap_name  :  name of the colormap
    x_min      :  the minimum value to plot this colorbar between
    x_max      :  the maximum value to plot this colorbar between
    cbar_min   :  minimum value for the colormap (default 0)
    cbar_max   :  maximum value for the colormap (default 1)
    vert       :  whether the colorbar should be vertical (default False)
    label      :  the label for the colorbar (default: None)
    ticks      :  whether to put the tick values on the colorbar (default: True)
    '''
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    
    # Add an axis to the big_fig
    ax_cbar = plt.Subplot(big_fig, grid)
    big_fig.add_subplot(ax_cbar)
    
    # Normalise the colorbar so you have the correct upper and
    # lower limits and define the three ticks you want to show
    norm = mpl.colors.Normalize(vmin=cbar_min, vmax=cbar_max)

    if show_ticks:
        ticks = [y_min, np.average([y_min, y_max]), y_max]
    else:
        ticks=[]
            
    # Figure out the orientation
    if vert:
        orientation='vertical'
    else:
        orientation='horizontal'
        
    # Add in your colorbar:
    cb = mpl.colorbar.ColorbarBase(ax_cbar, 
                                       cmap=cmap_name,
                                       norm=norm,
                                       orientation=orientation,
                                       ticks=ticks,
                                       boundaries=np.linspace(y_min, y_max, 300))
                                       
    if label:
        cb.set_label(label)
        
    return big_fig
    
def add_cells_picture(figures_dir, big_fig):
    
    # Get the file name and read it in as an image
    f_name = os.path.join(figures_dir, '../..', 'CorticalLayers_schematic_cells.jpg')
    img = mpimg.imread(f_name)
    img_cropped = img[30:, :]
    
    # Add an axis in the bottom left corner
    grid = gridspec.GridSpec(1, 1)
    grid.update(left=0.36, right=0.51, top=0.47, bottom=0.1, wspace=0, hspace=0)
    ax = plt.Subplot(big_fig, grid[0])
    big_fig.add_subplot(ax)

    # Show the picture and turn the axis off
    ax.imshow(img_cropped)
    ax.axis('off')
    
    # Add in the laminar labels
    boundary_values = [ 0, 113, 166, 419, 499, 653, 945, 1170 ]
    
    numerals = [ 'I', 'II', 'III', 'IV', 'V', 'VI', 'WM' ]

    for top, bottom, numeral in zip(boundary_values[0:], boundary_values[1:], numerals):
        x_pos = -0.15 * img_cropped.shape[1]
        y_pos = np.mean([top, bottom])
        ax.text(x_pos, y_pos, numeral,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=25)
                  
    '''
    ax.text(1.1, 0.5, 'Schematic of cortical laminae',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=30, 
                    transform=ax.transAxes, 
                    rotation=270)
    '''
    return big_fig
    

def figure_1(measure_dict, figures_dir, results_dir, mpm='MT'):
    
    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=3.5)

    # Get the various min and max values:
    min_max_dict = get_min_max_values(measure_dict)
    axis_label_dict = get_axis_label_dict()
    
    # Create the big figure
    big_fig = plt.figure(figsize=(46, 30), facecolor='white')
    
    #==== FOUR ROWS OF DATA ======================================
    # Make a list of the file names for the left lateral image
    left_lat_fname_list = [ os.path.join(results_dir, 'PNGS', 
                                    'SlopeAge_at14_CT_lh_pial_lateral.png'),
                               os.path.join(results_dir, 'PNGS', 
                                    'SlopeAge_FDRmask_CT_lh_pial_lateral.png'),
                               os.path.join(results_dir, 'PNGS', 
                                    'SlopeAge_at14_MT_projfrac+030_lh_pial_lateral.png'),
                               os.path.join(results_dir, 'PNGS', 
                                    'SlopeAge_FDRmask_MT_projfrac+030_lh_pial_lateral.png') ]
    
    # List the var names that will be used to get the axis labels
    # and min/max values
    var_name_list = [ ( 'CT_all_slope_age_at14', 'CT_global_mean' ),
                        ( 'CT_all_slope_age', 'CT_all_slope_age' ),
                        ( 'MT_projfrac+030_all_slope_age_at14', 'MT_projfrac+030_global_mean' ),
                        ( 'MT_projfrac+030_all_slope_age', 'MT_projfrac+030_all_slope_age' ) ]
    
    # List the colorbar names
    cmap_name_list = [ 'jet', 'winter_r', 'jet', 'autumn' ]
    
    # List of min max color tuples
    min_max_color_list = [ ('blue', 'red'), ('red', 'blue'), ('red', 'blue'), ('blue', 'red') ]
    
    # Scatter and von economo grid
    grid = gridspec.GridSpec(4, 2)
    grid.update(left=0.55, bottom=0.06, top=0.97, right=0.99, hspace=0.5, wspace=0.3)
                    
    ax_list = []
    for g_loc in grid:
        ax_list += [ plt.Subplot(big_fig, g_loc) ]
        big_fig.add_subplot(ax_list[-1])
    
    for i, (left_lat_fname, 
                var_name, 
                cmap_name,
                min_max_color) in enumerate(zip(left_lat_fname_list, 
                                                                    var_name_list, 
                                                                    cmap_name_list,
                                                                    min_max_color_list)):
        
        #==== BRAIN IMAGES ======================================
        # Plot the braaaaains
        f_list = [ left_lat_fname, 
                    left_lat_fname.replace('lh_pial_lateral', 'lh_pial_medial'),
                    left_lat_fname.replace('lh_pial_lateral', 'rh_pial_lateral'),
                    left_lat_fname.replace('lh_pial_lateral', 'rh_pial_medial') ]
        
        grid = gridspec.GridSpec(1,4)
        
        grid.update(left=0.01, 
                        right=0.5,
                        bottom=0.81 - (i*0.25), 
                        top=1.01 - (i*0.25), 
                        wspace=0, 
                        hspace=0)
        
        # Put the four brains in a row
        big_fig = add_four_hor_brains(grid, f_list, big_fig)
        
        # Add a colorbar
        cb_grid = gridspec.GridSpec(1,1)
        
        cb_grid.update(left=0.11, 
                            right=0.4, 
                            bottom=0.81 - (i*0.25),
                            top=0.82 - (i*0.25), 
                            wspace=0, 
                            hspace=0)    
        
        big_fig = add_colorbar(cb_grid[0], big_fig, 
                                cmap_name=cmap_name, 
                                cbar_min=min_max_dict['{}_CBAR_min'.format(var_name[0])], 
                                cbar_max=min_max_dict['{}_CBAR_max'.format(var_name[0])],
                                y_min=min_max_dict['{}_CBAR_min'.format(var_name[0])],
                                y_max=min_max_dict['{}_CBAR_max'.format(var_name[0])],
                                label=axis_label_dict[var_name[0]])
    
        #==== VON ECONOMO BOX PLOTS =============================
        ax_list[i*2] = von_economo_boxes(measure_dict, figures_dir, 
                                        measure_dict['von_economo'], 
                                        measure=var_name[0],
                                        y_label=axis_label_dict[var_name[0]], 
                                        y_min=min_max_dict['{}_min'.format(var_name[0])], 
                                        y_max=min_max_dict['{}_max'.format(var_name[0])], 
                                        von_economo_colors=True,
                                        max_color=min_max_color[1],
                                        min_color=min_max_color[0],
                                        alpha=0,
                                        ax=ax_list[i*2],
                                        figure=big_fig)
    
        # Allign the y labels for each column    
        ax_list[i*2].yaxis.set_label_coords(-0.14, 0.5)
        
        # Make sure axis is in scientific format
        ax_list[i*2].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
                                      
        # Update the font size for the labels
        # to be a little smaller
        for lab in [ ax_list[i*2].yaxis.label, ax_list[i*2].xaxis.label ]:
            f_size = lab.get_fontsize()
            lab.set_fontsize(f_size * 0.88)        
        
        #==== SCATTER PLOTS =============================                            
        if not cmap_name == 'jet':
            color='k'
            x_var = var_name[1].replace('age', 'age_at14')
        else:
            color_measure = var_name_list[i+1][1]
            color_measure_cmap = cmap_name_list[i+1]
            norm = mpl.colors.Normalize(vmin=min_max_dict['{}_CBAR_min'.format(color_measure)], 
                                        vmax=min_max_dict['{}_CBAR_max'.format(color_measure)])
            cmap_converter = mpl.cm.ScalarMappable(norm=norm, cmap=color_measure_cmap)
            color = cmap_converter.to_rgba(measure_dict['{}'.format(var_name[1].replace('_mean', '_slope_age'))])
            x_var = 'age_scan'
            
        ax_list[(i*2)+1] = pretty_scatter(measure_dict[x_var], measure_dict[var_name[1]], 
                                            x_label=axis_label_dict[x_var], 
                                            y_label=axis_label_dict[var_name[1]], 
                                            x_min=min_max_dict['{}_min'.format(x_var)], 
                                            x_max=min_max_dict['{}_max'.format(x_var)], 
                                            y_min=min_max_dict['{}_min'.format(var_name[1])],
                                            y_max=min_max_dict['{}_max'.format(var_name[1])], 
                                            color=color,
                                            ax=ax_list[(i*2)+1],
                                            figure=big_fig)
                                            
        # Make sure axis is in scientific format
        ax_list[(i*2)+1].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
                                        
        # Allign the y labels for each column    
        ax_list[(i*2)+1].yaxis.set_label_coords(-0.14, 0.5)

        # Update the font size for the labels
        # to be a little smaller
        for lab in [ ax_list[(i*2)+1].yaxis.label, ax_list[(i*2)+1].xaxis.label ]:
            f_size = lab.get_fontsize()
            lab.set_fontsize(f_size * 0.88)     

    # Place the A, B, C and i, ii, iii labels
    let_list = [ 'A', 'B', 'C', 'D' ]
    rom_list = [ 'i', 'ii', 'iii' ]
    
    '''
    # For the first column put the letters in the top left corner
    for i, ax in enumerate(ax_list[:,0]):
        ax.text(-0.1, 0.95, '{}i'.format(let_list[i]),
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    fontsize=40,
                    transform=ax.transAxes,
                    weight='bold')

    for i, ax in enumerate(ax_list[:,2]):
        ax.text(-0.2, 0.95, '{}ii'.format(let_list[i]),
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    fontsize=40,
                    transform=ax.transAxes,
                    weight='bold')
                    
    for i, ax in enumerate(ax_list[:,3]):
        ax.text(-0.2, 0.95, '{}iii'.format(let_list[i]),
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    fontsize=40,
                    transform=ax.transAxes,
                    weight='bold')
    
    '''        
    # Save the figure
    filename = os.path.join(figures_dir, 'Figure1.png')
    big_fig.savefig(filename, bbox_inches=0, dpi=120)
    
    plt.close()
    
    
def figure_2(measure_dict, figures_dir, results_dir, mpm='MT'):
    
    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)

    # Get the various min and max values:
    min_max_dict = get_min_max_values(measure_dict)
    axis_label_dict = get_axis_label_dict()
    
    # Create the big figure
    big_fig, big_ax = plt.subplots(figsize=(32, 20), facecolor='white')
    big_fig.subplots_adjust(left=0.05, right=0.98, bottom=0.07, top=0.95, wspace=0.05, hspace=0.05)
    
    #=========================================================================
    # We're going to set up a grid for the top row so we can 
    # adjust the spacings without screwing up the spacings in the bottom row
    
    grid = gridspec.GridSpec(1, 3)
    grid.update(left=0.15, bottom=0.56, top=0.96, wspace=0.2, hspace=0)
    top_ax_list = []
    for g_loc in grid:
        top_ax_list += [ plt.Subplot(big_fig, g_loc) ]
        big_fig.add_subplot(top_ax_list[-1])

    #=========================================================================
    # Schematic for how we measured the different layers
    f_name = os.path.join(figures_dir, '../..', 'CorticalLayers_schematic_methods.jpg')
    img = mpimg.imread(f_name)
    ax = top_ax_list[0]
    ax.set_position([0.01, 0.52, 0.4, 0.45])
    ax.imshow(img)
    ax.axis('off')
    
    #=========================================================================
    # Nodal CT MT
    figure_name = os.path.join(figures_dir, 
                                'Nodal_CT_corr_{}_projfrac+030_slope_age_at14.png'.format(mpm))
                                    
    marker_color_dict = get_von_economo_color_dict(measure_dict['von_economo'])
    marker_shapes_dict = get_von_economo_shapes_dict(measure_dict['von_economo'])
    marker_colors = [ marker_color_dict[ve] for ve in measure_dict['von_economo'] ]
    marker_shapes = [ marker_shapes_dict[ve] for ve in measure_dict['von_economo'] ]
    
    pretty_scatter(measure_dict['CT_all_slope_age_at14'], measure_dict['{}_projfrac+030_all_slope_age_at14'.format(mpm)], 
                    x_label=axis_label_dict['CT_all_slope_age_at14'],
                    y_label=axis_label_dict['{}_projfrac+030_all_slope_age_at14'.format(mpm)], 
                    x_min=min_max_dict['CT_all_slope_age_at14_min'],
                    x_max=min_max_dict['CT_all_slope_age_at14_max'],
                    y_min=min_max_dict['{}_projfrac+030_all_slope_age_at14_min'.format(mpm)],
                    y_max=min_max_dict['{}_projfrac+030_all_slope_age_at14_max'.format(mpm)], 
                    color='k',
                    marker='^',
                    figure_name=figure_name)

    top_ax_list[1] = pretty_scatter(measure_dict['CT_all_slope_age_at14'], 
                    measure_dict['{}_projfrac+030_all_slope_age_at14'.format(mpm)], 
                    x_label=axis_label_dict['CT_all_slope_age_at14'],
                    y_label=axis_label_dict['{}_projfrac+030_all_slope_age_at14'.format(mpm)], 
                    x_min=min_max_dict['CT_all_slope_age_at14_min'],
                    x_max=min_max_dict['CT_all_slope_age_at14_max'],
                    y_min=min_max_dict['{}_projfrac+030_all_slope_age_at14_min'.format(mpm)],
                    y_max=min_max_dict['{}_projfrac+030_all_slope_age_at14_max'.format(mpm)], 
                    color='k',
                    marker='^',
                    ax=top_ax_list[1],
                    figure=big_fig)    
                    
    # NODAL SLOPE CT WITH SLOPE MT
    figure_name = os.path.join(figures_dir, 
                                'Nodal_CT_corr_{}_projfrac+030_slope_age.png'.format(mpm))
                                    
    pretty_scatter(measure_dict['CT_all_slope_age'], measure_dict['{}_projfrac+030_all_slope_age'.format(mpm)], 
                    x_label=axis_label_dict['CT_all_slope_age'],
                    y_label=axis_label_dict['{}_projfrac+030_all_slope_age'.format(mpm)], 
                    x_min=min_max_dict['CT_all_slope_age_min'],
                    x_max=min_max_dict['CT_all_slope_age_max'],
                    y_min=min_max_dict['{}_projfrac+030_all_slope_age_min'.format(mpm)],
                    y_max=min_max_dict['{}_projfrac+030_all_slope_age_max'.format(mpm)],  
                    color='k',
                    marker='^',
                    figure_name=figure_name)

    top_ax_list[2] = pretty_scatter(measure_dict['CT_all_slope_age'], 
                    measure_dict['{}_projfrac+030_all_slope_age'.format(mpm)], 
                    x_label=axis_label_dict['CT_all_slope_age'],
                    y_label=axis_label_dict['{}_projfrac+030_all_slope_age'.format(mpm)], 
                    x_min=min_max_dict['CT_all_slope_age_min'],
                    x_max=min_max_dict['CT_all_slope_age_max'],
                    y_min=min_max_dict['{}_projfrac+030_all_slope_age_min'.format(mpm)],
                    y_max=min_max_dict['{}_projfrac+030_all_slope_age_max'.format(mpm)],  
                    marker='^',
                    color='k',
                    ax=top_ax_list[2],
                    figure=big_fig)    
    
    # Make sure the y labels are the same distance from the axis
    # for these two plots    
    for ax in top_ax_list:
        ax.yaxis.set_label_coords(-0.1, 0.5)

    #=========================================================================
    # We're going to set up two separate grids for the bottom row so we can 
    # adjust the spacings without screwing up the spacings in the top row
    violin_ax_list = []
    
    # First a space for the first violin plot on the far left
    grid = gridspec.GridSpec(1, 1)
    grid.update(left=0.12, right=0.37, top=0.47, bottom=0.1, wspace=0, hspace=0)
    for g_loc in grid:
        violin_ax_list += [ plt.Subplot(big_fig, g_loc) ]
        big_fig.add_subplot(violin_ax_list[-1])
        
    # Next two spaces for the remaining two
    grid = gridspec.GridSpec(1, 2)
    grid.update(left=0.49, top=0.47, bottom=0.1, wspace=0.08, hspace=0)
    for g_loc in grid:
        violin_ax_list += [ plt.Subplot(big_fig, g_loc) ]
        big_fig.add_subplot(violin_ax_list[-1])
    
    #=========================================================================
    # Schematic for the different cytoarchitectonics for each layer
    big_fig = add_cells_picture(figures_dir, big_fig)
    
    #=========================================================================
    # MEAN MT ACROSS NODES at different depths

    figure_name = os.path.join(figures_dir, 
                                '{}_all_mean_DifferentDepths.png'.format(mpm))
    
    violin_mt_depths(measure_dict,
                        measure='all_mean',
                        y_label=axis_label_dict['{}_all_mean'.format(mpm)],
                        cmap='jet',
                        y_min=min_max_dict['{}_all_mean_min'.format(mpm)],
                        y_max=min_max_dict['{}_all_mean_max'.format(mpm)], 
                        cmap_min=min_max_dict['{}_all_mean_min'.format(mpm)],
                        cmap_max=min_max_dict['{}_all_mean_max'.format(mpm)],
                        figure_name=figure_name,
                        mpm=mpm,
                        vert=False,
                        cbar=True)

    violin_ax_list[0] = violin_mt_depths(measure_dict,
                        measure='all_mean',
                        y_label=axis_label_dict['{}_all_mean'.format(mpm)],
                        cmap='jet',
                        y_min=min_max_dict['{}_all_mean_min'.format(mpm)],
                        y_max=min_max_dict['{}_all_mean_max'.format(mpm)], 
                        cmap_min=min_max_dict['{}_all_mean_min'.format(mpm)],
                        cmap_max=min_max_dict['{}_all_mean_max'.format(mpm)],
                        lam_labels=False,
                        ax=violin_ax_list[0],
                        figure=big_fig,
                        mpm=mpm,
                        vert=False,
                        cbar=True)
                        
    violin_ax_list[0].set_ylabel('Summary across 308 cortical regions\nat increasing distances from pial surface',
                                    fontsize=30)
        
    # CORR WITH AGE ACROSS NODES at different depths
    figure_name = os.path.join(figures_dir, 
                                    '{}_all_slope_age_DifferentDepths.png'.format(mpm))
    
    violin_mt_depths(measure_dict,
                        measure='all_slope_age',
                        y_label=axis_label_dict['{}_all_slope_age'.format(mpm)],
                        cmap='RdBu_r',
                        y_min=min_max_dict['{}_all_slope_age_min'.format(mpm)], 
                        y_max=min_max_dict['{}_all_slope_age_max'.format(mpm)], 
                        cmap_min=min_max_dict['{}_all_slope_age_max'.format(mpm)]*-1/2.0, 
                        cmap_max=min_max_dict['{}_all_slope_age_max'.format(mpm)]/2.0,
                        figure_name=figure_name,
                        mpm=mpm,
                        vert=False,
                        cbar=True)

    violin_ax_list[1] = violin_mt_depths(measure_dict,
                        measure='all_slope_age',
                        y_label=axis_label_dict['{}_all_slope_age'.format(mpm)],
                        cmap='RdBu_r',
                        y_min=min_max_dict['{}_all_slope_age_min'.format(mpm)], 
                        y_max=min_max_dict['{}_all_slope_age_max'.format(mpm)], 
                        cmap_min=min_max_dict['{}_all_slope_age_max'.format(mpm)]*-1/2.0, 
                        cmap_max=min_max_dict['{}_all_slope_age_max'.format(mpm)]/2.0,
                        ax=violin_ax_list[1],
                        figure=big_fig,
                        lam_labels=False,
                        mpm=mpm,
                        vert=False,
                        cbar=True)
                         
    # CORR WITH CT ACROSS NODES at different depths
    figure_name = os.path.join(figures_dir, 
                                    '{}_all_slope_CT_DifferentDepths.png'.format(mpm))
    
    violin_mt_depths(measure_dict,
                        measure='all_slope_ct',
                        y_label=axis_label_dict['{}_all_slope_ct'.format(mpm)],
                        cmap='PRGn',
                        y_min=min_max_dict['{}_all_slope_ct_min'.format(mpm)], 
                        y_max=min_max_dict['{}_all_slope_ct_max'.format(mpm)], 
                        cmap_min=min_max_dict['{}_all_slope_ct_max'.format(mpm)]*-1, 
                        cmap_max=min_max_dict['{}_all_slope_ct_max'.format(mpm)],
                        figure_name=figure_name,
                        mpm=mpm,
                        vert=False,
                        cbar=True)
                        
    violin_ax_list[2] = violin_mt_depths(measure_dict,
                        measure='all_slope_ct',
                        y_label=axis_label_dict['{}_all_slope_ct'.format(mpm)],
                        cmap='PRGn',
                        y_min=min_max_dict['{}_all_slope_ct_min'.format(mpm)], 
                        y_max=min_max_dict['{}_all_slope_ct_max'.format(mpm)], 
                        cmap_min=min_max_dict['{}_all_slope_ct_max'.format(mpm)]*-1, 
                        cmap_max=min_max_dict['{}_all_slope_ct_max'.format(mpm)],
                        lam_labels=False,                        
                        ax=violin_ax_list[2],
                        figure=big_fig,
                        mpm=mpm,
                        vert=False,
                        cbar=True)
                        

    # Turn off the axes for the first columns
    for ax in [ big_ax, top_ax_list[0] ]:
        ax.axis('off')
    
    # Also remove the y tick labels for the violin plots
    # that are not the first
    for ax in violin_ax_list[1:]:
        ax.set_yticklabels([])
    
    # Place the A, B, C and i, ii, iii labels
    let_list = [ 'A', 'B', 'C' ]
    rom_list = [ 'i', 'ii', 'iii', 'iv' ]
    
    # For the first column put the letters in the top left corner
    for i, ax in enumerate(top_ax_list):
        x_pos = ax.get_position().x0
        
        if i == 0:
            big_fig.text(0.04, 0.965, '{}'.format(let_list[i]),
                        horizontalalignment='left',
                        verticalalignment='top',
                        fontsize=40,
                        weight='bold',
                        color='white')
        else:
            big_fig.text(x_pos-0.03, 0.965, '{}'.format(let_list[i]),
                        horizontalalignment='right',
                        verticalalignment='top',
                        fontsize=40,
                        weight='bold',
                        color='k')
     
    # Add in the figure panel number for the first violin plot
    big_fig.text(0.04, 0.5, 'Di'.format(rom_list[i+1]),
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=40,
                weight='bold',
                color='k')
                    
    # Add in the figure panel number for the cells schematic
    big_fig.text(0.37, 0.5, 'Dii',
                    horizontalalignment='left',
                    verticalalignment='top',
                    fontsize=40,
                    weight='bold',
                    color='k')
                
    # And the figure panel numbers for the second two violin plots
    for i, ax in enumerate(violin_ax_list[1:]):
        x_pos = ax.get_position().x0
        
        big_fig.text(x_pos, 0.5, 'D{}'.format(rom_list[i+2]),
                    horizontalalignment='right',
                    verticalalignment='top',
                    fontsize=40,
                    weight='bold',
                    color='k')
                    
    # Save the figure
    filename = os.path.join(figures_dir, 'Figure2.png')
    big_fig.savefig(filename, bbox_inches=0, dpi=100)
    
    plt.close()
    
def figure_3(measure_dict, figures_dir, results_dir, mpm='MT'):
    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)

    # Get the set values
    min_max_dict = get_min_max_values(measure_dict)
    axis_label_dict = get_axis_label_dict()

    # Create the big figure
    big_fig, big_ax = plt.subplots(figsize=(30, 39), facecolor='white')
    big_ax.axis('off')
    # Set the list of network measures we care about for this figure
    network_measures_list = [ 'Degree', 'Closeness', 'AverageDist' ]
    
    # Loop through each network measure
    for i, network_measure in enumerate(network_measures_list):
    
        #=========================================================================
        # Put the four hor brains in one row
        #=========================================================================
        f_list = [ os.path.join(results_dir, 'PNGS',
                                    '{}_CT_covar_ones_all_COST_10_lh_pial_lateral.png'.format(network_measure)),
                    os.path.join(results_dir, 'PNGS', 
                                    '{}_CT_covar_ones_all_COST_10_lh_pial_medial.png'.format(network_measure)),
                    os.path.join(results_dir, 'PNGS', 
                                    '{}_CT_covar_ones_all_COST_10_rh_pial_medial.png'.format(network_measure)),
                    os.path.join(results_dir, 'PNGS', 
                                    '{}_CT_covar_ones_all_COST_10_rh_pial_lateral.png'.format(network_measure)) ]

        grid = gridspec.GridSpec(1, 4)
        bottom = (1 - ((5*i)+2)/ 15.0 ) -0.01
        top = (1 - (5*i)/15.0) -0.01
        grid.update(left=0, right=1, bottom=bottom, top=top, wspace=0, hspace=0)

        big_fig = add_four_hor_brains(grid, f_list, big_fig)
        
        #=========================================================================
        # Next put von economo box plots for this graph measure in the 
        # first column of the next row
        #=========================================================================
        network_measure_min = min_max_dict['{}_CT_covar_ones_all_COST_10_min'.format(network_measure)]
        network_measure_max = min_max_dict['{}_CT_covar_ones_all_COST_10_max'.format(network_measure)]
        y_label = axis_label_dict[network_measure]
        
        grid = gridspec.GridSpec(1, 3)
        bottom = ( 1 - ((5*i)+5)/ 15.0 ) + 0.03
        top = ( 1 - ((5*i)+2)/15.0 ) - 0.03
        grid.update(bottom=bottom, top=top, left=0.05, right=0.98, hspace=0.1)
        
        ax_list = []
        for g_loc in grid:
            ax = plt.Subplot(big_fig, g_loc)
            big_fig.add_subplot(ax)
            ax_list += [ax]
    
        ax_list[0] = von_economo_boxes(measure_dict, figures_dir, 
                                            measure_dict['von_economo'], 
                                            measure='{}_CT_covar_ones_all_COST_10'.format(network_measure),
                                            y_label=y_label, 
                                            y_min=network_measure_min, 
                                            y_max=network_measure_max, 
                                            max_color='red',
                                            min_color='blue',
                                            alpha=0,
                                            ax=ax_list[0],
                                            figure=big_fig)
                                            
        #=========================================================================
        # Finally put scatter plots of deltaCT, and deltaMT by the network
        # measure in the second and third spaces on the next row
        #=========================================================================
    
        measure_list = [ 'CT_all_slope_age',
                         '{}_projfrac+030_all_slope_age'.format(mpm) ]
                         
        for j, measure in enumerate(measure_list):

            # Get the appropriate min, max and label values
            # for the y axis
            measure_min = min_max_dict['{}_min'.format(measure)]
            measure_max = min_max_dict['{}_max'.format(measure)]
            y_label = axis_label_dict[measure]
            x_label = axis_label_dict[network_measure]
            
            ax_list[j+1] = pretty_scatter(measure_dict['{}_CT_covar_ones_all_COST_10'.format(network_measure)],
                                                measure_dict[measure], 
                                                x_label=x_label,
                                                y_label=y_label, 
                                                x_min=network_measure_min, x_max=network_measure_max,
                                                y_min=measure_min,y_max=measure_max, 
                                                color='k',
                                                marker='^',
                                                ax=ax_list[j+1],
                                                figure=big_fig)    
                    
    #=========================================================================
    # Add in the letters for each panel
    #=========================================================================
    # Place the A, B, C and i, ii, iii labels
    let_list = [ 'A', 'B', 'C' ]
    rom_list = [ 'i', 'ii', 'iii', 'iv' ]

    # Get all the axes
    ax_list = big_fig.get_axes()
    
    # Put the right letter and the first roman numeral
    # next to each row of brains
    for i, ax in enumerate(ax_list[1::7]):
        x_pos = ax.get_position().x0 + 0.01
        y_pos = ax.get_position().y1
        big_fig.text(x_pos, y_pos, 
                    '{}i'.format(let_list[i]),
                    horizontalalignment='left',
                    verticalalignment='top',
                    fontsize=40,
                    weight='bold',
                    color='k')
        # For the same letter put ii, iii and iv next to each
        # of the plots
        for j in range(3):
            ax = ax_list[5 + i*7 + j]
            x_pos = ax.get_position().x0 - 0.04
            y_pos = ax.get_position().y1
            big_fig.text(x_pos, y_pos, 
                        '{}{}'.format(let_list[i], rom_list[j+1]),
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        fontsize=40,
                        weight='bold',
                        color='k')

    #=========================================================================
    # And finally clean everything up and save the figure
    #=========================================================================
    # Save the figure
    filename = os.path.join(figures_dir, 'Figure3.png')
    big_fig.savefig(filename, bbox_inches=0, dpi=100)

    plt.close()
    
def figure_3_separate(measure_dict, figures_dir, results_dir, mpm='MT', network_measure='Degree'):
    
    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)

    # Get the set values
    min_max_dict = get_min_max_values(measure_dict)
    axis_label_dict = get_axis_label_dict()
    
    x_label = axis_label_dict[network_measure]
    
    # Create the big figure
    big_fig, ax_list = plt.subplots(3,4, figsize=(40, 24), facecolor='white')
    plt.subplots_adjust(hspace=0.5)
    
    #=========================================================================
    # Start by putting the high degree nodes in the top row
    #=========================================================================
    f_list = [ os.path.join(results_dir, 'PNGS',
                                '{}_CT_covar_ones_all_COST_10_lh_pial_lateral.png'.format(network_measure)),
                os.path.join(results_dir, 'PNGS', 
                                '{}_CT_covar_ones_all_COST_10_lh_pial_medial.png'.format(network_measure)),
                os.path.join(results_dir, 'PNGS', 
                                '{}_CT_covar_ones_all_COST_10_rh_pial_medial.png'.format(network_measure)),
                os.path.join(results_dir, 'PNGS', 
                                '{}_CT_covar_ones_all_COST_10_rh_pial_lateral.png'.format(network_measure)) ]

    grid = gridspec.GridSpec(1, 4)
    grid.update(left=0, right=1, bottom=0.66, top=1, wspace=0, hspace=0)

    big_fig = add_four_hor_brains(grid, f_list, big_fig)

    #=========================================================================
    # Next put scatter plots of CT, deltaCT, MT and deltaMT by the network
    # measure you've chosen
    #=========================================================================
    network_measure_min = min_max_dict['{}_CT_covar_ones_all_COST_10_min'.format(network_measure)]
    network_measure_max = min_max_dict['{}_CT_covar_ones_all_COST_10_max'.format(network_measure)]
    
    measure_list = [ 'CT_all_slope_age_at14',
                     'CT_all_slope_age',
                     '{}_projfrac+030_all_slope_age_at14'.format(mpm),
                     '{}_projfrac+030_all_slope_age'.format(mpm) ]
                         
    for i, measure in enumerate(measure_list):

        # Get the appropriate min, max and label values
        # for the y axis
        measure_min = min_max_dict['{}_min'.format(measure)]
        measure_max = min_max_dict['{}_max'.format(measure)]
        y_label = axis_label_dict[measure]
        
        figure_name = os.path.join(figures_dir, 
                                    '{}_corr_{}.png'.format(network_measure, measure))
                                    
        pretty_scatter(measure_dict['{}_CT_covar_ones_all_COST_10'.format(network_measure)],
                        measure_dict[measure], 
                        x_label=x_label,
                        y_label=y_label, 
                        x_min=network_measure_min, x_max=network_measure_max,
                        y_min=measure_min,y_max=measure_max, 
                        color='k',
                        marker='^',
                        figure_name=figure_name)
    
        ax_list[1,i] = pretty_scatter(measure_dict['{}_CT_covar_ones_all_COST_10'.format(network_measure)],
                                        measure_dict[measure], 
                                        x_label=x_label,
                                        y_label=y_label, 
                                        x_min=network_measure_min, x_max=network_measure_max,
                                        y_min=measure_min,y_max=measure_max, 
                                        color='k',
                                        marker='^',
                                        ax=ax_list[1, i],
                                        figure=big_fig)    

    #=========================================================================
    # Next put von economo box plots of four different graph measures
    # split up by von economo type
    #=========================================================================
    measure_list = [ 'Degree', 'PC', 'AverageDist', 'Closeness' ]
    
    for i, measure in enumerate(measure_list):
        
        measure_min = min_max_dict['{}_CT_covar_ones_all_COST_10_min'.format(measure)]
        measure_max = min_max_dict['{}_CT_covar_ones_all_COST_10_max'.format(measure)]
        y_label = axis_label_dict[measure]
        
        figure_name = os.path.join(figures_dir,
                        'VonEconomo_{}_CT_covar_ones_all_COST_10.png'.format(measure))
        
        von_economo_boxes(measure_dict, figures_dir, 
                            measure_dict['von_economo'], 
                            measure='{}_CT_covar_ones_all_COST_10'.format(measure),
                            y_label=y_label, 
                            y_min=measure_min, y_max=measure_max, 
                            max_color='red',
                            min_color='blue',
                            alpha=0,
                            figure_name=figure_name)
        
        ax_list[2, i] = von_economo_boxes(measure_dict, figures_dir, 
                                            measure_dict['von_economo'], 
                                            measure='{}_CT_covar_ones_all_COST_10'.format(measure),
                                            y_label=y_label, 
                                            y_min=measure_min, y_max=measure_max, 
                                            max_color='red',
                                            min_color='blue',
                                            alpha=0,
                                            ax=ax_list[2, i],
                                            figure=big_fig)
                                    
    #=========================================================================
    # And finally clean everything up and save the figure
    #=========================================================================
    # Turn off the axes for the first row
    for ax in ax_list[0,:].reshape(-1):
        ax.axis('off')
    
    # Nice tight layout
    big_fig.tight_layout()

    # Place the A, B, C and i, ii, iii labels
    let_list = [ 'A', 'B', 'C' ]
    rom_list = [ 'i', 'ii', 'iii', 'iv' ]

    # Put A in the top left corner
    x_pos = ax_list[0,0].get_position().x0
    y_pos = ax_list[0,0].get_position().y1
    big_fig.text(x_pos-0.015, y_pos-0.02, 'A',
                horizontalalignment='right',
                verticalalignment='top',
                fontsize=40,
                weight='bold',
                color='k')

    # And the figure panel numbers for the second and third rows
    for i, ax in enumerate(ax_list[1:,:].reshape(-1)):
        x_pos = ax.get_position().x0
        y_pos = ax.get_position().y1
        
        big_fig.text(x_pos-0.015, y_pos+0.03, '{}{}'.format(let_list[(i/4)+1], rom_list[i%4]),
                    horizontalalignment='right',
                    verticalalignment='top',
                    fontsize=40,
                    weight='bold',
                    color='k')

    # Save the figure
    filename = os.path.join(figures_dir, 'Figure3_{}.png'.format(network_measure))
    big_fig.savefig(filename, bbox_inches=0, dpi=100)

    plt.close()

    
def figure_4(measure_dict, graph_dict, figures_dir, results_dir, mpm='MT'):
        
    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)

    # Get the set values
    min_max_dict = get_min_max_values(measure_dict)
    axis_label_dict = get_axis_label_dict()
    
    # Create the big figure
    big_fig, ax_list = plt.subplots(5,4, figsize=(30, 40), facecolor='white')

    #=========================================================================
    # The first row should be two sagittal views of the network in an 
    # anatomical layout - column 1: anatomical module and column2: von economo
    # classes
    #=========================================================================
    G = graph_dict['CT_covar_ones_all_COST_10']
    G_02 = graph_dict['CT_covar_ones_all_COST_02']
    
    rich_edges, rich_nodes = rich_edges_nodes(G, thresh=85)
    
    grid = gridspec.GridSpec(1, 2)
    grid.update(left=0.03, right=0.97, top=0.97, bottom=0.7, wspace=0.02, hspace=0)
    ax = plt.Subplot(big_fig, grid[0])
    big_fig.add_subplot(ax)
    
    ax = plot_anatomical_network(G_02, measure_dict, 
                                    measure='renum_module',
                                    orientation='sagittal',
                                    edge_width=1,
                                    cmap_name='jet_r',
                                    node_size=800,
                                    ax=ax)
    ax = plot_anatomical_network(G_02, measure_dict, 
                                    measure='renum_module',
                                    orientation='sagittal',
                                    cmap_name='jet_r',
                                    edge_list = [], 
                                    node_list = rich_nodes,
                                    node_shape='s', 
                                    node_size=1000,
                                    ax=ax)
                                   
    ax = plt.Subplot(big_fig, grid[1])
    big_fig.add_subplot(ax)
    
    ax = plot_anatomical_network(G_02, measure_dict, 
                                    measure='von_economo',
                                    orientation='sagittal',
                                    cmap_name='von_economo',
                                    edge_width=0.5,
                                    node_size=800,
                                    ax=ax)
    ax = plot_anatomical_network(G_02, measure_dict, 
                                    measure='von_economo',
                                    orientation='sagittal',
                                    cmap_name='von_economo',
                                    edge_list = [], 
                                    node_list = rich_nodes,
                                    node_size=1000,
                                    node_shape='s', 
                                    ax=ax)
    
    #=========================================================================
    # The middle row is of the circular layout and a box plot of 
    # von economo value by module
    #=========================================================================
    grid = gridspec.GridSpec(1, 2)
    grid.update(left=0.03, right=0.97, top=0.7, bottom=0.45, wspace=0.02, hspace=0)
    ax = plt.Subplot(big_fig, grid[0])
    big_fig.add_subplot(ax)
    
    ax = plot_circular_network(G, measure_dict, 
                                    sort_measure='renum_module',
                                    sort_cmap_name='jet_r',
                                    show_wedge=False,
                                    edge_list = rich_edges,
                                    node_size=1000,
                                    ax=ax)
    
    #=========================================================================
    # Put box plots of CT, deltaCT, MT and deltaMT and von economo class 
    # split up by the network modules
    #=========================================================================    
    measure_list = [ 'von_economo',
                     'CT_all_slope_age_at14',
                     'CT_all_slope_age',
                     '{}_projfrac+030_all_slope_age_at14'.format(mpm),
                     '{}_projfrac+030_all_slope_age'.format(mpm) ]

    for i, measure in enumerate(measure_list):
        
        measure_min = min_max_dict['{}_min'.format(measure)]
        measure_max = min_max_dict['{}_max'.format(measure)]
        
        y_label = axis_label_dict[measure]        
        
        if i == 0:
            ax = ax_list[2,2]
        else:
            ax = ax_list[3,i-1]
        
        ax = von_economo_boxes(measure_dict, figures_dir, 
                                            measure_dict['Renumbered_Module_CT_covar_ones_all_COST_10'], 
                                            measure=measure,
                                            y_label=y_label, 
                                            group_label='Module',
                                            y_min=measure_min, y_max=measure_max, 
                                            von_economo_colors=False,
                                            cmap_name='jet_r',
                                            alpha=1.0,
                                            ax=ax,
                                            figure=big_fig)
    
    #=========================================================================
    # Put box plots of CT, deltaCT, MT and deltaMT and von economo class 
    # split up by the rich club nodes
    #=========================================================================    
    deg = np.copy(measure_dict['Degree_CT_covar_ones_all_COST_10'])
    hub_thresh = np.percentile(deg, 85)
    rc = np.zeros_like(deg)
    rc[deg>hub_thresh] = 1
    
    for i, measure in enumerate(measure_list):
        
        measure_min = min_max_dict['{}_min'.format(measure)]
        measure_max = min_max_dict['{}_max'.format(measure)]
        
        if i == 0:
            ax = ax_list[2,3]
        else:
            ax = ax_list[4,i-1]
        
        y_label = axis_label_dict[measure]        
        
        if (i == 1) or (i == 4):
            max_color = 'red'
            min_color = 'blue'
        else:
            max_color = 'blue'
            min_color = 'red'
            
        ax = von_economo_boxes(measure_dict, figures_dir, 
                                            rc, 
                                            measure=measure,
                                            y_label=y_label, 
                                            group_label='Rich Club Member',
                                            y_min=measure_min, y_max=measure_max, 
                                            von_economo_colors=False,
                                            cmap_name='coolwarm',
                                            alpha=0,
                                            max_color=max_color,
                                            min_color=min_color,
                                            ax=ax,
                                            figure=big_fig)
                                            
    #=========================================================================
    # Clean everything up and save the figure
    #=========================================================================
    # Turn off the axes you aren't using
    for ax in ax_list.reshape(-1)[:-10]:
        ax.axis('off')
        
    # Nice tight layout
    big_fig.tight_layout()

    # Save the figure
    filename = os.path.join(figures_dir, 'New_Figure4.png')
    big_fig.savefig(filename, bbox_inches=0, dpi=100)

    plt.close()
    
def calc_min_max(x, pad=0.05):
    '''
    Find min and max values such that
    all the data lies within 90% of
    of the axis range
    '''
    try:
        r = np.max(x) - np.min(x)
        if r > 0:
            x_min = np.min(x) - pad * r
            x_max = np.max(x) + pad * r
        else:
            x_min = np.mean(x)
            x_max = np.mean(x)
    except:
        x_min = np.nan
        x_max = np.nan
    return x_min, x_max
    
def get_min_max_values(measure_dict):
    '''
    These are the appropriate min and max values for the 
    discovery cohort
    '''
    
    min_max_dict = {}
        
    for measure_name, measure_data in measure_dict.items():
        measure_min, measure_max = calc_min_max(measure_data, pad=0.05)
        min_max_dict['{}_min'.format(measure_name)] = measure_min
        min_max_dict['{}_max'.format(measure_name)] = measure_max
    
    min_max_dict['CT_all_slope_age_CBAR_min'] = -0.035
    min_max_dict['CT_all_slope_age_CBAR_max'] = -0.015
    min_max_dict['CT_all_slope_age_at14_CBAR_min'] = 2.5
    min_max_dict['CT_all_slope_age_at14_CBAR_max'] = 3.5
    min_max_dict['MT_projfrac+030_all_slope_age_CBAR_min'] = 0.005
    min_max_dict['MT_projfrac+030_all_slope_age_CBAR_max'] = 0.01
    min_max_dict['MT_projfrac+030_all_slope_age_at14_CBAR_min'] = 0.8
    min_max_dict['MT_projfrac+030_all_slope_age_at14_CBAR_max'] = 1.0
    min_max_dict['MT_all_mean_min'] = 0.4
    min_max_dict['MT_all_mean_max'] = 1.8 
    min_max_dict['MT_all_slope_age_min'] = -0.01
    min_max_dict['MT_all_slope_age_max'] = 0.016
    min_max_dict['MT_all_slope_ct_min'] = -5.5
    min_max_dict['MT_all_slope_ct_max'] = 2.2 
    
    '''
    # Set the various min and max values:
    min_max_dict['age_min'] = 14
    min_max_dict['age_max'] = 25
    min_max_dict['CT_global_mean_min'] = 2.4
    min_max_dict['CT_global_mean_max'] = 3.0
    min_max_dict['CT_all_slope_age_at14_min'] = 1.9
    min_max_dict['CT_all_slope_age_at14_max'] = 4.0    
    min_max_dict['CT_all_slope_age_min'] = -0.055
    min_max_dict['CT_all_slope_age_max'] = 0.015
    min_max_dict['MT_projfrac+030_global_mean_min'] = 0.8
    min_max_dict['MT_projfrac+030_global_mean_max'] = 1.05
    min_max_dict['MT_projfrac+030_all_slope_age_at14_min'] = 0.75
    min_max_dict['MT_projfrac+030_all_slope_age_at14_max'] = 1.1
    min_max_dict['MT_projfrac+030_all_slope_age_min'] = -0.004
    min_max_dict['MT_projfrac+030_all_slope_age_max'] = 0.016
    min_max_dict['Degree_min'] = 0
    min_max_dict['Degree_max'] = 110
    min_max_dict['PC_min'] = 0
    min_max_dict['PC_max'] = 1
    min_max_dict['AverageDist_min'] = 10
    min_max_dict['AverageDist_max'] = 120
    min_max_dict['Clustering_min'] = 0
    min_max_dict['Clustering_max'] = 1  
    min_max_dict['Closeness_min'] = 0.19
    min_max_dict['Closeness_max'] = 0.58
    min_max_dict['von_economo_min'] = 0.5
    min_max_dict['von_economo_max'] = 5.5
    min_max_dict['InterhemProp_min'] = 0.0
    min_max_dict['InterhemProp_max'] = 1.0
    
    '''
    return min_max_dict
    
def get_axis_label_dict():

    axis_label_dict = {}
    
    axis_label_dict['Degree'] = 'Degree'
    axis_label_dict['von_economo'] = 'Cortical Lamination Pattern'
    axis_label_dict['PC'] = 'Participation Coefficient'
    axis_label_dict['AverageDist'] = 'Average Distance (mm)'
    axis_label_dict['Clustering'] = 'Clustering'
    axis_label_dict['Closeness'] = 'Closeness'
    axis_label_dict['InterhemProp'] = 'Interhemispheric Connections'
    axis_label_dict['CT_all_slope_age_at14'] = 'CT at 14 yrs (mm)'
    axis_label_dict['CT_all_slope_age'] =  r'$\Delta$CT (mm/year)'
    axis_label_dict['MT_projfrac+030_all_slope_age_at14'] = 'MT at 14 yrs (AU)'
    axis_label_dict['MT_projfrac+030_all_slope_age'] = r'$\Delta$MT (AU/year)'
    axis_label_dict['age_scan'] = 'Age (years)'
    axis_label_dict['CT_global_mean'] = 'Global CT (mm)'
    axis_label_dict['MT_projfrac+030_global_mean'] = 'Global MT (AU)'
    axis_label_dict['MT_all_mean'] = 'Mean MT across regions (AU)'
    axis_label_dict['MT_all_slope_ct'] = 'Correlation MT vs CT (AU/mm)'
    axis_label_dict['MT_all_slope_age'] = r'$\Delta$MT with age (AU/year)'
    
    return axis_label_dict
    
    
def corr_by_agebin(measure_dict_dict, paper_dir, x_measure='Degree_CT_covar_ones_all_COST_10', y_measure='CT_all_slope_age', ax=None, fig=None, label=None):

    y = np.array(measure_dict_dict['COMPLETE_EXCLBAD'][y_measure])
    
    m_array = np.zeros(5)
    r_array = np.zeros(5)
    p_array = np.zeros(5)
    
    for i, age_bin in enumerate(range(1,6)):
        cohort = 'AGE_BIN_{}_EXCLBAD'.format(age_bin)
        print cohort
        measure_dict = measure_dict_dict[cohort]
        x = np.array(measure_dict[x_measure])
        
        m,c,r,p,sterr,p_perm = permutation_correlation(x, y)
        m_array[i] = m
        r_array[i] = r
        p_array[i] = p
        
    if not ax:
        fig, ax = plt.subplots()

    ax.plot(range(1,6), m_array, c='b')    
    ax.scatter(range(1,6), m_array, s=70, c='b')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    if label:
        ax.set_ylabel(label)
    ax.set_xticklabels(['', '14-15', '16-17', '18-19', '20-21', '22-24'], rotation=45)

    sns.despine()
    
    return ax
        

def get_circular_layout(G, df):

    # Create two empty dictionaries for the
    # positions and the normal angle to each
    # position (in degrees)
    pos_dict = {}
    theta_dict = {}
    
    # Make a list of theta values that
    # start at 90 and go round the circle
    # in a clockwise direction
    theta_list = [ t%360 for t in np.arange(450, 90, -360.0/len(df['node'])) ]

    # And then fill in those dictionaries!
    for i, key in enumerate(df['node'].values):
        theta = theta_list[i] * np.pi / 180.0
        pos_dict[key] = np.array([np.cos(theta)*0.5, np.sin(theta)*0.5])
        theta_dict[key] = theta_list[i]
    
    return pos_dict, theta_dict
    
def setup_color_list(df, cmap_name='jet', measure='module'):
    '''
    Use a colormap to set colors for each value in the 
    sort_measure and return a list of colors for each node
    '''
    
    colors_dict = {}
    
    n = np.float(len(set(df[measure])))
    
    if type(cmap_name) is str:
        cmap = plt.get_cmap(cmap_name)
    else:
        cmap = cmap_name
        
    for i, mod in enumerate(sorted(set(df[measure]))):
        colors_dict[mod] = cmap((i+0.5)/n)

    colors_list = [ colors_dict[mod] for mod in df[measure].values ]

    return colors_list
    
def plot_circular_network(G, measure_dict, sort_measure='module', wedge_measure='von_economo', sort_cmap_name='jet_r', wedge_cmap_name='von_economo', node_size=500, edge_list=None, edge_color='k', edge_width=0.2, figure=None, ax=None, show_wedge=False):

    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)
    
    if not edge_list:
        edge_list = G.edges()
        
    # Put the measures you care about together
    # in a data frame
    df = pd.DataFrame({ 'degree' : measure_dict['Degree_CT_covar_ones_all_COST_10'] ,
                        'module' : measure_dict['Module_CT_covar_ones_all_COST_10'],
                        'renum_module' : measure_dict['Renumbered_Module_CT_covar_ones_all_COST_10'],
                        'von_economo' : measure_dict['von_economo'],
                        'lobes' : measure_dict['lobes'],
                        'x' : measure_dict['centroids'][:,0],
                        'y' : measure_dict['centroids'][:,1],
                        'z' : measure_dict['centroids'][:,2]})
    df['node'] = range(len(df['degree']))
    
    # First get the module and wedge color lists in node order
    # (This has to be done before you sort the data frame)
    von_economo_colors = get_von_economo_color_dict(measure_dict['von_economo'])
    if sort_cmap_name == 'von_economo':
        sort_cmap_name =  mpl.colors.ListedColormap(von_economo_colors.values())
    if wedge_cmap_name == 'von_economo':
        wedge_cmap_name =  mpl.colors.ListedColormap(von_economo_colors.values())
    
    node_colors_list = setup_color_list(df, cmap_name=sort_cmap_name, measure=sort_measure)
    wedge_colors_list = setup_color_list(df, cmap_name=wedge_cmap_name, measure=wedge_measure)
    
    # Now sort the df by the measure you care about
    df.sort(columns=[sort_measure, wedge_measure, 'node'], inplace=True)
    
    # Get the positions of node and the normal angle to each position
    pos_dict, theta_dict = get_circular_layout(G, df)
    
    # If you've given this code an axis and figure then use those
    # otherwise just create your own
    if not ax:
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = figure
    
    nx.draw_networkx(G, 
                    pos=pos_dict, 
                    node_color=node_colors_list, 
                    node_size=node_size,
                    edgelist=edge_list,
                    width=edge_width,
                    edge_color = edge_color,
                    with_labels=False, 
                    ax=ax)
    
    if show_wedge:
        ax = add_wedge(df, theta_dict, wedge_colors_list, wedge_measure=wedge_measure, ax=ax)
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-0.75, 0.75)
    else:
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
    ax.axis('off')

    return ax
    
def add_wedge(df, theta_dict, wedge_colors_list, wedge_measure='von_economo', ax=None):

    theta_adj = 360.0/(2*len(df['node']))
    
    df.sort(['node'], inplace=True)
    
    for node in df['node'].values:
        wedge = mpatches.Wedge((0,0), 
                                r = 0.65, width = 0.1,
                                theta1=theta_dict[node]-theta_adj,
                                theta2=theta_dict[node]+theta_adj,
                                facecolor=wedge_colors_list[node],
                                edgecolor='none')
        ax.add_patch(wedge)
        
    return ax
    
def plot_anatomical_network(G, measure_dict, measure='module', orientation='sagittal', cmap_name='jet_r', edge_list=None, edge_color='k', edge_width=0.2, node_list=None, node_shape='o', node_size=500, figure=None, ax=None):
    
    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)
    
    if edge_list is None:
        edge_list = list(G.edges())
        
    if node_list is None:
        node_list = G.nodes()
        node_list = sorted(node_list)
        
    # Put the measures you care about together
    # in a data frame
    df = pd.DataFrame({ 'degree' : measure_dict['Degree_CT_covar_ones_all_COST_10'] ,
                        'module' : measure_dict['Module_CT_covar_ones_all_COST_10'],
                        'renum_module' : measure_dict['Renumbered_Module_CT_covar_ones_all_COST_10'],
                        'von_economo' : measure_dict['von_economo'],
                        'lobes' : measure_dict['lobes'],
                        'x' : measure_dict['centroids'][:,0],
                        'y' : measure_dict['centroids'][:,1],
                        'z' : measure_dict['centroids'][:,2]})
    df['node'] = range(len(df['degree']))
            
    # Get the module and wedge color lists
    # (This has to be done before you sort the data frame)
    von_economo_colors = get_von_economo_color_dict(measure_dict['von_economo'])
    if cmap_name == 'von_economo':
        cmap_name =  mpl.colors.ListedColormap(von_economo_colors.values())

    colors_list = setup_color_list(df, cmap_name=cmap_name, measure=measure)
    
    # Eliminate all nodes that aren't in nodelist
    df = df.loc[df['node'].isin(node_list)]

    # Sort by x so that the nodes are plotted in a sensible order
    sort_dict = {}
    sort_dict['axial'] = 'z'
    sort_dict['coronal'] = 'y'
    sort_dict['sagittal'] = 'x'
    
    df.sort(columns=[sort_dict[orientation]], inplace=True)
        
    # Get the positions of nodes
    pos_dict = {}
    pos_dict['axial'], pos_dict['sagittal'], pos_dict['coronal'] = get_anatomical_layouts(G, df)
    pos = pos_dict[orientation]
    
    # If you've given this code an axis and figure then use those
    # otherwise just create your own
    if not ax:
        # Create a figure
        fig_size_dict = {}
        fig_size_dict['axial'] = (9,12)
        fig_size_dict['sagittal'] = (12,8)
        fig_size_dict['coronal'] = (9,8)
        
        fig, ax = plt.subplots(figsize=fig_size_dict[orientation])
    else:
        fig = figure
    
    nx.draw_networkx(G, 
                        pos=pos, 
                        node_color=colors_list, 
                        node_shape=node_shape,
                        node_size=node_size,
                        nodelist=node_list,
                        edgelist=edge_list,
                        width=edge_width,
                        edge_color=edge_color,
                        with_labels=False, 
                        ax=ax)
    
    axis_limits_dict = {}
    axis_limits_dict['axial'] = [ -70, 70, -105, 70]
    axis_limits_dict['coronal'] = [ -70, 70, -45, 75 ]
    axis_limits_dict['sagittal'] = [ -105, 70, -45, 75 ]
    
    ax.set_xlim(axis_limits_dict[orientation][0],axis_limits_dict[orientation][1])
    ax.set_ylim(axis_limits_dict[orientation][2],axis_limits_dict[orientation][3])
    ax.axis('off')
    
    return ax
    
def get_anatomical_layouts(G, df):

    axial_dict = {}
    sagittal_dict = {}
    coronal_dict = {}
    
    for node in df['node'].values:
        axial_dict[node] = np.array([df['x'].loc[df['node']==node].values[0], 
                                        df['y'].loc[df['node']==node].values[0]])
        coronal_dict[node] = np.array([df['x'].loc[df['node']==node].values[0], 
                                        df['z'].loc[df['node']==node].values[0]])
        sagittal_dict[node] = np.array([df['y'].loc[df['node']==node].values[0],
                                        df['z'].loc[df['node']==node].values[0]])
        
    return axial_dict, sagittal_dict, coronal_dict
    
def set_conn_types(G, G_edge=None, thresh=75):

    if not G_edge:
        G_edge = G
        
    # Figure out the degrees from the main graph (G)
    deg = G.degree().values()

    # Now calculate the threshold that you're going
    # to use to designate a node as a hub or not
    hub_thresh = np.percentile(deg, thresh)

    # Loop through the edges of the G_edge graph and 
    # assign the connection type as 2 (hub-hub),
    # 1 (hub-peripheral; feeder) or 0 (peripheral-peripheral)
    for node1, node2 in G_edge.edges():
        if deg[node1] > hub_thresh and deg[node2] > hub_thresh:
            G_edge.edge[node1][node2]['conn_type'] = 2
        elif deg[node1] > hub_thresh or deg[node2] > hub_thresh:
            G_edge.edge[node1][node2]['conn_type'] = 1
        else:
            G_edge.edge[node1][node2]['conn_type'] = 0
            
    # Return G_edge
    return G_edge
    
def rich_edges_nodes(G, thresh=75):
    # Figure out the degrees from the main graph (G)
    deg = G.degree().values()

    # Now calculate the threshold that you're going
    # to use to designate a node as a hub or not
    hub_thresh = np.percentile(deg, thresh)

    G = set_conn_types(G, thresh=thresh)
    
    rich_edges = [ (node1, node2) for node1, node2 in G.edges() if G[node1][node2]['conn_type']==2 ]
    rich_nodes = [ node for node in G.nodes() if deg[node] > hub_thresh ]
    
    return rich_edges, rich_nodes
    