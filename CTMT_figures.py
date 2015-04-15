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


def plot_rich_club(rc, rc_rand, ax=None, figure_name=None, x_max=200, y_max=1.2, color=sns.color_palette()[0]):
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
    
    # Plot the real rich club data
    sns.tsplot(rc, color=color, ax=ax)
    
    # Plot the random rich club data with confidence intervals error bars
    sns.tsplot(rc_rand.T, err_style='ci_bars', color='grey', ci=95, ax=ax)
    
    # Fix the x and y axis limits
    ax.set_xlim((0, x_max))
    ax.set_ylim((0, y_max))
    # Make sure there aren't too many bins!
    plt.locator_params(nbins=5)
    
    # Set the x and y axis labels
    ax.set_xlabel("Degree")
    ax.set_ylabel("Rich Club")
    
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
    
def plot_network_measures(measures_dict, ax=None, figure_name=None, y_max=2.5, y_min=-0.5, color=sns.color_palette()[0]):
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
    df = pd.DataFrame(measures_dict)
    
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

def pretty_scatter(x, y, x_label='x', y_label='y', x_max=None, x_min=None, y_max=None, y_min=None, figure_name=None, ax=None, figure=None, color=sns.color_palette()[0]):
    '''
    This function creates a scatter plot with a regression line
    for the y variable against the degrees of graph G
    '''
    # Import what you need
    import matplotlib.pylab as plt
    import seaborn as sns
    
    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)

    # Load the data into a data frame
    df =  pd.DataFrame({x_label : x,
                        y_label : y})
        
    # Create the figure if you need to
    if not ax:
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = figure
        
    # Create the linear regression plot
    ax = sns.regplot(x_label, y_label, df, ci=95, ax=ax, color=color)
    
    # Fix the x and y axis limits
    if np.isscalar(x_max) and np.isscalar(x_min):
        ax.set_xlim((x_min, x_max))
    if np.isscalar(y_max) and np.isscalar(y_min):
        ax.set_ylim((y_min, y_max))
    
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))

    # Make sure there aren't too many bins!
    ax.locator_params(axis='y', nbins=4)
    
    # Put a line at y = 0
    ax.axhline(0, linewidth=0.5, color='black', linestyle='--')

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
        

def degree_r_values(graph_dict, y):
    
    r_array = np.ones([30])
    p_array = np.ones([30])
    
    cost_list = range(1,31)
    
    for i, cost in enumerate(cost_list):
        measure = 'CT'
        covars = 'ones'
        group = 'all'
        cost = np.float(cost)
        
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
    depth_list = np.hstack([np.arange(100,-1,-10), np.arange(-20, -101, -20)])

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

def create_violin_data(measure_dict, map='MT', measure='all_slope_age', cmap='RdBu_r', cmap_min=-7, cmap_max=7):
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
    n_values = len(measure_dict['{}_projfrac+000_{}'.format(map, measure)])
    df =  pd.DataFrame({'index' : range(n_values)})
    color_list = []
    
    # Set up the color mapping
    cm = plt.get_cmap(cmap)
    cNorm  = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max)            #### THIS PROBABLY NEEDS ADJUSTING!
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)

    # Create a list of all the depths you care about
    depth_list = np.hstack([np.arange(100,-1,-10), np.arange(-20, -101, -20)])
    
    # Loop through all the depths
    for i in depth_list:
        
        # Fill in the appropriate data
        if i >= 0:
            m_array = measure_dict['{}_projfrac{:+04.0f}_{}'.format(map, i, measure)]
        else:
            m_array = measure_dict['{}_projdist{:+04.0f}_{}'.format(map, i, measure)]

        df['{}'.format(i)] = m_array/1000.0

        color_list += [scalarMap.to_rgba(np.mean(m_array))]

    return df, color_list


def violin_mt_depths(measure_dict, map='MT', measure='all_slope_age', cmap='PRGn', cmap_min=-7, cmap_max=7, y_max=None, y_min=None, figure_name=None, ax=None, figure=None):
    '''
    INPUTS:
        data_dir --------- where the PARC_*_behavmerge.csv files are saved
        measure_dict
    '''
    
    # Import what you need
    import matplotlib.pylab as plt
    import seaborn as sns
    
    # Set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)
    
    # Get the data, colors and labels
    df, color_list = create_violin_data(measure_dict, map=map, measure=measure, 
                                                cmap=cmap, cmap_min=cmap_min, cmap_max=cmap_max)
    
    labels_list = create_violin_labels()
        
    # Create the figure if you need to
    if not ax:
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = figure
        
    # Create the box plot
    ##### You could change this here to a violin plot if you wanted to...
    ax = sns.boxplot(df[df.columns[1:]], color=color_list, names=labels_list, ax=ax)
    
    # Fix the y axis limits
    if np.isscalar(y_max) and np.isscalar(y_min):
        ax.set_ylim((y_min, y_max))
    
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))

    # Make sure there aren't too many bins!
    ax.locator_params(axis='y', nbins=4)
    
    # Re-do the tick labels so that they're rotated
    ax.set_xticklabels(labels_list, rotation=90)

    # Put a line at the grey white matter boundary
    # and another at y=0
    ax.axvline(11, linewidth=0.5, color='black', linestyle='--')
    ax.axhline(0, linewidth=0.5, color='black', linestyle='-')

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

def figure_1(graph_dict, figures_dir, n=10):

    big_fig, ax_list = plt.subplots(4,4, figsize=(40, 25.2), facecolor='white', sharey='row')
    
    cost_list = [ 5, 10, 15, 20 ]
    
    for i, cost in enumerate(cost_list):
        measure = 'CT'
        covars = 'ones'
        group = 'all'
        cost = np.float(cost)
        
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
        plot_degree_dist(G, figure_name=figure_name, x_max=150, y_max=0.1, color=sns.color_palette()[0])
        ax_list[1, i] = plot_degree_dist(G, ax=ax_list[1, i], x_max=150, y_max=0.1, color=sns.color_palette()[0])

        #============= RICH CLUB ==============================
        figure_name = os.path.join(figures_dir, 
                                        '{}_covar_{}_{}_richclub_COST_{:02.0f}.png'.format(measure,
                                                                                                    covars, 
                                                                                                    group.upper(), 
                                                                                                    cost))
        deg, rc, rc_rand = rich_club(G, R_list, n=n)
        plot_rich_club(rc, rc_rand, figure_name=figure_name, x_max=150, y_max=1.2, color=sns.color_palette()[0])    
        ax_list[2, i] = plot_rich_club(rc, rc_rand, ax=ax_list[2, i], x_max=150, y_max=1.2, color=sns.color_palette()[0])    
        
        #============= NETWORK MEASURES =======================
        figure_name = os.path.join(figures_dir, 
                                        '{}_covar_{}_{}_networkmeasures_COST_{:02.0f}.png'.format(measure,
                                                                                                    covars, 
                                                                                                    group.upper(), 
                                                                                                    cost))
        measures_dict = calculate_network_measures(G, R_list, n=n)
        plot_network_measures(measures_dict, figure_name=figure_name, y_max=2.5, y_min=-0.5, color=sns.color_palette()[0])
        ax_list[3, i] = plot_network_measures(measures_dict, ax=ax_list[3, i], y_max=2.5, y_min=-0.5, color=sns.color_palette()[0])

        
        
    # Get rid of y axis labels for columns that aren't on the left side
    [ a.set_ylabel('') for a in ax_list[:,1:].reshape(-1) ]
    
    # RAAAANDOMLY - and I don't know why this is happening
    # set the x limits for the very last plot to those of the one
    # next to it - HMMMMMM
    ax_list[3,i].set_xlim( ax_list[3,i-1].get_xlim() )
    
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
    filename = os.path.join(figures_dir, 'Figure1.png')
    big_fig.savefig(filename, bbox_inches=0, dpi=100)
    
    plt.close()

def figure_2(ct_data_file, mt_data_file, measure_dict, figures_dir):

    big_fig, ax_list = plt.subplots(3,3, figsize=(30, 18), facecolor='white')
    
    #==== CORRELATE GLOBAL CT WITH AGE =============================
    figure_name = os.path.join(figures_dir, 
                                    'Global_CT_corr_Age.png'.format(measure))

    df_ct = read_in_df(ct_data_file)
    
    color=sns.color_palette('RdBu_r', 10)[1]
    
    pretty_scatter(df_ct['age_scan'], df_ct['Global'], 
                    x_label='Age (years)', y_label='Cortical Thickness (mm)', 
                    x_max=25, x_min=14, 
                    y_max=3.0, y_min=2.4, 
                    figure_name=figure_name,
                    color=color)
                            
    ax_list[0, 0] = pretty_scatter(df_ct['age_scan'], df_ct['Global'], 
                    x_label='Age (years)', y_label='Cortical Thickness (mm)', 
                    x_max=25, x_min=14, 
                    y_max=3.0, y_min=2.4, 
                    color=color,
                    ax=ax_list[0, 0],
                    figure=big_fig)
                        
    #==== CORRELATE GLOBAL MT(70) WITH AGE =============================
    figure_name = os.path.join(figures_dir, 
                                    'Global_MT_projfrac+030_corr_Age.png')

    df_mt = read_in_df(mt_data_file)
    
    color=sns.color_palette('PRGn_r', 10)[1]

    pretty_scatter(df_mt['age_scan'], df_mt['Global']/1000, 
                    x_label='Age (years)', y_label='Magnetisation Transfer', 
                    x_max=25, x_min=14, 
                    y_max=1.05, y_min=0.8, 
                    figure_name=figure_name,
                    color=color)
                            
    ax_list[1, 0] = pretty_scatter(df_mt['age_scan'], df_mt['Global']/1000, 
                    x_label='Age (years)', y_label='Magnetisation Transfer', 
                    x_max=25, x_min=14, 
                    y_max=1.05, y_min=0.8, 
                    color=color,
                    ax=ax_list[1, 0],
                    figure=big_fig)

    #==== CORRELATE GLOBAL MT(70) WITH CT =============================
    figure_name = os.path.join(figures_dir, 
                                    'Global_MT_projfrac+030_corr_CT.png')

    df_mt = read_in_df(mt_data_file)
    
    color=sns.color_palette('PRGn', 10)[1]

    pretty_scatter(df_ct['Global'], df_mt['Global']/1000, 
                    x_label='Cortical Thickness (mm)', y_label='Magnetisation Transfer', 
                    x_max=30, x_min=2.4, 
                    y_max=1.05, y_min=0.8, 
                    figure_name=figure_name,
                    color=color)
                            
    ax_list[2, 0] = pretty_scatter(df_ct['Global'], df_mt['Global']/1000, 
                    x_label='Cortical Thickness (mm)', y_label='Magnetisation Transfer', 
                    x_max=3.0, x_min=2.4, 
                    y_max=1.05, y_min=0.8, 
                    color=color,
                    ax=ax_list[2, 0],
                    figure=big_fig)
    
    
    #==== SHOW CORR WITH AGE AT DIFFERENT DEPTHS ======================
    figure_name = os.path.join(figures_dir, 
                                    'MT_projfrac+030_corr_Age_DifferentDepths.png')
    
    violin_mt_depths(measure_dict,
                        measure='all_slope_age',
                        cmap='PRGn',
                        y_max=0.015, y_min=-0.010, 
                        figure_name=figure_name)
                        
    ax_list[2, 1] = violin_mt_depths(measure_dict,
                                        measure='all_slope_age',
                                        y_max=0.015, y_min=-0.010, 
                                        ax=ax_list[2, 1],
                                        figure=big_fig)
    
    #==== SHOW CORR WITH CT AT DIFFERENT DEPTHS ======================
    figure_name = os.path.join(figures_dir, 
                                    'MT_projfrac+030_corr_CT_DifferentDepths.png')
    
    violin_mt_depths(measure_dict,
                        measure='all_slope_ct',
                        cmap='PRGn',
                        y_min=-0.000007,
                        y_max=0.000003,
                        cmap_min=-0.003,
                        cmap_max=0.003,
                        figure_name=figure_name)

    ax_list[2, 2] = violin_mt_depths(measure_dict,
                                        measure='all_slope_ct',
                                        cmap='PRGn',
                                        y_min=-0.000007,
                                        y_max=0.000003,
                                        cmap_min=-0.003,
                                        cmap_max=0.003,
                                        ax=ax_list[2, 2],
                                        figure=big_fig)

    # Nice tight layout
    big_fig.tight_layout()
    
    # Save the figure
    filename = os.path.join(figures_dir, 'Figure2.png')
    big_fig.savefig(filename, bbox_inches=0, dpi=100)
    
    plt.close()

    
def figure_3(graph_dict, pc_dict, measures_dict, figures_dir):

    import matplotlib.pylab as plt
    import numpy as np
    import networkx as nx
    
    big_fig, ax_list = plt.subplots(2,3, figsize=(30, 12), facecolor='white')
    
    cost = 10    
    covars = 'ones'
    group = 'all'
    cost = np.float(cost)
    measure = 'CT'

    key = '{}_covar_{}_{}_COST_{:02.0f}'.format(measure, covars, group, cost)

    G = graph_dict[key]    
    pc = np.array(pc_dict[key].values())
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
    
    ax_list[0,0].text(0, 1, 
                        'B', 
                        horizontalalignment='center',
                        verticalalignment='top',
                        fontsize=80,
                        transform=ax.transAxes,
                        weight='bold')

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
                    y_max=20, y_min=-10, 
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
    filename = os.path.join(figures_dir, 'Figure3.png')
    big_fig.savefig(filename, bbox_inches=0, dpi=100)
    
    plt.close()
    
    
def partial_volume_fig(measure_dict, figures_dir):

    big_fig, ax_list = plt.subplots(2, 2, figsize=(20, 20), facecolor='white')
        
    #==== SHOW MEAN MT AT DIFFERENT DEPTHS ======================                            
    ax_list[0, 0] = violin_mt_depths(measure_dict,
                                        map='MT',
                                        measure='all_mean',
                                        y_min=0,
                                        y_max=2.0,
                                        cmap='jet',
                                        cmap_min=0,
                                        cmap_max=2000.0,
                                        ax=ax_list[0, 0],
                                        figure=big_fig)
                                        
    ax_list[0, 1] = violin_mt_depths(measure_dict,
                                        map='synthetic',
                                        measure='all_mean',
                                        y_min=0,
                                        y_max=2.0,
                                        cmap='jet',
                                        cmap_min=0,
                                        cmap_max=2000.0,
                                        ax=ax_list[0, 1],
                                        figure=big_fig)
    
    #==== SHOW STD AT DIFFERENT DEPTHS ======================                            
    ax_list[1, 0] = violin_mt_depths(measure_dict,
                                        map='MT',
                                        measure='all_std',
                                        y_min=0,
                                        y_max=0.2,
                                        cmap='jet',
                                        cmap_min=0.0,
                                        cmap_max=200.0,
                                        ax=ax_list[1, 0],
                                        figure=big_fig)
                                        
    ax_list[1, 1] = violin_mt_depths(measure_dict,
                                        map='synthetic',
                                        measure='all_std',
                                        y_min=0,
                                        y_max=0.2,
                                        cmap='jet',
                                        cmap_min=0,
                                        cmap_max=200.0,
                                        ax=ax_list[1, 1],
                                        figure=big_fig)
                        
    # Nice tight layout
    big_fig.tight_layout()
    
    # Save the figure
    filename = os.path.join(figures_dir, 'PartialVolumeFig.png')
    big_fig.savefig(filename, bbox_inches=0, dpi=100)
    
    plt.close()
