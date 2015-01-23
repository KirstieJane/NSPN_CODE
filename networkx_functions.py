#!/usr/bin/env python

# This needs a whole bunch of commenting, but it's basically
# a whole pile of useful functions for graph analyses
# Copied over from KW's ipython notebook on 11th December 2014

def save_mat(M, M_text_name):
    import numpy as np
    # Save the matrix as a text file
    np.savetxt(M_text_name,
                   M,
                   fmt='%.5f',
                   delimiter='\t',
                   newline='\n')
                   
def sort_partition(partition, axial_pos):
    
    import numpy as np
    
    average_y_list = []
    
    module_list = set(partition.values())
    
    for i, mod in enumerate(module_list) :

        # Create a list of nodes that are in each module
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == mod]

        # Find their average Y position
        y_values = []
        for node in list_nodes:
            y_values.append(axial_pos[node][1])
            
        average_y_list.append(np.average(y_values))
        
    # Now sort the module_list and return it
    module_list = [ mod for (av_y, mod) in sorted(zip(average_y_list, module_list))]

    return module_list
    
    
    
def graph_at_cost(M, cost):
    
    import numpy as np
    import networkx as nx
    
    # Make a copy of the matrix
    thr_M = np.copy(M)

    # Set all diagonal values to 0
    thr_M[np.diag_indices_from(thr_M)] = 0

    # Multiply all values by -1 because the minimum spanning tree
    # looks for the smallest distance - not the largest correlation!
    thr_M = thr_M*-1
    
    # Read this full matrix into a graph G
    G = nx.from_numpy_matrix(thr_M)
    
    # Make a list of all the sorted edges in the full matrix
    G_edges_sorted = [ edge for edge in sorted(G.edges(data = True), key = lambda (a, b, dct): dct['weight']) ]
    
    # Calculate minimum spanning tree and make a list of the mst_edges
    mst = nx.minimum_spanning_tree(G)
    mst_edges = mst.edges(data = True)
    
    # Create a list of edges that are *not* in the mst
    # (because you don't want to add them in twice!)
    G_edges_sorted_notmst = [ edge for edge in G_edges_sorted if not edge in mst_edges ]
    
    # Figure out the number of edges you want to keep for this
    # particular cost. You have to round this number because it
    # won't necessarily be an integer, and you have to subtract
    # the number of edges in the minimum spanning tree because we're
    # going to ADD this number of edges to it 
    n_edges = (cost/100.0) * len(G_edges_sorted)
    n_edges = np.int(np.around(n_edges))
    n_edges = n_edges - len(mst.edges())

    # If your cost is so small that your minimum spanning tree already covers it
    # then you can't do any better than the MST and you'll just have to return
    # it with an accompanying error message
    if n_edges < 0:
        print 'Unable to calculate matrix at this cost - minimum spanning tree is too large'

    # Otherwise, add in the appropriate number of edges (n_edges)
    # from your sorted list (G_edges_sorted_notmst)
    else:
        mst.add_edges_from(G_edges_sorted_notmst[:n_edges])

    # And return the *updated* minimum spanning tree
    # as your graph
    return mst

    
def full_graph(M):
    
    import numpy as np
    import networkx as nx
    
    # Make a copy of the matrix
    thr_M = np.copy(M)

    # Set all diagonal values to 0
    thr_M[np.diag_indices_from(thr_M)] = 0

    # Read this full matrix into a graph G
    G = nx.from_numpy_matrix(thr_M)
    
    return G

        
def threshold_matrix(M, cost):
    '''
    M is the full association matrix.
    cost is the percentage (0 to 100) at which you'd like to threshold
    
    threshold_matrix first creates a copy of the input matrix, then
    sets all diagonal values to 0. It next calculates the minimum spanning tree,
    and ensures that those edges are *always* included in the thresholded
    matrix.
    
    then sets all values below the 
    appropriate percentile to 0
    '''
    # Make a copy of the matrix
    thr_M = np.copy(M)
    
    # Set all diagonal values to -999    
    thr_M[np.diag_indices_from(thr_M)] = -999
    
    # Calculate minmum spanning tree
    G = nx.from_numpy_matrix(M)
    mst = nx.minimum_spanning_tree(G, weight='weight'*-1)
    
    # Calculate the threshold value
    thr = np.percentile(thr_M[np.triu_indices_from(thr_M, k=1)], cost)
    
    # Set all values that are less than the threshold to 0
    thr_M[thr_M < thr] = 0
       
    # Set all values that are not zero to 1
    thr_M[thr_M != 0] = 1

    return thr_M
    
    
def partial_r(x, y, covars):

    import numpy as np
    from scipy.stats import pearsonr
    
    res_i = residuals(covars, x)
    res_j = residuals(covars, y)
    part_r = pearsonr(res_i, res_j)[0]
    return part_r
    
    
def residuals(x, y):
    '''
    A useful little function that correlates
    x and y together to give their residual
    values. These can then be used to calculate
    partial correlation values
    '''
    import numpy as np
    
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    A = np.vstack([x, np.ones(x.shape[-1])]).T
    B = np.linalg.lstsq(A, y)[0]
    m = B[:-1]
    c = B[-1]
    pre = np.sum(m * x.T, axis=1) + c
    res = y - pre
    return res
    
    
def calc_modularity(G):
    
    import numpy as np
    import networkx as nx
    import community
    
    # Binarize both of the graphs
    for u,v,d in G.edges(data=True):
        d['weight']=1
            
    # Compute the best partition based on the threshold you've specified in cost
    partition = community.best_partition(G)

    modularity = community.modularity(partition, G)    
    
    return modularity

def calc_clustering(G):
    
    import numpy as np
    import networkx as nx
    import community
    
    # Binarize both of the graphs
    for u,v,d in G.edges(data=True):
        d['weight']=1
            
    # Compute the best partition based on the threshold you've specified in cost
    partition = community.best_partition(G)

    modularity = community.modularity(partition, G)    
    
    return modularity
    
    
def plot_modules(G, 
                 G_edge,
                 axial_pos,
                 coronal_pos,
                 sagittal_pos,
                 cost_mod=20,
                 cost_edge=2,
                 cmap_name='jet',
                 title='',
                 integer_adjust=3,
                 fractional_adjust=2.5):
    
    import matplotlib.pylab as plt
    import numpy as np
    import networkx as nx
    import community
    
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
        
    # Figure out the spring positions
    spring_pos = nx.spring_layout(G)
    
    # Set up the four figures
    fig, ax_list = plt.subplots(1,4,figsize=(20,5))
    
    # com is a marker for the modules. It'll loop through every module, so if you have
    # eg: 4 modules this loop will run 4 times.
    for i, mod in enumerate(module_list) :
        
        # Create a list of nodes that are in this module        
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == mod]

        # Axial view
        nx.draw_networkx_nodes(G, axial_pos,
                                    list_nodes,
                                    node_size = 3 + 2.5 * np.array(G.degree().values()),
                                    node_color = cmap(i / size),
                                    ax = ax_list[0])
        # Sagittal view
        nx.draw_networkx_nodes(G, sagittal_pos,
                                    list_nodes, 
                                    node_size = integer_adjust + fractional_adjust * np.array(G.degree().values()),
                                    node_color = cmap(i / size),
                                    ax = ax_list[1])
        # Coronal view
        nx.draw_networkx_nodes(G, coronal_pos, 
                                    list_nodes,
                                    node_size = integer_adjust + fractional_adjust * np.array(G.degree().values()),
                                    node_color = cmap(i / size),
                                    ax = ax_list[2])
        # Sprint layout
        nx.draw_networkx_nodes(G, spring_pos,
                                    list_nodes,
                                    node_size = integer_adjust + fractional_adjust * np.array(G.degree().values()),
                                    node_color = cmap(i / size),
                                    ax = ax_list[3])

    # Add in all the edges
    nx.draw_networkx_edges(G_edge, axial_pos, alpha=0.2, ax = ax_list[0])
    nx.draw_networkx_edges(G_edge, sagittal_pos, alpha=0.2, ax = ax_list[1])
    nx.draw_networkx_edges(G_edge, coronal_pos, alpha=0.2, ax = ax_list[2])
    nx.draw_networkx_edges(G, spring_pos, alpha=0.5, ax = ax_list[3])

    # Change the x and y limits to make the images look a bit better
    ax_list[0].set_xlim(-75, 75)
    ax_list[0].set_ylim(-110, 70)

    ax_list[1].set_xlim(-110, 70)
    ax_list[1].set_ylim(-70, 100)

    ax_list[2].set_xlim(-75, 75)
    ax_list[2].set_ylim(-50, 90)

    # Turn the axis labels off
    for ax in ax_list:
        ax.set_axis_off()

    # Add in the overall title
    fig.suptitle(title)
    
    # And add in the cost infomation
    cost_text = 'N modules: {:.0f}\nModule cost: {:.0f}%\nEdges cost: {:.0f}%'.format(size, cost_mod, cost_edge)
    fig.text(0.5, 0.9,
                    cost_text,
                     horizontalalignment='center',
                     verticalalignment='top',
                     #transform = ax_list[2].transAxes
                     )
    
    # Finally, make it clear that the last image is a spring layout
    ax_list[3].text(0.5, 0,
                    'spring layout\nedge cost as module cost',
                     horizontalalignment='center',
                     verticalalignment='bottom',
                     transform = ax_list[3].transAxes)
    
    return fig
    
    
def create_mat(df, aparc_names, covar, demean=False):
    '''
    df contains the data you're going to correlate
    aparc_names are all the regions you care about
    covar needs to be either a column in df OR a 
    list of columns
    '''
    import numpy as np
    from scipy.stats import pearsonr

    mat_corr = df[aparc_names].corr().iloc[:,:]

    mat_corr_covar = np.ones_like(mat_corr)
    
    if len(covar) > 1:
        x = np.vstack([df[covar]])
    else:
        x = df[covar]

    triu_i, triu_j = np.triu_indices(len(aparc_names))
    for i, j in zip(triu_i, triu_j):
        if i%20 == 0 and j == len(aparc_names)-1:
            print 'Processing row {}'.format(i)
        
        if demean:
            res_i = residuals(x.T, df[aparc_names[i]] - df[aparc_names].mean(axis=1).values)
            res_j = residuals(x.T, df[aparc_names[j]] - df[aparc_names].mean(axis=1).values)
            mat_corr_covar[i, j] = pearsonr(res_i, res_j)[0]        
        else:
            res_i = residuals(x.T, df[aparc_names[i]])
            res_j = residuals(x.T, df[aparc_names[j]])
            mat_corr_covar[i, j] = pearsonr(res_i, res_j)[0]

    mat_corr = mat_corr * mat_corr.T
    mat_corr_covar = mat_corr_covar * mat_corr_covar.T
    
    return mat_corr, mat_corr_covar
    
    
def assign_node_attr(G, centroids, aparc_names):

    # Assign names and x,y,z coordinates to the nodes
    for i, node in enumerate(G.nodes()):
        G.node[node]['x'] = centroids[i, 0]
        G.node[node]['y'] = centroids[i, 1]
        G.node[node]['z'] = centroids[i, 2]
        G.node[node]['name_500'] = aparc_names[i]
        G.node[node]['name_DK'] = aparc_names[i].rsplit('_',1)[0]
        G.node[node]['hemi'] = aparc_names[i].split('_',1)[0]

    # Set a counter for the interhemispheric edges
    interhem_count=0
    
    # Loop through the edges
    for node1, node2 in G.edges():
        
        # If the two hemispheres are the same then interhem == 0
        if G.node[node1]['hemi'] == G.node[node2]['hemi']:

            G.edge[node1][node2]['interhem'] = 0
            
        # If the two hemispheres are different then interhem == 1
        else:
            
            G.edge[node1][node2]['interhem'] = 1
            interhem_count +=1

    # Now we want to map this back to each node
    
            
            
    # Assign a graph attribute of the proportion of edges that are interhemispheric
    G.graph['interhem_proportion'] = interhem_count*100.0/len(G.edges())
    
    # Save this back to the graph_dict
    # (otherwise all is lost!)
    graph_dict['{}_covar_{}_{}_COST_{:02.0f}'.format(measure, covars, group, cost)] = G