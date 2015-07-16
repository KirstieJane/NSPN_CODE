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


def calc_efficiency(G): 

    import networkx as nx
    
    E=0.0
    for node in G:
        path_length=nx.single_source_shortest_path_length(G, node)
        E += 1.0/sum(path_length.values())

    return E
    
def closeness(G):
    import networkx as nx
    
    closeness_dict = closeness_centrality(G)


    
def shortest_path(G):
    import networkx as nx
    import numpy as np
    
    shortestpl_dict_dict = nx.shortest_path_length(G)
    
    shortestpl_dict = {}
    
    for node in G.nodes():
        shortestpl_dict[node] = np.average(shortestpl_dict_dict[node].values())
        
    return shortestpl_dict
    

def assign_nodal_distance(G, centroids):

    '''
    Give each node in the graph their
    x, y, z coordinates and then calculate the eucledian
    distance for every edge that connects to each node
    
    Also calculate the number of interhemispheric edges
    (defined as edges which different signs for the x 
    coordinate
    
    Returns the graph
    '''
    import networkx as nx
    import numpy as np
    from scipy.spatial import distance
    
    # First assign the x, y, z values to each node
    for i, node in enumerate(G.nodes()):
        G.node[node]['x'] = centroids[i, 0]
        G.node[node]['y'] = centroids[i, 1]
        G.node[node]['z'] = centroids[i, 2]
        G.node[node]['centroids'] = centroids[i, :]
    
    # Loop through every node in turn
    for i, node in enumerate(G.nodes()):
        # Loop through the edges connecting to this node
        # Note that "node1" should always be exactly the same
        # as "node", I've just used another name to keep 
        # the code clear (which I may not have achieved given
        # that I thought this comment was necesary...)
        for node1, node2 in G.edges(nbunch=[node]):
        
            # Calculate the eulidean distance for this edge
            cent1 = G.node[node1]['centroids']
            cent2 = G.node[node2]['centroids']
            
            dist = distance.euclidean(cent1, cent2)
            
            # And assign this value to the edge
            G.edge[node1][node2]['euclidean'] = dist 
        
            # Also figure out whether this edge is interhemispheric
            # by multiplying the x values. If x1 * x2 is negative
            # then the nodes are in different hemispheres.
            x1 = G.node[node1]['x']
            x2 = G.node[node2]['x']
            
            if x1*x2 > 0:
                G.edge[node1][node2]['interhem'] = 0
            else:
                G.edge[node1][node2]['interhem'] = 1
            
        # Create two nodal attributes (average distance and 
        # total distance) by summarizing the euclidean distance
        # for all edges which connect to the node
        euc_list = [ G.edge[m][n]['euclidean'] for m, n in G.edges(nbunch=node) ]
        
        G.node[node]['average_dist'] = np.mean(euc_list)
        G.node[node]['total_dist'] = np.sum(euc_list)
        
        # Create an interhem nodal attribute by getting the average
        # of the interhem values for all edges which connect to the node
        
        interhem_list = [ G.edge[m][n]['interhem'] for m, n in G.edges(nbunch=node) ]

        G.node[node]['interhem_proportion'] = np.mean(interhem_list)
        
    return G
    
def participation_coefficient(G):
    '''
    Computes the participation coefficient for each node (Guimera et al. 2005).
    
    HEAVILY STOLEN FROM https://github.com/mb3152/brain-graphs
    
    BUT that link is now gone *and* it had a mistake in it!
    Nightmare
    
    However, Petra V saved the day so now we're back on track
    
    ------
    Inputs
    ------
    graph = Networkx graph
    ------
    Output
    ------
    Dictionary of the participation coefficient for each node.
    '''
    # Import the modules you'll need
    import networkx as nx
    import numpy as np
    import community as community
    
    # Binarize both of the graphs
    for u,v,d in G.edges(data=True):
        d['weight']=1

    # Calculate the best modular partition
    nodal_partition = community.best_partition(G)

    # Reverse the dictionary because the output of Louvain is "backwards"
    # meaning it saves the module per node, rather than the nodes in each
    # module
    module_partition = {}
    for m,n in zip(nodal_partition.values(),nodal_partition.keys()):
        try:
            module_partition[m].append(n)
        except KeyError:
            module_partition[m] = [n]

    # Create an empty dictionary for the participation
    # coefficients
    pc_dict = {}
    all_nodes = set(G.nodes())
    
    # Loop through modules
    for m in module_partition.keys():
        
        # Print a little note to the screen because it can take a long
        # time to run this code
        print 'Calculating pc for nodes in module {} of {}'.format(m, len(module_partition.keys()))
        
        # Get the set of nodes in this module
        mod_list = set(module_partition[m])
        
        # Loop through each node (source node) in this module
        for source in mod_list:
        
            # Calculate the degree for the source node
            degree = float(nx.degree(G=G, nbunch=source))
            
            # Calculate the number of these connections
            # that are to nodes in *other* modules
            count = 0

            for target in mod_list:
                
                # If the edge is in there then increase the counter by 1
                if (source, target) in G.edges():
                    count += 1

            # This gives you the within module degree
            wm_degree = float(count)

            # The participation coeficient is 1 - the square of 
            # the ratio of the within module degree and the total degree
            pc = 1 - ((float(wm_degree) / float(degree))**2)
                
            # Save the participation coefficient to the dictionary
            pc_dict[source] = pc
            
    return nodal_partition, pc_dict
    
    
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
        # Spring layout
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
    import pandas as pd

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

def assign_node_names(G, aparc_names):

    # Assign names to the nodes
    for i, node in enumerate(G.nodes()):
        G.node[node]['name_500'] = aparc_names[i]
        G.node[node]['name_DK'] = aparc_names[i].rsplit('_',1)[0]
        G.node[node]['hemi'] = aparc_names[i].split('_',1)[0]
    
    return G
    
def assign_euclidean_distances(G, centroids):
    '''
    Assign positions to each node and then calculate
    distance measures for each
    
    Returns the graph. You can access the node attributes
    by typing eg: G.node['x']
    '''
    
    from scipy.spatial import distance
    
    # Assign names and x,y,z coordinates to the nodes
    for i, node in enumerate(G.nodes()):
        G.node[node]['x'] = centroids[i, 0]
        G.node[node]['y'] = centroids[i, 1]
        G.node[node]['z'] = centroids[i, 2]

    # Set a counter for the total interhemispheric
    # edges for the whole graph
    total_interhem_count = 0
    
    # Loop through the edges connecting to each node
    for i, node in enumerate(G.nodes()):
    
        # Set a counter for the interhemispheric edges for THIS node
        interhem_count=0
        
        for node1, node2 in G.edges(nbunch=[node]):
        
            # If the two hemispheres are the same then interhem == 0
            if G.node[node1]['hemi'] == G.node[node2]['hemi']:

                G.edge[node1][node2]['interhem'] = 0
                
            # If the two hemispheres are different then interhem == 1
            else:
                
                G.edge[node1][node2]['interhem'] = 1
                interhem_count +=1

        # Nodal proportion of interhemispheric connections
        G.node[node]['interhem_proportion'] = interhem_count*1.0 / G.degree()[node]
        
        # Update the total interhemispheric counter
        total_interhem_count += interhem_count
        
    # Assign a graph attribute of the proportion of edges that are interhemispheric
    # You have to divide the total_interhem_count by 2 because each one has
    # been counted twice in the loop above
    G.graph['interhem_proportion'] = total_interhem_count*1.0/(len(G.edges())*2.0)
    
    return G
    
    
def rich_club(G, R_list=None, n=10):
    '''
    This calculates the rich club coefficient for each degree
    value in the graph (G).
    
    Inputs:
        G ------ networkx graph
        R_list - list of random graphs with matched degree distribution
                   if R_list is None then a random graph is calculated
                   within the code
                   if len(R_list) is less than n then the remaining random graphs
                   are calculated within the code
                 Default R_list = None 
        n ------ number of random graphs for which to calculate rich club
                   coefficients
                 Default n = 10
           
    Returns:
        rc ------ dictionary of rich club coefficients for the real graph
        rc_rand - array of rich club coefficients for the n random graphs
    '''
    # Import the modules you'll need
    import networkx as nx
    import numpy as np
    
    # First, calculate the rich club coefficient for the regular graph
    rc_dict = nx.rich_club_coefficient(G, normalized=False)
    
    # Save the degrees as a numpy array
    deg = np.array(rc_dict.keys())
    
    # Save the rich club coefficients as a numpy array
    rc = np.array(rc_dict.values())
    
    # Calculate n different random graphs and their 
    # rich club coefficients
    
    # Start by creating an empty array that will hold
    # the n random graphs' rich club coefficients
    rc_rand = np.ones([len(rc), n])
    
    for i in range(n):
        # If you haven't already calculated random graphs
        # or you haven't given this function as many random
        # graphs as it is expecting then calculate a random
        # graph here
        if not R_list or len(R_list) <= i:
            R = random_graph(G)
        # Otherwise just use the one you already made
        else:
            R = R_list[i]
            
        # Calculate the rich club coefficient
        rc_rand_dict = nx.rich_club_coefficient(R, normalized=False)
        
        # And save the values to the numpy array you created earlier
        rc_rand[:, i] = rc_rand_dict.values()
        
    return deg, rc, rc_rand

def random_graph(G, Q=10):
    '''
    Create a random graph that preserves degree distribution
    by swapping pairs of edges (double edge swap).
    
    Inputs:
        G: networkx graph
        Q: constant that determines how many swaps to conduct
           for every edge in the graph
           Default Q =10

    Returns:
        R: networkx graph
    
    CAVEAT: If it is not possible in 15 attempts to create a
    connected random graph then this code will just return the
    original graph (G). This means that if you come to look at
    the values that are an output of calculate_global_measures
    and see that the values are the same for the random graph
    as for the main graph it is not necessarily the case that 
    the graph is random, it may be that the graph was so low cost
    (density) that this code couldn't create an appropriate random
    graph!
    
    This should only happen for ridiculously low cost graphs that
    wouldn't make all that much sense to investigate anyway...
    so if you think carefully it shouldn't be a problem.... I hope!
    '''
    
    import networkx as nx
    # Copy the graph
    R = G.copy()
    
    # Calculate the number of edges and set a constant
    # as suggested in the nx documentation
    E = R.number_of_edges()
    
    # Start with assuming that the random graph is not connected
    # (because it might not be after the first permuatation!)
    connected=False
    attempt=0
    
    # Keep making random graphs until they are connected!
    while not connected and attempt < 15:
        # Now swap some edges in order to preserve the degree distribution
        print 'Creating random graph - may take a little while!'
        nx.double_edge_swap(R,Q*E,max_tries=Q*E*10)

        # Check that this graph is connected! If not, start again
        connected = nx.is_connected(R)
        if not connected:
            attempt +=1
    
    if attempt == 15:
        print 'Giving up - can not randomise graph'
        print '==================================='
        R = G.copy()
        
    return R
    

def calculate_global_measures(G, R_list=None, n=10):
    '''
    A wrapper function that calls a bunch of useful functions
    and reports a plethora of network measures for the real graph
    G, and for n random graphs that are matched on degree distribution
    (unless otherwise stated)
    
    This USED to be called calculate_network_measures. It was
    changed on 2nd July because another loop which calculated nodal
    measures was created!
    '''
    import networkx as nx
    import numpy as np
    
    #==== SET UP ======================
    # If you haven't already calculated random graphs
    # or you haven't given this function as many random
    # graphs as it is expecting then calculate a random
    # graph here
    if R_list is None:
        R_list = []
    
    for i in range(n):
        if len(R_list) <= i:
            R_list += [ random_graph(G) ]

    #==== MEASURES ====================
    network_measures_dict = {}
    
    #---- Clustering coefficient ------
    network_measures_dict['C'] = nx.average_clustering(G)
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = nx.average_clustering(R_list[i])
    network_measures_dict['C_rand'] = rand_array
    
    #---- Shortest path length --------
    network_measures_dict['L'] = nx.average_shortest_path_length(G)
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = nx.average_shortest_path_length(R_list[i])
    network_measures_dict['L_rand'] = rand_array
    
    #---- Assortativity ---------------
    network_measures_dict['a'] = np.mean(nx.degree_assortativity_coefficient(G))
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = np.mean(nx.degree_assortativity_coefficient(R_list[i]))
    network_measures_dict['a_rand'] = rand_array

    #---- Modularity ------------------
    network_measures_dict['M'] = calc_modularity(G)
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = calc_modularity(R_list[i])
    network_measures_dict['M_rand'] = rand_array
    
    #---- Efficiency ------------------
    network_measures_dict['E'] = calc_efficiency(G)
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = calc_efficiency(R_list[i])
    network_measures_dict['E_rand'] = rand_array
     
    #---- Small world -----------------
    sigma_array = np.ones(n)
    for i in range(n):
        sigma_array[i] = ( ( network_measures_dict['C'] / network_measures_dict['C_rand'][i] )
                            / ( network_measures_dict['L'] / network_measures_dict['L_rand'][i] ) )
    network_measures_dict['sigma'] = sigma_array
    network_measures_dict['sigma_rand'] = 1.0

    return network_measures_dict

def calculate_nodal_measures(G, centroids, aparc_names):
    '''
    A function which returns a dictionary of numpy arrays for a graph's
        * degree
        * participation coefficient
        * average distance
        * total distance
        * clustering
        * closeness
        * interhemispheric proportion
        * name
        * hemisphere
    '''
    
    import numpy as np
    import networkx as nx
    
    #==================================
    # Create the dictionary
    nodal_dict = {}
    
    #==================================
    # Degree
    deg = G.degree().values()    
    nodal_dict['degree'] = np.array(deg)
    
    #==================================
    # Closeness
    closeness = nx.closeness_centrality(G).values()
    nodal_dict['closeness'] = np.array(closeness)

    #==================================
    # Shortest path length
    L = shortest_path(G).values()
    nodal_dict['shortest_path'] = np.array(L)
    
    #==================================
    # Clustering
    clustering = nx.clustering(G).values()
    nodal_dict['clustering'] = np.array(clustering)

    #==================================
    # Participation coefficent and 
    # module assignment
    partition, pc_dict = participation_coefficient(G)
    nodal_dict['module'] = np.array(partition.values())
    nodal_dict['pc'] = np.array(pc_dict.values())
    
    #==================================
    # Euclidean distance and
    # interhem proportion
    G = assign_nodal_distance(G, centroids)
    average_dist = nx.get_node_attributes(G, 'average_dist').values()
    total_dist = nx.get_node_attributes(G, 'total_dist').values()
    interhem_prop = nx.get_node_attributes(G, 'interhem_proportion').values()
    
    nodal_dict['average_dist'] = np.array(average_dist)
    nodal_dict['total_dist'] = np.array(total_dist)
    nodal_dict['interhem_prop'] = np.array(interhem_prop)
    
    #=================================
    # Names
    G = assign_node_names(G, aparc_names)
    name_500 = nx.get_node_attributes(G, 'name_500').values()
    name_DK = nx.get_node_attributes(G, 'name_DK').values()
    hemi = nx.get_node_attributes(G, 'hemi').values()
    nodal_dict['name_500'] = np.array(name_500)
    nodal_dict['name_DK'] = np.array(name_DK)
    nodal_dict['hemi'] = np.array(hemi)
    
    return nodal_dict
    
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
    