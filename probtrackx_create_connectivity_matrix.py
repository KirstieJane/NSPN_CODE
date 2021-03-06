#!/usr/bin/env python

'''
This code looks for all the outputs of freesurfer_probtrackx 
and combines them into a text file connectivity matrix
'''

#=============================================================================
# IMPORTS
#=============================================================================
import os
import sys
import numpy as np
from glob import glob
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm

#=============================================================================
# FUNCTIONS
#=============================================================================

def usage():
    import sys
    print "USAGE: probtrackx_create_connectivity_matrix.py <data_dir> <subid> <occ>"
    sys.exit()
   
def save_png(M, M_name):
    if not os.path.exists(M_name):

        fig, ax = plt.subplots(figsize=(4,4))    
        # Plot the matrix on a log scale
        axM = ax.matshow(M, 
                        interpolation='nearest',
                        cmap='jet',
                        norm=LogNorm())
        
        # Add a colorbar
        cbar = fig.colorbar(axM)

        fig.savefig(M_name, bbox_inches=0, dpi=600)

#=============================================================================
# READ IN COMMANDLINE ARGUMENTS
#=============================================================================

data_dir=sys.argv[1]
subid = sys.argv[2]
occ = sys.argv[3]

probtrackx_dir = os.path.join(data_dir, 
                                'SUB_DATA', 
                                subid, 
                                'SURFER', 
                                'MRI{}'.format(occ), 
                                'probtrackx')
                                
if not os.path.isdir(probtrackx_dir):
    print '{} does not exist'.format(probtrackx_dir)
    usage
                                
#=============================================================================
# FIND ALL THE TEXT FILES AND COMBINE
#=============================================================================

matrix_file_list =  glob(os.path.join(probtrackx_dir, 'Seg*/matrix_seeds_to_all_targets'))

matrix_file_list.sort()

outfile_mean = os.path.join(probtrackx_dir, 'mean_connectivity.txt')
outfile_sum = os.path.join(probtrackx_dir, 'sum_connectivity.txt')
outfile_prob = os.path.join(probtrackx_dir, 'prob_connectivity.txt')

if not os.path.isfile(outfile_prob):
    mean_matrix = np.zeros([308, 308])
    sum_matrix = np.zeros([308, 308])

    for i, m_file in enumerate(matrix_file_list):
        data = np.loadtxt(m_file)
        m_data = data.mean(axis=0)
        mean_matrix[i,:] = m_data
        s_data = data.sum(axis=0)
        sum_matrix[i,:] = s_data
    
    prob_matrix = mean_matrix / 5000.0
    
    np.savetxt(outfile_mean, mean_matrix, fmt='%10.10f')
    np.savetxt(outfile_prob, prob_matrix, fmt='%10.10f')
    np.savetxt(outfile_sum, sum_matrix, fmt='%10.10f')

else:
    mean_matrix = np.loadtxt(outfile_mean)
    sum_matrix = np.loadtxt(outfile_sum)
    prob_matrix = np.loadtxt(outfile_prob)
    
#=============================================================================
# MAKE A NICE PICTURE
#=============================================================================

figfile_mean = os.path.join(probtrackx_dir, 'mean_connectivity.png')
figfile_sum = os.path.join(probtrackx_dir, 'sum_connectivity.png')
figfile_prob = os.path.join(probtrackx_dir, 'prob_connectivity.png')

save_png(mean_matrix, figfile_mean)
save_png(sum_matrix, figfile_sum)
save_png(prob_matrix, figfile_prob)

#=============================================================================
# THE END
# Hello from the Lake District
# Kx, LoonieSpoonie and SpillieVanillie
#=============================================================================
