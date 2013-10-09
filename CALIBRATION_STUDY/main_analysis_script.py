#!/usr/bin/env python

'''
This is the wrapper for the calibration study DTI analysis
Created by Kirstie Whitaker
Email: kw401@cam.ac.uk
'''

#==============================================================================
# IMPORTS
import os
import numpy as np
import itertools as it
#------------------------------------------------------------------------------
from get_b0_orders import get_b0_orders
from run_registrations import run_registrations
from run_roistats import run_roistats
from wrangle_text_files import wrangle_text_files
from read_in_data import read_in_data
from set_colors_shapes import set_colors_shapes
from plot_data import plot_data
#==============================================================================

#==============================================================================
# Define the data directory
data_dir = '/work/imagingD/NSPN/workspaces/kw401/BOOTSTRAPPING_ANALYSES'
#==============================================================================

#==============================================================================
# Define some variables
#  subs: subject ids
#
#  locs: scanner locations
#
#  scans: which of the DTI scans you're using
#       (This only applies to the UCL data where there are two.
#       At all other sites there is only one.)
#
#  incl_excl: either the original data or data excluding volume 31
#       which is not the same at CBSU as it is at the other sites
#
#  n_b0s: number of b0s used to calculate tensor
#
#  ec_b0: the b0 that was used as the eddy_correct target volume
#
#  b0_order: the order that the b0s were in the 4D file
#
#  sep_av: whether the b0s were averaged before the tensor was fit
#       or kept separate
#
#  transform: the registration path from MNI to DTI space
#
#  roi_name: the name of the roi
#------------------------------------------------------------------------------
subs = [ '06', '07', '08' ]
locs = [ 'WBIC', 'CBSU', 'UCL' ]
scans = [ 'DTI_2A', 'DTI_2B' ]
incl_excl_list = [ 'EXCL_VOL31' ]
n_b0s_list = [ '1', '2', '3', '4', '5', '6' ]
#ec_b0 = [ '00', '14', '27', '41', '54', '68' ]
#b0_order = [ '00_14_27_41_54_68' ]
sep_av_list = [ 'SEPARATE' ] 
transform = [ 'MNI_DIFF_FA_DIRECT', 'MNI_DIFF_VIA_HIGHRES_LIN', 'MNI_DIFF_VIA_HIGHRES_NL_BBR' ]
roi_name = [ 'lcing', 'rcing', 'wholebrain', 'bodycc' ]
#==============================================================================

#==============================================================================
# And now get going:
"""
b0_orders = get_b0_orders(1)
    
for incl_excl, sep_av, b0_order in it.product(incl_excl_list, sep_av_list, b0_orders):
    '''
    Loop through the combinations of include/exclude volume 31,
    whether the b0s were combined first or not,
    and the ways of ordering just 1 b0 (6)
    '''
    run_registrations(data_dir, incl_excl, 1, b0_order, sep_av, subs, locs, scans)

for incl_excl, sep_av, n_b0s in it.product(incl_excl_list, sep_av_list, n_b0s_list):
    '''
    Loop through the combinations of include/exclude volume 31,
    whether the b0s were combined first or not,
    and the number of b0s that were included
    '''
    b0_orders = get_b0_orders(np.int(n_b0s))
    
    for b0_order in b0_orders:
        '''
        Also loop through the different orders of the b0s
        '''
        run_roistats(data_dir, incl_excl, n_b0s, b0_order, sep_av, subs, locs, scans)

"""
    
for incl_excl, n_b0s, sep_av, transform, roi_name in it.product(incl_excl, n_b0s, sep_av, transform, roi_name):
    '''
    Loop through all the combinations of include/exclude volume 31,
        the number of b0s,
        whether the b0s were combined first or not,
        how the roi was transformed from standard space,
        and all the rois
    '''

    b0_orders = get_b0_orders(np.int(n_b0s))

    for b0_order in b0_orders:
    
        results_file, results_dir = wrangle_text_files(data_dir, incl_excl, n_b0s,
                                            b0_order, sep_av, transform,
                                            roi_name, subs, locs, scans)
        
        data = read_in_data(results_file)
        
        colors, shapes = set_colors_shapes()
        
        plot_data(data, results_dir, roi_name, colors, shapes)


