#!/usr/bin/env python

'''
This is the wrapper for the calibration study DTI analysis
Created by Kirstie Whitaker
Email: kw401@cam.ac.uk
'''

#==============================================================================
# IMPORTS
import os
import itertools as it
#==============================================================================

#==============================================================================
# Define the data directory
data_dir = '/work/imagingD/NSPN/workspaces/kw401/BOOTSTRAPPING_ANALYSES'
#==============================================================================

#==============================================================================
# Define some variables
#  subs: subject ids

#  locs: scanner locations

#  scans: which of the DTI scans you're using
#       (This only applies to the UCL data where there are two.
#       At all other sites there is only one.)

#  incl_excl: either the original data or data excluding volume 31
#       which is not the same at CBSU as it is at the other sites

#  n_b0s: number of b0s used to calculate tensor

#  ec_b0: the b0 that was used as the eddy_correct target volume

#  b0_order: the order that the b0s were in the 4D file

#  sep_av: whether the b0s were averaged before the tensor was fit
#       or kept separatesu

#  transform: the registration path from MNI to DTI space

#  roi_name: the name of the roi

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
#==============================================================================

#==============================================================================
# And now get going:

for incl_excl, n_b0s, ec_b0, b0_order, sep_av, transform, roi_name in it.product(incl_excl, n_b0s, ec_b0, b0_order, sep_av, transform, roi_name):
    
    results_file, results_dir = wrangle_text_files(data_dir, incl_excl, n_b0s,
                                        ec_b0, b0_order, sep_av, transform,
                                        roi_name, subs, locs, scans)
    
    data = read_in_data(results_file)
    
    colors, shapes = set_colors_shapes()
    
    plot_data(results_file, results_dir)

