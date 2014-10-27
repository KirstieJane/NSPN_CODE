#!/usr/bin/env python

#=============================================================================
# Created by Kirstie Whitaker
# September 2014
# Contact: kw401@cam.ac.uk
#=============================================================================

#=============================================================================
# IMPORTS
#=============================================================================
import os
import sys
import argparse
import numpy as np
import scipy.io as sio
from glob import glob

import pandas as pd
import nibabel as nib
from surfer import Brain, io

import itertools as it
from scipy.stats.stats import linregress

import matplotlib.pylab as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


#=============================================================================
# FUNCTIONS
#=============================================================================
def setup_argparser():
    '''
    Code to read in arguments from the command line
    Aso allows you to change some settings
    '''
    # Build a basic parser.
    help_text = ('Plot values on a freesurfer surface')
    
    sign_off = 'Author: Kirstie Whitaker <kw401@cam.ac.uk>'
    
    parser = argparse.ArgumentParser(description=help_text, epilog=sign_off)
    
    # Now add the arguments
    # Required argument: dti_dir
    parser.add_argument(dest='overlay_file', 
                            type=str,
                            metavar='overlay_file',
                            help='overlay file in with the hemisphere in the first two letters')
                            
    parser.add_argument(dest='output_dir', 
                            type=str,
                            metavar='output_dir',
                            help='output directory')
                          
    parser.add_argument('--subject_id',
                            type=str,
                            metavar='subject id',
                            help='freesurfer subject id',
                            default='fsaverage')
                            
    parser.add_argument('-sd', '--subjects_dir',
                            type=str,
                            metavar='subjects_dir',
                            help='freesurfer subjects dir',
                            default=os.environ["SUBJECTS_DIR"])
                            
    parser.add_argument('-c', '--cmap',
                            type=str,
                            metavar='cmap',
                            help='colormap',
                            default='RdBu_r')
                            
    parser.add_argument('--center',
                            action='store_true',
                            help='center the color bar around 0')
                            
    parser.add_argument('-t', '--thresh',
                            type=float,
                            metavar='thresh',
                            help='mask values below this value',
                            default=-98)
                            
    parser.add_argument('-l', '--lower',
                            type=float,
                            metavar='lowerthr',
                            help='lower limit for colorbar',
                            default=None)

    parser.add_argument('-u', '--upper',
                            type=float,
                            metavar='upperthr',
                            help='upper limit for colorbar',
                            default=None)
                            
    
    parser.add_argument('-s', '--surface',
                            type=str,
                            metavar='surface',
                            help='surface - one of "pial", "inflated" or "both"',
                            default='both')
                            
    arguments = parser.parse_args()
    
    return arguments, parser

#------------------------------------------------------------------------------
def plot_surface(vtx_data, subject_id, subjects_dir, hemi, surface, output_dir, prefix, l, u, cmap, center, thresh):
    # Open up a brain in pysurfer
    brain = Brain(subject_id, hemi, surface,
                  subjects_dir = subjects_dir,
                  config_opts=dict(background="white",
                                   height=665,
                                   width=800))

    # Figure out the min and max for the color bar
    if l == None:
        l = roi_data[roi_data>-99].min()
        l = np.floor(l*20)/20.0
    if u == None:
        u = roi_data[roi_data>-99].max()
        u = np.ceil(u*20)/20.0
    
    if center:
        # Make sure the colorbar is centered
        if l**2 < u **2:
            l = u*-1
        else:
            u = l*-1
    
    # Add your data to the brain
    brain.add_data(vtx_data,
                    l, 
                    u,
                    thresh = thresh,
                    colormap=cmap,
                    alpha=.8)
    
    # Save the images for medial and lateral
    # putting a color bar on all of them
    brain.save_imageset(prefix = os.path.join(output_dir, prefix),
                        views = views_list, 
                        colorbar = range(len(views_list)) )

#----------------------------------------------------------------------------- 
def combine_pngs(measure, surface, output_dir):
    '''
    Find four images and combine them into one nice picture
    '''
    figsize = (5,4)
    fig = plt.figure(figsize = figsize, facecolor='white')

    grid = gridspec.GridSpec(2, 2)
    grid.update(left=0, right=1, top=1, bottom = 0.08, wspace=0, hspace=0)

    f_list = [ '_'.join([os.path.join(output_dir, measure), 'lh', surface, 'lateral.png']),
               '_'.join([os.path.join(output_dir, measure), 'rh', surface, 'lateral.png']),
               '_'.join([os.path.join(output_dir, measure), 'lh', surface, 'medial.png']),
               '_'.join([os.path.join(output_dir, measure), 'rh', surface, 'medial.png']) ]

    # Plot each figure in turn
    for g_loc, f in zip(grid, f_list):
        ax = plt.Subplot(fig, g_loc)
        fig.add_subplot(ax)
        img = mpimg.imread(f)
        # Crop the figures appropriately
        # NOTE: this can change depending on which system you've made the 
        # images on originally - it's a bug that needs to be sorted out!
        if 'lateral' in f:
            img_cropped = img[75:589,55:(-50),:]
        else:
            img_cropped = img[45:600,25:(-25),:]
        ax.imshow(img_cropped, interpolation='none')
        ax.set_axis_off()

    # Add the bottom of one of the images as the color bar
    # at the bottom of the combo figure
    grid_cbar = gridspec.GridSpec(1,1)
    grid_cbar.update(left=0, right=1, top=0.08, bottom=0, wspace=0, hspace=0)
    ax = plt.Subplot(fig, grid_cbar[0])
    fig.add_subplot(ax)
    img = mpimg.imread(f)
    img_cbar = img[600:,:]
    ax.imshow(img_cbar, interpolation='none')
    ax.set_axis_off()
    
    # Save the figure
    filename = os.path.join(output_dir, '{}_{}_combined.png'.format(measure, surface))
    print filename
    fig.savefig(filename, bbox_inches=0, dpi=300)


#=============================================================================
# SET SOME VARIABLES
#=============================================================================
# Read in the arguments from argparse
arguments, parser = setup_argparser()

overlay_file = arguments.overlay_file
output_dir = arguments.output_dir
subject_id = arguments.subject_id
subjects_dir = arguments.subjects_dir
l = arguments.lower
u = arguments.upper
cmap = arguments.cmap
center = arguments.center
surface = arguments.surface
thresh = arguments.thresh

if surface == 'both':
    surface_list = [ "inflated", "pial" ]
elif surface == 'inflated':
    surface_list = [ "inflated" ]
elif surface == 'pial':
    surface_list = [ "pial" ]
else:
    print "Do not recognise surface. Check {}".format(surface)
    parser.print_help()
    sys.exit()
    
hemi_list = [ "lh", "rh" ]
views_list = [ 'medial', 'lateral' ]

# Check that the inputs exist:
for hemi in hemi_list:
    f = os.path.join(os.path.dirname(overlay_file), hemi + os.path.basename(overlay_file)[2:])
    if not os.path.isfile(f):
        print "{} overlay file doesn't exist".format(hemi)
        sys.exit()

# Make the output directory if it doesn't already exist    
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


#=============================================================================
# OVERLAY THE DATA
#=============================================================================
for hemi, surface in it.product(hemi_list, surface_list):

    prefix = '_'.join([hemi, surface])
    
    f = os.path.join(os.path.dirname(overlay_file), hemi + os.path.basename(overlay_file)[2:])
        
    vtx_data = io.read_scalar_data(f)
    
    # Show this data on a brain
    plot_surface(vtx_data, subject_id, subjects_dir,
                     hemi, surface, 
                     output_dir, prefix,
                     l, u, cmap, center,
                     thresh)

#============================================================================= 
# COMBINE THE IMAGES
#============================================================================= 
for surface in surface_list:
    combine_pngs(measure, surface, output_dir)
