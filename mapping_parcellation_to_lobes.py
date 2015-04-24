#!/usr/bin/env python

import argparse

#=============================================================================
# Set up the argparser so you can read arguments from the command line
#=============================================================================
def setup_argparser():
    '''
    # Code to read in arguments from the command line
    '''
    
    # Build a basic parser.
    help_text = ( 'Map the name of a region to the appropriate freesurfer lobe \
                   as defined in https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation')
    
    sign_off = 'Author: Kirstie Whitaker <kw401@cam.ac.uk>'
    
    parser = argparse.ArgumentParser(description=help_text, epilog=sign_off)
    
    # Now add the arguments
    # Required argument: names_file
    parser.add_argument(dest='names_file', 
                            type=str,
                            metavar='names_file',
                            help='file with a list of names for each region')
    
    # Required argument: lobes_file
    parser.add_argument(dest='lobes_file', 
                            type=str,
                            metavar='lobes_file',
                            help='file that will be created with the appropriate lobe for each region in names_file')
    
    arguments = parser.parse_args()
    
    return arguments, parser


#=============================================================================
# Define some variables
#=============================================================================
# Read in the arguments from argparse
arguments, parser = setup_argparser()

# Make a list of names
names = [ line.strip() for line in open(arguments.names_file) ]

# Create a list of blanks for the lobes
# (you'll fill this in in a second)
lobes = ['.']*len(names)


#=============================================================================
# Loop through and fill in the lobes list
#=============================================================================
for i, name in enumerate(names):
    
    frontal_list = [ 'frontal', 'anteriorcingulate', 'paracentral', 'precentral', 'pars', 'insula' ]
    parietal_list = [ 'parietal', 'isthmuscingulate', 'postcentral', 'posteriorcingulate', 'precuneus', 'supramarginal' ]
    occipital_list = [ '_cuneus', 'occipital', 'lingual', 'pericalcarine' ]
    temporal_list = [ 'bankssts', 'entorhinal', 'fusiform', 'temporal', 'parahippocampal' ]
    
    for key_word in frontal_list:
        
        if key_word in name:
            
            lobes[i] = 'frontal'
    
    for key_word in occipital_list:
        
        if key_word in name:
            
            lobes[i] = 'occipital'
            
    for key_word in temporal_list:
        
        if key_word in name:
            
            lobes[i] = 'temporal'
    
    for key_word in parietal_list:
        
        if key_word in name:
            
            lobes[i] = 'parietal'


#=============================================================================
# Save the lobes list
#=============================================================================
with open(arguments.lobes_file, 'w') as f:
    f.write("\n".join(lobes))
    
    
#=============================================================================
# ALL DONE!
#=============================================================================