#!/bin/bash

#=============================================================================
# Created by Kirstie Whitaker on 4th December 2014
# Contact kw401@cam.ac.uk
#
# This code projects a quantitative volume from a variety of different depths
# onto it's corresponding freesurfer surface.
#
#-----------------------------------------------------------------------------
# USAGE: freesurfer_project_to_surface.sh <subject_dir> <input_volume_name>
#
# Note that this code assumes that recon-all has been run and that the input
# volume has already been registered to freesurfer space
#=============================================================================

#=============================================================================
# Define usage function
#=============================================================================
function usage {

    echo "freesurfer_project_to_surface.sh <subject_dir> <input_volume>"
    echo "    subject_dir is the freesurfer subject directory"
    echo "    input_volume is the volume in <subject_dir>/mri that you want"
    echo "        to project to the surface"
    exit
}

#=============================================================================
# Read in command line arguments
#=============================================================================

# sub_dir is the freesurfer subject directory
sub_dir=$1

# input_vol is the volume in <subject_dir>/mri that you want to project
# to the surface
input_vol=$2

#=============================================================================
# Check that the files all exist etc
#=============================================================================
if [[ ! -d ${sub_dir} ]]; then
    echo "SUBJECT DIRECTORY does not exist. Check ${sub_dir}"
    print_usage=1
fi

if [[ ! -f ${input_vol} ]]; then
    echo "INPUT VOLUME does not exist. Check ${input_vol}"
    print_usage=1
fi

if [[ ${print_usage} == 1 ]]; then
    usage
fi

#=============================================================================
# Define a couple of variables
#=============================================================================
# Set the freesurfer SUBJECTS_DIR environmental variable
export SUBJECTS_DIR=`dirname ${subject_dir}`

# Figure out the measure name from the input volume
measure_name=`basename ${input_vol} .mgz`

#=============================================================================
# Get started
#-----------------------------------------------------------------------------
# Resample all the subjects in the Freesurfer group descriptor file
# to fsaverage space for each hemisphere separately
#=============================================================================

# Process for each hemisphere separately
for hemi in lh rh; do

    for frac in `seq -f %+02.2f -1 0.1 1`; do

        if [[ ! -f ${hemi}.${measure_name}_projfrac${frac/.}.mgh ]]; then
        
            mri_vol2surf --mov ${input_vol} \
                            --o ${sub_dir}/surf/${hemi}.${measure_name}_projfrac${frac}.mgh \
                            --regheader \
                            --projfrac ${frac} \
                            --interp nearest \
                            --surf white \
                            --hemi ${hemi} 
        fi

    done # Close the fraction of cortical thickness loop
    
    for dist in `seq -f %+02.2f -5 0.2 0`; do

        if [[ ! -f ${hemi}.${measure_name}_projdist${abs}.mgh ]]; then
        
            mri_vol2surf --mov ${input_vol} \
                            --o ${sub_dir}/surf/${hemi}.${measure_name}_projdist${dist}.mgh \
                            --regheader \
                            --projdist ${dist} \
                            --interp nearest \
                            --surf pial \
                            --hemi ${hemi} 
                            
        fi
        
    done # Close the absolute distance loop
    
done # Close the hemi loop