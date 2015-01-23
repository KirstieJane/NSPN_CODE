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
# to the surface (for example: MT.mgz)
input_vol_name=$2
input_vol=${sub_dir}/mri/${input_vol_name}

#=============================================================================
# Check that the files all exist etc
#=============================================================================
if [[ ! -d /${sub_dir} ]]; then
    if [[ -d ${sub_dir} ]]; then
        sub_dir=`pwd`/${sub_dir}
    else
        echo "SUBJECT DIRECTORY does not exist. Check ${sub_dir}"
        print_usage=1
    fi
fi

if [[ ! -f ${input_vol} ]]; then
    if [[ ! -f ${input_vol}.mgz ]]; then
        echo "INPUT VOLUME does not exist. Check ${input_vol}"
        print_usage=1
    else
        input_vol=${input_vol}.mgz
    fi
fi

if [[ ${print_usage} == 1 ]]; then
    usage
fi

#=============================================================================
# Define a couple of variables
#=============================================================================
# Set the freesurfer SUBJECTS_DIR environmental variable
export SUBJECTS_DIR=`dirname ${sub_dir}`
sub=`basename ${sub_dir}`

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

    # Measure at 10 fractional depths between the grey and white matter boundary
    # and the pial surface
    for frac in `seq -f %+02.2f 0 0.1 1`; do

        if [[ ! -f ${sub_dir}/surf/${hemi}.${measure_name}_projfrac${frac/.}.mgh ]]; then
        
            mri_vol2surf --mov ${input_vol} \
                            --o ${sub_dir}/surf/${hemi}.${measure_name}_projfrac${frac}.mgh \
                            --regheader ${sub} \
                            --projfrac ${frac} \
                            --interp nearest \
                            --surf white \
                            --hemi ${hemi} 
        fi

    done # Close the fraction of cortical thickness loop
    
    # Measure at 25 different depths starting from the pial surface and descending
    # by 0.2mm for each step
    for dist in `seq -f %+02.2f -5 0.2 0`; do

        if [[ ! -f ${sub_dir}/surf/${hemi}.${measure_name}_projdist${dist}.mgh ]]; then
        
            mri_vol2surf --mov ${input_vol} \
                            --o ${sub_dir}/surf/${hemi}.${measure_name}_projdist${dist}.mgh \
                            --regheader ${sub} \
                            --projdist ${dist} \
                            --interp nearest \
                            --surf pial \
                            --hemi ${hemi} 
                            
        fi
        
    done # Close the absolute distance loop
    
    # Measure at 10 different depths starting from the grey/white matter
    # boundary and descending by 0.2mm into white matter for each step
    for dist in `seq -f %+02.2f -2 0.2 0`; do
    
        if [[ ! -f ${sub_dir}/surf/${hemi}.${measure_name}_projdist${dist}_fromBoundary.mgh ]]; then
        
            mri_vol2surf --mov ${input_vol} \
                            --o ${sub_dir}/surf/${hemi}.${measure_name}_projdist${dist}_fromBoundary.mgh \
                            --regheader ${sub} \
                            --projdist ${dist} \
                            --interp nearest \
                            --surf white \
                            --hemi ${hemi} 
                            
        fi
        
    done # Close the absolute distance from the boundary loop

done # Close the hemi loop