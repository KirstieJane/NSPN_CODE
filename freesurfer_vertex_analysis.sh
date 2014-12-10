#!/bin/bash

#=============================================================================
# Created by Kirstie Whitaker on 27th October 2014
# Contact kw401@cam.ac.uk
#
# Edited on 26th November 2014 to limit smoothing kernels to 10mm and 15mm
#
# This code completes analysis of any surface measure at 10mm and 15mm
# smoothing kernels of subjects defined in the fsgd file.
#
# It corrects for multiple comparisons at p < 0.05 for both the cluster 
# forming and cluster testing thresholds as well as cluster forming p < 0.01
# and cluster testing p < 0.05.
#
#-----------------------------------------------------------------------------
# USAGE: freesurfer_vertex_analysis.sh <analysis_dir> <fsgd> <contrast> <measure>
#
# Note that this code assumes that recon-all has been run for all subjects
# in the fsgd file and that the appropriate SUBJECTS_DIR has already been set
# as an enviromental variable
#=============================================================================

#=============================================================================
# Define usage function
#=============================================================================
function usage {

    echo "freesurfer_vertex_analysis.sh <analysis_dir> <fsgd> <contrast> <measure> <dods_doss>"
    echo "    analysis_dir is wherever you want to save your output"
    echo "    fsgd is the freesurfer group descriptor file - must end is .fsgd"
    echo "    contrast file contains the contrast of interest - must end in .mtx"
    echo "    measure is whatever surface measure you're interested in - eg: thickness"
    echo "    dods_doss is one of dods and doss depending on whether you model interactions"
    exit
}

#=============================================================================
# Read in command line arguments
#=============================================================================

# analysis_dir is wherever you want to save your output
analysis_dir=$1

# fsgd is the freesurfer group descriptor file - must end in .fsgd
fsgd=$2

# contrast file contains your contrast of interest - must end in .mtx
contrast_file=$3

# measure is whatever surface measure you're interested in - eg: thickness
measure=$4
measure_name=${measure%.*}

# dods_doss is which of different onset different slope (DODS)
# or different onset same slope (DOSS) you'd like to run
dods_doss=$5

#=============================================================================
# Check that the files all exist etc
#=============================================================================
if [[ ! -f ${fsgd} ]]; then
    echo "FS GROUP DESCRIPTOR FILE does not exist. Check ${fsgd}"
    print_usage=1
fi

if [[ ! -f ${contrast_file} ]]; then
    echo "CONTRAST FILE does not exist. Check ${contrast_file}"
    print_usage=1
fi

if [[ -z ${measure} ]]; then
    echo "MEASURE not set - assuming thickness"
    measure=thickness
    measure_name=${measure%.*}
fi

if [[ -z ${dods_doss} ]]; then
    echo "DODS DOSS not set - assuming different onset different slope"
    dods_doss=dods
fi

if [[ ${print_usage} == 1 ]]; then
    usage
fi

#=============================================================================
# Define a couple of variables
#=============================================================================
# Figure out the analysis name from the Freesurfer group descriptor file
analysis_name=`basename ${fsgd} .fsgd`

# Figure out the contrast name from the contrast file
contrast_name=`basename ${contrast_file} .mtx`

#=============================================================================
# Get started
#-----------------------------------------------------------------------------
# Resample all the subjects in the Freesurfer group descriptor file
# to fsaverage space for each hemisphere separately
#=============================================================================

# Process for each hemisphere separately
for hemi in lh rh; do

    # Process the individual data as defined in the fsgd unless it's already
    # been completed
    if [[ ! -f ${analysis_dir}/${hemi}.${analysis_name}.${measure_name}.00.mgh ]]; then
    
        mris_preproc \
          --fsgd ${fsgd}      `# Freesurfer group descriptor file` \
          --target fsaverage  `# Target file to which all inputs will be aligned` \
          --hemi ${hemi}      `# Hemisphere` \
          --meas ${measure}   `# Surface measure to be represented in target space` \
          --fwhm 0            `# Smooth after registration to fsaverage with a gaussian kernel of ${fwhm} mm` \
          --out ${analysis_dir}/${hemi}.${analysis_name}.${measure_name}.00.mgh
          
    fi
    
    # Smooth the data at a variety of gaussian kernel sizes
    for fwhm in 10 15; do
    
        # Because all these analyses are going to get messy we should create
        # an appropriately named directory
        glm_dir=${analysis_dir}/GLM.${hemi}.${analysis_name}.${measure_name}.${fwhm}
        
        mkdir -p ${glm_dir}
        
        if [[ ! -f ${glm_dir}/${hemi}.${analysis_name}.${measure_name}.${fwhm}.mgh ]]; then
        
            mri_surf2surf \
              --hemi ${hemi}  `# Hemisphere` \
              --s fsaverage   `# Source and target subject are the same` \
              --fwhm ${fwhm}  `# Smooth surface to full width half maximum of (eg) 10` \
              --cortex        `# Only smooth vertices that are within the cortex label` \
              --sval ${analysis_dir}/${hemi}.${analysis_name}.${measure_name}.00.mgh \
                              `# Input surface file` \
              --tval ${glm_dir}/${hemi}.${analysis_name}.${measure_name}.${fwhm}.mgh \
                              `# Output surface file - will be same dimensions as the input file`
                              
        fi
        
        # Calculate the mean across all subjects for visualisation purposes
        
        if [[ ! -f ${glm_dir}/${hemi}.${analysis_name}.${measure_name}.${fwhm}.MEAN.mgh ]]; then
            
            mri_concat ${glm_dir}/${hemi}.${analysis_name}.${measure_name}.${fwhm}.mgh \
                        --o ${glm_dir}/${hemi}.${analysis_name}.${measure_name}.${fwhm}.MEAN.mgh \
                        --mean
                        
        fi
        
        # Now run the general linear model
        
        if [[ ! -f ${glm_dir}/${contrast_name}/sig.mgh ]]; then
        
            mri_glmfit \
                --y ${glm_dir}/${hemi}.${analysis_name}.${measure_name}.${fwhm}.mgh \
                                     `# Input surface data` \
                --fsgd ${fsgd}       `# Freesurfer group descriptor file` \
                     ${dods_doss}    `# dods stands for different offset different slope - usually the right choice` \
                --C ${contrast_file}      `# Contrast file` \
                --surf fsaverage     `# Common space surface file` \
                            ${hemi}  `# Hemisphere` \
                --cortex             `# only test within the cortex label` \
                --glmdir ${glm_dir}  `# GLM directory for output data`
                
        fi

        # And finally calculate cluster correction for the p values
        
        # Calculate both positive and negative findings
        
        for direction in pos neg; do 
        
            for thresh in 1.3 2; do
            
                if [[ ! -f ${glm_dir}/${contrast_name}/cache.th${thresh/.}.${direction}.sig.cluster.mgh ]]; then
        
                    # Here we're using a cached simulation
                    # see the documentation to run a permutation test
                    mri_glmfit-sim \
                        --glmdir ${glm_dir}  `# GLM directory - contains the output of the glm fit` \
                        --cache ${thresh}    `# Set the cluster forming threshold of -log10(p) [ 2 <--> 0.01; 3 <--> 0.001 etc ]` \
                        ${direction}         `# Consider positive or negative results separately` \
                        --cwp 0.05           `# Keep clusters that have p < 0.05` \
                        --2spaces            `# Correct for the fact that you have two hemispheres` 
                                            
                    mris_calc -o ${glm_dir}/${contrast_name}/gamma_th${thresh/.}.${direction}.sig.cluster.mgh \
                                ${glm_dir}/${contrast_name}/gamma.mgh \
                                masked \
                                ${glm_dir}/${contrast_name}/cache.th${thresh/.}.${direction}.sig.cluster.mgh

                fi
                
            done # Close the thresh loop
                            
        done # Close the direction loop
                    
    done # Close fwhm loop
done # Close hemi loop

echo "All done!"

