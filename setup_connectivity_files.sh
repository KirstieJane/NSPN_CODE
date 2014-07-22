#!/bin/bash

# A little wrapper script that gets various files ready to run the calculate_connectivity_matrix code

# USAGE
#wrapper_connectivity_setup.sh <parcellation_file> <dti_dir> <reg_dir> <surfer_dir>

# INPUTS
parcellation_file=$1
dti_dir=$2
reg_dir=$3
surfer_dir=$4

parcellation_to_dti_transform=${reg_dir}/freesurfer_TO_diffB0_BBR.mat

parcellation_file_DTIspace=${dti_dir}/CONNECTIVITY/`basename ${parcellation_file} .nii.gz`_DTIspace.nii.gz
white_matter_file=${surfer_dir}/mri/wm.nii
white_matter_file_DTIspace=${dti_dir}/CONNECTIVITY/`basename ${white_matter_file} .nii`_DTIspace.nii.gz
parcellation_file_WhiteMatterUnderCortex_DTIspace=${dti_dir}/CONNECTIVITY/`basename ${parcellation_file} .nii.gz`_WhiteMatterUnderCortex_DTIspace.nii.gz

# RUN THE CODE
# Make the connectivity directory if it doesn't already exist
mkdir -p ${dti_dir}/CONNECTIVITY/

# Flirt the parcellation file to DTI space if it doesn't yet exist
if [[ ! -f ${parcellation_file_DTIspace} ]]; then
    flirt -in ${parcellation_file} \
            -ref ${dti_dir}/dti_ec_brain.nii.gz \
            -applyxfm \
            -init ${parcellation_to_dti_transform} \
            -out ${parcellation_file_DTIspace} \
            -interp nearestneighbour
fi

# Flirt the white matter mask to DTI space if it doesn't yet exist
if [[ ! -f ${white_matter_file_DTIspace} ]]; then
    flirt -in ${white_matter_file} \
            -ref ${dti_dir}/dti_ec_brain.nii.gz \
            -applyxfm \
            -init ${parcellation_to_dti_transform} \
            -out ${white_matter_file_DTIspace} \
            -interp nearestneighbour
fi

# Create the intersection file of the parcellation and white matter if it doesn't yet exist
if [[ ! -f ${parcellation_file_WhiteMatterUnderCortex_DTIspace} ]]; then
    fslmaths ${white_matter_file_DTIspace} -bin -mul ${parcellation_file_DTIspace} ${parcellation_file_WhiteMatterUnderCortex_DTIspace}
fi
