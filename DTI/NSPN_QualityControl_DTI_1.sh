#!/bin/bash

#==============================================================================
#               NAME:  NSPN_QualityControl_DTI_1.sh
#
#        DESCRIPTION:  This script takes an input directory, searches
#                      for the DTI data, bvals and bvecs files, and checks
#                      that they have the right number of data points.
#                      It also creates a png of the slices so you can
#                      look at the brains.
#
#        PARAMETER 1:  Subject nifti folder
#                       eg: /work/imagingA/NSPN/data/rawdata/study/ucl ...
#                               ... /nii4d/DTI/12385/MRI0
#
#              USAGE:  NSPN_QualityControl_DTI_1.sh <sub_nifti_folder>
#                          eg: NSPN_QualityControl_DTI_1.sh ${dir}
#
#             AUTHOR:  Kirstie Whitaker
#                          kw401@cam.ac.uk
#            CREATED:  8th May 2013
#==============================================================================

sub_dir=$1
sub_dir=/work/imagingA/NSPN/data/rawdata/study/ucl/nii4d/DTI/12385/MRI0

# First thing to do is find the .nii files in each folder
nii_files=(`ls -d ${sub_dir}/*/*nii 2> /dev/null`)

dti_data_found=0

if [[ ${#nii_files[@]} == 0 ]]; then
    print "No DTI data found"
else
    for file in ${nii_files[@]}; do
        n_vols=`fslnvols ${file}`
        if [[ ${n_vols} == 69 ]]; then
            print "DTI data has 69 volumes"
            mkdir ${sub_dir}/DTI_SPLIT
            fslsplit ${file} ${sub_dir}/DTI_SPLIT/dti_
            slicesdir ${sub_dir}/DTI_SPLIT/*
            # Create a little marker to show that we've found a
            # nii file with 69 volumes
            let dti_data_found=${dti_data_found}+1
        fi
    done
fi

# Now lets just check that we only have ONE nii file with
# 69 volumes
if [[ ${dti_data_found} > 1 ]]; then
    echo "ERROR: too many nii files"
fi

# Now do a quick check on the bvals file
bvals_files=(`ls -d ${sub_dir}/*/*bvals`)
if [[ ${#bvals_files[@]} == 0 ]]; then
    print "No bvals file found"
elif [[ ${#bvals_files[@]} == 1 ]]; then
    n_bvals=`cat ${bvals_files[0]} | wc -w`
    if [[ ${n_bvals} == 69 ]]; then
        echo "Bvals data has 69 entries"
    else
        echo "ERROR: Bvals data does not have the correct number of entries"
else
    echo "ERROR: Too many bvals files"
fi

# And finally a similar check on the bvecs file
bvecs_files=(`ls -d ${sub_dir}/*/*bvecs`)
if [[ ${#bvecs_files[@]} == 0 ]]; then
    print "No bvecs file found"
elif [[ ${#bvecs_files[@]} == 1 ]]; then
    n_bvecs=`cat ${bvecs_files[0]} | wc -w`
    if [[ ${n_bvecs} == 207 ]]; then
        echo "Bvecs data has 207 (69 * 3) entries"
    else
        echo "ERROR: Bvecs data does not have the correct number of entries"
else
    echo "ERROR: Too many bvecs files"
fi
