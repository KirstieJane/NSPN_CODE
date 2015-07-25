#!/bin/bash

#====================================================================
# Created by Kirstie a while ago but only 
# commented on July 25th 2015
#
# This code takes the output of the MPM processing
# and masks the background to have values of 0s
# 
# For each calculated file - A, MT, MTR, MRTdiff, R1, R2s and synt
# there is an _brain.nii.gz file and an _head.nii.gz file.
# 
# The brain file is the output of a bet command in FSL. It may be
# ok but it may be too harsh.
#
# The head file gives you the data within the head, but masks
# all the background to zeros. This is the file that should
# be passed to subsequent analyses if they include brain
# extraction (skull stripping) in their pipelines.
#
# For any questions or comments please see the HappyPenguin/NSPN_CODE/ 
# wiki at:
# https://github.com/HappyPenguin/NSPN_CODE/wiki/Masking-background-of-MPM-files 
# and leave a message for Kirstie (HappyPenguin) if you're unsure.
#
# Kx
#====================================================================

#====================================================================
# Check the input file is there
#====================================================================
pdw_file=$1

if [[ ! -f ${pdw_file} ]]; then
    echo "PDw file does not exist"
    echo "Check ${pdw_file}"
    echo "USAGE: mpm_masking_brainextraction.sh <pdw_file_in_mpm_dir>"
    exit    
fi 

#====================================================================
# Set a couple of variables
#====================================================================
mpm_dir=`dirname ${pdw_file}`

bet_dir=${mpm_dir}/PDw_BrainExtractionOutput/

orig_filename_list=(PDw T1w MTw)
calc_filename_list=(A MT MTR MTRdiff R1 R2s synt)

#====================================================================
# First convert all the input files to .nii.gz
# and make sure they're in FSL standard orientation
#====================================================================
for f_name in ${orig_filename_list[@]} ${calc_filename_list[@]}; do

    mpm_file=(`ls -d ${mpm_dir}/*${f_name}.nii*`)
    
    fslreorient2std ${mpm_file} ${mpm_dir}/${f_name}.nii.gz    
    
done

#====================================================================
# Do the brain extraction on the PDw file
#====================================================================
mkdir -p ${bet_dir}
  
bet ${mpm_dir}/PDw.nii.gz ${bet_dir}/PDw_brain.nii.gz –A
            
#====================================================================
# Now make the brain and head files for each of the
# calculated MPM files
#====================================================================
for f_name in PDw ${calc_filename_list[@]}; do
    fslmaths ${bet_dir}/PDw_brain.nii.gz \
                -bin \
                -mul ${mpm_dir}/${f_name}.nii.gz \
                ${mpm_dir}/${f_name}_brain.nii.gz
                
    fslmaths ${bet_dir}/PDw_brain_outskin_mask.nii.gz \
                -bin \
                -mul ${mpm_dir}/${f_name}.nii.gz \
                ${mpm_dir}/${f_name}_head.nii.gz

done # Close the mpm calculated file loop

#====================================================================
# All done!
#====================================================================