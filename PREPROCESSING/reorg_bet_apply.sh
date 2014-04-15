#!/bin/bash

#===================================================================
# This is a little script that copies over the output from 
# MPM processing, reorients the nifti files to standard orientation,
# brain extracts the PDw file and applies that brain extraction to 
# all the calculated files.
#
# Note that there is currently NO brain extraction applied to the T1w
# and MTw files because the optimum values haven't yet been 
# figured out (by KW). They are't in the same space as the PDw image 
# and therefore the PDw brain extractions can't easily be applied.
# However, the registration parameters ARE saved in the qform/sform
# fields of the NIFTI file so *if* someone felt very motivated to
# figure out how to apply the PDw brain extractions in the future
# that would be a superdooper plan.
#
# For now though, we're only interested in the quantitative maps. So
# PDw is the way to go :)

#-------------------------------------------------------------------

# Created by Kirstie Whitaker
# on 15th April 2014
# Please contact kw401@cam.ac.uk with any questions

#===================================================================
# DEFINE USAGE

function usage {
    echo "USAGE: reorg_bet_apply.sh <current_dir> <target_dir>"
    echo "    current_dir should contain the output from the MPM"
    echo "    processing pipeline, specifically nii files ending in"
    echo "        T1w, PDw, MTw, A, MT, MTR, MTRdiff, R1, R2s, synt"
    echo "    The new directory will strip strings from the start"
    echo "    of these file names such that all participants are"
    echo "    named in a standard way" 
    exit
}

#===================================================================
# DEFINE VARIABLES

current_dir=$1
target_dir=$2
bet_dir=${target_dir}/PDw_BrainExtractionOutput/

orig_filename_list=(PDw T1w MTw)
calc_filename_list=(A MT MTR MTRdiff R1 R2s synt)

# Check that the current directory exists
if [[ ! -d ${current_dir} ]]; then
    echo "Current directory doesn't exist - please check!"
    usage
fi

# If no arguments are passed then just print the usage and exit
if [[ $# == 0 ]]; then
    usage
fi

# Make the target and bet directory if it doesn't yet exist
mkdir -p ${bet_dir}

#===================================================================
# COPY OVER AND REORIENT ALL MPM OUTPUT FILES

echo -n "Reorienting and copying "

for filename in ${orig_filename_list[@]} ${calc_filename_list[@]}; do

    echo -n "${filename} "

    # Find the appropriate MPM output file in the current directory
    mpm_file=(`ls -d ${current_dir}/*${filename}.nii`)

    fslreorient2std ${mpm_file} ${target_dir}/${filename}.nii.gz    
done

echo "   Done!"

#===================================================================
# ONLY BRAIN EXTRACT THE PDw FILE

echo "Brain extracting PDw file"

bet ${target_dir}/PDw.nii.gz ${bet_dir}/PDw_brain.nii.gz -A

#===================================================================
# APPLY PDw BRAIN AND HEAD EXTRACTIONS TO CALCULATED FILES

echo -n "Applying brain extraction "
for filename in PDw ${calc_filename_list[@]}; do

    echo -n "${filename} "

    fslmaths ${bet_dir}/PDw_brain.nii.gz \
                -bin \
                -mul ${target_dir}/${filename}.nii.gz \
                ${target_dir}/${filename}_brain.nii.gz
                
    fslmaths ${bet_dir}/PDw_brain_outskin_mask.nii.gz \
                -bin \
                -mul ${target_dir}/${filename}.nii.gz \
                ${target_dir}/${filename}_head.nii.gz
                
done

echo "   Done!"

#===================================================================
# ALL DONE! CONGRATULATIONS.
# Fun fact - My initials (KW) were only added to the Portuguese
# alphabet in 2009.
#===================================================================

