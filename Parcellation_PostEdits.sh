#!/bin/bash

#================================================================
# Parcellation post-edits
# Created by Kirstie Whitaker
# On 3rd September 2015
# Heavily based on Rafael R-G's subParcellation code
# Contact kw401@cam.ac.uk with questions
#================================================================

#==============================================================================
# USAGE: Parcellation_PostEdits.sh <study_dir> <sub>
#==============================================================================
function usage {

    echo "USAGE: Parcellation_PostEdits.sh <study_dir> <sub>"
    echo "       Where <study_dir> is the parent directory to"
    echo "       the SUB_DATA directory and expects to find"
    echo "       SUB_DATA inside it and then the standard"
    echo "       NSPN directory structure and <sub> is an"
    echo "       NSPN subject ID that corresponds to a folder in"
    echo "       the SUB_DATA directory"
    echo ""
    echo "DESCRIPTION: This code creates the parcellation for each person"
    echo "             in their freesurfer space, and should be applied"
    echo "             after recon-all edits have been completed"
    exit
}

#=============================================================================
# READ IN COMMAND LINE ARGUMENTS
#=============================================================================

study_dir=$1
sub=$2

if [[ ! -d ${study_dir} ]]; then
    echo "**** STUDY DIRECTORY does not exist ****"
    usage
fi

if [[ -z ${sub} ]]; then
    echo "**** No subject ID given ****"
    usage
fi
    
#================================================================
# Set some useful variables
#================================================================
sub_data_dir=${study_dir}/SUB_DATA
fsaverage_subid=fsaverageSubP

#================================================================
# Transform the fsaverage parcellations
#================================================================
   
SUBJECTS_DIR=${sub_data_dir}
subjid=MRI0
    
for hemi in lh rh; do
    
    if [[ ! -f ${SUBJECTS_DIR}/${sub}/SURFER/${subjid}/label/${hemi}.500.aparc ]]; then
    
        # Transform the surface parcellation from fsaverage space 
        # to indiviual native space
        mri_surf2surf --srcsubject ${fsaverage_subid} \
                        --sval-annot ${SUBJECTS_DIR}/${fsaverage_subid}/label/${hemi}.500.aparc \
                        --trgsubject ${sub}/SURFER/${subjid} \
                        --trgsurfval ${SUBJECTS_DIR}/${sub}/SURFER/${subjid}/label/${hemi}.500.aparc \
                        --hemi ${hemi}
    fi
done

if [[ ! -f ${SUBJECTS_DIR}/${sub}/SURFER/${subjid}/parcellation/500.aparc.nii.gz ]]; then
    # Transform indivual surface parcellation to individual volume parcellation
    mkdir -p ${SUBJECTS_DIR}/${sub}/SURFER/${subjid}/parcellation/
    mri_aparc2aseg --s ${sub}/SURFER/${subjid} \
                    --o ${SUBJECTS_DIR}/${sub}/SURFER/${subjid}/parcellation/500.aparc.nii.gz \
                    --annot 500.aparc \
                    --rip-unknown \
                    --hypo-as-wm
fi
#================================================================

