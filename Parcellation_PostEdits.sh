#!/bin/bash

#================================================================
# Parcellation post-edits
# Created by Kirstie Whitaker
# On 3rd September 2015
# Heavily based on Rafael R-G's subParcellation code
# Contact kw401@cam.ac.uk with questions
#================================================================

#==============================================================================
# USAGE: Parcellation_PostEdits.sh <data_dir>
#==============================================================================
function usage {

    echo "USAGE: Parcellation_PostEdits.sh <study_dir> <sublist>"
    echo "       Where <study_dir> is the parent directory to"
    echo "       the SUB_DATA directory and expects to find"
    echo "       SUB_DATA inside it and then the standard"
    echo "       NSPN directory structure and <sublist> is a text"
    echo "       file containing the NSPN ids and should be inside"
    echo "       <study_dir>"
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
sublist=$2

if [[ ! -d ${study_dir} ]]; then
    echo "**** STUDY DIRECTORY does not exist ****"
    usage
fi

if [[ ! -f ${sublist} ]]; then
    echo "**** SUBLIST does not exist ****"
    usage
fi
    
#================================================================
# Set some useful variables
#================================================================
sub_data_dir=${study_dir}/SUB_DATA
fsaverage_subid=fsaverageSubP

#================================================================
# Transform the fsaverage parcellations for each subject
# in sublist
#================================================================
for sub in `cat ${sublist}`; do
    
    SUBJECTS_DIR=${sub_data_dir}
    subjid=MRI0
        
    for hemi in lh rh; do
        
        # Transform the surface parcellation from fsaverage space 
        # to indiviual native space
        mri_surf2surf --srcsubject ${fsaverage_subid} \
                        --sval-annot ${SUBJECTS_DIR}/${fsaverage_subid}/label/${hemi}.500.aparc \
                        --trgsubject ${sub}/SURFER/${subjid} \
                        --trgsurfval ${SUBJECTS_DIR}/${sub}/SURFER/${subjid}/label/${hemi}.500.aparc \
                        --hemi ${hemi}
	done
    
	# Transform indivual surface parcellation to individual volume parcellation
	mkdir -p ${SUBJECTS_DIR}/${sub}/SURFER/${subjid}/parcellation/
	mri_aparc2aseg --s ${sub}/SURFER/${subjid} \
                    --o ${SUBJECTS_DIR}/${sub}/SURFER/${subjid}/parcellation/500.aparc.nii.gz \
                    --annot 500.aparc \
                    --rip-unknown \
                    --hypo-as-wm
   
done

#================================================================

