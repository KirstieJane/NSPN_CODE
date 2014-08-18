#!/bin/bash

# This code takes a freesurfer directory and
# runs the appropriate commands to parcellate the volume into
# lobes

#=============================================================================
# READ IN COMMAND LINE ARGUMENTS
#=============================================================================
surfer_dir=$1
ctab=$2

#=============================================================================
# CHECK THE INPUTS
#=============================================================================
if [[ -d ${surfer_dir} ]]; then
    echo "Surf dir doesn't exist - CHECK ${surfer_dir}"
    exit
fi

if [[ -f ${ctab} ]]; then
    echo "color look up table doesn't exist - CHECK ${ctab}"
    exit
fi

#=============================================================================
# DEFINE VARIABLES
#=============================================================================
# Set the subjects dir and subject id variables
SUBJECTS_DIR=${surfer_dir}/../
surf_sub=`basename ${surfer_dir}`

#=============================================================================
# COMBINE LABELS
#=============================================================================
# First you have to pull together the labels
# from the surface annotation files
for hemi in lh rh; do
    if [[ ! -f ${surfer_dir}/label/${hemi}.lobesStrict ]]; then
        mri_annotation2label --subject ${surf_sub} \
                             --hemi ${hemi} \
                             --lobesStrict \
                             ${surfer_dir}/label/${hemi}.lobesStrict
    fi
done

#=============================================================================
# LABEL WHITE MATTER
#=============================================================================
# Transform the surface annotation into a segmentation volume
# and label the white matter up to 5mm beneath the lobes
if [[ ! -f ${surfer_dir}/mri/lobes+aseg.mgz ]]; then

    mri_aparc2aseg --s ${surf_sub} \
                --labelwm \
                --rip-unknown \
                --annot lobesStrict \
                --o ${surfer_dir}/mri/lobes+aseg.mgz
fi

#=============================================================================
# EXTRACT STATISTICS
#=============================================================================
# Extract some statistics :)
if [[ ! -f ${surfer_dir}/stats/lobesStrict.stats ]]; then
    mri_segstats --i ${surfer_dir}/mri/norm.mgz \
                 --seg ${surfer_dir}/mri/lobes+aseg.mgz \
                 --sum ${surfer_dir}/stats/lobesStrict.stats \
                 --pv ${surfer_dir}/mri/norm.mgz \
                 --ctab ${ctab}

fi

