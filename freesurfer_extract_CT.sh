#!/bin/bash

#==============================================================================
# freesurfer_extract_CT.sh <surfer_dir>
# Writen by Kirstie Whitaker on 24th March 2015
#==============================================================================

#==============================================================================
function usage {

    echo "USAGE: freesurfer_extract_CT.sh <surfer_dir>"
    echo "DESCRIPTION: This code extracts surface stats from the"
    echo "aparc, 500.aparc and lobesStrict parcellation schemes"
    exit
}
#==============================================================================
# Read in the inputs
surfer_dir=${1}

if [[ ! -f ${surfer_dir}/mri/T1.mgz ]]; then
    usage()
fi

#==============================================================================
# Set some variables
lobes_ctab=`dirname ${0}`/LobesStrictLUT.txt
parc500_ctab=`dirname ${0}`/parc500LUT.txt

SUBJECTS_DIR=${surfer_dir}/../
surf_sub=`basename ${surfer_dir}`

#==============================================================================
# Loop over both left and right hemispheres
for hemi in lh rh; do
    # Loop over parcellations
    for parc in aparc 500.aparc lobesStrict; do

        # First extract just the thickness & curvature values
        if [[ ! -f ${surfer_dir}/stats/${hemi}.${parc}.stats \
                && -f ${surfer_dir}/label/${hemi}.${parc}.annot ]]; then
            mris_anatomical_stats -a ${surfer_dir}/label/${hemi}.${parc}.annot \
                                    -f ${surfer_dir}/stats/${hemi}.${parc}.stats \
                                    ${surf_sub} \
                                    ${hemi}
        fi
    done
done
#==============================================================================
