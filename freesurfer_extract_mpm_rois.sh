#!/bin/bash

# This code takes the MT, PDw and R2* files from the mpm_dir,
# converts them to freesurfer space (in the surfer_dir) and then
# extracts a bunch of different values from each of them

# USAGE: mpm_freesurfer_roi_extract <mpm_dir> <surfer_dir>

# INPUTS
mpm_dir=$1
surfer_dir=$2

lobes_ctab=`dirname ${0}`/LobesStrictLUT.txt
                     
# Loop through the mpm outputs that you're interested in
for mpm in MT R2s; do

    if [[ ! -f ${mpm_dir}/${mpm}_head.mgz ]]; then
        # Convert the mpm nii file to mgz format
        mri_convert ${mpm_dir}/${mpm}_head.nii.gz ${mpm_dir}/${mpm}_head.mgz
    fi
    
    if [[ ! -f ${surfer_dir}/mri/${mpm}.mgz ]]; then
        # Align the mgz file to "freesurfer" anatomical space
        mri_vol2vol --mov ${mpm_dir}/${mpm}_head.mgz \
                    --targ ${surfer_dir}/mri/T1.mgz \
                    --regheader \
                    --o ${surfer_dir}/mri/${mpm}.mgz \
                    --no-save-reg
    fi
    
    # Extract roi values
    #=== wmparc
    if [[ ! -f ${surfer_dir}/stats/${mpm}_wmparc.stats ]]; then
        mri_segstats --i ${surfer_dir}/mri/${mpm}.mgz \
                     --seg ${surfer_dir}/mri/wmparc.mgz \
                     --ctab ${FREESURFER_HOME}/WMParcStatsLUT.txt \
                     --sum ${surfer_dir}/stats/${mpm}_wmparc.stats \
                     --pv ${surfer_dir}/mri/norm.mgz
    fi
    
    #=== aseg
    if [[ ! -f ${surfer_dir}/stats/${mpm}_aseg.stats ]]; then
        mri_segstats --i ${surfer_dir}/mri/${mpm}.mgz \
                     --seg ${surfer_dir}/mri/aseg.mgz \
                     --ctab ${FREESURFER_HOME}/ASegStatsLUT.txt \
                     --sum ${surfer_dir}/stats/${mpm}_aseg.stats \
                     --pv ${surfer_dir}/mri/norm.mgz
    fi
    
    #=== lobesStrict
    if [[ ! -f ${surfer_dir}/stats/${mpm}_lobesStrict.stats ]]; then
        mri_segstats --i ${surfer_dir}/mri/${mpm}.mgz \
                     --seg ${surfer_dir}/mri/lobes+aseg.mgz \
                     --ctab ${lobes_ctab} \
                     --sum ${surfer_dir}/stats/${mpm}_lobesStrict.stats \
                     --pv ${surfer_dir}/mri/norm.mgz
    fi
   
    #=== 500.aparc_cortical_expanded_consecutive_WMoverlap
    # Only run this if there is a 500 cortical parcellation
    if [[ ! -f ${surfer_dir}/stats/${mpm}_500cortExpConsecWMoverlap.stats \
            && -f ${surfer_dir}/parcellation/500.aparc_cortical_expanded_consecutive.nii.gz ]]; then
        
        # Create the overlap file if it doesn't already exist
        if [[ ! -f ${surfer_dir}/parcellation/500.aparc_cortical_expanded_consecutive_WMoverlap.nii.gz ]]; then
        
            fslmaths ${surfer_dir}/parcellation/500.aparc_whiteMatter.nii.gz \
                        -bin \
                        -mul ${surfer_dir}/parcellation/500.aparc_cortical_expanded_consecutive.nii.gz \
                        ${surfer_dir}/parcellation/500.aparc_cortical_expanded_consecutive_WMoverlap.nii.gz
        fi
        
        mri_segstats --i ${surfer_dir}/mri/${mpm}.mgz \
                     --seg ${surfer_dir}/parcellation/500.aparc_cortical_expanded_consecutive_WMoverlap.nii.gz \
                     --sum ${surfer_dir}/stats/${mpm}_500cortExpConsecWMoverlap.stats \
                     --pv ${surfer_dir}/mri/norm.mgz
    fi   
done

