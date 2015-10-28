#!/bin/bash

#================================================================
# Freesurfer Extract ROIs
# Created by Kirstie Whitaker
# On 3rd September 2015
# Must be run after parcellation and recon-all
# Contact kw401@cam.ac.uk with questions
#================================================================

#==============================================================================
# USAGE: Extract_ParcROIs.sh <study_dir> <sub>
#==============================================================================
function usage {

    echo "USAGE: Extract_ParcROIs.sh <study_dir> <sub>"
    echo "       Where <study_dir> is the parent directory to"
    echo "       the SUB_DATA directory and expects to find"
    echo "       SUB_DATA inside it and then the standard"
    echo "       NSPN directory structure and <sub> is an"
    echo "       NSPN subject ID that corresponds to a folder in"
    echo "       the SUB_DATA directory"
    echo ""
    echo "DESCRIPTION: This code extracts CT and MT values for a bunch of"
    echo "             different depths using the surface based parcellations"
    echo "             aparc, 500.aparc, and lobesStrict"
    exit
}


#================================================================
# Get the color tables
#================================================================
# These files needs to be in the same directory as this script
# Fine if you download the git repository but not fine 
# if you've only take the script itself!
#================================================================
lobes_ctab=`dirname ${0}`/LobesStrictLUT.txt
parc500_ctab=`dirname ${0}`/parc500LUT.txt

if [[ ! -f ${lobes_ctab} ]]; then
    echo "Can't find 500 parcellation color look up table file"
    echo "Check that parc500LUT.txt is in the same directory"
    echo "as this script"
    usage
fi

if [[ ! -f ${lobes_ctab} ]]; then
    echo "Can't find lobes color look up table file"
    echo "Check that LobesStrictLUT.txt is in the same directory"
    echo "as this script"
    usage
fi
    
#================================================================
# READ IN COMMAND LINE ARGUMENTS
#================================================================
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

surfer_dir=${sub_data_dir}/${sub}/SURFER/MRI0/
mpm_dir=${sub_data_dir}/${sub}/MPM/MRI0/
dti_dir=${sub_data_dir}/${sub}/DTI/MRI0/
reg_dir=${sub_data_dir}/${sub}/REG/MRI0/

SUBJECTS_DIR=${surfer_dir}/../
surf_sub=`basename ${surfer_dir}`


#================================================================
# TRANSFORM MPM MEASURES FILES TO FREESURFER SPACE
#================================================================
# If the mpm measure file doesn't exist yet in the 
# <surfer_dir>/mri folder then you have to make it

# Loop through the mpm outputs that you're interested in
for mpm in MT R1 R2s R1R2s A; do
    mpm_file=${mpm_dir}/${mpm}_head.nii.gz

    if [[ -f ${mpm_file} ]]; then
        # If the measure file has particularly small values
        # then multiply this file by 1000 first
        if [[ ${mpm} == "R2s" || ${mpm} == "MT" ]]; then
            if [[ ! -f ${mpm_file/.nii/_mul1000.nii} ]]; then
                fslmaths ${mpm_file} -mul 1000 ${mpm_file/.nii/_mul1000.nii}
            fi
            mpm_file=${mpm_file/.nii/_mul1000.nii}
        fi
        
    elif [[ ${mpm} == R1R2s ]]; then
        # If the mpm label is R1R2s then you need to create
        # the file
        if [[ ! -f ${mpm_file} ]]; then
            fslmaths ${mpm_dir}/R1_head.nii.gz \
                     -div ${mpm_dir}/R2s_head_mul1000.nii.gz \
                     ${mpm_file}
        fi

    else
        echo "${mpm} file doesn't exist in mpm_dir, skipping"
        continue
    fi
    
    # Align the mgz file to "freesurfer" anatomical space
    if [[ ! -f ${surfer_dir}/mri/${mpm}.mgz ]]; then
        echo "Registering ${mpm} file to freesurfer space"
        mri_vol2vol --mov ${mpm_file} \
                    --targ ${surfer_dir}/mri/T1.mgz \
                    --regheader \
                    --o ${surfer_dir}/mri/${mpm}.mgz \
                    --no-save-reg
    else
        echo  "${mpm} file already in freesurfer space"
    fi
done


#=============================================================================
# PROCESS THE DTI DATA IF IT EXISTS
#=============================================================================
# REGISTER B0 TO FREESURFER SPACE
#=============================================================================
# The first step is ensuring that the dti_ec (B0) file
# has been registered to freesurfer space
if [[ ! -f ${reg_dir}/diffB0_TO_surf.dat && -f ${dti_dir}/dti_ec.nii.gz ]]; then
    bbregister --s ${surf_sub} \
               --mov ${dti_dir}/dti_ec.nii.gz \
               --init-fsl \
               --reg ${reg_dir}/diffB0_TO_surf.dat \
               --t2
fi


#=============================================================================
# TRANSFORM DTI MEASURES FILES TO FREESURFER SPACE
#=============================================================================
# If the dti measure file doesn't exist yet in the <surfer_dir>/mri folder
# then you have to make it
for measure in FA MD MO L1 L23 sse; do

    measure_file_dti=`ls -d ${dti_dir}/FDT/*_${measure}.nii.gz 2> /dev/null`
    if [[ ! -f ${measure_file_dti} ]]; then 
        echo "${measure} file doesn't exist in dti_dir, skipping"
        continue
    fi
    
    # If the measure file has particularly small values
    # then multiply this file by 1000 first
    if [[ "MD L1 L23" =~ ${measure} ]]; then
        if [[ ! -f ${measure_file_dti/.nii/_mul1000.nii} ]]; then
            fslmaths ${measure_file_dti} -mul 1000 ${measure_file_dti/.nii/_mul1000.nii}
        fi
        measure_file_dti=${measure_file_dti/.nii/_mul1000.nii}
    fi
    
    # Now transform this file to freesurfer space
    if [[ ! -f ${surfer_dir}/mri/${measure}.mgz ]]; then
        
        echo "Registering ${measure} file to freesurfer space"
        mri_vol2vol --mov ${measure_file_dti} \
                    --targ ${surfer_dir}/mri/T1.mgz \
                    --o ${surfer_dir}/mri/${measure}.mgz \
                    --reg ${reg_dir}/diffB0_TO_surf.dat \
                    --no-save-reg

    else
        echo "${measure} file already in freesurfer space"
       
    fi
done

#================================================================
# Extract the stats from the surface parcellations
#================================================================
# Specifically this will loop through the following 
# parcellations:
#     aparc
#     500.aparc
#     lobesStrict
#=================================================================
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
        
        # Next loop through all the different MPM and DTI files
        for measure in MT R1 R2s R1R2s A FA MD MO L1 L23 sse; do

            if [[ ! -f ${surfer_dir}/mri/${measure}.mgz ]]; then
                echo "Not extracting stats from ${measure} file"
                continue
            fi
            # First take the average across all of cortex
            
            # Project the values to the surface
            if [[ ! -f ${surfer_dir}/surf/${hemi}.${measure}_cortexAv.mgh ]]; then
            
                mri_vol2surf --mov ${surfer_dir}/mri/${measure}.mgz \
                                --o ${surfer_dir}/surf/${hemi}.${measure}_cortexAv.mgh \
                                --regheader ${surf_sub} \
                                --projfrac-avg 0 1 0.1 \
                                --interp nearest \
                                --surf white \
                                --hemi ${hemi} 
            fi

            # Calculate the stats
            if [[ ! -f ${surfer_dir}/stats/${hemi}.${parc}.${measure}_cortexAv.stats \
                        && -f ${surfer_dir}/label/${hemi}.${parc}.annot ]]; then
                        
                mris_anatomical_stats -a ${surfer_dir}/label/${hemi}.${parc}.annot \
                                        -t ${surfer_dir}/surf/${hemi}.${measure}_cortexAv.mgh \
                                        -f ${surfer_dir}/stats/${hemi}.${parc}.${measure}_cortexAv.stats \
                                        ${surf_sub} \
                                        ${hemi}
            fi
            
            # Then average across white matter
            # Project the values to the surface
            if [[ ! -f ${surfer_dir}/surf/${hemi}.${measure}_wmAv.mgh ]]; then
            
                mri_vol2surf --mov ${surfer_dir}/mri/${measure}.mgz \
                                --o ${surfer_dir}/surf/${hemi}.${measure}_wmAv.mgh \
                                --regheader ${surf_sub} \
                                --projdist-avg -2 0 0.1 \
                                --interp nearest \
                                --surf white \
                                --hemi ${hemi} 
            fi

            # Calculate the stats
            if [[ ! -f ${surfer_dir}/stats/${hemi}.${parc}.${measure}_wmAv.stats \
                        && -f ${surfer_dir}/label/${hemi}.${parc}.annot ]]; then
                        
                mris_anatomical_stats -a ${surfer_dir}/label/${hemi}.${parc}.annot \
                                        -t ${surfer_dir}/surf/${hemi}.${measure}_wmAv.mgh \
                                        -f ${surfer_dir}/stats/${hemi}.${parc}.${measure}_wmAv.stats \
                                        ${surf_sub} \
                                        ${hemi}
            fi
            
            # Loop through a bunch of different fractional depths 
            # from the white matter surface
            for frac in `seq -f %+02.2f 0 0.1 1`; do

                # Project the values to the surface
                if [[ ! -f ${surfer_dir}/surf/${hemi}.${measure}_projfrac${frac}.mgh ]]; then
                
                    mri_vol2surf --mov ${surfer_dir}/mri/${measure}.mgz \
                                    --o ${surfer_dir}/surf/${hemi}.${measure}_projfrac${frac}.mgh \
                                    --regheader ${surf_sub} \
                                    --projfrac ${frac} \
                                    --interp nearest \
                                    --surf white \
                                    --hemi ${hemi} 
                fi

                # Calculate the stats
                if [[ ! -f ${surfer_dir}/stats/${hemi}.${parc}.${measure}_projfrac${frac}.stats \
                            && -f ${surfer_dir}/label/${hemi}.${parc}.annot ]]; then
                            
                    mris_anatomical_stats -a ${surfer_dir}/label/${hemi}.${parc}.annot \
                                            -t ${surfer_dir}/surf/${hemi}.${measure}_projfrac${frac}.mgh \
                                            -f ${surfer_dir}/stats/${hemi}.${parc}.${measure}_projfrac${frac}.stats \
                                            ${surf_sub} \
                                            ${hemi}
                fi
        
            done # Close the fraction of cortical thickness loop
            
            # Now loop through the different absolute depths
            # **from the grey/white matter boundary**
            for dist in `seq -f %+02.2f -2 0.1 0`; do

                if [[ ! -f ${surfer_dir}/surf/${hemi}.${measure}_projdist${dist}_fromBoundary.mgh ]]; then
                
                    mri_vol2surf --mov ${surfer_dir}/mri/${measure}.mgz \
                                    --o ${surfer_dir}/surf/${hemi}.${measure}_projdist${dist}_fromBoundary.mgh \
                                    --regheader ${surf_sub} \
                                    --projdist ${dist} \
                                    --interp nearest \
                                    --surf white \
                                    --hemi ${hemi} 
                
                fi

                # Calculate the stats
                if [[ ! -f ${surfer_dir}/stats/${hemi}.${parc}.${measure}_projdist${dist}_fromBoundary.stats \
                            && -f ${surfer_dir}/label/${hemi}.${parc}.annot ]]; then
                            
                    mris_anatomical_stats -a ${surfer_dir}/label/${hemi}.${parc}.annot \
                                            -t ${surfer_dir}/surf/${hemi}.${measure}_projdist${dist}_fromBoundary.mgh \
                                            -f ${surfer_dir}/stats/${hemi}.${parc}.${measure}_projdist${dist}_fromBoundary.stats \
                                            ${surf_sub} \
                                            ${hemi}
                fi
            done # Close the absolute distance **from boundary** loop
        done # Close the measure loop
    done # Close parcellation loop
done # Close hemi loop

