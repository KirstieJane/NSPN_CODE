#!/bin/bash

#==============================================================================
# Combine stats measures of surface parcellations and segmentations for
# NSPN MPM and DTI data for all subjects
# Created by Kirstie Whitaker
# Contact kw401@cam.ac.uk
#==============================================================================

#==============================================================================
# USAGE: freesurfer_combine_rois_stats.sh <data_dir>
#==============================================================================
function usage {

    echo "USAGE: freesurfer_combine_rois_stats.sh <data_dir>"
    echo "Note that data dir expects to find SUB_DATA within it"
    echo "and then the standard NSPN directory structure"
    echo ""
    echo "DESCRIPTION: This code looks for the output of freesurfer_extract_rois.sh"
    echo "in each subject's directory and then combines that information together"
    echo "in the FS_ROIS folder within DATA_DIR"
    exit
}

#=============================================================================
# READ IN COMMAND LINE ARGUMENTS
#=============================================================================

data_dir=$1

#=============================================================================
# GET STARTED
#=============================================================================

mkdir -p ${data_dir}/FS_ROIS/

#=============================================================================
# SEGMENTATIONS
#=============================================================================
# Loop through the various segmentations
for seg in aseg wmparc lobesStrict 500cortExpConsecWMoverlap 500cortConsec; do

    for measure in R1 MT R2s A FA MD MO L1 L23 sse; do
    
        # Find all the individual stats files for that segmentation
        inputs=(`ls -d ${data_dir}/SUB_DATA/*/SURFER/MRI0/stats/${measure}_${seg}.stats 2> /dev/null `)

        if [[ ${#inputs[@]} -gt 0 ]]; then
            if [[ ${measure} == R1 ]]; then
                # Write out the volume for each segment
                # Note that we aren't going to do this again - once is enough
                # it doesn't change!
                asegstats2table --inputs ${inputs[@]} \
                                -t ${data_dir}/FS_ROIS/${seg}_volume_temp.csv \
                                -d comma \
                                --common-segs \
                                --meas volume
            fi
        
            # Now write out the mean value for the measure
            asegstats2table --inputs ${inputs[@]} \
                            -t ${data_dir}/FS_ROIS/${measure}_${seg}_mean_temp.csv \
                            -d comma \
                            --common-segs \
                            --meas mean
                        
            # Create the first two columns:
            # nspn_id and occ (which is always 0 at the moment)
            echo "nspn_id,occ" > ${data_dir}/FS_ROIS/nspn_id_col
            for sub in ${inputs[@]}; do
                sub=${sub/${data_dir}/}
                echo ${sub:10:5},0 >> ${data_dir}/FS_ROIS/nspn_id_col
            done
        
            # Now paste the data together
            paste -d , ${data_dir}/FS_ROIS/nspn_id_col \
                        ${data_dir}/FS_ROIS/${measure}_${seg}_mean_temp.csv \
                            > ${data_dir}/FS_ROIS/${measure}_${seg}_mean.csv

            # And replace all '-' with '_' because statsmodels in python
            # likes that more :P
            sed -i "s/-/_/g" ${data_dir}/FS_ROIS/${measure}_${seg}_mean.csv
            sed -i "s/_0/-0/g" ${data_dir}/FS_ROIS/${measure}_${seg}_mean.csv
            sed -i "s/://g" ${data_dir}/FS_ROIS/${measure}_${seg}_mean.csv
                                    
            # Don't forget to paste the nspn_ids in for the volume file
            if [[ ${measure} == R1 ]]; then
                paste -d , ${data_dir}/FS_ROIS/nspn_id_col \
                            ${data_dir}/FS_ROIS/${seg}_volume_temp.csv \
                                > ${data_dir}/FS_ROIS/${seg}_volume.csv
                # And replace '-' with '_'
                sed -i "s/-/_/g" ${data_dir}/FS_ROIS/${seg}_volume.csv
                sed -i "s/://g" ${data_dir}/FS_ROIS/${seg}_volume.csv
            fi
            
            # Remove the temporary files
            rm ${data_dir}/FS_ROIS/*temp.csv
            rm ${data_dir}/FS_ROIS/nspn_id_col
        
        else
            echo "    No input files for ${measure}_${seg}!"
        fi
    done
done

#=============================================================================
# PARCELLATIONS
#=============================================================================
# Loop through the various parcellations

subjects=(`ls -d ${data_dir}/SUB_DATA/*/SURFER/MRI0/ 2> /dev/null`)

for parc in aparc 500.aparc lobesStrict; do

    for measure in area volume thickness meancurv gauscurv foldind curvind; do
    
        for hemi in lh rh; do
        
            # Combine stats for all subjects for each measure and for each 
            # hemisphere separately
            aparcstats2table --hemi ${hemi} \
                                --subjects ${subjects[@]} \
                                --parc ${parc} \
                                --meas ${measure} \
                                -d comma \
                                --common-parcs \
                                -t ${data_dir}/FS_ROIS/${parc}_${measure}_${hemi}_temp.csv 
        done
        
        # Create the first two columns:
        # nspn_id and occ (which is always 0 at the moment)
        echo "nspn_id,occ" > ${data_dir}/FS_ROIS/nspn_id_col
        for sub in ${subjects[@]}; do
            sub=${sub/${data_dir}/}
            echo ${sub:10:5},0 >> ${data_dir}/FS_ROIS/nspn_id_col
        done
        
        # Now paste the data together
        paste -d , ${data_dir}/FS_ROIS/nspn_id_col \
                ${data_dir}/FS_ROIS/${parc}_${measure}_lh_temp.csv \
                ${data_dir}/FS_ROIS/${parc}_${measure}_rh_temp.csv \
                    > ${data_dir}/FS_ROIS/${parc}_${measure}.csv
        
        # And replace all '-' with '_' because statsmodels in python
        # likes that more :P
        sed -i "s/-/_/g" ${data_dir}/FS_ROIS/${parc}_${measure}.csv
        sed -i "s/_0/-0/g" ${data_dir}/FS_ROIS/${parc}_${measure}.csv
        sed -i "s/://g" ${data_dir}/FS_ROIS/${parc}_${measure}.csv
                                
        # Remove the temporary files
        rm ${data_dir}/FS_ROIS/*temp.csv
        rm ${data_dir}/FS_ROIS/nspn_id_col
        
        # NOTE - need to drop the columns that are called ${hemi}.${parc}.${measure}
        # left in for now because it'll be easier to drop them at the combined stage
        
    done
done


