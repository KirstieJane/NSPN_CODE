#!/bin/bash

# freesurfer_combine_rois_stats.sh <data_dir>

data_dir=$1

# Write out the various freesurfer information
mkdir -p ${data_dir}/FS_ROIS/

# Loop through the various segmentations
for seg in aseg wmparc lobes+aseg 500cortExpConsecWMoverlap ; do

    for measure in R1 MT R2s FA MD MO L1 L23 sse; do
    
        if [[ ${measure} == R1 ]]; then
            # Find all the stats files for the original segmentation
            # which was created based on the MPM R1 scans
            inputs=(`ls -d ${data_dir}/SUB_DATA/*/SURFER/MRI0/stats/${seg}.stats`)        
        else
            # Find all the individual stats files for that segmentation
            inputs=(`ls -d ${data_dir}/SUB_DATA/*/SURFER/MRI0/stats/${measure}_${seg}.stats`)
        fi
    
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
                    
        # Create the first column of nspn_ids
        echo "nspn_id" > ${data_dir}/FS_ROIS/nspn_id_col
        for sub in ${inputs[@]}; do
            sub=${sub/${data_dir}/}
            echo ${sub:10:5} >> ${data_dir}/FS_ROIS/nspn_id_col
        done
    
        # Now paste the data together
        paste -d , ${data_dir}/FS_ROIS/nspn_id_col \
                    ${data_dir}/FS_ROIS/${measure}_${seg}_mean_temp.csv \
                        > ${data_dir}/FS_ROIS/${measure}_${seg}_mean.csv

        # Don't forget to paste the nspn_ids in for the volume file
        if [[ ${measure} == R1 ]]; then
            paste -d , ${data_dir}/FS_ROIS/nspn_id_col \
                        ${data_dir}/FS_ROIS/${seg}_volume_temp.csv \
                            > ${data_dir}/FS_ROIS/${seg}_volume.csv
        fi
        
        # Remove the temporary files
        rm ${data_dir}/FS_ROIS/*temp.csv
        rm ${data_dir}/FS_ROIS/nspn_id_col

    done
done 