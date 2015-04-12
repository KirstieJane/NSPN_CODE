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

if [[ ! -d ${data_dir} ]]; then
    usage
fi
    
#=============================================================================
# GET STARTED
#=============================================================================

mkdir -p ${data_dir}/FS_ROIS/

#=============================================================================
# SEGMENTATIONS
#=============================================================================
# Loop through the various segmentations
#for seg in aseg wmparc lobesStrict 500cortExpConsecWMoverlap 500cortConsec; do
for seg in aseg
    
    for measure in R1 MT R2s A FA MD MO L1 L23 sse; do
    
        # Find all the individual stats files for that segmentation
        inputs=(`ls -d ${data_dir}/SUB_DATA/*/SURFER/MRI0/stats/${measure}_${seg}.stats 2> /dev/null `)

        if [[ ${#inputs[@]} -gt 0 ]]; then
            if [[ ${measure} == R1 ]]; then
                # Write out the volume for each segment
                # Note that we aren't going to do this again - once is enough
                # it doesn't change!
                asegstats2table --inputs ${inputs[@]} \
                                -t ${data_dir}/FS_ROIS/SEG_${seg}_volume_temp.csv \
                                -d comma \
                                --common-segs \
                                --meas volume
            fi
        
            # Now write out the mean value for the measure
            asegstats2table --inputs ${inputs[@]} \
                            -t ${data_dir}/FS_ROIS/SEG_${measure}_${seg}_mean_temp.csv \
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
                        ${data_dir}/FS_ROIS/SEG_${measure}_${seg}_mean_temp.csv \
                            > ${data_dir}/FS_ROIS/SEG_${measure}_${seg}_mean.csv

            # And replace all '-' with '_' because statsmodels in python
            # likes that more :P
            sed -i "s/-/_/g" ${data_dir}/FS_ROIS/SEG_${measure}_${seg}_mean.csv
            sed -i "s/_0/-0/g" ${data_dir}/FS_ROIS/SEG_${measure}_${seg}_mean.csv
            sed -i "s/://g" ${data_dir}/FS_ROIS/SEG_${measure}_${seg}_mean.csv
                                    
            # Don't forget to paste the nspn_ids in for the volume file
            if [[ ${measure} == R1 ]]; then
                paste -d , ${data_dir}/FS_ROIS/nspn_id_col \
                            ${data_dir}/FS_ROIS/SEG_${seg}_volume_temp.csv \
                                > ${data_dir}/FS_ROIS/SEG_${seg}_volume.csv
                # And replace '-' with '_'
                sed -i "s/-/_/g" ${data_dir}/FS_ROIS/SEG_${seg}_volume.csv
                sed -i "s/://g" ${data_dir}/FS_ROIS/SEG_${seg}_volume.csv
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

subjects=(`ls -d ${data_dir}/SUB_DATA/*/SURFER/MRI?/ 2> /dev/null`)

for parc in aparc 500.aparc lobesStrict; do

    # Start by pulling out the standard measures 
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
                                --skip \
                                -t ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_${hemi}_temptemp.csv 
                                
            # Drop the first column because it isn't necessary
            cut -d, -f2- ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_${hemi}_temptemp.csv \
                    > ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_${hemi}_temp.csv 
            
            # But save it for later!
            cut -d, -f1 ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_${hemi}_temptemp.csv \
                    > ${data_dir}/FS_ROIS/nspn_id_col
        done
        
        sed -i "s|${data_dir}/SUB_DATA/||g" ${data_dir}/FS_ROIS/nspn_id_col
        sed -i "s|/SURFER/MRI|,|g" ${data_dir}/FS_ROIS/nspn_id_col
        sed -i "s|/||g" ${data_dir}/FS_ROIS/nspn_id_col
        sed -i "s|${hemi}.${parc}.${measure}|nspn_id,occ|g" ${data_dir}/FS_ROIS/nspn_id_col

        # Now paste the data together
        paste -d , ${data_dir}/FS_ROIS/nspn_id_col \
                ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_lh_temp.csv \
                ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_rh_temp.csv \
                    > ${data_dir}/FS_ROIS/PARC_${parc}_${measure}.csv
        
        # And replace all '-' with '_' because statsmodels in python
        # likes that more :P
        sed -i "s/-/_/g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}.csv
        sed -i "s/_0/-0/g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}.csv
        sed -i "s/://g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}.csv
        
        # Remove the temporary files
        rm ${data_dir}/FS_ROIS/*temp.csv
        rm ${data_dir}/FS_ROIS/nspn_id_col

    done # Close the measure loop
    
    # Next extract "thickness" values from the projected maps
    #for measure in R1 MT R2s A FA MD MO L1 L23 sse synthetic; do
    for measure in synthetic; do
            
        for frac in `seq -f %+02.2f -1 0.05 1`; do
        
            for hemi in lh rh; do

                # Combine stats for all subjects for each measure and for each 
                # hemisphere separately
                aparcstats2table --hemi ${hemi} \
                                    --subjects ${subjects[@]} \
                                    --parc ${parc}.${measure}_projfrac${frac} \
                                    --meas thickness \
                                    -d comma \
                                    --common-parcs \
                                    --skip \
                                    -t ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_${hemi}_temptemp.csv 
                                    
                # Drop the first column because it isn't necessary
                cut -d, -f2- ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_${hemi}_temptemp.csv \
                        > ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_${hemi}_temp.csv 
                
                # But save it for later!
                cut -d, -f1 ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_${hemi}_temptemp.csv \
                        > ${data_dir}/FS_ROIS/nspn_id_col
            done
            
            sed -i "s|${data_dir}/SUB_DATA/||g" ${data_dir}/FS_ROIS/nspn_id_col
            sed -i "s|/SURFER/MRI|,|g" ${data_dir}/FS_ROIS/nspn_id_col
            sed -i "s|/||g" ${data_dir}/FS_ROIS/nspn_id_col
            sed -i "s|${hemi}.${parc}.${measure}_projfrac${frac}.thickness|nspn_id,occ|g" ${data_dir}/FS_ROIS/nspn_id_col
            
            # Now paste the data together
            paste -d , ${data_dir}/FS_ROIS/nspn_id_col \
                    ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_lh_temp.csv \
                    ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_rh_temp.csv \
                        > ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}.csv
            
            # And replace all '-' with '_' because statsmodels in python
            # likes that more :P
            sed -i "s/-/_/g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}.csv
            sed -i "s/_0/-0/g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}.csv
            sed -i "s/://g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}.csv
            
            # Remove the temporary files
            rm ${data_dir}/FS_ROIS/*temp.csv
            rm ${data_dir}/FS_ROIS/nspn_id_col
        
        done # Close frac loop
        
        for dist in `seq -f %+02.2f 0 -0.2 -2`; do
        
            for hemi in lh rh; do

                # Combine stats for all subjects for each measure and for each 
                # hemisphere separately
                aparcstats2table --hemi ${hemi} \
                                    --subjects ${subjects[@]} \
                                    --parc ${parc}.${measure}_projdist${dist}_fromBoundary \
                                    --meas thickness \
                                    -d comma \
                                    --common-parcs \
                                    --skip \
                                    -t ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_${hemi}_temptemp.csv 
                                    
                # Drop the first column because it isn't necessary
                cut -d, -f2- ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_${hemi}_temptemp.csv \
                        > ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_${hemi}_temp.csv 
                
                # But save it for later!
                cut -d, -f1 ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_${hemi}_temptemp.csv \
                        > ${data_dir}/FS_ROIS/nspn_id_col
            done
            
            sed -i "s|${data_dir}/SUB_DATA/||g" ${data_dir}/FS_ROIS/nspn_id_col
            sed -i "s|/SURFER/MRI|,|g" ${data_dir}/FS_ROIS/nspn_id_col
            sed -i "s|/||g" ${data_dir}/FS_ROIS/nspn_id_col
            sed -i "s|${hemi}.${parc}.${measure}_projdist${dist}_fromBoundary.thickness|nspn_id,occ|g" ${data_dir}/FS_ROIS/nspn_id_col
            
            # Now paste the data together
            paste -d , ${data_dir}/FS_ROIS/nspn_id_col \
                    ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_lh_temp.csv \
                    ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_rh_temp.csv \
                        > ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary.csv
            
            # And replace all '-' with '_' because statsmodels in python
            # likes that more :P
            sed -i "s/-/_/g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary.csv
            sed -i "s/_0/-0/g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary.csv
            sed -i "s/://g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary.csv
            
            # Remove the temporary files
            rm ${data_dir}/FS_ROIS/*temp.csv
            rm ${data_dir}/FS_ROIS/nspn_id_col
        
        done # Close dist loop
    done # Close measure loop
done # Close parc loop


