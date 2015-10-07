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
for seg in aseg wmparc lobesStrict 500cortExpConsecWMoverlap 500cortConsec; do
    
    #for measure in R1 MT R2s A FA MD MO L1 L23 sse synthetic; do
    for measure in MT R1 R2s R1R2s; do
    
        # Find all the individual stats files for that segmentation
        inputs=(`ls -d ${data_dir}/SUB_DATA/*/SURFER/MRI?/stats/${measure}_${seg}.stats 2> /dev/null `)

        if [[ ${#inputs[@]} -gt 0 ]]; then
        
            #===== NSPN_ID AND OCC VALUES ====================================
            # We need to edit the first two columns so they're nice and easily
            # readable with the nspn_ids etc
            echo "nspn_id,occ" > ${data_dir}/FS_ROIS/nspn_id_col
            for sub in ${inputs[@]}; do
                sub=${sub/${data_dir}/}
                
                # nspn_id is the 5 characters starting at the 10th once you've
                # taken away the data directory, and occ is the 26th character.
                echo ${sub:10:5},${sub:26:1} >> ${data_dir}/FS_ROIS/nspn_id_col
            done
        
            # Write out each statistic
            # This is silly because it loops over volume many times
            # but to be honest, I think the code was looking super messy when
            # I had it being faster. So just be patient and don't worry about
            # speed ;)
            
            for stat in mean std volume; do
                # Now write out the mean values for the measure
                asegstats2table --inputs ${inputs[@]} \
                                -t ${data_dir}/FS_ROIS/SEG_${measure}_${seg}_${stat}_temp.csv \
                                -d comma \
                                --all-segs \
                                --meas ${stat}
                                
                # Now paste the data together
                paste -d , ${data_dir}/FS_ROIS/nspn_id_col \
                            ${data_dir}/FS_ROIS/SEG_${measure}_${seg}_${stat}_temp.csv \
                                > ${data_dir}/FS_ROIS/SEG_${measure}_${seg}_${stat}.csv
                                
                # And replace all '-' with '_' because statsmodels in python
                # likes that more :P but only for the first line
                sed -i "1 s/-/_/g" ${data_dir}/FS_ROIS/SEG_${measure}_${seg}_${stat}.csv
                # And replace the : marker
                sed -i "s/://g" ${data_dir}/FS_ROIS/SEG_${measure}_${seg}_${stat}.csv

                # Remove the temporary files
                rm ${data_dir}/FS_ROIS/*temp.csv
            done
        
        # Get rid of the nspn_id_col file ready for the next loop
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
    for measure in area volume thickness thicknessstd meancurv gauscurv foldind curvind; do
    
        for hemi in lh rh; do
        
            # Combine stats for all subjects for each measure and for each 
            # hemisphere separately
            aparcstats2table --hemi ${hemi} \
                                --subjects ${subjects[@]} \
                                --parc ${parc} \
                                --meas ${measure} \
                                -d comma \
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
        # likes that more :P but only for the first line
        sed -i "1 s/-/_/g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}.csv
        # And replace the : marker
        sed -i "s/://g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}.csv
        
        # Remove the temporary files
        rm ${data_dir}/FS_ROIS/*temp.csv
        rm ${data_dir}/FS_ROIS/nspn_id_col

    done # Close the measure loop
    
    # Next extract "thickness" and "thicknessstd" 
    # values from the projected maps
    for stat in thickness thicknessstd; do
    
        # Come up with some readable names for the files
        if [[ ${stat} == thickness ]]; then
            stat_name=mean
        else
            stat_name=std
        fi
        
        # Now loop through all the measures
        #for measure in R1 MT R2s A FA MD MO L1 L23 sse synthetic; do
        for measure in MT R1 R2s R1R2s; do
                
            #for frac in `seq -f %+02.2f -1 0.05 1`; do
            for frac in `seq -f %+02.2f 0 0.1 1`; do
            
                for hemi in lh rh; do

                    # Combine stats for all subjects for each measure and for each 
                    # hemisphere separately
                    aparcstats2table --hemi ${hemi} \
                                        --subjects ${subjects[@]} \
                                        --parc ${parc}.${measure}_projfrac${frac} \
                                        --meas ${stat} \
                                        -d comma \
                                        --skip \
                                        -t ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_${stat_name}_${hemi}_temptemp.csv 
                                        
                    # Drop the first column because it isn't necessary
                    cut -d, -f2- ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_${stat_name}_${hemi}_temptemp.csv \
                            > ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_${stat_name}_${hemi}_temp.csv 
                    
                    # But save it for later!
                    cut -d, -f1 ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_${stat_name}_${hemi}_temptemp.csv \
                            > ${data_dir}/FS_ROIS/nspn_id_col
                done
                
                sed -i "s|${data_dir}/SUB_DATA/||g" ${data_dir}/FS_ROIS/nspn_id_col
                sed -i "s|/SURFER/MRI|,|g" ${data_dir}/FS_ROIS/nspn_id_col
                sed -i "s|/||g" ${data_dir}/FS_ROIS/nspn_id_col
                sed -i "s|${hemi}.${parc}.${measure}_projfrac${frac}.${stat}|nspn_id,occ|g" ${data_dir}/FS_ROIS/nspn_id_col
                
                # Now paste the data together
                paste -d , ${data_dir}/FS_ROIS/nspn_id_col \
                        ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_${stat_name}_lh_temp.csv \
                        ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_${stat_name}_rh_temp.csv \
                            > ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_${stat_name}.csv
                
                # And replace all '-' with '_' because statsmodels in python
                # likes that more :P - but only on the first line!
                sed -i "1 s/-/_/g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_${stat_name}.csv
                # and get rid of the : marker
                sed -i "s/://g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projfrac${frac}_${stat_name}.csv
                
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
                                        --meas ${stat} \
                                        -d comma \
                                        --skip \
                                        -t ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_${stat_name}_${hemi}_temptemp.csv 
                                        
                    # Drop the first column because it isn't necessary
                    cut -d, -f2- ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_${stat_name}_${hemi}_temptemp.csv \
                            > ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_${stat_name}_${hemi}_temp.csv 
                    
                    # But save it for later!
                    cut -d, -f1 ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_${stat_name}_${hemi}_temptemp.csv \
                            > ${data_dir}/FS_ROIS/nspn_id_col
                done
                
                sed -i "s|${data_dir}/SUB_DATA/||g" ${data_dir}/FS_ROIS/nspn_id_col
                sed -i "s|/SURFER/MRI|,|g" ${data_dir}/FS_ROIS/nspn_id_col
                sed -i "s|/||g" ${data_dir}/FS_ROIS/nspn_id_col
                sed -i "s|${hemi}.${parc}.${measure}_projdist${dist}_fromBoundary.${stat}|nspn_id,occ|g" ${data_dir}/FS_ROIS/nspn_id_col
                
                # Now paste the data together
                paste -d , ${data_dir}/FS_ROIS/nspn_id_col \
                        ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_${stat_name}_lh_temp.csv \
                        ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_${stat_name}_rh_temp.csv \
                            > ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_${stat_name}.csv
                
                # And replace all '-' with '_' because statsmodels in python
                # likes that more :P but only on the first row
                sed -i "1 s/-/_/g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_${stat_name}.csv
                # And replace the : marker
                sed -i "s/://g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_projdist${dist}_fromBoundary_${stat_name}.csv
                
                # Remove the temporary files
                rm ${data_dir}/FS_ROIS/*temp.csv
                rm ${data_dir}/FS_ROIS/nspn_id_col
            
            done # Close dist loop
            
            # Loop over the two different averages
            for av in cortexAv wmAv; do
            
                for hemi in lh rh; do

                    # Combine stats for all subjects for each measure and for each 
                    # hemisphere separately
                    aparcstats2table --hemi ${hemi} \
                                        --subjects ${subjects[@]} \
                                        --parc ${parc}.${measure}_${av} \
                                        --meas ${stat} \
                                        -d comma \
                                        --skip \
                                        -t ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_${av}_${stat_name}_${hemi}_temptemp.csv 
                                        
                    # Drop the first column because it isn't necessary
                    cut -d, -f2- ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_${av}_${stat_name}_${hemi}_temptemp.csv \
                            > ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_${av}_${stat_name}_${hemi}_temp.csv 
                    
                    # But save it for later!
                    cut -d, -f1 ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_${av}_${stat_name}_${hemi}_temptemp.csv \
                            > ${data_dir}/FS_ROIS/nspn_id_col
                done
                
                sed -i "s|${data_dir}/SUB_DATA/||g" ${data_dir}/FS_ROIS/nspn_id_col
                sed -i "s|/SURFER/MRI|,|g" ${data_dir}/FS_ROIS/nspn_id_col
                sed -i "s|/||g" ${data_dir}/FS_ROIS/nspn_id_col
                sed -i "s|${hemi}.${parc}.${measure}_${av}.${stat}|nspn_id,occ|g" ${data_dir}/FS_ROIS/nspn_id_col
                
                # Now paste the data together
                paste -d , ${data_dir}/FS_ROIS/nspn_id_col \
                        ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_${av}_${stat_name}_lh_temp.csv \
                        ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_${av}_${stat_name}_rh_temp.csv \
                            > ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_${av}_${stat_name}.csv
                
                # And replace all '-' with '_' because statsmodels in python
                # likes that more :P but only on the first row
                sed -i "1 s/-/_/g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_${av}_${stat_name}.csv
                # And replace the : marker
                sed -i "s/://g" ${data_dir}/FS_ROIS/PARC_${parc}_${measure}_${av}_${stat_name}.csv
                
                # Remove the temporary files
                rm ${data_dir}/FS_ROIS/*temp.csv
                rm ${data_dir}/FS_ROIS/nspn_id_col
            
            done # Close dist loop

        done # Close measure loop
    done # Close stat loop
done # Close parc loop

# If there are any empty files delete them
# otherwise they'll screw up the behaviour merge code
for file in `ls -d ${data_dir}/FS_ROIS/*`; do

    lines=(`cat ${file} | wc -l`)
    
    if [[ ${lines} == 0 ]]; then
        rm ${file}
    fi
done

