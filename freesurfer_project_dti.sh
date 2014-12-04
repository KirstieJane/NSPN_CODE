#!/bin/bash

# Resample DTI measures to surface

data_dir=/home/kw02/BrainWorks/
sublist=${data_dir}/sublist

for sub in `cat ${sublist}`; do

    export SUBJECTS_DIR=${data_dir}/SUB_DATA/${sub}/MPRAGE/

    if [[ -f ${SUBJECTS_DIR}/SURF/mri/orig.mgz ]]; then
    
        for dti_scan in DTI_64D_1A ; do 
        
            for dti_number in DTI_1 DTI_2; do
            
                if [[ -f ${data_dir}/SUB_DATA/${sub}/${dti_scan}/${dti_number}/FDT/${sub}_FA.nii.gz ]]; then
            
                    echo ${sub}
                    
                    for hemi in lh rh; do
                    
                        for frac in 0.5 0 -0.5 -1; do 
                        
                            if [[ ! -f ${data_dir}/SUB_DATA/${sub}/MPRAGE/SURF/surf/${hemi}.${dti_scan}_${dti_number}_FA_${frac/.}.mgz ]]; then
                            
                                mri_vol2surf --mov ${data_dir}/SUB_DATA/${sub}/${dti_scan}/${dti_number}/FDT/${sub}_FA.nii.gz \
                                                --reg ${data_dir}/SUB_DATA/${sub}/REG/${dti_scan}/${dti_number}/diffB0_TO_surf.dat \
                                                --o ${data_dir}/SUB_DATA/${sub}/MPRAGE/SURF/surf/${hemi}.${dti_scan}_${dti_number}_FA_${frac/.}.mgz \
                                                --ref ${data_dir}/SUB_DATA/${sub}/MPRAGE/SURF/mri/orig.mgz \
                                                --projfrac ${frac} \
                                                --interp nearest \
                                                --surf white \
                                                --hemi ${hemi} 
                                                
                                #mv ${data_dir}/SUB_DATA/${sub}/MPRAGE/SURF/surf/${hemi}.${dti_scan}_${dti_number}_FA_${frac/.}.mgz \
                                #${data_dir}/SUB_DATA/${sub}/MPRAGE/SURF/surf/${hemi}.${dti_scan}_${dti_number}_FA_${frac/.}
                            fi
                        done
                    done
                fi
            done
        done
    fi
done
