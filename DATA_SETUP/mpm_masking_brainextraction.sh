#!/bin/bash

####
# UPDATE CURRENT MPM_DIR
####
current_mpm_dir=/work/imaging8/rt337/uchange.2014/MPM/

preprocessed_dir=/work/imagingG/NSPN/data_ready/1.0-preprocessed/uchange.remainder/MPM/

orig_filename_list=(PDw T1w MTw)
calc_filename_list=(A MT MTR MTRdiff R1 R2s synt)

for sub in `ls -d ${current_mpm_dir}`; do
#for sub in 10736; do

    for occ in 0 1; do
        echo "Processing SUB: ${sub}, OCC: ${occ}"
    
        current_dir=${current_mpm_dir}/${sub}/MRI${occ}/
        target_dir=${preprocessed_dir}/${sub}/MRI${occ}/
        
        bet_dir=${target_dir}/PDw_BrainExtractionOutput/
        
        test=(`ls -d ${current_dir}/*_PDw.nii 2> /dev/null`)
        
        if [[ ${#test[@]} == 1 ]]; then
    
            mkdir -p ${bet_dir}

            for filename in ${orig_filename_list[@]} ${calc_filename_list[@]}; do
            
                mpm_file=(`ls -d ${current_dir}/*${filename}.nii`)
                
                fslreorient2std ${mpm_file} ${target_dir}/${filename}.nii.gz    
                
            done
            
            bet ${target_dir}/PDw.nii.gz ${bet_dir}/PDw_brain.nii.gz –A
            
            for filename in PDw ${calc_filename_list[@]}; do
                fslmaths ${bet_dir}/PDw_brain.nii.gz \
                            -bin \
                            -mul ${target_dir}/${filename}.nii.gz \
                            ${target_dir}/${filename}_brain.nii.gz
                            
                fslmaths ${bet_dir}/PDw_brain_outskin_mask.nii.gz \
                            -bin \
                            -mul ${target_dir}/${filename}.nii.gz \
                            ${target_dir}/${filename}_head.nii.gz
         
            done # Close the mpm calculated file loop
            
            echo "    wooo - all done!"
            
        else
            echo "    no data, moving on"
        
        fi # End the test if statement
            

    done # Close the occ loop
done # Close the sub loop
