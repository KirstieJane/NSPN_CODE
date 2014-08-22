#!/bin/bash

data_dir=/home/kw401/scratch/UCHANGE_INTERIM/

for sub in `cat ${data_dir}/sublist`; do
    surfer_dir=${data_dir}/SUB_DATA/${sub}/SURFER/MRI0/

    export SUBJECTS_DIR=`dirname ${surfer_dir}`

    mkdir -p ${surfer_dir}/probtrackx/seeds_targets/
    rm -rf ${surfer_dir}/probtrackx/targets.txt
    
    label_list=(`dirname ${0}`/destrieux_labels_order.txt)
    labels=(`cat ${label_list}`)
    
    # CREATE INDIVIDUAL LABELS FOR THE ANNOTATION FILE
    # In this example we're looking at the aparc.a2009s scheme
    for hemi in lh rh; do
        for label in ${labels[@]}; do
            if [[ ! -f ${surfer_dir}/probtrackx/seeds_targets/${hemi}.${label}.label ]]; then
                mri_annotation2label --subject MRI0 \
                                        --hemi ${hemi} \
                                        --annotation ${surfer_dir}/label/${hemi}.aparc.a2009s.annot \
                                        --outdir ${surfer_dir}/probtrackx/seeds_targets/ \
                                        --surface white
            fi
            
            if [[ ! -f ${surfer_dir}/probtrackx/seeds_targets/${hemi}.${label}.gii ]]; then
                echo "converting $label to $vol"
                mri_label2vol --label ${surfer_dir}/probtrackx/seeds_targets/${hemi}.${label}.label \
                                --temp ${surfer_dir}/dlabel/anat/anat_brain_mask.nii.gz \
                                --o ${surfer_dir}/probtrackx/seeds_targets/${hemi}.${label}.gii \
                                --identity \
                                --fillthresh 0.5
                                
                mri_label2vol --label ${surfer_dir}/probtrackx/seeds_targets/${hemi}.${label}.label \
                                --temp ${surfer_dir}/dlabel/anat/anat_brain_mask.nii.gz \
                                --o ${surfer_dir}/probtrackx/seeds_targets/${hemi}.${label}.nii.gz \
                                --identity \
                                --fillthresh 0.5
            fi
        done # Close label list
    done # Close hemi loop
    
done # Close sub loop


