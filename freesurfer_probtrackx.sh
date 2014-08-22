#!/bin/bash

data_dir=/home/kw401/scratch/UCHANGE_INTERIM/

for sub in `cat ${data_dir}/sublist`; do
    surfer_dir=${data_dir}/SUB_DATA/${sub}/SURFER/MRI0/

    export SUBJECTS_DIR=`dirname ${surfer_dir}`

    
    # CREATE INDIVIDUAL LABELS FOR THE ANNOTATION FILE
    # In this example we're looking at the aparc.a2009s scheme
    for hemi in lh rh; do
        if [[ ! -f ${SUBJECTS_DIR}/MRI0/label/${hemi}.Pole_occipital.label ]]; then
            mri_annotation2label --subject MRI0 \
                                    --hemi ${hemi} \
                                    --annotation ${SUBJECTS_DIR}/MRI0/label/${hemi}.aparc.a2009s.annot \
                                    --outdir ${SUBJECTS_DIR}/MRI0/label \
                                    --surface white                             
        fi
        
        if [[ ! -f ${SUBJECTS_DIR}/MRI0/label/${hemi}.Pole_occipital.gii ]]; then
            for label in `ls -d ${SUBJECTS_DIR}/MRI0/label/${hemi}.*label`; do
                vol=${label/%.label/.gii}
                echo "converting $label to $vol"
                mri_label2vol --label ${label} \
                                --temp ${SUBJECTS_DIR}/MRI0/dlabel/anat_brain_mask.nii.gz \
                                --o ${vol} \
                                --identity \
                                --fillthresh 0.5
            done
        fi
    done
    
done


