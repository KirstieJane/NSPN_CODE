data_dir=/work/imagingG/NSPN/workspaces/kw401/UCHANGE/INTERIM_ANALYSIS/
sublist=${data_dir}/sublist
for sub in `cat ${sublist}`; do
    highres_dir=${data_dir}/SUB_DATA/${sub}/HIGHRES/MRI0/
    SUBJECTS_DIR=${highres_dir}
    for hemi in lh rh; do
        mri_annotation2label --subject SURF_R1 \
                             --hemi ${hemi} \
                             --lobesStrict \
                             ${highres_dir}/SURF_R1/label/${hemi}.lobesStrict
                             
        mris_anatomical_stats -a ${highres_dir}/SURF_R1/label/${hemi}.lobesStrict.annot \
                                -f ${highres_dir}/SURF_R1/stats/${hemi}.lobesStrict.stats \
                                SURF_R1 ${hemi}
    done
done