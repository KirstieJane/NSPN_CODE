#!/bin/bash

surfer_dir=$1

SUBJECTS_DIR=${surfer_dir}/../
surf_sub=`basename ${surfer_dir}`

# Try a few different dmax values just to see
# how that value changes the results
for dmax in 5 10 15 20 100; do
    mri_aparc2aseg --s ${surf_sub} \
                --labelwm \
                --wmparc-dmax 100 \
                --rip-unknown \
                --annot lobes_${dmax100}

    mri_segstats --i ${surfer_dir}/mri/norm.mgz \
                 --seg ${surfer_dir}/mri/lobes_${dmax}+aseg.mgz \
                 --sum ${surfer_dir}/stats/lobes_${dmax}+aseg.stats \
                 --pv ${surfer_dir}/mri/norm.mgz

done


#===================================================================================
#### OLD! DON'T USE ###
# But useful if you're looking to figure out how to extract
# stats from the surface!!
#===================================================================================
# data_dir=/work/imagingG/NSPN/workspaces/kw401/UCHANGE/INTERIM_ANALYSIS/
# sublist=${data_dir}/sublist
# for sub in `cat ${sublist}`; do
    # highres_dir=${data_dir}/SUB_DATA/${sub}/HIGHRES/MRI0/
    # SUBJECTS_DIR=${highres_dir}
    # for hemi in lh rh; do
        # mri_annotation2label --subject SURF_R1 \
                             # --hemi ${hemi} \
                             # --lobesStrict \
                             # ${highres_dir}/SURF_R1/label/${hemi}.lobesStrict
#                             
        # mris_anatomical_stats -a ${highres_dir}/SURF_R1/label/${hemi}.lobesStrict.annot \
                                # -f ${highres_dir}/SURF_R1/stats/${hemi}.lobesStrict.stats \
                                # SURF_R1 ${hemi}
    # done
# done
