#!/bin/bash

surfer_dir=$1

SUBJECTS_DIR=${surfer_dir}/../
surf_sub=`basename ${surfer_dir}`

# First you have to pull together the labels
# from the surface annotation files
for hemi in lh rh; do
    mri_annotation2label --subject ${surf_sub} \
                         --hemi ${hemi} \
                         --lobesStrict \
                         ${surfer_dir}/label/${hemi}.lobesStrict
done

# Try a few different dmax values just to see
# how that value changes the results
mri_aparc2aseg --s ${surf_sub} \
            --labelwm \
            --rip-unknown \
            --annot lobesStrict \
            --o ${surfer_dir}/mri/lobes+aseg.mgz

mri_segstats --i ${surfer_dir}/mri/norm.mgz \
             --seg ${surfer_dir}/mri/lobes+aseg.mgz \
             --sum ${surfer_dir}/stats/lobes+aseg.stats \
             --pv ${surfer_dir}/mri/norm.mgz \
             --id 1001 1003 1004 1005 1006 1007 \
                  2001 2003 2004 2005 2006 2007 \
                  3001 3003 3004 3005 3006 3007 \
                  4001 4003 4004 4005 4006 4007


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
