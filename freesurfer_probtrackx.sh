#!/bin/bash

#======================================================
# USAGE: freesurfer_probtrackx <data_dir> <subid>
#======================================================


#======================================================
# READ IN COMMAND LINE ARGUMENTS
#======================================================
data_dir=$1
sub=$2

#======================================================
# CHECK THE COMMAND LINE ARGUMENTS
#======================================================

if [[ ! -d ${data_dir} ]]; then
    echo "DATA DIR doesn't exist - check ${data_dir}"
    print_usage=1
fi

if [[ -z ${sub} ]]; then
    echo "SUBID is blank"
    print_usage=1
fi

if [[ ${print_usage} == 1 ]]; then
    echo "USAGE: freesurfer_probtrackx <data_dir> <subid>"
    exit
fi

#======================================================
# DEFINE SOME USEFUL VARIABLES
#======================================================
surfer_dir=${data_dir}/SUB_DATA/${sub}/SURFER/MRI0/
export SUBJECTS_DIR=`dirname ${surfer_dir}`

label_list=(`dirname ${0}`/destrieux_labels_order.txt)
labels=(`cat ${label_list}`)


#======================================================
# AND GET GOING
#======================================================

mkdir -p ${surfer_dir}/probtrackx/seeds_targets/

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

#======================================================
# Make the seeds_targets.txt file for each individual
#======================================================
rm -rf ${surfer_dir}/probtrackx/seeds_targets_list_gii.txt
rm -rf ${surfer_dir}/probtrackx/seeds_targets_list_nii.txt

for hemi in lh rh; do
    for label in ${labels[@]}; do
        echo ${surfer_dir}/probtrackx/seeds_targets/${hemi}.${label}.gii >> ${surfer_dir}/probtrackx/seeds_targets_list_gii.txt
        echo ${surfer_dir}/probtrackx/seeds_targets/${hemi}.${label}.nii.gz >> ${surfer_dir}/probtrackx/seeds_targets_list_nii.txt
    done
done

#======================================================
# Create the probtrackx command for each seed
#======================================================
mkdir -p ${surfer_dir}/probtrackx/commands_dir/

for label in `cat  ${surfer_dir}/probtrackx/seeds_targets_list_gii.txt`; do
    label_name=`basename ${label} .gii`
    echo "#!/bin/bash" > ${data_dir}/probtrackx/commands_dir/${sub}_${label_name}.sh
    echo "probtrackx -s ${surfer_dir}/dmri.bedpostX/merged \
                     -m ${surfer_dir}/dmri.bedpostX/nodif_brain_mask \
                     -x ${label} \
                     --dir=${surfer_dir}/probtrackx/${label_name}/ \
                     --forcedir \
                     --opd \
                     --os2t \
                     --s2tastext \
                     --targetmasks=${surfer_dir}/probtrackx/seeds_targets_list_gii \
                     --waypoints=${surfer_dir}/anatorig/White-Matter++.nii.gz \
                     -l \
                     --onewaycondition \
                     --omatrix1 \
                     --xfm=${surfer_dir}/dmri/xfms/anatorig2diff.bbr.mat \
                     --meshspace=freesurfer " >> ${data_dir}/probtrackx/commands_dir/${sub}_${label_name}.sh

done # Close label loop
