#!/bin/bash

#======================================================
# USAGE: freesurfer_probtrackx <data_dir> <subid>
#======================================================
function usage {
    echo "USAGE: freesurfer_probtrackx <data_dir> <subid>"
    exit
}
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
    usage
fi

#======================================================
# DEFINE SOME USEFUL VARIABLES
#======================================================
surfer_dir=${data_dir}/SUB_DATA/${sub}/SURFER/MRI0/
export SUBJECTS_DIR=`dirname ${surfer_dir}`

#label_list=(`dirname ${0}`/destrieux_labels_order.txt)
#labels=(`cat ${label_list}`)

#======================================================
# Split up the input parcellation scheme and 
# make the seeds_targets.txt file for each individual
#======================================================

rm -rf ${surfer_dir}/probtrackx/seeds_targets_list.txt

mkdir -p ${surfer_dir}/probtrackx/seeds_targets_500cortExpConsecWMoverlap/

i=1
while [[ ${i} -le 308 ]]; do
    
    # If the region hasn't yet been created on its own then do that now
    if [[ ! -f ${surfer_dir}/probtrackx/seeds_targets_500cortExpConsecWMoverlap/Seg`printf %04d $i`.nii.gz ]]; then
        fslmaths ${surfer_dir}/parcellation/500.aparc_cortical_expanded_consecutive_WMoverlap.nii.gz \
                        -thr ${i} \
                        -uthr ${i} \
                        -bin \
                        ${surfer_dir}/probtrackx/seeds_targets_500cortExpConsecWMoverlap/Seg`printf %04d $i`.nii.gz
    fi

    # Always write the region to the seeds_targets_list file
    echo ${surfer_dir}/probtrackx/seeds_targets_500cortExpConsecWMoverlap/Seg`printf %04d $i`.nii.gz \
                            >> ${surfer_dir}/probtrackx/seeds_targets_list.txt
    let i=${i}+1
done                    

#======================================================
# Create the probtrackx command for each seed
#======================================================
mkdir -p ${surfer_dir}/probtrackx/commands_dir/

for region in `cat  ${surfer_dir}/probtrackx/seeds_targets_list.txt`; do

    region_name=`basename ${region} .nii.gz`

    echo "#!/bin/bash" > ${surfer_dir}/probtrackx/commands_dir/${sub}_${region_name}.sh
    echo "probtrackx -s ${surfer_dir}/dmri.bedpostX/merged \
                     -m ${surfer_dir}/dmri.bedpostX/nodif_brain_mask \
                     -x ${region} \
                     --dir=${surfer_dir}/probtrackx/${region_name}/ \
                     --forcedir \
                     --opd \
                     --os2t \
                     --s2tastext \
                     --targetmasks=${surfer_dir}/probtrackx/seeds_targets_list.txt \
                     --waypoints=${surfer_dir}/dlabel/anatorig/White-Matter++.nii.gz \
                     -l \
                     --xfm=${surfer_dir}/dmri/xfms/anatorig2diff.bbr.mat " \
                            >> ${surfer_dir}/probtrackx/commands_dir/${sub}_${region_name}.sh
                            
done # Close region loop

#======================================================
# Run the scripts!
#======================================================

scripts_dir=`dirname ${0}`

if [[ -f ${scripts_dir}/WRAPPERS/SubmitAllScriptsInDirectory.sh ]]; then
    ${scripts_dir}/WRAPPERS/SubmitAllScriptsInDirectory.sh ${surfer_dir}/probtrackx/commands_dir/
else
    echo "======= Can't find SubmitAllScripts wrapper! ======="
    echo "= You'll have to submit these manually             ="
fi
#=============================================================================