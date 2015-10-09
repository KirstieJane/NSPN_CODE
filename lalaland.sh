### Using the DEV version of freesurfer on the HPC

FS_OVERRIDE=1

FREESURFER_HOME=/home/kw401/DRIVERS/FREESURFER_DEV/freesurfer/

source ${FREESURFER_HOME}/SetUpFreeSurfer.sh




# RECON-ALL WITH BET BRAIN MASK

sub=12345

SUBJECTS_DIR=/scratch/kw401/UCHANGE_DEV_FREESURFER/SUB_DATA/${sub}/SURFER/

subjid=MRI0

recon-all -i ${SUBJECTS_DIR}/../MPM/MRI0/R1_head.nii.gz \
            -subjid MRI0 \
            -sd ${SUBJECTS_DIR} \
            -T2 ${SUBJECTS_DIR}/../MPM/MRI0/R2s_head.nii.gz \
            -motioncor \
            -talairach \
            -normalization \
            -deface

mri_convert ${SUBJECTS_DIR}/../MPM/MRI0/R1_brain.nii.gz \
            ${SUBJECTS_DIR}/MRI0/mri/brainmask.mgz \
            --conform

mri_em_register -mask ${SUBJECTS_DIR}/MRI0/mri/brainmask.mgz \
                ${SUBJECTS_DIR}/MRI0/mri/orig_nu.mgz \
                $FREESURFER_HOME/average/RB_all_2014-08-21.gca \
                ${SUBJECTS_DIR}/MRI0/mri/transforms/talairach_with_skull.lta
                
mri_watershed -T1 -brain_atlas $FREESURFER_HOME/average/RB_all_withskull_2014-08-21.gca \
                ${SUBJECTS_DIR}/MRI0/mri/transforms/talairach_with_skull.lta \
                ${SUBJECTS_DIR}/MRI0/mri/T1.mgz \
                ${SUBJECTS_DIR}/MRI0/mri/brainmask.auto.mgz 
                
cp ${SUBJECTS_DIR}/MRI0/mri/brainmask.auto.mgz ${SUBJECTS_DIR}/MRI0/mri/brainmask.mgz 

recon-all -subjid MRI0 -sd ${SUBJECTS_DIR} -nuintensitycor -autorecon2 –autorecon3

