#!/bin/bash

#===================================================================
# A wrapper script that calls a couple of commands to generate pngs
# and gif movies so that it's easy to check the brain extractions
# and freesurfer reconstructions.
#

#-------------------------------------------------------------------

# Created by Kirstie Whitaker
# on 22nd April 2014
# Please contact kw401@cam.ac.uk with any questions

#===================================================================
# SET VARIABLES

nspn_dir=/work/imagingG/NSPN/
data_ready_dir=${nspn_dir}/data_ready/
preprocessed_dir=${data_ready_dir}/1.0-preprocessed/
study_dir=${preprocessed_dir}/uchange.interim.2013/
MPM_preprocessed_dir=${study_dir}/MPM/
surfer_dir=${study_dir}/SURFER
quality_dir=${study_dir}/QUALITY_CONTROL

### ONE DAY THIS CODE DIRECTORY WILL ALL BE IN ONE SENSIBLE PLACE!!
code_dir=${nspn_dir}/workspaces/kw401/UCHANGE/INTERIM_ANALYSIS/SCRIPTS/

#===================================================================
# DOWNLOAD THE CODE

# Create the SCRIPTS directory
mkdir -p ${code_dir}

# Download the code from GitHub
rm -f ${code_dir}/FROM_GITHUB
wget -O ${code_dir}/FROM_GITHUB https://github.com/HappyPenguin/DESCRIBING_DATA/archive/master.zip --no-check-certificate

# Unzip the FROM_GITHUB
#+ -o option forces overwrite
unzip -o ${code_dir}/FROM_GITHUB -d ${code_dir}/

####
# Make all files executable
chmod +x ${code_dir}/DESCRIBING_DATA-master/*
chmod +x ${code_dir}/DESCRIBING_DATA-master/*/*

####
# Convert all files from dos to unix
dos2unix ${code_dir}/DESCRIBING_DATA-master/*
dos2unix ${code_dir}/DESCRIBING_DATA-master/*/*

# This is a really important step btw 
#+ you get unintelligible error messages if you don't run it!

#===================================================================
# RUN THE CODE
make_pngs_script=${code_dir}/DESCRIBING_DATA-master/VISUALIZING_MRI_DATA/MakePngs_HighRes.py

for age_group_dir in `ls -d ${MPM_preprocessed_dir}/*`; do
    age_group=`basename ${age_group_dir}`
    echo "== AGE_GROUP: ${age_group} =="

    for sub_dir in `ls -d ${age_group_dir}/*`; do
        sub=`basename ${sub_dir}`
        echo "    SUB: ${sub}"

        for occ_dir in `ls -d ${sub_dir}/*`; do
            occ=`basename ${occ_dir}`
            echo "        OCC: ${occ}"

            sub_mpm_dir=${MPM_preprocessed_dir}/${age_group}/${sub}/${occ}/
            sub_surf_dir=${surfer_dir}/${age_group}/${sub}/${occ}/mri/
            sub_quality_dir=${quality_dir}/${age_group}/${sub}/${occ}/
            
            mkdir -p ${sub_quality_dir}
            
            if [[ ! -f ${sub_quality_dir}/check_fsl_brain_head.gif ]]; then
                echo -n "          Checking FSL brain and head extraction"
                ${make_pngs_script} -tc2 white \
                                    -ax \
                                    ${sub_mpm_dir}/R1_head.nii.gz \
                                    ${sub_mpm_dir}/R1_brain.nii.gz \
                                    ${sub_quality_dir}/
                      
                cd ${sub_quality_dir}
                ffmpeg -i axial_slice_%04d.png -pix_fmt rgb24 check_fsl_brain_head.gif
                rm ${sub_quality_dir}/*png
                cd ${data_ready_dir}
            fi
            
            if [[ ! -f ${sub_quality_dir}/check_surfer_brain.gif ]]; then
                echo -n "          Checking Freesurfer brain extraction"
                ${make_pngs_script} -tc2 white \
                                    -ax \
                                    -cm2 winter \
                                    ${sub_surf_dir}/T1.nii \
                                    ${sub_surf_dir}/brainmask.nii \
                                    ${sub_quality_dir}/
                                    
                cd ${sub_quality_dir}
                ffmpeg -i axial_slice_%04d.png -pix_fmt rgb24 check_surfer_brain.gif
                rm ${sub_quality_dir}/*png
                cd ${data_ready_dir}
            fi
            
        done
    done
done
