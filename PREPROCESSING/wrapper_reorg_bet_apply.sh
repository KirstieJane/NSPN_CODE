#!/bin/bash

#===================================================================
# A wrapper script that calls the little script that copies over
# the MPM preprocessing output to the data_ready/1.0-preprocessed
# directory.
#
# The specific script it calls is NSPN_reorg_bet_apply.sh which can
# be found at Kirstie's GitHub page: XXXX <---- INSERT HERE KW!!
#
# This will need some editing to fit into the pipeline, specifically
# editing of where the MPM files are originally created, but
# conceptually the processing will stay very similar.

#-------------------------------------------------------------------

# Created by Kirstie Whitaker
# on 10th April 2014
# Please contact kw401@cam.ac.uk with any questions

#===================================================================
# SET VARIABLES

nspn_dir=/work/imagingG/NSPN/
data_ready_dir=${nspn_dir}/data_ready/
preprocessed_dir=${data_ready_dir}/1.0-preprocessed/
study_dir=${preprocessed_dir}/uchange.interim.2013/
MPM_preprocessed_dir=${study_dir}/MPM/

### THIS LOCATION MAY CHANGE
current_data_dir=${study_dir}/dicom/MPM/

### ONE DAY THIS CODE DIRECTORY WILL ALL BE IN ONE SENSIBLE PLACE!!
code_dir=${nspn_dir}/workspaces/kw401/UCHANGE/INTERIM_ANALYSIS/SCRIPTS/

#===================================================================
# DOWNLOAD THE CODE

# Create the SCRIPTS directory
mkdir -p ${code_dir}

# Download the code from GitHub
rm -f ${code_dir}/FROM_GITHUB
wget -O ${code_dir}/FROM_GITHUB https://github.com/HappyPenguin/NSPN_CODE/archive/master.zip --no-check-certificate

# Unzip the FROM_GITHUB
#+ -o option forces overwrite
unzip -o ${code_dir}/FROM_GITHUB -d ${code_dir}/

####
# Make all files executable
chmod +x ${code_dir}/NSPN_CODE-master/*
chmod +x ${code_dir}/NSPN_CODE-master/*/*

####
# Convert all files from dos to unix
dos2unix ${code_dir}/NSPN_CODE-master/*
dos2unix ${code_dir}/NSPN_CODE-master/*/*

# This is a really important step btw 
#+ you get unintelligible error messages if you don't run it!

#===================================================================
# RUN THE CODE
reorg_bet_apply_script=${code_dir}/NSPN_CODE-master/PREPROCESSING/reorg_bet_apply.sh

for age_group_dir in `ls -d ${current_data_dir}/*`; do
    age_group=`basename ${age_group_dir}`
    echo "== AGE_GROUP: ${age_group} =="

    for sub_dir in `ls -d ${age_group_dir}/*`; do
        sub=`basename ${sub_dir}`
        echo "    SUB: ${sub}"

        for occ_dir in `ls -d ${sub_dir}/*`; do
            occ=`basename ${occ_dir}`
            echo "        OCC: ${occ}"

            target_dir=${MPM_preprocessed_dir}/${age_group}/${sub}/${occ}/

            ${reorg_bet_apply_script} ${occ_dir} ${target_dir}

        done
    done
done
