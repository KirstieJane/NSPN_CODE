#!/bin/bash

# Checking which participants have had their data deleted
# by Kirstie doing a crazy reckless rm -r :(
# on 26th September 2016

cd /work/imagingQ/NSPN/data_ready/1.0-preprocessed_imagingQ/MEPI/VALIDATION/MEPI

for sub in `ls -d */?????`;  do

    # Start off by assuming that everything's fine
    sub_error=0

    # Put in a little test so you don't print out the subject ID more than once
    # if they're missing a whole bunch of files
    print_sub=0

    #==========================================================================
    # Main input/output files
    #==========================================================================
    # Check that there are the following files in the subject's MRI0 folder

    for test_f in echo123_mefc.nii.gz \
                    echo123_mefl.nii.gz \
                    echo123_noign_medn.nii.gz \
                    echo123_tsoc.nii.gz \
                    echo1.nii echo2.nii echo3.nii \
                    echo123_ctab.txt \
                    brainmask_do.nii.gz brainmask.nii \
                    500.aparc_cortical_consecutive.nii.gz \
                    500.aparc_renum_subMask.nii.gz \
                    _meica_echo123.sh ; do

        if [[ ! -f ${sub}/MRI0/${test_f} ]]; then
            sub_error=1;
            if [[ ${print_sub} == 0 ]]; then
                echo ${sub}
                print_sub=1
            fi
            echo "    Missing ${test_f}"
        fi

    done

    #==========================================================================
    # 500.aparc_cortical_consecutive folder
    #==========================================================================
    # Check that there are 15 files in the 500.aparc_cortical_consecutive folder
    d=${sub}/MRI0/500.aparc_cortical_consecutive

    # If the directory exists then check how many files are in it
    if [[ -d ${d} ]]; then

        n=(`ls ${d} | wc -l`)
        if [[ ${n} != 15 ]]; then
            sub_error=1
            if [[ ${print_sub} == 0 ]]; then
                echo ${sub}
                print_sub=1
            fi
            echo "    500.aparc_cortical_consecutive folder incorrect length: ${n}"
        fi

        # Also check the unsortedParcels folder

        n=(`ls ${d}/unsortedParcels | wc -l`)
        if [[ ${n} != 8 ]]; then
            sub_error=1
            if [[ ${print_sub} == 0 ]]; then
                echo ${sub}
                print_sub=1
            fi
            echo "    unsorted parcels folder in 500.aparc_cortical_consecutive incorrect length: ${n}"
        fi

    # Otherwise print an error
    else
        sub_error=1
        if [[ ${print_sub} == 0 ]]; then
            echo ${sub}
            print_sub=1
        fi
        echo "    Missing 500.aparc_cortical_consecutive folder"
    fi


    #==========================================================================
    # 500.aparc_renum_subMask folder
    #==========================================================================
    # Check that there are 15 files in the 500.aparc_renum_subMask folder
    d=${sub}/MRI0/500.aparc_renum_subMask

    # If the directory exists then check how many files are in it
    if [[ -d ${d} ]]; then

        n=(`ls ${d} | wc -l`)
        if [[ ${n} != 15 ]]; then
            sub_error=1
            if [[ ${print_sub} == 0 ]]; then
                echo ${sub}
                print_sub=1
            fi
            echo "    500.aparc_renumSubMask folder incorrect length: ${n}"
        fi

        # Also check the unsortedParcels folder

        n=(`ls ${d}/unsortedParcels | wc -l`)
        if [[ ${n} != 8 ]]; then
            sub_error=1
            if [[ ${print_sub} == 0 ]]; then
                echo ${sub}
                print_sub=1
            fi
            echo "    unsorted parcels folder in 500.aparc_renumSubMask incorrect length: ${n}"
        fi

    # Otherwise print an error
    else
        sub_error=1
        if [[ ${print_sub} == 0 ]]; then
            echo ${sub}
            print_sub=1
        fi
        echo "    Missing 500.aparc_renumSubMask folder"
    fi

    #==========================================================================
    # meica.echo123 folder
    #==========================================================================
    # Check that there are 54 files in the meica.echo123 folder
    d=${sub}/MRI0/meica.echo123

    # If the directory exists then check how many files are in it
    if [[ -d ${d} ]]; then

        n=(`ls ${d} | wc -l`)
        if [[ ${n} != 54 ]]; then
            sub_error=1
            if [[ ${print_sub} == 0 ]]; then
                echo ${sub}
                print_sub=1
            fi
            echo "    meica.echo123 folder incorrect length: ${n}"
        fi

    # Otherwise print an error
    else
        sub_error=1
        if [[ ${print_sub} == 0 ]]; then
            echo ${sub}
            print_sub=1
        fi
        echo "    Missing meica.echo123 folder"
    fi


    #===========================================================================
    # Add the subject to the need to check list
    #===========================================================================
    if [[ ${sub_error} == 1 ]]; then
        echo ${sub} >> KW_rerun
    fi

done
