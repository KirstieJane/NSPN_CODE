for cohort in [ 'DISCOVERY', 'VALIDATION', 'COMPLETE' ]:

    measure_dict = measure_dict_dict['{}_ALL'.format(cohort)]

    df_PV = pd.DataFrame(data=aparc_names, columns=['ROI_name'])
    df_PV['von_economo'] = von_economo
    df_PV['lobe'] = lobes

    df_PV['module'] = measure_dict['Module_CT_covar_ones_all_COST_10'] + 1
    df_PV['degree'] = measure_dict['Degree_CT_covar_ones_all_COST_10']
    df_PV['pc'] = measure_dict['PC_CT_covar_ones_all_COST_10']
    df_PV['closeness'] = measure_dict['Closeness_CT_covar_ones_all_COST_10']
    df_PV['clustering'] = measure_dict['Clustering_CT_covar_ones_all_COST_10']
    df_PV['L'] = measure_dict['ShortestPath_CT_covar_ones_all_COST_10']
    df_PV['d_euc_av'] = measure_dict['AverageDist_CT_covar_ones_all_COST_10']
    df_PV['d_euc_tot'] = measure_dict['TotalDist_CT_covar_ones_all_COST_10']
    df_PV['interhem_proportion'] = measure_dict['InterhemProp_CT_covar_ones_all_COST_10']

    df_PV['CT_all_mean'] = measure_dict['CT_all_mean']
    df_PV['MT_projfrac+030_all_mean'] = measure_dict['MT_projfrac+030_all_mean']
    df_PV['MT_projdist-100_all_mean'] = measure_dict['MT_projdist-100_all_mean']
    df_PV['CT_all_slope_age'] = measure_dict['CT_all_slope_age']
    df_PV['MT_projfrac+030_all_slope_age'] = measure_dict['MT_projfrac+030_all_slope_age']
    df_PV['MT_projdist-100_all_slope_age'] = measure_dict['MT_projdist-100_all_slope_age']
    df_PV['MT_projfrac+030_all_slope_ct'] = measure_dict['MT_projfrac+030_all_slope_ct']
    df_PV['MT_projdist-100_all_slope_ct'] = measure_dict['MT_projdist-100_all_slope_ct']
    df_PV['CT_all_slope_age_at14'] = measure_dict['CT_all_slope_age_at14']
    df_PV['MT_projfrac+030_all_slope_age_at14'] = measure_dict['MT_projfrac+030_all_slope_age_at14']
    df_PV['MT_projdist-100_all_slope_age_at14'] = measure_dict['MT_projdist-100_all_slope_age_at14']

    #measure_dict_1415 = measure_dict_dict['AGE_BIN_1_EXCLBAD']
    #df_PV['CT_1415_mean'] = measure_dict_1415['CT_all_mean']
    #df_PV['MT_projfrac+030_1415_mean'] = measure_dict_1415['MT_projfrac+030_all_mean']

    #measure_dict_2224 = measure_dict_dict['AGE_BIN_5_EXCLBAD']
    #df_PV['CT_2224_mean'] = measure_dict_2224['CT_all_mean']
    #df_PV['MT_projfrac+030_2224_mean'] = measure_dict_2224['MT_projfrac+030_all_mean']

    df_PV.to_csv('CTMT_data_20151010_{}_ALL.csv'.format(cohort))
