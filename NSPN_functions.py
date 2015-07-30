#!/usr/bin/env python

'''
A random collection of useful code
'''
from regional_correlation_functions import *
import numpy as np
import pandas as pd

def permutation_correlation(x_orig, y_orig, n_perm=1000):
    '''
    A simple permutation test for linear regression
    between x and y
    '''
    import numpy as np
    from scipy.stats import linregress 
    import pandas as pd
    
    # Make a copy of the original data
    # because the shuffle command does this in place
    # and we don't want any mistakes!
    x = np.copy(x_orig)
    y = np.copy(y_orig)
    
    # Run the unpermuted correlation
    m, c, r, p, sterr = linregress(x, y)
    
    # Create an m_array that will hold the shuffled
    # slope values
    m_array = np.ones([n_perm])
    
    # Now loop through all the shuffles and calculate
    # the regression for each, saving the slopes in the
    # m_array you created above
    for i in range(n_perm):
        np.random.shuffle(y)
        m_shuff, c_shuff, r_shuff, p_shuff, sterr_shuff = linregress(x, y)
        m_array[i] = m_shuff
    
    # If the true slope is positive then we want to look
    # for the proportion of shuffled slopes that are
    # larger than the true slope
    if m < 0:
        perm_p = len(m_array[m_array<m])/np.float(n_perm)
    # If the true slope is negative then we want to look
    # for the proportion of shuffled slopes that are
    # *more negative* than the true slope
    if m > 0:
        perm_p = len(m_array[m_array>m])/np.float(n_perm)
    
    # We're doing a 2 tailed test so we have to multiply
    # the perm_p value by 2
    perm_p = perm_p*2.0
    
    return m, c, r, p, sterr, perm_p

def permutation_multiple_correlation(x_orig, y_orig, covars=[], n_perm=1000):
    '''
    Define a permuation test for multiple regression
    in which we first calculate the real model fit,
    and then extract the slope from 1000 random shuffles
    of the dependent variable. Note that it is only the 
    y variable that is shuffled, all the x data remain
    the same.
    '''
    
    import statsmodels.api as sm
    import numpy as np
    import pandas as pd
    
    # Make a copy of the original data
    # because the shuffle command does this in place
    # and we don't want any mistakes!
    x = np.copy(x_orig)
    y = np.copy(y_orig)
    
    # Create the data frame
    df =  pd.DataFrame({'x' : x,
                        'y' : y})
                        
    # Create your formula
    formula = 'y ~ x'

    # Add in the covariates
    for i in range(len(covars)):
        df['c_{}'.format(i)] = covars[i]
        formula += ' + c_{}'.format(i)
    
    # Fit the model
    model = sm.OLS.from_formula(formula, df)
    results = model.fit()
    m = results.params['x']

    # Create an m_array that will hold the shuffled
    # slope values
    m_array = np.ones([n_perm])
    
    # And repeat the test n_perm times with shuffled
    # y values
    for i in range(n_perm):
        np.random.shuffle(y)
        df['y'] = y
        model_shuf = sm.OLS.from_formula(formula, df)
        results_shuf = model_shuf.fit()
        m_array[i] = results_shuf.params['x']
        
    # If the true slope is positive then we want to look
    # for the proportion of shuffled slopes that are
    # larger than the true slope
    if m < 0:
        perm_p = len(m_array[m_array<m])/np.float(n_perm)
    # If the true slope is negative then we want to look
    # for the proportion of shuffled slopes that are
    # *more negative* than the true slope
    if m > 0:
        perm_p = len(m_array[m_array>m])/np.float(n_perm)
    
    # We're doing a 2 tailed test so we have to multiply
    # the perm_p value by 2
    perm_p = perm_p*2.0
    
    return results, perm_p
    
def read_in_df(data_file, aparc_names):
    '''
    A very useful command for NSPN behavmerge data frames
    Beware though - this is quite specific and there are 
    a few versions floating around! Be careful
    '''
    import pandas as pd
    import numpy as np
    import os
    
    # Read in the data file
    df = pd.read_csv(data_file, sep=',')
    
    # Only keep the first scan!
    df = df.loc[df.occ==0, :]

    # Strip "thickness" or "thicknessstd" from the column
    # names so they match with the aparc_names names
    data_cols = [ x.replace('_{}'.format('thicknessstd'), '') for x in df.columns ]
    df.columns = data_cols
    data_cols = [ x.replace('_{}'.format('thickness'), '') for x in df.columns ]
    df.columns = data_cols
    
    # Define a few variables you'll want in the data frame
    df['ones'] = df['age_scan'] * 0 + 1
    df['age'] = df['age_scan']
    
    df['Global'] = df[aparc_names].mean(axis=1)
    df['Global_std'] = df[aparc_names].mean(axis=1)

    # If there is a corresponding standard deviation
    # file then read in the standard deviation
    if 'mean' in data_file:
        std_data_file = data_file.replace('mean', 'std')
    else:
        std_data_file = data_file.replace('thickness', 'thicknessstd')
    
    if os.path.isfile(std_data_file):
    
        # Repeating the steps really
        # Read in the file
        df_std = pd.read_csv(std_data_file, sep=',')
        # Only keep the first occ
        df_std = df_std.loc[df_std.occ==0, :]
        # Change the names so they match up
        data_cols = [ x.replace('_{}'.format('thicknessstd'), '') for x in df_std.columns ]
        df_std.columns = data_cols
        data_cols = [ x.replace('_{}'.format('thickness'), '') for x in df_std.columns ]
        df_std.columns = data_cols
        
        # Now write the std across all aparc names into the original data frame
        # by averaging the variances
        df['Global_std'] = np.sqrt(np.average(df_std[aparc_names]**2, axis=1))
    
    # Convert the values to floats
    df[aparc_names] = df[aparc_names].astype('float')
    
    # If this is an MT, R2s, synthetic, MD, L1 or L23 file
    # then you have to divide the values by 1000
    # However there have been problems here in the past with
    # mixing multiplied with non-multiplied values
    # so we'll actually just check for values greater than a
    # reasonable maximum and divide those ones.
    cols_list = aparc_names+['Global']+['Global_std']
    if 'MT' in os.path.basename(data_file):
        df.loc[df['Global']>50, cols_list] = df.loc[df['Global']>50, cols_list]/1000.0
    if 'synthetic' in os.path.basename(data_file):
        df.loc[df['Global']>50, cols_list] = df.loc[df['Global']>50, cols_list]/1000.0
    if 'R2s' in os.path.basename(data_file):
        df.loc[df['Global']>1, cols_list] = df.loc[df['Global']>1, cols_list]/1000.0
    if 'L1' in os.path.basename(data_file):
        df.loc[df['Global']>0.01, cols_list] = df.loc[df['Global']>0.01, cols_list]/1000.0
    if 'L23' in os.path.basename(data_file):
        df.loc[df['Global']>0.01, cols_list] = df.loc[df['Global']>0.01, cols_list]/1000.0
    if 'MD' in os.path.basename(data_file):
        df.loc[df['Global']>0.01, cols_list] = df.loc[df['Global']>0.01, cols_list]/1000.0
    
    return df
    
def create_pysurfer_command(roi_file,
                            scripts_dir, 
                            sub_data_dir, 
                            c='jet', 
                            l=None, 
                            u=None, 
                            t=-99, 
                            s='pial', 
                            center=False):
    '''
    Create a text string containing the appropriate options for
    the pysurfer command
    '''
    import os
    
    # Create the command for pysurfer and run it
    # start by putting in the name of the code with its path
    command_list = [ os.path.join(scripts_dir, 
                                  'DESCRIBING_DATA',
                                  'VISUALIZING_SURFACE_MRI_DATA',
                                  'pysurfer_plot_500parcellation_surface_values.py') ]
    
    # Set the subject directory
    command_list += [ '-sd {}'.format(sub_data_dir) ]
    
    # Set the surface
    if s:
        command_list += [ '-s {}'.format(s) ]

    # Set the colormap
    if c:
        command_list += [ '-c {}'.format(c) ]

    # Set the colormap limits
    if l:
        command_list += [ '-l {}'.format(l) ]
    if u:
        command_list += [ '-u {}'.format(u) ]
        
    # Set the threshold
    if t:
        command_list += [ '-t {}'.format(t) ]

    # Center if necessary
    if center:
        command_list += [ '--center' ]
        
    # And add the filename
    command_list += [ roi_file ]
        
    # Finally join it all together into one string
    command = ' '.join(command_list)

    return command
    
    
def renumber_modules(measure_dict):

    import numpy as np
    
    module = measure_dict['Module_CT_covar_ones_all_COST_10']
    ct_14 = measure_dict['CT_all_slope_age_at14']
    
    med_ct_dict = {}
    mods = sorted(set(module))
    for mod in mods:
        med_ct14 = np.percentile(ct_14[module==mod], 50)
        med_ct_dict[mod] = med_ct14
        
    new_mods = sorted(med_ct_dict, key=med_ct_dict.get)
    
    new_module = np.copy(module)
    
    for i, mod in enumerate(module):
        new_module[i] = new_mods.index(mod) + 1
        
    measure_dict['Renumbered_Module_CT_covar_ones_all_COST_10'] = new_module
    
    return measure_dict
    
    
def save_name_lists(measure_dict, aparc_names, lobes, von_economo, von_economo_3, centroids):

    # ROI names
    measure_dict['aparc_names'] = aparc_names
    measure_dict['dk_names_34'] = sorted(list(set([ roi.split('_')[1] for roi in aparc_names ])))
    measure_dict['dk_names_68'] = sorted(list(set([ roi.rsplit('_', 1)[0] for roi in aparc_names ])))        
    
    # ROI hemispheres
    measure_dict['hemi'] = np.array([ name[0] for name in aparc_names])
    measure_dict['hemi_34'] = np.array([ name[0] for name in measure_dict['dk_names_34']])
    measure_dict['hemi_68'] = np.array([ name[0] for name in measure_dict['dk_names_68']])
    
    # ROI lobes, von_economo labels and von_economo_3 labels
    measure_dict['lobes'] = lobes
    measure_dict['von_economo'] = von_economo
    measure_dict['von_economo_3'] = von_economo_3
    
    for measure in [ 'lobes', 'von_economo', 'von_economo_3' ]:
        # 34 regions
        list_34 = []
        for roi in measure_dict['dk_names_34']:
            i = np.where(np.array(aparc_names) =='lh_{}_part1'.format(roi))
            list_34 += [ measure_dict[measure][i[0]] ]
        measure_dict['{}_34'.format(measure)] = np.array(list_34)
        
        # 68 regions
        list_68 = []
        for roi in measure_dict['dk_names_68']:
            i = np.where(np.array(aparc_names) =='{}_part1'.format(roi))
            list_68 += [ measure_dict[measure][i[0]] ]
        measure_dict['{}_68'.format(measure)] = np.array(list_68)

    # Centroids - for 308 only at the moment!
    measure_dict['centroids'] = centroids
    
    # Record the number of subregions for each DK atlas region
    measure_dict['N_SubRegions'] = np.ones(308)
    # 34
    n_subregions_34 = []
    for roi in measure_dict['dk_names_34']:
        n_subregions_34 += [ len([ x for x in measure_dict['aparc_names'] if roi in x ]) ]
    measure_dict['N_SubRegions_34'] = np.array(n_subregions_34)
    # 68
    n_subregions_68 = []
    for roi in measure_dict['dk_names_68']:
        n_subregions_68 += [ len([ x for x in measure_dict['aparc_names'] if roi in x ]) ]
    measure_dict['N_SubRegions_68'] = np.array(n_subregions_68)
    
    return measure_dict
    
def append_collapsed_across_regions(df, measure_dict):
    '''
    This code adds in additional columns to the end of
    the data frame collapsing across regions but 
    separating the hemispheres (68) and again combining
    the two hemispheres (34)
    
    df is the data frame read in from the FS_ROIS output
    measure_dict must contain the dk_names and the aparc_names
    '''
    for roi in measure_dict['dk_names_34']:
        roi_list = [ x for x in measure_dict['aparc_names'] if roi in x ]
        df['{}'.format(roi)] = df[roi_list].mean(axis=1)
    for roi in measure_dict['dk_names_68']:
        roi_list = [ x for x in measure_dict['aparc_names'] if roi in x ]
        df['{}'.format(roi)] = df[roi_list].mean(axis=1)
    
    return df
    
def save_regional_values(measure_name, measure_dict, df, df_ct):

    names_dict = { 308 : measure_dict['aparc_names'],
                    68 : measure_dict['dk_names_68'],
                    34 : measure_dict['dk_names_34'] }
                    
    for n in [ 308, 68, 34 ]:
        
        # Set the suffix for the name
        # 308 is currently blank because it has always
        # been named without a suffix. This could change
        # in the future...but would cause a bunch of problems
        # at the moment!
        if n == 68:
            suff='_68'
        elif n == 34:
            suff='_34'
        else:
            suff = ''
            
        # MEAN
        measure_dict['{}_all_mean{}'.format(measure_name, suff)] = df[names_dict[n]].mean(axis=0).values
        
        # STD
        measure_dict['{}_all_std{}'.format(measure_name, suff)] = df[names_dict[n]].std(axis=0).values

        # CORR W AGE
        (m_array, c_array, r_array, 
            p_array, p_fdr_array,
            m_mask_array, m_fdr_mask_array) = regional_linregress(df, 'age_scan', names_dict[n])
    
        measure_dict['{}_all_slope_age{}'.format(measure_name, suff)] = m_array
        measure_dict['{}_all_slope_age_c{}'.format(measure_name, suff)] = c_array
        measure_dict['{}_all_slope_age_at14{}'.format(measure_name, suff)] = c_array + 14*m_array
        measure_dict['{}_all_slope_age_at25{}'.format(measure_name, suff)] = c_array + 25*m_array
        measure_dict['{}_all_slope_age_r{}'.format(measure_name, suff)] = r_array
        measure_dict['{}_all_slope_age_p{}'.format(measure_name, suff)] = p_array
        measure_dict['{}_all_slope_age_p_fdr{}'.format(measure_name, suff)] = p_fdr_array
        measure_dict['{}_all_slope_age_m_mask{}'.format(measure_name, suff)] = m_mask_array
        measure_dict['{}_all_slope_age_m_fdr_mask{}'.format(measure_name, suff)] = m_fdr_mask_array
        measure_dict['{}_all_slope_age_m_fdr_mask{}'.format(measure_name, suff)] = m_fdr_mask_array
        
        # CORR SLOPE VS AT 14
        m, c, r, p, sterr, perm_p = permutation_correlation(c_array + 14*m_array, m_array)
    
        measure_dict['{}_all_slope_age_vs_at14{}'.format(measure_name, suff)] = m
        measure_dict['{}_all_slope_age_vs_at14_c{}'.format(measure_name, suff)] = c
        measure_dict['{}_all_slope_age_vs_at14_r{}'.format(measure_name, suff)] = r
        measure_dict['{}_all_slope_age_vs_at14_p{}'.format(measure_name, suff)] = p
        measure_dict['{}_all_slope_age_vs_at14_p_perm{}'.format(measure_name, suff)] = perm_p
        
        # CORR W CT
        if not measure_name == 'CT':
            (m_array, c_array, r_array, 
                p_array, p_fdr_array,
                m_mask_array, m_fdr_mask_array) = regional_linregress_byregion(df, df_ct, names_dict[n])
    
            measure_dict['{}_all_slope_ct{}'.format(measure_name, suff)] = m_array
            measure_dict['{}_all_slope_ct_c{}'.format(measure_name, suff)] = c_array
            measure_dict['{}_all_slope_ct_r{}'.format(measure_name, suff)] = r_array
            measure_dict['{}_all_slope_ct_p{}'.format(measure_name, suff)] = p_array
            measure_dict['{}_all_slope_ct_p_fdr{}'.format(measure_name, suff)] = p_fdr_array
            measure_dict['{}_all_slope_ct_m_mask{}'.format(measure_name, suff)] = m_mask_array
            measure_dict['{}_all_slope_ct_m_fdr_mask{}'.format(measure_name, suff)] = m_fdr_mask_array
            
            # CORR SLOPE VS AT 14
            m, c, r, p, sterr, perm_p = permutation_correlation(c_array + 14*m_array, m_array)
        
            measure_dict['{}_all_slope_ct_vs_at14{}'.format(measure_name, suff)] = m
            measure_dict['{}_all_slope_ct_vs_at14_c{}'.format(measure_name, suff)] = c
            measure_dict['{}_all_slope_ct_vs_at14_r{}'.format(measure_name, suff)] = r
            measure_dict['{}_all_slope_ct_vs_at14_p{}'.format(measure_name, suff)] = p
            measure_dict['{}_all_slope_ct_vs_at14_p_perm{}'.format(measure_name, suff)] = perm_p
            
            # CORR SLOPE MT WITH SLOPE CT
            slope_ct = measure_dict['CT_all_slope_age{}'.format(suff)]
            slope_mt = measure_dict['{}_all_slope_age{}'.format(measure_name, suff)]
            
            m, c, r, p, sterr, perm_p = permutation_correlation(slope_ct, slope_mt)
            measure_dict['{}_vs_CT_all_slope_age{}'.format(measure_name, suff)] = m
            measure_dict['{}_vs_CT_all_slope_age_c{}'.format(measure_name, suff)] = c
            measure_dict['{}_vs_CT_all_slope_age_r{}'.format(measure_name, suff)] = r
            measure_dict['{}_vs_CT_all_slope_age_p{}'.format(measure_name, suff)] = p
            measure_dict['{}_vs_CT_all_slope_age_p_perm{}'.format(measure_name, suff)] = perm_p
            
            # CORR BASELINE MT WITH BASELINE CT
            baseline_ct = measure_dict['CT_all_slope_age_at14{}'.format(suff)]
            baseline_mt = measure_dict['{}_all_slope_age_at14{}'.format(measure_name, suff)]
            
            m, c, r, p, sterr, perm_p = permutation_correlation(baseline_ct, baseline_mt)
            measure_dict['{}_vs_CT_all_slope_age_at14{}'.format(measure_name, suff)] = m
            measure_dict['{}_vs_CT_all_slope_age_at14_c{}'.format(measure_name, suff)] = c
            measure_dict['{}_vs_CT_all_slope_age_at14_r{}'.format(measure_name, suff)] = r
            measure_dict['{}_vs_CT_all_slope_age_at14_p{}'.format(measure_name, suff)] = p
            measure_dict['{}_vs_CT_all_slope_age_at14_p_perm{}'.format(measure_name, suff)] = perm_p

            # CORR AGE 25 MT WITH AGE 25 CT
            age25_ct = measure_dict['CT_all_slope_age_at25{}'.format(suff)]
            age25_mt = measure_dict['{}_all_slope_age_at25{}'.format(measure_name, suff)]
            
            m, c, r, p, sterr, perm_p = permutation_correlation(age25_ct, age25_mt)
            measure_dict['{}_vs_CT_all_slope_age_at25{}'.format(measure_name, suff)] = m
            measure_dict['{}_vs_CT_all_slope_age_at25_c{}'.format(measure_name, suff)] = c
            measure_dict['{}_vs_CT_all_slope_age_at25_r{}'.format(measure_name, suff)] = r
            measure_dict['{}_vs_CT_all_slope_age_at25_p{}'.format(measure_name, suff)] = p
            measure_dict['{}_vs_CT_all_slope_age_at25_p_perm{}'.format(measure_name, suff)] = perm_p

    return measure_dict
    
    
def save_network_values(measure_dict, G_name, graph_dict):
    nodal_dict = graph_dict['{}_NodalMeasures'.format(G_name)]
    global_dict = graph_dict['{}_GlobalMeasures'.format(G_name)]
    
    measure_dict['Degree_{}'.format(G_name)] = nodal_dict['degree']
    measure_dict['PC_{}'.format(G_name)] = nodal_dict['pc']
    measure_dict['Module_{}'.format(G_name)] = nodal_dict['module'] + 1
    measure_dict['Closeness_{}'.format(G_name)] = nodal_dict['closeness']
    measure_dict['ShortestPath_{}'.format(G_name)] = nodal_dict['shortest_path']
    measure_dict['Clustering_{}'.format(G_name)] = nodal_dict['clustering']
    measure_dict['AverageDist_{}'.format(G_name)] = nodal_dict['average_dist']
    measure_dict['TotalDist_{}'.format(G_name)] = nodal_dict['total_dist']
    measure_dict['InterhemProp_{}'.format(G_name)] = nodal_dict['interhem_prop']
    
    return measure_dict
    
def save_global_values(measure_dict, measure_name, df, df_ct):
    # MEAN
    measure_dict['{}_global_mean'.format(measure_name)] = df['Global'].values

    # STD
    measure_dict['{}_global_std'.format(measure_name)] = df['Global_std'].values
        
    # VAR across regions
    v_nodal = df[aparc_names].var(axis=0)
    p75 = np.percentile(v_nodal, 75)
    p25 = np.percentile(v_nodal, 25)
    IQR = p75 - p25
    upper_limit = p75 + 1.5*IQR
    measure_dict['{}_low_var_names'.format(measure_name)] = [ name for name in measure_dict['aparc_names'] if v_nodal[name] < upper_limit ]
    measure_dict['{}_high_var_names'.format(measure_name)] = [ name for name in measure_dict['aparc_names'] if v_nodal[name] >= upper_limit ]
    v_all = df[aparc_names].var(axis=1)
    measure_dict['{}_allregions_var'.format(measure_name)] = v_all
    v_low = df[measure_dict['{}_low_var_names'.format(measure_name)]].var(axis=1)
    measure_dict['{}_low_var_regions_var'.format(measure_name)] = v_low
    v_high = df[measure_dict['{}_low_var_names'.format(measure_name)]].var(axis=1)
    measure_dict['{}_high_var_regions_var'.format(measure_name)] = v_high

    # CORR GLOBAL W AGE
    m, c, r, p, sterr, perm_p = permutation_correlation(df['age_scan'], df['Global'].values)

    measure_dict['{}_global_slope_age'.format(measure_name)] = m
    measure_dict['{}_global_slope_age_c'.format(measure_name)] = c
    measure_dict['{}_global_slope_age_at14'.format(measure_name)] = c + 14*m
    measure_dict['{}_global_slope_age_at25'.format(measure_name)] = c + 25*m
    measure_dict['{}_global_slope_age_r'.format(measure_name)] = r
    measure_dict['{}_global_slope_age_p'.format(measure_name)] = p
    measure_dict['{}_global_slope_age_p_perm'.format(measure_name)] = perm_p

    # CORR VAR (ALL REGIONS) W AGE
    m, c, r, p, sterr, perm_p = permutation_correlation(df['age_scan'], v_all)

    measure_dict['{}_allregions_var_slope_age'.format(measure_name)] = m
    measure_dict['{}_allregions_var_slope_age_c'.format(measure_name)] = c
    measure_dict['{}_allregions_var_slope_age_at14'.format(measure_name)] = c + 14*m
    measure_dict['{}_allregions_var_slope_age_at25'.format(measure_name)] = c + 25*m
    measure_dict['{}_allregions_var_slope_age_r'.format(measure_name)] = r
    measure_dict['{}_allregions_var_slope_age_p'.format(measure_name)] = p
    measure_dict['{}_allregions_var_slope_age_p_perm'.format(measure_name)] = perm_p
    
    # CORR VAR (LOW VAR REGIONS) W AGE
    m, c, r, p, sterr, perm_p = permutation_correlation(df['age_scan'], v_low)

    measure_dict['{}_low_var_regions_var_slope_age'.format(measure_name)] = m
    measure_dict['{}_low_var_regions_var_slope_age_c'.format(measure_name)] = c
    measure_dict['{}_low_var_regions_var_slope_age_at14'.format(measure_name)] = c + 14*m
    measure_dict['{}_low_var_regions_var_slope_age_at25'.format(measure_name)] = c + 25*m
    measure_dict['{}_low_var_regions_var_slope_age_r'.format(measure_name)] = r
    measure_dict['{}_low_var_regions_var_slope_age_p'.format(measure_name)] = p
    measure_dict['{}_low_var_regions_var_slope_age_p_perm'.format(measure_name)] = perm_p

    #  CORR W CT
    if not measure_name == 'CT':
        
        df_ct_mt = df_ct.merge(df, 
                                on=['nspn_id', 'occ'],
                                how='inner', 
                                suffixes=['_ct', '_mt'])
                                
        m, c, r, p, sterr, perm_p = permutation_correlation(df_ct_mt['Global_ct'].values,
                                                             df_ct_mt['Global_mt'].values)

        measure_dict['{}_global_slope_ct'.format(measure_name)] = m
        measure_dict['{}_global_slope_ct_c'.format(measure_name)] = c
        measure_dict['{}_global_slope_ct_at14'.format(measure_name)] = c + 14*m
        measure_dict['{}_global_slope_ct_at25'.format(measure_name)] = c + 25*m
        measure_dict['{}_global_slope_ct_r'.format(measure_name)] = r
        measure_dict['{}_global_slope_ct_p'.format(measure_name)] = p
        measure_dict['{}_global_slope_ct_p_perm'.format(measure_name)] = perm_p

    return measure_dict
    