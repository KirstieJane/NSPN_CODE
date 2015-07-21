#!/usr/bin/env python

'''
A random collection of useful code
'''

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