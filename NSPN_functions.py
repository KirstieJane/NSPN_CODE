#!/usr/bin/env python

'''
A random collection of useful code
'''

def permutation_correlation(x_orig, y_orig, n_perm=1000):
    
    import numpy as np
    from scipy.stats import linregress 
    
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
    
    