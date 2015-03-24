#!/usr/bin/env python

def permutation_ols(df, formula, n=500):
    '''
    INPUTS:
        df      - data frame
        formula - text string containing pasty style formula
                  referring to columns in data frame
        n       - number of permutations
                    default = 500
                    
    RETURNS:
        t_values - a numpy array of n+1 t values (with the first being
                   the true t-value) for each of the regressors in the model
                   Note that these t-values are tests of the partial correlation
                   between the regressor and the dependent variable *after*
                   taking into account any variance described by the other
                   regressors
        p_values - the permutation test p-values for each regressor.
                     p < 0.05 --> significantly greater than the null
                     p > 0.95 --> significantly smaller than the null

    '''
    import pandas as pd
    from statsmodels.formula.api import ols
    import numpy as np
    
    # First calculate the true linear model
    lm_true = ols(formula, df).fit()
    
    # Write the values you need into numpy arrays
    t_values = np.copy(lm_true.tvalues)

    # Make a copy of the endog (y) and exog (x) values
    # (These are the data you sent to the linear model)
    x = np.copy(lm_true.model.exog)
    y = np.copy(lm_true.model.endog)
    
    for i in range(n):
        # Now shuffle y while keeping x the same
        np.random.shuffle(y)
        
        # Recombine your data into a data frame
        df_shuff = pd.DataFrame(np.append(y[..., None], x, 1))
        df_shuff.columns = [lm_true.model.endog_names] + lm_true.model.exog_names
        
        lm_shuff = ols(formula, df_shuff).fit()
        
        t_values = np.vstack([t_values, lm_shuff.tvalues])
        
    # Now calculate the permuted p value for each column in x.
    p_values = np.ones(t_values.shape[1])
    
    for x in range(x.shape[1]):
        p_values[x] = np.sum(t_values[1:,x] < t_values[0,x]) / np.float(n)
        
    return t_values, p_values
        

def bootstrap_ols(df, formula, n=500):
    '''
    INPUTS:
        df      - data frame
        formula - text string containing pasty style formula
                  referring to columns in data frame
        n       - number of samples with replacement
                    default = 500
                    
    RETURNS:
        
    '''
    import pandas as pd
    from statsmodels.formula.api import ols
    import numpy as np
    
    # First calculate the true linear model
    lm_true = ols(formula, df).fit()
    
    # Write the values you need into numpy arrays
    t_values = np.copy(lm_true.tvalues)

    # Make a copy of the endog (y) and exog (x) values
    # (These are the data you sent to the linear model)
    x = np.copy(lm_true.model.exog)
    y = np.copy(lm_true.model.endog)
    
    for i in range(n):
        # Now choose *with replacement* a sample of y measures
        # that is the same length as the y measure itself
        ids = np.random.choice(len(y), len(y))
        
        # Recombine your data into a data frame
        df_shuff = pd.DataFrame(np.append(y[ids, None], x[ids], 1))
        df_shuff.columns = [lm_true.model.endog_names] + lm_true.model.exog_names
        
        lm_shuff = ols(formula, df_shuff).fit()
        
        t_values = np.vstack([t_values, lm_shuff.tvalues])
        
    # Now calculate the permuted p value for each column in x.
    p_values = np.ones(t_values.shape[1])
    
    for x in range(x.shape[1]):
        p_values[x] = np.sum(t_values[1:,x] < t_values[0,x]) / np.float(n)
        
    return t_values, p_values
        
        
