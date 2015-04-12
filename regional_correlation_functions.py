#!/usr/bin/env python

def regional_linregress(df, x, aparc_names):
    '''
    regional_linregress
    
    INPUTS: 
        df ------------- pandas data frame
        x -------------- independent variable name (must be column in df)
        aparc_names ---- list of variable names (columns in df) to loop
                           through as dependent variables for the regression
                           
    RETURNS:
        m_array -------------- numpy array containing slopes for each region
        r_array -------------- numpy array containing pearson r values for each region
        p_array -------------- numpy array containing raw p values for each region
        p_fdr_array ---------- numpy array containing fdr corrected p values for each region
        m_masked_array ------- numpy array containing the slope values for regions which
                                 are indivudially significant otherwise -99 markers
        m_fdr_masked_array --- numpy array containing the slope values for regions which
                                 pass fdr correction otherwise -99 markers
    '''
    
    # Import what you need
    from statsmodels.sandbox.stats.multicomp import fdrcorrection0 as fdr
    import numpy as np
    from scipy.stats import linregress
    
    # Set up some empty arrays
    # to contain the slope of the regression line (m)
    # the list of raw p values (p)
    # and the r values (r) for each region.
    m_array = np.ones(len(aparc_names))
    p_array = np.ones(len(aparc_names))
    r_array = np.ones(len(aparc_names))

    # Loop through all the regions and record m, p and r for each region
    for i, roi in enumerate(aparc_names):
        m, c, r, p, std_err = linregress(df[x].values,
                                            df[roi].values)
        m_array[i] = m
        r_array[i] = r    
        p_array[i] = p
        
    # Calculate the fdr p values
    p_fdr_array = fdr(p_array)[1]
    p_fdr_mask = fdr(p_array)[0]
    
    # Create two masked versions of the slope array
    m_masked_array = np.copy(m_array)
    m_masked_array[p_array>0.05] = -99
    
    m_fdr_masked_array = np.copy(m_array)
    m_fdr_masked_array[p_fdr_array>0.05] = -99
    
    # Return the arrays
    return m_array, r_array, p_array, p_fdr_array, m_masked_array, m_fdr_masked_array
        
    
def regional_linregress_byregion(df_x, df_y, aparc_names):
    '''
    regional_linregress
    
    INPUTS: 
        df_x ------------- pandas data frame containing x axis values
        df_y ------------- pandas data frame containing y axis values
        aparc_names ------ list of variable names (columns in df_x and df_y)
                             to loop though and conduct pairwise regressions
                           
    RETURNS:
        m_array -------------- numpy array containing slopes for each region
        r_array -------------- numpy array containing pearson r values for each region
        p_array -------------- numpy array containing raw p values for each region
        p_fdr_array ---------- numpy array containing fdr corrected p values for each region
        m_masked_array ------- numpy array containing the slope values for regions which
                                 are indivudially significant otherwise -99 markers
        m_fdr_masked_array --- numpy array containing the slope values for regions which
                                 pass fdr correction otherwise -99 markers
    '''
    
    # Import what you need
    from statsmodels.sandbox.stats.multicomp import fdrcorrection0 as fdr
    import numpy as np
    from scipy.stats import linregress
    
    # Set up some empty arrays
    # to contain the slope of the regression line (m)
    # the list of raw p values (p)
    # and the r values (r) for each region.
    m_array = np.ones(len(aparc_names))
    p_array = np.ones(len(aparc_names))
    r_array = np.ones(len(aparc_names))

    df = df_x.merge(df_y, on='nspn_id')
    
    # Loop through all the regions and record m, p and r for each region
    for i, roi in enumerate(aparc_names):
        m, c, r, p, std_err = linregress(df['{}_x'.format(roi)].values,
                                            df['{}_y'.format(roi)].values)
        m_array[i] = m
        r_array[i] = r    
        p_array[i] = p
        
    # Calculate the fdr p values
    p_fdr_array = fdr(p_array)[1]
    p_fdr_mask = fdr(p_array)[0]
    
    # Create two masked versions of the slope array
    m_masked_array = np.copy(m_array)
    m_masked_array[p_array>0.05] = -99
    
    m_fdr_masked_array = np.copy(m_array)
    m_fdr_masked_array[p_fdr_array>0.05] = -99
    
    # Return the arrays
    return m_array, r_array, p_array, p_fdr_array, m_masked_array, m_fdr_masked_array
    