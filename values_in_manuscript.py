#!/usr/bin/env python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

def format_p(p):
    '''
    A tiny little function to re-format p
    as a string because if it's smaller than
    0.001 you aren't going to write the whole number
    in - you'll just say < 0.001
    '''
    if p < 0.001:
        p = '< 0.001'
    else:
        p = '= {:2.3f}'.format(p)
    return p
    
measure_dict = measure_dict_dict['DISCOVERY_EXCLBAD']

ct_14_min = np.min(measure_dict['CT_all_slope_age_at14'])
ct_14_max = np.max(measure_dict['CT_all_slope_age_at14'])
print 'Cortical thickness at 14 years of age, estimated as the intercept of a simple linear regression on age, ranged between {:2.1f} and {:2.1f} mm across different cortical areas.'.format(ct_14_min, ct_14_max)

print '================='

m = measure_dict['CT_global_slope_age']
r = measure_dict['CT_global_slope_age_r']
p_perm = format_p(measure_dict['CT_global_slope_age_p_perm'])

print 'Cortical shrinkage was estimated for each regional node as the linear reduction of cortical thickness as a function of increasing age. On average over all nodes, there was evidence for significant cortical shrinkage (r2 = {:2.2f}, P {}, estimated global rate of shrinkage = {:2.4f} mm/year; Figure 1Aiii). There were no significant main effects of gender, or scanner site, on global cortical shrinkage; and no significant interactions between age and either gender or site.'.format(r**2, p_perm, m)

print '================='

m_fdr = measure_dict['CT_all_slope_age_m_fdr_mask']
n = m_fdr[m_fdr > -99].shape[0]
print 'Many but not all regions showed significant shrinkage: {:2.0f} of 308 nodes separately tested at P < 0.05, FDR corrected for multiple comparisons, demonstrated significant shrinkage'.format(n)

print '================='

m = measure_dict['CT_all_slope_age']
ve = np.array(measure_dict['von_economo'])

df = pd.DataFrame( { 'm' : m, 've' : ve })
formula = 'm ~ C(ve)'
model = smf.ols(formula, data=df).fit()
table = sm.stats.anova_lm(model, typ=2)
F = table.F['C(ve)']
dof_ve = table.df['C(ve)']
dof_res = table.df['Residual']
p = format_p(table['PR(>F)']['C(ve)'])

m_ve2 = np.mean(m[ve==2])
m_ve5 = np.mean(m[ve==5])

print 'Convergently, cortical lamination types had significantly different rates of cortical shrinkage (F[{:1.0f}, {:1.0f}] = {:2.2f}, P {}): von Economo type 2 had the fastest rate ({:2.3f} mm/year) and type 5 had the slowest rate ({:2.3f} mm/year) of cortical shrinkage.'.format(dof_ve, dof_res, F, p, m_ve2, m_ve5)

print '================='
m = measure_dict['CT_all_slope_age_vs_at14']
r = measure_dict['CT_all_slope_age_vs_at14_r']
p_perm = format_p(measure_dict['CT_all_slope_age_vs_at14_p_perm'])

print 'Cortical thickness at baseline (14 years) and adolescent cortical shrinkage (linear reduction in thickness over the age range 14-25 years) were negatively correlated (r2 = {:2.2f}, P {}, β = {:2.1f}x10^-3; Figure 1Biii).'.format(r**2, p_perm, m*1000)

print '================='
m = measure_dict['CT_low_var_regions_var_slope_age']
r = measure_dict['CT_low_var_regions_var_slope_age_r']
p_perm = format_p(measure_dict['CT_low_var_regions_var_slope_age_p_perm'])

print 'The net effect of this baseline-biased shrinkage process was “regression to the mean” or reduction of the variability in cortical thickness between regions (r2 = {:2.2f}, P {}, β = {:2.1f}x10^-3)'.format(r**2, p_perm, m*1000)

print '================='
high_var_names = measure_dict['CT_high_var_regions']
print 'The regions that had high variablility were: {}'.format('\n'.join(high_var_names))

print '================='
m = measure_dict['MTall_global_slope_age']
r = measure_dict['MTall_global_slope_age_r']
p_perm = format_p(measure_dict['MTall_global_slope_age_p_perm'])
print 'Magnetization transfer estimated over the full thickness of cortex significantly increased as a function of age globally over all regions (r2 = {:2.2f}, P {}, β = {:2.2f}x10^-3).'.format(r**2, p_perm, m*1000)

print '================='
pia_dist = np.mean(measure_dict['CT_global_mean'])*0.7
wm_dist = np.mean(measure_dict['CT_global_mean'])*0.3
m = measure_dict['MT_projfrac+030_global_slope_age']
r = measure_dict['MT_projfrac+030_global_slope_age_r']
p_perm = format_p(measure_dict['MT_projfrac+030_global_slope_age_p_perm'])
print 'The location of the maximal rate of age-related increase in MT was at 70% of the depth of cortex, approximately {:2.1f}mm below the pial surface and {:2.2f}mm above the boundary with white matter (r2 = {:2.2f}, P {}, β = {:2.2f}x10^-3).'.format(pia_dist, wm_dist, r**2, p_perm, m*1000)

print '================='
m_fdr = measure_dict['MT_projfrac+030_all_slope_age_m_fdr_mask']
n = m_fdr[m_fdr > -99].shape[0]
print 'The linear relationship between age and MT was significantly greater than zero in {:2.0f} of the 308 cortical regions individually tested (P < 0.05, FDR corrected for multiple comparisons)'.format(n)

print '================='
m = measure_dict['MT_projfrac+030_all_slope_age_vs_at14']
r = measure_dict['MT_projfrac+030_all_slope_age_vs_at14_r']
p_perm = format_p(measure_dict['MT_projfrac+030_all_slope_age_vs_at14_p_perm'])
print 'Cortical areas that had higher myelination at 14 years had slower rates of further increase in myelination, whereas less myelinated areas at baseline had greater rates of increasing myelination in the adolescent period, 14-25 years (r2 = {:2.2f}, P {}, β = {:2.1f}x10^-3; Figure 1Div'.format(r**2, p_perm, m*1000)


print '================='
m_14 = measure_dict['MT_projfrac+030_vs_CT_all_slope_age_at14']
r_14 = measure_dict['MT_projfrac+030_vs_CT_all_slope_age_at14_r']
p_perm_14 = format_p(measure_dict['MT_projfrac+030_vs_CT_all_slope_age_at14_p_perm'])
m_25 = measure_dict['MT_projfrac+030_vs_CT_all_slope_age_at25']
r_25 = measure_dict['MT_projfrac+030_vs_CT_all_slope_age_at25_r']
p_perm_25 = format_p(measure_dict['MT_projfrac+030_vs_CT_all_slope_age_at25_p_perm'])
m_delta = measure_dict['MT_projfrac+030_vs_CT_all_slope_age']
r_delta = measure_dict['MT_projfrac+030_vs_CT_all_slope_age_r']
p_perm_delta = format_p(measure_dict['MT_projfrac+030_vs_CT_all_slope_age_p_perm'])

print '\n\nUnsurprisingly perhaps, there was strong evidence on a regional level for an association between cortical thickness and intra-cortical myelination at age 14 (r2 = {:2.2f}, P {}, β = {:2.3f}mm-1; Figure 2B) and at age 25 (r2 = {:2.2f}, P {}, β = {:2.3f}mm-1) along with a strong correlation between the rates of cortical shrinkage and intra-cortical myelination (r2 = {:2.2f}, P {}, β = {:2.3f}mm-1)'.format(r_14**2, p_perm_14, m_14, r_25**2, p_perm_25, m_25, r_delta**2, p_perm_delta, m_delta )

print '================='
m = measure_dict['MT_projfrac+030_global_slope_ct']
r = measure_dict['MT_projfrac+030_global_slope_ct_r']
p_perm = format_p(measure_dict['MT_projfrac+030_global_slope_ct_p_perm'])

print '\n\nThe association between cortical thickness and myelination was also present when we correlated each region across our participants (r2 = {:2.2f}, P {}, β = {:2.3f}mm-1; Figure 2Diii).'.format(r**2, p_perm, m) 

print '================='
m_fdr = measure_dict['MT_projfrac+030_all_slope_ct_m_fdr_mask']
n = m_fdr[m_fdr > -99].shape[0]
print 'The linear relationship between age and MT was significantly greater than zero in {:2.0f} of the 308 cortical regions individually tested (P < 0.05, FDR corrected for multiple comparisons)'.format(n)


#==================================================================================================
# CHANGE TO THE COMPLETE DATA SET (EXCL_BAD)
#==================================================================================================
measure_dict = measure_dict_dict['DISCOVERY_EXCLBAD']

print '================='
deg = measure_dict['Degree_CT_covar_ones_all_COST_10']
pc = measure_dict['PC_CT_covar_ones_all_COST_10']
av_dist = measure_dict['AverageDist_CT_covar_ones_all_COST_10']
clust = measure_dict['Clustering_CT_covar_ones_all_COST_10']
closeness = measure_dict['Closeness_CT_covar_ones_all_COST_10']

m, c, r_deg_close, p, sterr, p_perm_deg_close = permutation_correlation(deg, closeness)
m, c, r_deg_pc, p, sterr, p_perm_deg_pc = permutation_correlation(deg, pc)
m, c, r_deg_avdist, p, sterr, p_perm_deg_avdist = permutation_correlation(deg, av_dist)
m, c, r_deg_clust, p, sterr, p_perm_deg_clust = permutation_correlation(deg, clust)

print "\n\nAt a nodal level of analysis, degree centrality was positively correlated with closeness centrality (r = {:2.2f}, P {}), participation coefficient (r = {:2.2f}, P {}) and mean connection distance (r = {:2.2f}, P {}); and negatively correlated with clustering (r = {:2.2f}, P {}); see SI for correlation matrix of all nodal graph metrics. This means that high degree hubs were efficiently and hierarchically connected to the rest of the network, mediated inter-modular connections to nodes in other modules, and were more costly in the sense that they were connected by longer distance edges.".format(r_deg_close, format_p(p_perm_deg_close), r_deg_pc, format_p(p_perm_deg_pc), r_deg_avdist, format_p(p_perm_deg_avdist), r_deg_clust, format_p(p_perm_deg_clust) )


print '================='
deg = measure_dict['Degree_CT_covar_ones_all_COST_10']
ct_14 = measure_dict['CT_all_slope_age_at14']
delta_ct = measure_dict['CT_all_slope_age']
mt_14 = measure_dict['MT_projfrac+030_all_slope_age_at14']
delta_mt = measure_dict['MT_projfrac+030_all_slope_age']

m_ct_14, c, r_ct_14, p, sterr, p_perm_ct_14 = permutation_correlation(deg, ct_14)
m_delta_ct, c, r_delta_ct, p, sterr, p_perm_delta_ct = permutation_correlation(deg, delta_ct)
m_mt_14, c, r_mt_14, p, sterr, p_perm_mt_14 = permutation_correlation(deg, mt_14)
m_delta_mt, c, r_delta_mt, p, sterr, p_perm_delta_mt = permutation_correlation(deg, delta_mt)

print "\n\nDegree was negatively correlated with rate of cortical shrinkage (r2 = {:2.2f}, P {}, β = {:2.3f}x10-33mm.year-1/degree; Figure 3Bii) although not with baseline cortical thickness at age 14 (r2 = {:2.2f}, P {}, β = {:2.3f}x10-33mm/degree; Figure 3Bi). Convergently, degree was negatively correlated with baseline MT at age 14 (r2 = {:2.2f}, P {}, β = {:2.3f}x10-3degree-1) and positively correlated with the rate of increase in MT over the course of adolescence (r2 = {:2.2f}, P {}, β = {:2.1f}x10-6year-1/degree). ".format(r_delta_ct**2, format_p(p_perm_delta_ct), m_delta_ct*1000, r_ct_14**2, format_p(p_perm_ct_14), m_ct_14*1000, r_mt_14**2, format_p(p_perm_mt_14), m_mt_14*1000,r_delta_mt**2, format_p(p_perm_delta_mt), m_delta_mt*1000000 )


print '================='
for measure in [ 'Degree', 'PC', 'AverageDist' ]:
    print '\nMEASURE: {}'.format(measure)
    m = measure_dict['{}_CT_covar_ones_all_COST_10'.format(measure)]
    ve = np.array(measure_dict['von_economo'])

    df = pd.DataFrame( { 'm' : m, 've' : ve })
    formula = 'm ~ C(ve)'
    model = smf.ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    F = table.F['C(ve)']
    dof_ve = table.df['C(ve)']
    dof_res = table.df['Residual']
    p = format_p(table['PR(>F)']['C(ve)'])

    print table

    m_ve1 = np.mean(m[ve==1])
    m_ve23 = np.mean(np.hstack([m[ve==2], m[ve==3]]))
    m_ve45 = np.mean(m[ve>3])

    print '1: {:2.2f}, 2&3: {:2.2f}, 4&5: {:2.2f}'.format(m_ve1, m_ve23, m_ve45)

    print 'As a corollary of these correlational results, there were significant differences of degree centrality and other network metrics between cortical lamination types (degree: F[{:1.0f}, {:1.0f}] = {:2.2f}, P {}; participation coefficient): granular isocortex (types 2 and 3) had higher degree centrality, participation coefficient and mean connection distance than agranular cortex (type 1) or koniocortex (type 5).'.format(dof_ve, dof_res, F, p, m_ve2, m_ve5)
