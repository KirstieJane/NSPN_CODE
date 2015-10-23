#!/usr/bin/env python

"""    
#==================
Code to make tables for the NSPN Cortical Myelination paper

Created July 2015 by Kirstie Whitaker
Contact: kw401@cam.ac.uk or www.github.com/KirstieJane
#==================

Suggested Usage:
#==================
table_filename_308 = 'test_308.tex'
caption = 'All Regional Measures (N=308) - Discovery Cohort'
create_latex_tables(measure_dict, table_filename_308, sort_col='MT_projfrac+030_all_slope_age', n=308, caption=caption)
                    
table_filename_34 = 'test_34.tex'
caption = 'Regional Measures (N=34 atlas regions) - Discovery Cohort'
create_latex_tables(measure_dict, table_filename_34, sort_col='MT_projfrac+030_all_slope_age', n=34, caption=caption)

table_filename_68 = 'test_68.tex'
caption = 'Regional Measures (N=68 atlas regions - Discovery Cohort'
create_latex_tables(measure_dict, table_filename_68, sort_col='MT_projfrac+030_all_slope_age', n=68, caption=caption)
                    
"""
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
import itertools as it
from scipy.stats import linregress 
import networkx as nx
from NSPN_functions import *
from networkx_functions import *

#==============================================================================
# FUNCTIONS
#==============================================================================

#------------------------------------------------------------------------------
# Define your formatter functions
#------------------------------------------------------------------------------
def s(x):
    '''
    Simply return as a string
    '''
    return '{}'.format(x)
    
def i(x):
    '''
    Return as string no decimal places
    '''
    return '{:.0f}'.format(np.float(x))

def f_dp2_exp_0(x):
    '''
    Return as string wiht 2 decimal places
    '''
    return '{:.2f}'.format(x)

def f_dp1_exp_0(x):
    '''
    Return as string wiht 1 decimal place
    '''
    return '{:.1f}'.format(x)
    
def f_p(x):
    '''
    Return as string with 3 decimal places
    and no leading 0 unless smaller than
    0.001 then return <.001
    '''
    p = '{:.3f}'.format(x)
    p = p[1:]
    if x < 0.001:
        p = '\\textless.001'
    return p
    
#------------------------------------------------------------------------------
# Define your latex header and footer functions
#------------------------------------------------------------------------------
def create_header_footer():
    '''
    Return text that needs to go at the beginning and end of the latex file
    (the table string will go in between)
    This header and footer should be the same for every table
    '''
    latex_header = '\n'.join([ '\\documentclass{article}', 
                            '\\usepackage{booktabs}', 
                            '\\usepackage[a4paper, left={1cm}, right={1cm}, top={1.5cm}, bottom={1cm}, landscape]{geometry}',
                            '\\usepackage{longtable}',
                            '\\usepackage{array}',
                            '\\begin{document}',
                            '\\newcolumntype{C}[1]{>{\\centering\let\\newline\\\\\\arraybackslash\hspace{0pt}}b{#1}}',
                            '\\newcolumntype{R}[1]{>{\\raggedleft\\arraybackslash\let\\newline\\\\\\arraybackslash\hspace{0pt}}b{#1}}',
                            '\\newcolumntype{L}[1]{>{\\raggedright\\arraybackslash\let\\newline\\\\\\arraybackslash\hspace{0pt}}b{#1}}',
                            '' ])

    latex_footer = '\n\\end{document}\n'

    return latex_header, latex_footer
    
#------------------------------------------------------------------------------
# Set all the parameters for each measure
#------------------------------------------------------------------------------
def get_dicts(measure_dict, n=308):
    '''
    Create a dict of dicts that contains all the parameters you want to set
    by hand. N represents the number of regions you want to access. It can be
    either 308 (all), 68 (collapsing within region), or 34 (collapsing within
    region and across hemisphere).
    '''
    table_dict = {}
    format_dict = {}
    align_title_dict = {}
    align_col_dict = {}
    top_title_dict = {}
    bottom_title_dict = {}
    multi_column_dict = {}

    if n == 68:
        suff = '_68'
        names = 'dk_names_68'
    elif n == 34:
        suff='_34'
        names = 'dk_names_34'
    else:
        suff=''
        names = 'aparc_names'
        
    table_dict['Lobe'] = measure_dict['lobes{}'.format(suff)]
    format_dict['Lobe'] = s
    align_title_dict['Lobe'] = 'c'
    align_col_dict['Lobe'] = 'l'
    top_title_dict['Lobe'] = 'Lobe'
    bottom_title_dict['Lobe'] = ''
    multi_column_dict['Lobe'] = 1

    if n > 34:
        table_dict['Region'] = [ x.split('_')[1] for x in measure_dict[names] ]
    else:
        table_dict['Region'] = measure_dict[names]
    format_dict['Region'] = s
    align_title_dict['Region'] = 'c'
    align_col_dict['Region'] = 'l'
    top_title_dict['Region'] = 'Region'
    bottom_title_dict['Region'] = ''
    multi_column_dict['Region'] = 1

    table_dict['Hemi'] = measure_dict['hemi{}'.format(suff)]
    format_dict['Hemi'] = s
    align_title_dict['Hemi'] = 'c'
    align_col_dict['Hemi'] = 'r'
    top_title_dict['Hemi'] = 'Hemi'
    bottom_title_dict['Hemi'] = ''
    multi_column_dict['Hemi'] = 1

    table_dict['SubRegion'] = [ x.split('part')[-1] for x in measure_dict[names] ]
    format_dict['SubRegion'] = i
    align_title_dict['SubRegion'] = 'C{1.3cm}'
    align_col_dict['SubRegion'] = 'R{1.3cm}'
    top_title_dict['SubRegion'] = 'Sub Region'
    bottom_title_dict['SubRegion'] = ''
    multi_column_dict['SubRegion'] = 1

    table_dict['N_SubRegions'] = measure_dict['N_SubRegions{}'.format(suff)]
    format_dict['N_SubRegions'] = i
    align_title_dict['N_SubRegions'] = 'C{1.5cm}'
    align_col_dict['N_SubRegions'] = 'R{1.5cm}'
    top_title_dict['N_SubRegions'] = 'N Sub Regions'
    bottom_title_dict['N_SubRegions'] = ''
    multi_column_dict['N_SubRegions'] = 1

    table_dict['CT_all_slope_age_at14'] = measure_dict['CT_all_slope_age_at14{}'.format(suff)]
    format_dict['CT_all_slope_age_at14'] = f_dp2_exp_0
    align_title_dict['CT_all_slope_age_at14'] = 'C{1cm}'
    align_col_dict['CT_all_slope_age_at14'] = 'R{1cm}'
    top_title_dict['CT_all_slope_age_at14'] = 'CT at 14'
    bottom_title_dict['CT_all_slope_age_at14'] = '(mm)'
    multi_column_dict['CT_all_slope_age_at14'] = 1

    table_dict['CT_all_slope_age'] = measure_dict['CT_all_slope_age{}'.format(suff)] * 1000
    format_dict['CT_all_slope_age'] = f_dp2_exp_0
    align_title_dict['CT_all_slope_age'] = 'C{1.3cm}'
    align_col_dict['CT_all_slope_age'] = 'R{1.3cm}'
    top_title_dict['CT_all_slope_age'] = '$\\Delta$CT with age'
    bottom_title_dict['CT_all_slope_age'] = '(mm/year) $\\times10^{-3}$'
    multi_column_dict['CT_all_slope_age'] = 2

    table_dict['CT_all_slope_age_p'] = measure_dict['CT_all_slope_age_p{}'.format(suff)]
    format_dict['CT_all_slope_age_p'] = f_p
    align_title_dict['CT_all_slope_age_p'] = 'C{1cm}'
    align_col_dict['CT_all_slope_age_p'] = 'R{1cm}'
    top_title_dict['CT_all_slope_age_p'] = ''
    bottom_title_dict['CT_all_slope_age_p'] = 'P'
    multi_column_dict['CT_all_slope_age_p'] = 0

    table_dict['MT_projfrac+030_all_slope_age_at14'] = measure_dict['MT_projfrac+030_all_slope_age_at14{}'.format(suff)]
    format_dict['MT_projfrac+030_all_slope_age_at14'] = f_dp2_exp_0
    align_title_dict['MT_projfrac+030_all_slope_age_at14'] = 'C{1cm}'
    align_col_dict['MT_projfrac+030_all_slope_age_at14'] = 'R{1cm}'
    top_title_dict['MT_projfrac+030_all_slope_age_at14'] = 'MT at 14'
    bottom_title_dict['MT_projfrac+030_all_slope_age_at14'] = '(AU)'
    multi_column_dict['MT_projfrac+030_all_slope_age_at14'] = 1

    table_dict['MT_projfrac+030_all_slope_age'] = measure_dict['MT_projfrac+030_all_slope_age{}'.format(suff)] * 1000
    format_dict['MT_projfrac+030_all_slope_age'] = f_dp2_exp_0
    align_title_dict['MT_projfrac+030_all_slope_age'] = 'C{1.3cm}'
    align_col_dict['MT_projfrac+030_all_slope_age'] = 'R{1.3cm}'
    top_title_dict['MT_projfrac+030_all_slope_age'] = '$\\Delta$MT with age'
    bottom_title_dict['MT_projfrac+030_all_slope_age'] = '(AU/year) $\\times10^{-3}$'
    multi_column_dict['MT_projfrac+030_all_slope_age'] = 2

    table_dict['MT_projfrac+030_all_slope_age_p'] = measure_dict['MT_projfrac+030_all_slope_age_p{}'.format(suff)]
    format_dict['MT_projfrac+030_all_slope_age_p'] = f_p
    align_title_dict['MT_projfrac+030_all_slope_age_p'] = 'C{1cm}'
    align_col_dict['MT_projfrac+030_all_slope_age_p'] = 'R{1cm}'
    top_title_dict['MT_projfrac+030_all_slope_age_p'] = ''
    bottom_title_dict['MT_projfrac+030_all_slope_age_p'] = 'P'
    multi_column_dict['MT_projfrac+030_all_slope_age_p'] = 0

    table_dict['Degree'] = measure_dict['Degree_CT_covar_ones_all_COST_10{}'.format(suff)]
    if n == 308:
        format_dict['Degree'] = i
    else:
        format_dict['Degree'] = f_dp1_exp_0
    align_title_dict['Degree'] = 'C{1.3cm}'
    align_col_dict['Degree'] = 'R{1.3cm}'
    top_title_dict['Degree'] = 'Degree'
    bottom_title_dict['Degree'] = ''
    multi_column_dict['Degree'] = 1
    
    table_dict['Closeness'] = measure_dict['Closeness_CT_covar_ones_all_COST_10{}'.format(suff)]
    format_dict['Closeness'] = f_dp2_exp_0
    align_title_dict['Closeness'] = 'C{1.8cm}'
    align_col_dict['Closeness'] = 'R{1.8cm}'
    top_title_dict['Closeness'] = 'Closeness'
    bottom_title_dict['Closeness'] = ''
    multi_column_dict['Closeness'] = 1
    
    table_dict['AverageDist'] = measure_dict['AverageDist_CT_covar_ones_all_COST_10{}'.format(suff)]
    format_dict['AverageDist'] = f_dp2_exp_0
    align_title_dict['AverageDist'] = 'C{1.8cm}'
    align_col_dict['AverageDist'] = 'R{1.8cm}'
    top_title_dict['AverageDist'] = 'Average Distance'    
    bottom_title_dict['AverageDist'] = '(mm)'
    multi_column_dict['AverageDist'] = 1
    
    col_list = ['Lobe', 'Region', 
                'Hemi', 'SubRegion', 
                'CT_all_slope_age_at14', 'CT_all_slope_age', 'CT_all_slope_age_p',
                'MT_projfrac+030_all_slope_age_at14', 'MT_projfrac+030_all_slope_age', 'MT_projfrac+030_all_slope_age_p',
                'Degree', 'Closeness', 'AverageDist' ]

    if n==34:
        col_list = col_list[0:2] + ['N_SubRegions'] + col_list[4:]
    elif n==68:
        col_list = col_list[0:3] + ['N_SubRegions'] + col_list[4:]
        
    # Put all these dicts into a dict of dicts
    table_dict_dict = { 'table_dict' : table_dict,
                        'format_dict' : format_dict,
                        'align_title_dict' : align_title_dict,
                        'align_col_dict' : align_col_dict,
                        'top_title_dict' : top_title_dict,
                        'bottom_title_dict' : bottom_title_dict,
                        'multi_column_dict' : multi_column_dict,
                        'col_list' : col_list }
    
    return table_dict_dict
    
#------------------------------------------------------------------------------
# Put the table dictionary into a pandas data frame
#------------------------------------------------------------------------------
def dict_to_df(table_dict, sort_col='MT_projfrac+030_all_slope_age', ascending=False):
    '''
    Put the table_dict into a data frame and sort by col
    '''
    table_df = pd.DataFrame(table_dict)

    table_df.sort(columns=sort_col, 
                    inplace=True,
                    ascending=ascending)

    return table_df

#------------------------------------------------------------------------------
# We're going to summarize this in two different ways
# Collapsing by region (308 --> 68)
# Collapsing by region and hemisphere (308 -->34)
#============ PROBABLY NEEDS TO BE DELETED!! ===========
#------------------------------------------------------------------------------
def get_df_34(table_df):
    table_df_34 = table_df.groupby('Region').mean()
    table_df_34['N_SubRegions'] = table_df.groupby('Region')['SubRegion'].count()
    table_df_34['Lobe'] = table_df.groupby('Region')['Lobe'].first()
    table_df_34['Region'] = table_df.groupby('Region')['Region'].first()
    table_df_34.sort(columns=['MT_projfrac+030_all_slope_age'], 
                    inplace=True,
                    ascending=False)
    return table_df_34
    
def get_df_68(table_df):
    table_df_68 = table_df.groupby(['Region', 'Hemi']).mean()
    table_df_68['N_SubRegions'] = table_df.groupby(['Region', 'Hemi'])['SubRegion'].count()
    table_df_68['Lobe'] = table_df.groupby(['Region', 'Hemi'])['Lobe'].first()
    table_df_68['Region'] = table_df.groupby(['Region', 'Hemi'])['Region'].first()
    table_df_68['Hemi'] = table_df.groupby(['Region', 'Hemi'])['Hemi'].first()
    table_df_68.sort(columns=['MT_projfrac+030_all_slope_age'], 
                    inplace=True,
                    ascending=False)
    return table_df_68
    
#------------------------------------------------------------------------------
# Define your latex adjust column spacings function
#------------------------------------------------------------------------------
def adjust_spacings(latex_table, col_list, align_col_dict):
    '''
    Replace the spacings for the main columns that are output
    on the top line of the latex table string with those in the align_col_dict
    '''
    spacings_list = [ align_col_dict[col] for col in col_list ]
    spacings_list = ''.join(['\\begin{longtable}{'] + spacings_list + ['}'])
    latex_table_list = latex_table.split('\n')
    latex_table_list[0] = spacings_list          
    latex_table = '\n'.join(latex_table_list)

    return latex_table
    
#------------------------------------------------------------------------------
# Define your latex add in caption function
#------------------------------------------------------------------------------
def add_caption(latex_table, caption):
    '''
    Just add in a row on the second line down to include the caption (title)
    for this table
    '''
    latex_table_list = latex_table.split('\n')
    latex_table_list[0] = latex_table_list[0] + '\n\\caption*{{{}}} \\\\'.format(caption)         
    latex_table = '\n'.join(latex_table_list)

    return latex_table
    
#------------------------------------------------------------------------------
# Define your latex adjust header function
#------------------------------------------------------------------------------
def adjust_header(latex_table, align_title_dict, multi_column_dict, top_title_dict, bottom_title_dict):
    '''
    Replace the standard header spacings to give two rows - the first being bold
    and containing some multicolumns and the second being in regular text
    '''
    latex_table_list = latex_table.split('\n')
    titles_list, lineend = [ x.split('&') for x in latex_table_list[2].rsplit(' ',1) ]
    top_titles_list = []
    bottom_titles_list = []
    for title in titles_list:
        title = title.strip()
        if multi_column_dict[title] > 0:
            # Add 1 cm to the width for each additional column:
            align_title_parts = align_title_dict[title].split('{')
            if (len(align_title_parts) > 1) and (multi_column_dict[title] > 1):
                width = np.float(align_title_parts[1].strip('cm}') ) + 1 * (multi_column_dict[title] - 1)
                width_str = ''.join([align_title_parts[0], '{', np.str(width), 'cm}'])
            else:
                width_str = align_title_dict[title]
            top_titles_list += [ '\\multicolumn{{{}}}{{{}}}{{\\textbf{{{}}}}}'.format( multi_column_dict[title], 
                                                                                        width_str, 
                                                                                        top_title_dict[title] ) ]
                                                                                        
        bottom_titles_list += [ '\\multicolumn{{1}}{{{}}}{{{}}}'.format( align_title_dict[title], 
                                                                            bottom_title_dict[title] ) ]

    top_titles_str = ' & '.join(top_titles_list)
    bottom_titles_str = ' & '.join(bottom_titles_list)
    latex_table_list[2] = ' '.join([top_titles_str, lineend[0], '\n', bottom_titles_str, lineend[0]])

    latex_table = '\n'.join(latex_table_list)

    return latex_table
    
#------------------------------------------------------------------------------
# Save the data frame as a latex string
#------------------------------------------------------------------------------
def save_df_to_latex(latex_header, latex_footer, latex_table, output_filename):
    '''
    As it says on the tin, save the data frame to the filename
    '''
    # Write the file to the table_filename
    with open(output_filename, 'w') as f:
        f.write(latex_header)
        f.write(latex_table)
        f.write(latex_footer)
    
#------------------------------------------------------------------------------
# Write the overall wrapper function
#------------------------------------------------------------------------------    
def create_latex_tables(measure_dict, output_filename, caption=False, sort_col='MT_projfrac+030_all_slope_age', ascending=False, n=308):
    '''
    The overall wrapper function
    '''
    # Get the measures
    table_dict_dict = get_dicts(measure_dict, n=n)
    
    # Create a data frame
    table_df = dict_to_df(table_dict_dict['table_dict'], sort_col=sort_col, ascending=ascending)
        
    # Get the appropriate formatting functions
    formatters = [ table_dict_dict['format_dict'][col] for col in table_dict_dict['col_list'] ]

    # Create the latex_table text string
    latex_table = table_df.to_latex(longtable=True, 
                                        index=False, 
                                        columns=table_dict_dict['col_list'],
                                        formatters=formatters,
                                        escape=False)

    # Adjust the spacings
    latex_table = adjust_spacings(latex_table, 
                                    table_dict_dict['col_list'], 
                                    table_dict_dict['align_col_dict'])

    # Adjust the header alignments and make the text bold
    latex_table = adjust_header(latex_table, 
                                table_dict_dict['align_title_dict'], 
                                table_dict_dict['multi_column_dict'], 
                                table_dict_dict['top_title_dict'], 
                                table_dict_dict['bottom_title_dict'])

    # Add in caption
    if caption:
        latex_table = add_caption(latex_table, caption)
    
    # Get your latex document header and footer
    latex_header, latex_footer = create_header_footer()
    
    # Save to output_file
    save_df_to_latex(latex_header, latex_footer, latex_table, output_filename)
    
    # Save to output_file without the header and footer
    save_df_to_latex('', '', latex_table, output_filename.replace('.tex', '.txt'))    


