#!/usr/bin/env python

#=============================================================================
# Created by Kirstie Whitaker
# 27th July 2015
# Contact: kw401@cam.ac.uk
#
# This code creates the table of statistical results for figure 1 in the
# for the NSPN Cortical Myelination manuscript
#=============================================================================

#=============================================================================
# IMPORTS
#-----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
import requests
from glob import glob

#=============================================================================
# FUNCTIONS
#=============================================================================
#------------------------------------------------------------------------------
# Define your latex add in caption function
#------------------------------------------------------------------------------
def add_caption(latex_table, caption):
    '''
    Just add in a row on the second line down to include the caption (title)
    for this table
    '''
    latex_table_list = latex_table.split('\n')
    latex_table_list[0] = latex_table_list[0] + '\n\\caption{{{}}} \\\\'.format(caption)         
    latex_table = '\n'.join(latex_table_list)

    return latex_table

def format_p(x):
    '''
    If p is less than 0.001 then return a string of <.001
    '''
    p = '{:.3f}'.format(x)
    p = '$P = {}$'.format(p[1:])
    if x < 0.001:
        p = '$P < .001$'
    return p
        

#=============================================================================
# CREATE A DICT FOR THE TWO COHORTS
#-----------------------------------------------------------------------------
def create_dict(measure_dict_dict):
    name_dict = { 'DISCOVERY_EXCLBAD' : 'Discovery',
                  'VALIDATION_EXCLBAD' : 'Validation'}

    table_dict = {}

    for name, measure_dict in measure_dict_dict.items():
        print '==={}==='.format(name_dict[name])
        
        data_list = []
        col_list = []
        
        #=======================================================
        # Global CT vs Age
        #=======================================================    
        m = measure_dict['CT_global_slope_age']
        r = measure_dict['CT_global_slope_age_r']
        p = measure_dict['CT_global_slope_age_p_perm']

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{Global CT vs Age}}']
        col_list += [ '' ] * 2

        #=======================================================
        # Delta CT vs CT at 14
        #=======================================================    
        m = measure_dict['CT_all_slope_age_vs_at14']
        r = measure_dict['CT_all_slope_age_vs_at14_r']
        p = measure_dict['CT_all_slope_age_vs_at14_p_perm']

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$CT vs CT at 14}}']
        col_list += [ '' ] * 2

        table_dict['\\textbf{{{}}}'.format(name_dict[name])] = data_list

        #=======================================================
        # Global MT vs Age
        #=======================================================    
        m = measure_dict['MT_projfrac+030_global_slope_age']
        r = measure_dict['MT_projfrac+030_global_slope_age_r']
        p = measure_dict['MT_projfrac+030_global_slope_age_p_perm']

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{Global MT vs Age}}']
        
        col_list += [ '' ] * 2

        #=======================================================
        # Delta CT vs CT at 14
        #=======================================================    
        m = measure_dict['MT_projfrac+030_all_slope_age_vs_at14']
        r = measure_dict['MT_projfrac+030_all_slope_age_vs_at14_r']
        p = measure_dict['MT_projfrac+030_all_slope_age_vs_at14_p_perm']

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$MT vs MT at 14}}']
        col_list += [ '' ] * 2

        table_dict['\\textbf{{{}}}'.format(name_dict[name])] = data_list

    return table_dict, col_list
    
#=============================================================================
# MAKE THIS DICT INTO A PANDAS DATA FRAME
#-----------------------------------------------------------------------------
def make_table_df(table_dict, col_list):

    table_df = pd.DataFrame(table_dict, index=col_list)
    table_df = table_df.loc[:, ['\\textbf{Discovery}', '\\textbf{Validation}']]

    return table_df
    
#=============================================================================
# WRITE DATA FRAME TO A LATEX FILE
#-----------------------------------------------------------------------------
def write_latex_table(table_df, table_file, caption):
    latex_table = table_df.to_latex(longtable=True, 
                                        index=True, 
                                        escape=False)

    latex_table = latex_table.replace('lll', 'lcc')
    latex_table = latex_table.replace('\\multirow', '\\rule{0pt}{4ex} \\multirow')
    latex_table = add_caption(latex_table, caption)

    latex_header = '\n'.join([ '\\documentclass{article}', 
                                '\\usepackage{booktabs}', 
                                '\\usepackage[a4paper, left={1cm}, right={1cm}, top={1.5cm}, bottom={1.5cm}, portrait]{geometry}',
                                '\\usepackage{longtable}',
                                '\\usepackage{array}',
                                '\\usepackage{multirow}'
                                '\\begin{document}',
                                '' ])

    latex_footer = '\n\\end{document}\n'

    # Write the file to the table_filename
    with open(table_file, 'w') as f:
        f.write(latex_header)
        f.write(latex_table)
        f.write(latex_footer)
        
    # Write the file to the table_filename without the latex header and footer
    with open(table_file.replace('.tex', '.txt'), 'w') as f:
        f.write(latex_table)
        
#=============================================================================
# PUT IT ALL TOGETHER
#-----------------------------------------------------------------------------
def make_replication_table_figure1(measure_dict_dict, paper_dir):
    #=============================================================================
    # INPUTS
    #-----------------------------------------------------------------------------
    measure_dict_D = measure_dict_dict['DISCOVERY_EXCLBAD']
    measure_dict_V = measure_dict_dict['VALIDATION_EXCLBAD']

    #=============================================================================
    # GET STARTED BY SETTING THE CAPTION AND NAME YOUR OUTPUT FILE
    #-----------------------------------------------------------------------------
    caption = 'Associated statistics for Figure 1'

    table_file = os.path.join(paper_dir, 'Replication_Table_Figure1.tex')

    #=============================================================================
    # Make the data frame
    #-----------------------------------------------------------------------------
    table_dict, col_list = create_dict(measure_dict_dict)
    table_df = make_table_df(table_dict, col_list)
    write_latex_table(table_df, table_file, caption)