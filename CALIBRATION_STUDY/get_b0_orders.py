#!/usr/bin/env python

def get_b0_orders(n):
    """
    Created by Kirstie Whitaker
    Email: kw401@cam.ac.uk
    
    Inputs: number of b0s
    Returns:    list of orders of b0s
    """
    #==========================================================================
    import numpy as np
    from itertools import combinations
    #==========================================================================
    combo_list = []

    b0s = ['00', '14', '27', '41', '54', '68']

    for combination in combinations(b0s, n):
        for roll in np.arange(0, n, 1):
            combo_array = np.array(combination)
            combo = np.roll(combo_array, roll)

            combo_name = '_'
            for b0 in combo:
                combo_name+=b0 + '_'
            combo_name = combo_name[1:-1]
            combo_list.append(combo_name)
    
    return combo_list