#!/usr/bin/env python

def set_colors_shapes():
    """
    Created by Kirstie Whitaker
    Email: kw401@cam.ac.uk
    
    This is the code to change if you want to change the color scheme
    for the plots
    Inputs: none
    Returns:    colors: 3 x 5 array of colors
                shapes: 1 x 3 array of shapes
    """
    # Define your colors and shapes
    # 
    # Colors:
    # 
    # * Sub 6: pink
    # * Sub 7: green
    # * Sub 8: yellow

    colors = np.array([['#810051','#138900','#a6a300'],
                        ['#95256c','#389e28','#bfbc30'],
                        ['#c7007d', '#1dd300', '#fffa00'],
                        ['#e366b5','#7ae969','#fffc73']])

    # Shapes:
    # 
    # * Loc 1 (WBIC): circle
    # * Loc 2 (CBSU): diamond
    # * Loc 3 (UCL): triangle

    shapes = np.array(['o','D','^'])

    return colors, shapes