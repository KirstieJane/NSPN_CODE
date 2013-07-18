
# FROM: www.daniweb.com/software-development/python/threads/113058/using-wget-in-python
    import urllib, os
    # if you comment out this line, it will download to the directory from which you run the script.
    os.chdir('/directory/to/save/the/file/to')
    url = 'http://www.mydomain.com/myfile.txt'
    urllib.urlretrieve(url)

import itertools as it
import os
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator, AutoLocator

# Define the data directory

data_dir = '/work/imagingD/NSPN/workspaces/kw401/BOOTSTRAPPING_ANALYSES'

# Define some variables
# 
# *  subs: subject ids
# *  locs: scanner locations
# *  scans: which of the DTI scans you're using (This only applies to the UCL data where there are two. At all other sites there is only one.)
# *  incl_excl: either the original data or data excluding volume 31 which is not the same at CBSU as it is at the other sites
# *  n_b0s: number of b0s used to calculate tensor
# *  ec_b0: the b0 that was used as the eddy_correct target volume
# *  b0_order: the order that the b0s were in the 4D file
# *  sep_av: whether the b0s were averaged before the tensor was fit or kept separate
# *  transform: the registration path from MNI to DTI space
# *  roi_name: the name of the roi

subs = [ '06', '07', '08' ]
locs = [ 'WBIC', 'CBSU', 'UCL' ]
scans = [ 'DTI_2A', 'DTI_2B' ]
incl_excl = [ 'EXCL_VOL31']
n_b0s = [ '6' ]
ec_b0 = [ '00' ]
b0_order = [ '00_14_27_41_54_68' ]
sep_av = [ 'SEPARATE' ] 
transform = [ 'MNI_FNIRT_MPRAGE_BBR_B0' ]
roi_name = [ 'lcing', 'rcing', 'wholebrain', 'bodycc' ]

for incl_excl, n_b0s, ec_b0, b0_order, sep_av, transform, roi_name in it.product(incl_excl, n_b0s, ec_b0, b0_order, sep_av, transform, roi_name):
    
    results_file, results_dir = wrangle_text_files(incl_excl, n_b0s, ec_b0, b0_order, sep_av, transform, roi_name)
    
    data = read_in_data(results_file)
    
    colors, shapes = set_colors_shapes()
    
    plot_data(results_file, results_dir)

