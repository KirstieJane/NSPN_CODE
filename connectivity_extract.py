import os
import numpy as np
import scipy.io as sio
from glob import glob

data_dir = glob(os.path.join('/work/imaging*/NSPN/workspaces/kw401/UCHANGE/INTERIM_ANALYSIS'))[0]
mat_files = glob(os.path.join(data_dir, 'SUB_DATA', '*', 'DTI','MRI0','CONNECTIVITY','*mat'))


for mat_file in mat_files:
    meas = sio.loadmat(mat_file)
    subid = [ x.strip('Meas_') for x in meas.keys() if 'Meas' in x ][0]
    meas['Meas_'+subid]['cost'][0][0][0][0]
    
    print "===== {} =====".format(subid)
    for m in [ 'cost', 'k', 'a', 'arand', 'M', 'Mrand', 'C', 'Crand', 'L', 'Lrand', 'Sigma', 'E', 'Erand', 'CE', 'CErand' ]:
        print "{}: {}".format(m, meas['Meas_'+subid][m][0][0][0][0])
        
        
mat = loadmat('Meas_network_measures.mat') # Load in the .mat file
mdata = mat['Meas_network_measures']  # variable of interest in mat file
mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
ndata = {n: mdata[n][0, 0] for n in mdtype.names} # get all the data from mdata
columns = [n for n, v in ndata.iteritems() if v.size > 1 ]
df = pd.DataFrame(np.concatenate([ndata[c] for c in columns], axis=1), columns=columns)
df.describe()

'''
 'CE',
 'CErand',
 'Crand',
 'E',
 'Erand',
 'L',
 'Lrand',
 'M',
 'Mrand',
 'Sigma',
 'a',
 'arand',
 'cost',
 'k'
'''