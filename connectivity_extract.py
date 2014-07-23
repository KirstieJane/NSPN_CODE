#!/usr/bin/env python

import os
import numpy as np
import scipy.io as sio
from glob import glob
import pandas as pd

data_dir = glob(os.path.join('/work/imaging*/NSPN/workspaces/kw401/UCHANGE/INTERIM_ANALYSIS'))[0]

'''
mat_files = glob(os.path.join(data_dir, 'SUB_DATA', '*', 'DTI','MRI0','CONNECTIVITY','*mat'))


for mat_file in mat_files:
    meas = sio.loadmat(mat_file)
    subid = [ x.strip('Meas_') for x in meas.keys() if 'Meas' in x ][0]
    meas['Meas_'+subid]['cost'][0][0][0][0]
    
    print "===== {} =====".format(subid)
    for m in [ 'cost', 'k', 'a', 'arand', 'M', 'Mrand', 'C', 'Crand', 'L', 'Lrand', 'Sigma', 'E', 'Erand', 'CE', 'CErand' ]:
        print "{}: {}".format(m, meas['Meas_'+subid][m][0][0][0][0])
        
'''

fname = 'Meas_Co_wavelet3_thrNkeep02137'

mat = sio.loadmat(os.path.join(data_dir, 'SUB_DATA', '{}.mat'.format(fname))) # Load in the .mat file
mdata = mat[fname]  # variable of interest in mat file
mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
ndata = {n: mdata[n][0, 0] for n in mdtype.names} # get all the data from mdata
columns = [n for n, v in ndata.iteritems() if v.size == 39 ]
columns = [ 'E', 'M', 'C', 'nspn_id' ]
df1 = pd.DataFrame(np.concatenate([ndata[c].astype(np.float) for c in columns], axis=1), columns=columns)
print df1.describe()


df2 = pd.DataFrame(ndata['k'][:,0,:] )
df2['nspn_id'] = ndata['nspn_id']
df2.describe()

young_subs_file = os.path.join(data_dir, 'SUB_DATA', '14to15', 'subs')
young_subs_list = [ x.strip() for x in open(young_subs_file) ]
old_subs_file = os.path.join(data_dir, 'SUB_DATA', '22to25', 'subs')
old_subs_list = [ x.strip() for x in open(old_subs_file) ]

df1['young'] = 0
df1['old'] = 0
df2['young'] = 0
df2['old'] = 0

for sub in young_subs_list:
    df1['young'][df1['nspn_id']==np.float(sub)] = 1
    df2['young'][df2['nspn_id']==np.float(sub)] = 1

for sub in old_subs_list:
    df1['old'][df1['nspn_id']==np.float(sub)] = 1
    df2['old'][df2['nspn_id']==np.float(sub)] = 1

df1.to_csv(os.path.join(data_dir, 'SUB_DATA', '{}.csv'.format(fname))) # Save as CSV file

df2.to_csv(os.path.join(data_dir, 'SUB_DATA', '{}_degrees.csv'.format(fname))) # Save as CSV file
    
for measure_name, measure in zip(['Modularity', 'Clustering', 'Efficiency'],[ 'M' , 'C', 'E' ]):
    fig, ax = plt.subplots(figsize=(4,4))
    bp = ax.boxplot([df1[measure][df1.young==1], df1[measure][df1.old==1]], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['red', 'blue']):
        patch.set_facecolor(color)
    ax.set_ylabel(measure_name)
    labels = ['14to15', '22to25']
    ax.set_xticklabels(labels)
    plt.tight_layout()
    output_name = os.path.join(data_dir, 'SUB_DATA', 'FIGURES', 'FMRI_{}_box.png'.format(measure_name))
    fig.savefig(output_name, bbox_inches=0, dpi=600)

df1['old_excloutlier'] = df1['old']
df1['old_excloutlier'][df1['E'] > 0.4] = 0
measure = 'E'
measure_name = 'Efficiency_excloutlier'
fig, ax = plt.subplots(figsize=(4,4))
bp = ax.boxplot([df1[measure][df1.young==1], df1[measure][df1.old_excloutlier==1]], patch_artist=True)
for patch, color in zip(bp['boxes'], ['red', 'blue']):
    patch.set_facecolor(color)
ax.set_ylabel(measure_name)
labels = ['14to15', '22to25']
ax.set_xticklabels(labels)
plt.tight_layout()
output_name = os.path.join(data_dir, 'SUB_DATA', 'FIGURES', 'DTI_{}_box.png'.format(measure_name))
fig.savefig(output_name, bbox_inches=0, dpi=600)



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