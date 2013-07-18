#!/usr/bin/env python

import os
import sys
import dipy

from itertools import product
import numpy as np
import matplotlib.pylab as plt

import dipy.reconst.dti as dti

import nibabel as nib
from dipy.data import get_data

from nibabel import trackvis as tv
from dipy.tracking import metrics as tm
from dipy.segment.quickbundles import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.viz import fvtk

import vtk

def KW_record(ren=None, cam_pos=None, cam_focal=None, cam_view=None,
           out_path=None, path_numbering=False, n_frames=10, az_ang=10,
           magnification=1, size=(300, 300), bgr_color=(0, 0, 0),
           verbose=False):

    if ren == None:
        ren = vtk.vtkRenderer()
    ren.SetBackground(bgr_color)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(size[0], size[1])
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # ren.GetActiveCamera().Azimuth(180)

    ren.ResetCamera()

    renderLarge = vtk.vtkRenderLargeImage()
    renderLarge.SetInput(ren)
    renderLarge.SetMagnification(magnification)
    renderLarge.Update()

    writer = vtk.vtkPNGWriter()
    ang = 0

    if cam_pos != None:
        cx, cy, cz = cam_pos
        ren.GetActiveCamera().SetPosition(cx, cy, cz)
    if cam_focal != None:
        fx, fy, fz = cam_focal
        ren.GetActiveCamera().SetFocalPoint(fx, fy, fz)
    if cam_view != None:
        ux, uy, uz = cam_view
        ren.GetActiveCamera().SetViewUp(ux, uy, uz)

    cam = ren.GetActiveCamera()
    if verbose:
        print('Camera Position (%.2f,%.2f,%.2f)' % cam.GetPosition())
        print('Camera Focal Point (%.2f,%.2f,%.2f)' % cam.GetFocalPoint())
        print('Camera View Up (%.2f,%.2f,%.2f)' % cam.GetViewUp())
        
    for i in range(n_frames):
        ren.GetActiveCamera().Pitch(ang)
        renderLarge = vtk.vtkRenderLargeImage()
        renderLarge.SetInput(ren)
        renderLarge.SetMagnification(magnification)
        renderLarge.Update()
        writer.SetInputConnection(renderLarge.GetOutputPort())
        # filename='/tmp/'+str(3000000+i)+'.png'
        if path_numbering:
            if out_path == None:
                filename = str(1000000 + i) + '.png'
            else:
                filename = out_path + str(1000000 + i) + '.png'
        else:
            filename = out_path
        writer.SetFileName(filename)
        writer.Write()

        ang = +az_ang



# EDIT THIS FOR YOUR OWN IMAGE
data_dir = '/work/imagingA/mrimpact/workspaces/kw401/BBC/DIPY_ANALYSES/'

fimg = data_dir + 'data.nii.gz'
fbval = data_dir + 'bvals'
fbvec = data_dir + 'bvecs'
fmask = data_dir + 'data_brain_mask_ero.nii.gz'

print 'Load data'
import nibabel as nib
img = nib.load(fimg)

print 'Get data'
from dipy.data import get_data
data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

print 'Get bvals, bvecs'
from dipy.io import read_bvals_bvecs
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

print 'Create gradient table'
from dipy.core.gradients import gradient_table
gtab = gradient_table(bvals, bvecs)

if os.path.isfile(fmask):
    print 'Load mask'
    img_mask = nib.load(fmask)
    mask = img_mask.get_data()
else:
    print 'Create basic mask'
    mask = data[..., 0] > 50
    
print 'Fit tensor'
from dipy.reconst.dti import TensorModel
ten = TensorModel(gtab)
tenfit = ten.fit(data, mask)

print 'Calculate FA'
from dipy.reconst.dti import fractional_anisotropy
fa = fractional_anisotropy(tenfit.evals)

# Set the background values to 0
fa[np.isnan(fa)] = 0

print 'Calculate color FA'
from dipy.reconst.dti import color_fa
cfa = color_fa(fa, tenfit.evecs)

evecs = tenfit.evecs

#plt.imshow(cfa[:,:,35])
#plt.show()

# Save the FA and eigenvectors as nifti files
print 'Saving FA image'
fa_img = nib.Nifti1Image(fa, img.get_affine())
nib.save(fa_img, data_dir + 'tensor_fa.nii.gz')

print 'Saving eigenvectors file'
evecs_img = nib.Nifti1Image(tenfit.evecs, img.get_affine())
nib.save(evecs_img, data_dir + 'tensor_evecs.nii.gz')
'''
# Now, I suspect these are totally unnecessary here
# We already have them...
fa_img = nib.load(data_dir + 'tensor_fa.nii.gz')
fa = fa_img.get_data()
evecs_img = nib.load(data_dir + 'tensor_evecs.nii.gz')
evecs = evecs_img.get_data()

# TESTING
print fa.shape
print evecs.shape

print 'Get sphere'
from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

print 'Quantize evecs'
from dipy.reconst.dti import quantize_evecs
peak_indices = quantize_evecs(evecs, sphere.vertices)

# Run tractography
print 'Running tractography'
from dipy.tracking.eudx import EuDX
eu = EuDX(fa, peak_indices, odf_vertices = sphere.vertices, a_low=0.2)
tensor_streamlines = [ streamline for streamline in eu ]

print 'Save streamlines in trackvis format'
hdr = nib.trackvis.empty_header()
hdr['voxel_size'] = fa_img.get_header().get_zooms()[:3]
hdr['voxel_order'] = 'LAS'
hdr['dim'] = fa.shape

tensor_streamlines_trk = ((sl, None, None) for sl in tensor_streamlines)
ten_sl_fname = data_dir + 'tensor_streamlines.trk'
nib.trackvis.write(ten_sl_fname, tensor_streamlines_trk, hdr, points_space='voxel')

# Limit the length of the tensor_streamlines
long_streamlines = [ streamline for streamline in tensor_streamlines if dipy.tracking.metrics.length(streamline) > 10 ]

for j, streamlines in enumerate([ long_streamlines ]):
    print 'Number of streamlines {}'.format(len(streamlines))

    print 'Visualizing streamlines'
    try:
        from dipy.viz import fvtk
    except ImportError:
        raise ImportError('Python vtk module is not installed')
        sys.exit()

    r=fvtk.ren()
    from dipy.viz.colormap import line_colors

    fvtk.add(r, fvtk.line(streamlines, line_colors(streamlines)))

    fvtk.show(r)
    print('Saving illustration as tensor_tracks.png')
    #fvtk.record(r, n_frames=1, out_path=data_dir + 'tensor_tracking_{}.png'.format(j), size=(600, 600))

    print 'Performing quickbundles'
    qb = QuickBundles(streamlines, dist_thr=20., pts=5)

    print 'Show inital streamlines'
    r = fvtk.ren()
    fvtk.add(r, fvtk.line(streamlines, fvtk.white, opacity=1, linewidth=3))
    #fvtk.show(r)
    #fvtk.record(r, n_frames=1, out_path = data_dir + 'initial_tracks_{}.png'.format(j), size=(600, 600))

    print 'Show centroids'
    centroids = qb.centroids
    colormap = np.ones((len(centroids), 3))

    fvtk.clear(r)

    for i, centroid in enumerate(centroids):
        colormap[i] = np.random.rand(3)
        fvtk.add(r, fvtk.line(centroids, colormap, opacity=1., linewidth=5))

    #fvtk.show(r)
    #fvtk.record(r, n_frames=1, out_path = data_dir + 'centroids_{}.png'.format(j), size=(600, 600))

    print 'Show clusters'
    colormap_full = np.ones((len(streamlines), 3))
    for i, centroid in enumerate(centroids):
        inds = qb.label2tracksids(i)
        colormap_full[inds] = colormap[i]

    fvtk.clear(r)
    fvtk.add(r, fvtk.line(streamlines, colormap_full, opacity=0.5, linewidth=2))
    fvtk.show(r)
    #fvtk.record(r, n_frames=1, out_path = data_dir + 'clusters_{}.png'.format(j), size=(600, 600))
    
    '''
    fvtk.record(r, n_frames=36, path_numbering=True, out_path = data_dir + 'clusters_',
                cam_pos= None,
                cam_focal=None,
                cam_view=(1,0,0),
                verbose=True,
                size=(600, 600))
    '''