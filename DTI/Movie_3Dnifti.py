#/usr/bin/env python
"""
This script takes as an input a 3D nifti file
and creates three movies one for each orthogonal direction
(coronal, saggital, axial) that show all the slices
in that direction going from one side to another and
back again
"""

#==============================================================================
# IMPORTS
#==============================================================================
import matplotlib.animation as animation
import numpy as np
import matplotlib.pylab as plt
import os
import scipy
import nibabel as nib
#from dipy.data import get_data


#==============================================================================
# FUNCTIONS
#==============================================================================
def ani_frame(data, output_name):
    """
    This function is the important one that creates the animation
    from your data array.
    
    It will always cycle through the 3rd axis of the data array.

    It is heavily borrowed from:
        http://stackoverflow.com/a/13983801/2316203
    Thank you!
    
    Kirstie Whitaker
    May 3rd 2013
    """
 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    im = ax.imshow(data[:,:,0], interpolation=None, cmap=cmap)
    
    default_size = fig.get_size_inches()
    
    fig_size = default_size * scale
    
    fig.set_size_inches(fig_size)

    plt.tight_layout()

    def update_img(n):
        """
        This function moves the slice along and updates
        the image to that slice (while keeping all the other
        settings the same)
        
        Parameters:
            n - this is the frame of the movie
            timing_gap - this tells you the speed that you
            want your movie to update at. It's set as a default
            as 3, so you'll have a new slice every 3 frames of
            the movie
        """
        

        if n % timing_gap == 0:
            slice = n / timing_gap
            
            if slice < data.shape[2]:
                # If you're on the way "out" then increase the
                # slice by 1
                next = data[:,:, slice ]
            else:
                # If you're coming back through the slices
                # then start counting through negative
                # slice locations in order to reverse
                # the order
                next = data[:,:,data.shape[2] - slice ]
            
            # Update the image
            im.set_data(next)
            return im

    ani = animation.FuncAnimation(fig, update_img, data.shape[2] * timing_gap * 2, interval=30)
    writer = animation.writers['ffmpeg'](fps=30)
    ani.save(name, writer=writer, dpi=dpi)

    return ani

#==============================================================================
# READ IN YOUR ARGUMENTS:
#==============================================================================
nifti_file = sys.argv[1]
output_root = sys.argv[2]

#==============================================================================
# ADJUST SOME DEFAULTS:
#==============================================================================
# Set your dpi
dpi = 200

# Scale size of the figure
scale = 2

# Set the number of frames for each slice
timing_gap = 3

# Set the colormap you want to use
cmap = 'gray'

#==============================================================================
# HERE'S THE REAL STUFF:
#==============================================================================

# Load the data and write it to a numpy array
img = nib.load(nifti_file)
data = img.get_data()

axial = np.rot90(data, 3)
sagittal = np.flipud(np.swapaxes(data,0,2))
coronal = np.flipud(np.swapaxes(axial,0,2))

# Make sure you're in the right place
# (if anyone is reading this - don't worry who steve was
# I bought my computer second hand and I'm assuming that the dude
# I bought it from was called steve. I don't remember....
# Super bugs me that I can't change it though.)
#os.chdir('C://Users//steve//Drivers//FFmpeg')

# And GO!
ani = ani_frame(axial, output_root + '_axial.avi')
ani = ani_frame(sagittal, output_root + '_sagittal.avi')
ani = ani_frame(coronal, output_root + '_coronal.avi')
