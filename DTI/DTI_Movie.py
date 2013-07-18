#/usr/bin/env python
"""

"""

#==============================================================================
# IMPORTS
#==============================================================================
import matplotlib.animation as animation
import numpy as np
import matplotlib.pylab as plt
import os
import scipy

#==============================================================================
# FUNCTIONS
#==============================================================================
def ani_frame(data, name):
    """
    This one is the important one but it really needs to be
    re-written to be more flexible in other instances.
    Love it though!
    Stole it from:
        http://stackoverflow.com/a/13983801/2316203
    
    Kirstie Whitaker
    May 3rd 2013
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    im = ax.imshow(data[:,:,0], interpolation=None)
    #im.set_clim([0,1])
    
    fig.set_size_inches([5,5])

    plt.tight_layout()

    def update_img(n):
        if n%3 == 0:
            if n/3 < data.shape[2]:
                next = data[:,:,n/3]
            else:
                next = data[:,:,data.shape[2] - (n/3)]
            im.set_data(next)
            return im

    ani = animation.FuncAnimation(fig,update_img,data.shape[2]*3*2,interval=30)
    writer = animation.writers['ffmpeg'](fps=30)
    ani.save(name, writer=writer, dpi=dpi)

    return ani

#==============================================================================
# HERE'S THE REAL STUFF:
#==============================================================================
# Set your dpi
dpi = 200

cfa = np.load('colfa.npy')
cfa_rot = np.rot90(cfa, 3)
cfa_swap_1 = np.flipud(np.swapaxes(cfa,0,2))
cfa_swap_2 = np.flipud(np.swapaxes(cfa_rot,0,2))

# Make sure you're in the right place
# (if anyone is reading this - don't worry who steve was
# I bought my computer second hand and I'm assuming that the dude
# I bought it from was called steve. I don't remember....
# Super bugs me that I can't change it though.)
#os.chdir('C://Users//steve//Drivers//FFmpeg')

# And GO!
ani = ani_frame(cfa_rot, 'ColFA_Ax.avi')
ani = ani_frame(cfa_swap_1, 'ColFA_Sag.avi')
ani = ani_frame(cfa_swap_2, 'ColFA_Cor.avi')
