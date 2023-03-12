import sys
import os
userHome = os.path.expanduser('~')
utilsDir = os.path.join(userHome, 'software/galsim_analysis')
sys.path.append(utilsDir)

from galsimUtils import *

input_fluxes = [5000]
input_shears = [(0, 0)]
write_star_images(input_fluxes, input_shears, n_ims=1)
fName = 'analysis/adc/datasets/galsim/star_images/star_images.pkl'

with open(fName, 'rb') as f:
    im_dict = pkl.load(f)

plot_star_image(im_dict, input_shears[0], input_fluxes[0], 0)
