import sys
import os
import random
userHome = os.path.expanduser('~')
utilsDir = os.path.join(userHome, 'software/galsim_analysis')
sys.path.append(utilsDir)

from galsimUtils import *

seed = 1352
rng = np.random.default_rng(seed)

inputFluxes = [5000]
inputShears = [(0, 0)]

imagesFile = '/sdf/home/a/adriansh/analysis/adc/galsim_images/star_images.pkl'
#imagesFile = '/home/r/rejnicho/analysis/adc/galsim_images/star_images.pkl'
#dataset_file = '/home/r/rejnicho/analysis/adc/datasets/dnl/13144_R22_S00_dnl_dataset.pkl'


write_star_images(inputFluxes, inputShears, imagesFile, n_ims=1)

# Need to make the following line work
# bins = get_model_adc(dataset_file)

# Sampling gaussian to make bin edges (Temporary)
bins = np.arange(2**18 + 1).astype(np.float64)
std = 0.15
vals = rng.standard_normal(len(bins[1:-1]))*std
bins[1:-1] += vals

with open(imagesFile, 'rb') as f:
    imDict = pkl.load(f)

plot_star_image(imDict, inputShears[0], inputFluxes[0], 0, bins)
