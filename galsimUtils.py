import galsim
import os
import numpy as np
import pickle as pkl

class galsim_dnl_dataset():
    def __init__(self):#, ds_dict, n_shears):
        self.sigmas = defaultdict(dict) 
        self.fluxes = defaultdict(dict) 
        self.g1s = defaultdict(dict) 
        self.g2s = defaultdict(dict) 
        self.centroid_xs = defaultdict(dict) 
        self.centroid_ys = defaultdict(dict) 

def make_blank_image(flux, g1, g2, seed=None):
    # random seed
    if not seed:
        rng = galsim.BaseDeviate(seed)
    
    # model atmospheric turbulence as a VonKarman
    gprof = galsim.VonKarman(lam=700., r0=0.2, L0=10.0)
    
    # add 4.5 microns sigma of Gaussian to model diffusion
    # convert 4.5 microns to arcsec with factor 0.2"/10micron 
    pixscale = 0.2/10.e-6
    dprof = galsim.Gaussian(flux=flux, sigma=4.5e-6*pixscale).shear(g1=g1, g2=g2)
    
    # add optical term, making up reasonable Zernike terms
    oprof = galsim.OpticalPSF(lam=700.0, diam=8.4,
                              defocus=0.1, coma1=-0.1, coma2=0.005, astig1=0.1, astig2=0.005,
                              obscuration=0.4)
    
    # convolve these terms
    prof = galsim.Convolve([gprof, dprof, oprof])
    
    # draw image
    blank_image = galsim.Image(32, 32, scale=0.2, xmin=0, ymin=0, dtype=np.float64)  
    return blank_image, prof

def plot_star_image(imDict, inshear, influx, im_num):
    im_arr = imDict[inshear][influx][im_num]

    # DNL bins
    dnl_bins = make_dnl_bins()
    pedestal = 2**8

    star_image = galsim.Image(np.floor(im_arr), dtype=int, xmin=0, ymin=0)
    adc_image = make_adc_image(im_arr, pedestal, dnl_bins)
    f, axs = plt.subplots(1, 3, figsize=(33, 11))
    star_moments = star_image.FindAdaptiveMom(weight=None, strict=False)
    adc_moments = adc_image.FindAdaptiveMom(weight=None, strict=False)

    diff = star_image.array - adc_image.array
 
    im0 = axs[0].imshow(star_image.array, origin='lower', interpolation='None')
    im1 = axs[1].imshow(adc_image.array, origin='lower', interpolation='None')
    im2 = axs[2].imshow(diff, origin='lower', interpolation='None')
    axs[0].set_title('Ideal', fontsize=18)
    axs[1].set_title('DNL', fontsize=18)
    axs[2].set_title('Difference', fontsize=18)
    f.colorbar(im0, ax=axs[0], pad=0.2, orientation='horizontal')
    f.colorbar(im1, ax=axs[1], pad=0.2, orientation='horizontal')
    f.colorbar(im2, ax=axs[2], pad=0.2, orientation='horizontal')
    f.suptitle('Digitized Galaxy Images', fontsize=26)
    f.savefig('analysis/adc/plots/galsim/galaxies.png')

def write_star_images(input_fluxes, input_shears, n_ims=500):
    imDict = defaultdict(dict)
    write_dir = 'analysis/adc/datasets/galsim/star_images'
    fname = 'star_images.pkl'
    filename = os.path.join(write_dir, fname)

    for inshear in input_shears:
        for influx in input_fluxes:
            imDict[inshear][influx] = np.zeros((n_ims, 32, 32))
    for (g1, g2), fdict in imDict.items():
        for iflux, im_arrs in fdict.items():
            print(g1, g2, iflux)
            for n in range(n_ims):
                star_image = make_star_image(iflux, g1, g2)
                im_arrs[n] = star_image.array

    with open(filename, 'wb') as f:
        pkl.dump(imDict, f)
    print(f'Wrote {filename}')

def make_star_image(flux, g1, g2, seed=None):
    if seed:
        rng = galsim.BaseDeviate(seed)
    else:
        rng = galsim.BaseDeviate() # no seed

    blank_image, prof = make_blank_image(flux, g1, g2)
    star_image = prof.drawImage(image=blank_image, method='auto')
        
    # generate noise and add to image
    noise = galsim.CCDNoise(rng, gain=1.0, read_noise=5.0, sky_level=300.0)
    star_image.addNoise(noise)

    return star_image

def make_adc_image(im_array, pedestal, adc_bins):
    im_arr = np.digitize(im_array+pedestal, dnl_bins) - 1

    # subtract pedestal
    adc_image = galsim.Image(im_arr-pedestal, dtype=int, xmin=0, ymin=0)

    return adc_image

def make_dataset(imDict):
    imDictKeys = list(imDict.keys())
    shearDict = imDict[imDictKeys[0]]
    shearDictKeys = list(shearDict.keys())
    nIter = len(shearDict[shearDictKeys[0]]) 

    # DNL bins
    dnl_bins = make_dnl_bins()
    pedestal = 2**8

    # Make a dataset
    dataset = galsim_dnl_dataset()
    ds_dict = dict(Ideal = np.zeros(n_iter),
                   ADC = np.zeros(n_iter))

    for ish in range(len(input_shears)):
        for iflux in input_fluxes:
            dataset.sigmas[ish][iflux] = copy.deepcopy(ds_dict) 
            dataset.fluxes[ish][iflux] = copy.deepcopy(ds_dict) 
            dataset.g1s[ish][iflux] = copy.deepcopy(ds_dict) 
            dataset.g2s[ish][iflux] = copy.deepcopy(ds_dict) 
            dataset.centroid_xs[ish][iflux] = copy.deepcopy(ds_dict) 
            dataset.centroid_ys[ish][iflux] = copy.deepcopy(ds_dict) 

    for ishear, ((g1, g2), fdict) in enumerate(imDict.items()):
        print(g1, g2)
        for input_flux, im_arrs in fdict.items():
            sigmas = np.zeros((2, n_iter))
            fluxes = np.zeros((2, n_iter))
            g1s = np.zeros((2, n_iter))
            g2s = np.zeros((2, n_iter))
            cxs = np.zeros((2, n_iter))
            cys = np.zeros((2, n_iter))

            for n, arr in enumerate(im_arrs):

                # digitize and add pedestal
                int_arr = np.floor(arr)
                int_image = galsim.Image(int_arr, dtype=int, xmin=0, ymin=0)
                adc_image = make_adc_image(arr, pedestal, dnl_bins)
                images = [int_image, adc_image]

                for i_image, image in enumerate(images):
                    # calculate HSM moments (these are in pixel coordinates)
                    moments = image.FindAdaptiveMom(weight=None, strict=False)
                    sigmas[i_image, n] = moments.moments_sigma
                    fluxes[i_image, n] = moments.moments_amp
                    g1s[i_image, n] = moments.observed_shape.g1
                    g2s[i_image, n] = moments.observed_shape.g2
                    cxs[i_image, n] = moments.moments_centroid.x
                    cys[i_image, n] = moments.moments_centroid.y

            for ik, k in enumerate(ds_dict.keys()):
                dataset.sigmas[ishear][input_flux][k] = sigmas[ik] 
                dataset.fluxes[ishear][input_flux][k] = fluxes[ik] 
                dataset.g1s[ishear][input_flux][k] = g1s[ik] 
                dataset.g2s[ishear][input_flux][k] = g2s[ik] 
                dataset.centroid_xs[ishear][input_flux][k] = cxs[ik] 
                dataset.centroid_ys[ishear][input_flux][k] = cys[ik] 

    write_to = 'analysis/adc/datasets/galsim/'
    fname = 'galsim_dnl_dataset.pkl'
    filename = os.path.join(write_to, fname)
    save_dataset(dataset, filename)

# There are much better functions to make a plot like this
def plot_dataset():
    filename = 'analysis/adc/datasets/galsim/galsim_dnl_dataset.pkl'
    save_to = 'analysis/adc/plots/galsim/'
    #mean_bins = 50 #np.linspace(0.7, 0.8, 51)
    sigma_bins = 50 #np.linspace(1.5e-3, 4.5e-3, 51)
    flux_bins = 50 #np.linspace(20, 50, 51)
    shear_bins = 50 #np.linspace(-1.e-3, 1e-3, 51)
    centroid_bins = 50 #np.linspace(-1.5e-3, 1.5e-3, 51)

    with open(filename, 'rb') as f:
        dataset = pkl.load(f)
        ax_labels = ['Sigma', 'Flux', 'Shear', 'Centroid']

        hist_kwargs = [#dict(bins=mean_bins),
                       dict(bins=sigma_bins),
                       dict(bins=flux_bins),
                       dict(label=[f'g1', 
                                   f'g2'],
                            bins=shear_bins, fill=False, linewidth=3, 
                            histtype='step', stacked=False),
                       dict(label=[f'x',
                                   f'y'],  
                            bins=centroid_bins, fill=False, linewidth=3,
                            histtype='step', stacked=False)]

        fig, axs = plt.subplots(4, 4, figsize=(20, 20), sharex=False)

        for ishear, fdict in dataset.sigmas.items():
            input_shear = input_shears[ishear]

            for input_flux, sigmas in fdict.items():
                for ax in axs.ravel():
                    ax.cla()
                ideal_sigs = sigmas['Ideal']
                ideal_fluxes = dataset.fluxes[ishear][input_flux]['Ideal']
                ideal_g1s = dataset.g1s[ishear][input_flux]['Ideal']
                ideal_g2s = dataset.g2s[ishear][input_flux]['Ideal']
                ideal_cxs = dataset.centroid_xs[ishear][input_flux]['Ideal']
                ideal_cys = dataset.centroid_ys[ishear][input_flux]['Ideal']

                ideal_sig = np.mean(ideal_sigs)
                ideal_flux = np.mean(ideal_fluxes)
                ideal_g1 = ideal_g1s.mean()
                ideal_g2 = ideal_g2s.mean()
                ideal_cx = ideal_cxs.mean()
                ideal_cy = ideal_cys.mean() 

                hist_titles = [r'$\sigma_i$' + f' = {ideal_sig:.3E}', 
                               r'Flux$_i$' + f' = {ideal_flux:.3E}', 
                               r'$\gamma_{1,i}$' + f' = {ideal_g1:.3E}, ' + r'$\gamma_{2,i}$' + f' = {ideal_g2:.3E}, ',
                               r'$cx_i$' + f' = {ideal_cx:.3E}, ' + r'$cy_i$' + f' = {ideal_cy:.3E}, ']

                for key in ['ADC']:#, 'Corrected']:
                    data = [dataset.sigmas[ishear][input_flux][key] - ideal_sigs,
                            dataset.fluxes[ishear][input_flux][key] - ideal_fluxes, 
                            [dataset.g1s[ishear][input_flux][key] - ideal_g1s, 
                             dataset.g2s[ishear][input_flux][key] - ideal_g2s],
                            [dataset.centroid_xs[ishear][input_flux][key] - ideal_cxs, 
                             dataset.centroid_ys[ishear][input_flux][key] - ideal_cys]]


                for i in range(4):
                    axs[i, 0].set_ylabel(ax_labels[i], ha='right', fontsize=28)
                    axs[-1, i].set_xlabel(ax_labels[i], fontsize=28)

                    axs[i, i].hist(data[i], **hist_kwargs[i])
                    axs[i, i].set_title(hist_titles[i], fontsize=28)
                    axs[i, i].legend(fontsize=22)
                    plt.setp(axs[-1, i].get_xticklabels(), rotation=30, ha='right')

                    if i < 3:
                        axs[i, i].set_xticklabels([])

                    for j in range(0, i):
                        if i < 3:
                            axs[i, j].set_xticklabels([])
                        
                        if j > 0:
                            axs[i, j].set_yticklabels([])

                        #if j == 1:
                        #    plt.setp(axs[i, j].get_xticklabels(), rotation=30, ha='right')

                        if i == 2:
                            axs[i, j].scatter(data[j], data[i][0], label=f'g1', color='tab:blue', s=10)
                            axs[i, j].scatter(data[j], data[i][1], label=f'g2', color='tab:orange', s=10)
                        elif i == 3 and j < 2:
                            axs[i, j].scatter(data[j], data[i][0], label=f'cx', color='tab:blue', s=10)
                            axs[i, j].scatter(data[j], data[i][1], label=f'cy', color='tab:orange', s=10)
                        elif i == 3 and j == 2:
                            axs[i, j].scatter(data[j][0], data[i][0], label=f'g1, cx', color='tab:blue', s=10)
                            axs[i, j].scatter(data[j][0], data[i][1], label=f'g1, cy', color='tab:green', s=10)
                            axs[i, j].scatter(data[j][1], data[i][0], label=f'g2, cx', color='tab:orange', s=10)
                            axs[i, j].scatter(data[j][1], data[i][1], label=f'g2, cy', color='tab:purple', s=10)
                            axs[i, j].legend(fontsize=22)
                            #plt.setp(axs[i, j].get_xticklabels(), rotation=30, ha='right')
                        else:
                            axs[i, j].scatter(data[j], data[i], s=10)
                        axs[i, j].grid(visible=True)

                    # hide upper off-diagonals
                    for k in range(i+1, 4):
                        axs[i, k].set_frame_on(False)
                        axs[i, k].set_xticks([])
                        axs[i, k].set_yticks([])

                for ax in axs.ravel():
                    ax.tick_params(labelsize=24)

                fig.suptitle(f'{key} - Ideal for Input Shear = {input_shear}, Input Flux = {input_flux}', fontsize=36)
                gam1 = input_shear[0]
                gam2 = input_shear[1]
                fig.savefig(os.path.join(save_to, f'{key}_ideal_diff_{gam1}_{gam2}_{input_flux}.png'))
                print(f'Wrote {os.path.join(save_to, f"{key}_ideal_diff_{gam1}_{gam2}_{input_flux}.png")}') 


