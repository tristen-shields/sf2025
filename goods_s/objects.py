from .functions import *
from astropy.nddata import Cutout2D
from regions import CirclePixelRegion, PixCoord 
import pyimfit
from pysersic.multiband import FitMultiBandBSpline
from pysersic.priors import autoprior, SourceProperties, PySersicSourcePrior
from pysersic import FitSingle
from pysersic.loss import student_t_loss 
from jax.random import PRNGKey 
import bagpipes as bp
from astropy.visualization import simple_norm, ImageNormalize, ZScaleInterval, LinearStretch
from os import remove
from pysersic.results import plot_residual
from photutils.aperture import EllipticalAperture


'''
A class to store all of the information and functions needed for a given single object ID from PHOT_CATALOG (see loading.py).
'''


class obj(object):
    def __init__(self, objid):
        '''
        objid (int): ID of a single object from PHOT_CATALOG. 
        '''
        self.objid = objid 
        self.index = np.where(PHOT_CATALOG['OBJIDS'][()] == objid)[0][0] # Index of the given object ID in PHOT_CATALOG

        # Position
        self.ra = PHOT_CATALOG['RA'][self.index] # Object right ascension
        self.dec = PHOT_CATALOG['DEC'][self.index] # Object declination
        self.x = PHOT_CATALOG['X'][self.index] # Object x coordinate (pixels)
        self.y = PHOT_CATALOG['Y'][self.index] # Object y coordinate (pixels)
        self.pa = PHOT_CATALOG['SIZE/PA [deg]'][self.index] # Position angle of the object, in degrees
        
        # Size
        self.a = PHOT_CATALOG['SIZE/A [pix]'][self.index] # Object semi-major axis (pixels)
        self.b = PHOT_CATALOG['SIZE/B [pix]'][self.index] # Object semi-minor axis (pixels)
        self.Npix = PHOT_CATALOG['SIZE/N_PIX'][self.index] # Number of pixels belonging to the object in the JADES segmentation map
        
        # Redshift 
        if self.objid in Z_CATALOG_IDS:
            self.z_ind = np.where(Z_CATALOG_IDS == self.objid) # Index of the object's ID in the REDSHIFT catalog (Z_CATALOG)
            self.z_a = Z_A[self.z_ind] # Definition of photometric redshift used in the LSB paper 
        else:
            self.z_a = None # If object is in PHOT_CATALOG but not Z_CATALOG


    def Fnu(self, aperture, filt):
        '''
        Get a single flux density/flux density error reading for the object from PHOT_CATALOG, given an aperture and a filter.
            aperture (str): 'KRON', 'KRON_S', 'CIRC{i}' (see loading.py for key descriptions)
            filt (str): Any item from FILTER_NAMES (see filters.py)

            returns: 
                Fnu (float): Flux density in nJy
                Fnu_err (float): Error in measuring Fnu, in nJy
        '''
        Fnu = PHOT_CATALOG[f'{aperture}/{filt}/Fnu [nJy]'][self.index]
        Fnu_err = PHOT_CATALOG[f'{aperture}/{filt}/Fnu Error [nJy]'][self.index]

        return Fnu, Fnu_err
    

    def aperture_Fnu(self, aperture):
        '''
        Collect the object's flux densities + errors for all filters (in FILTER_NAMES) for a given aperture.
            aperture (str): 'KRON', 'KRON_S', 'CIRC{i}' (see loading.py for key descriptions)

            returns:
                Fnu (1darray): Flux density, in nJy
                Fnu_err (1darray): Error in measuring Fnu, in nJy
        '''
        Fnu = np.zeros(FILTER_NAMES.shape)
        Fnu_err = np.zeros(FILTER_NAMES.shape)
        for i, filt in enumerate(FILTER_NAMES):
            Fnu[i], Fnu_err[i] = self.Fnu(aperture, filt)
        
        return Fnu, Fnu_err
    

    def get_mosaic_cut(self, filt, n=100):
        '''
        Return a square cutout (data image and one standard deviation error image) of a given filter's mosaic centered around the object of length n pixels. 
        Also return the background of the mosaic from the header.
            filt (str): Filter name from FILTER_NAMES (filters.py)
            n (int): Length of the square cutout, in pixels

            returns:
                data (2darray): Data image of the cutout
                sigma (2darray): One standard deviation error image of the cutout
                bkg (float): Background of the chosen mosaic
                sr_pix (float): Steradians per pixel conversion for the mosaic
        '''

        if filt in MOSAIC_NAMES:
            mosaic_name = filt 
        else:
            print('Filter not in mosaic list')
            pass 

        i = np.where(MOSAIC_NAMES == mosaic_name)[0][0]
        mosaic = MOSAICS[i]
        hdul = fits.open(mosaic)

        data = Cutout2D(hdul[1].data, (self.x, self.y), n).data # Data image
        sigma = Cutout2D(hdul[2].data, (self.x, self.y), n).data # One standard deviation (sigma) error image
        try:
            bkg = hdul[0].header['BKGLEVEL'] # Background of the chosen mosaic
        except:
            print('No background level found for mosaic')
            bkg = 0
        sr_pix = hdul[0].header['PIXAR_SR'] # Pixels per steradian conversion for the mosaic

        return data, sigma, bkg, sr_pix
    

    def run_pyimfit(self, filt):
        '''
        Run a single-component Sersic surface brightness profile on this object in a given filter with the program pyimfit.
            filt (str): Filter name from FILTER_NAMES (see filters.py)
            
            returns:
                imfitter (obj): Fitting object returned from pyimfit (see pyimfit documentation)
                result (1darray): Data vector of parameters returned from the fit (x, y, PA, ellipticity, n, I_eff, R_eff)
                data (2darray): Original mosaic cutout of the object in the chosen filter
                sigma (2darray): One standard deviation error image associated with 'data'
                seg_mask (2darray): Mask applied to 'data' and 'sigma' that ignore any pixels that likely belong to other objects
        '''
        if np.isnan(self.a):
            print("the semi-major axis for this object doesn't exist in the catalog")
            return 0, 0, 0, 0, 0
        else:
            a = int(self.a) # pix

        data, sigma, bkg, __ = self.get_mosaic_cut(filt, n=12 * a)
        
        # Initial guesses for parameters
        ell0 = 1 - (self.b / self.a) # Initial guess for ellipticity
        x0 = int(np.round(6*a)) # Initial guess for position (both x and y)

        # Find a threshold for the segmentation map s.t. make_segmap() returns a "False" flag (see functions.py)
        thres = 1.5 # Initial threshold
        try:
            segmap0, cent0, flag = make_segmap(data, sigma, thres, x0, a)

            while flag == False:
                thres += 0.5 
                segmap0, cent0, flag = make_segmap(data, sigma, thres, x0, a)
                if cent0 == 0:
                    flag = True 
            if cent0 == 0:
                print(f'Segmap for {self.objid} is bad')
                return 0, 0, 0, 0, 0

            # Using the segmap from SEP, mask out any pixels belonging to objects other than the central one from the fitting process
            cent_mask = np.isin(segmap0, cent0) # Pixels belonging to central object
            mask0 = np.isin(segmap0, 0) # Pixels not belonging to any object
            seg_mask = ~(cent_mask + mask0) # Pixels that belong to an object, but not the center one
        except:
            print(f'Segmap for {self.objid} is bad')
            return 0, 0, 0, 0, 0 

        # Obtain a good initial guess for I_eff via a circular cutout of radius a 
        center = PixCoord(x0, x0)
        aperture = CirclePixelRegion(center, int(a))
        eff_mask = aperture.to_mask(mode='exact')
        eff_data = eff_mask.cutout(data)
        Ieff0 = np.average(eff_data) # Average brightness over the circular aperture of radius a

        model_desc = pyimfit.SimpleModelDescription() # Simple single-component Sersic model

        # Set initial positions on the image and limit them to plus or minus 5 pixels
        x1 = a * 6 # half of the 12*a length of square cutout [pix]
        model_desc.x0.setValue(x1, [x1 - 5, x1 + 5])
        model_desc.y0.setValue(x1, [x1 - 5, x1 + 5])

        # Create a Sersic model function and specify initial parameters plus bounds
        sersic = pyimfit.make_imfit_function('Sersic')
        sersic.PA.setValue(self.pa, [0, 180]) # Position angle [deg]
        sersic.ell.setValue(ell0, [0, 1]) # Ellipticity
        sersic.n.setValue(0.8, [0, 5]) # Sersic index n 
        sersic.I_e.setValue(Ieff0, [0, 10 * Ieff0]) # Effective brightness/intensity
        sersic.r_e.setValue(a, [0, 10 * a]) # Effective (half-light) radius
        model_desc.addFunction(sersic)

        # Create model fitter
        imfitter = pyimfit.Imfit(model_desc, psf=None)
        imfitter.loadData(data, error=sigma, mask=seg_mask, original_sky=bkg)


        # Run fit
        try:
            result = imfitter.doFit(ftol=1e-8) # Data vector: x, y, PA, ellipticity, n, I_eff, R_eff
        except:
            print(f"Imfitter couldn't fit OBJID {self.objid}")
            return 0, 0, 0, 0, 0

        return imfitter, result, data, sigma, seg_mask
    

    def run_pysersic_multi_band(self, bands):
        '''
        
        '''
        n = int(round(12 * self.a)) # side length of square cutout [pix]
        if n % 2 == 0: n += 1 # n needs to be odd for PSF convolution

        # Gather PSFs, image cutouts, and error cutouts for each band
        psf_dict = {}
        img_dict = {}
        err_dict = {}
        for i, band in enumerate(bands):
            psf_dict[band] = get_filter_PSF(band, n)
            img_dict[band], err_dict[band], bkg, __ = self.get_mosaic_cut(PSF_NAMES[np.isin(PSF_NAMES, bands)][i], n=n)
        
        # Gather masks through segmaps as in run_pyimfit(), but take the union of all band masks
        masks = []
        is_good = np.ones(len(bands), dtype=bool)
        for i, band in enumerate(bands):
            thres = 1.5
            segmap0, cent, flag = make_segmap(img_dict[band], err_dict[band], thres, int(n / 2), int(self.a))
            while flag == False:
                thres += 0.5
                segmap0, cent, flag = make_segmap(img_dict[band], err_dict[band], thres, int(n / 2), int(self.a))
                if cent == 0:
                    print(f'ID {self.objid} had a bad segmap in {band}')
                    is_good[i] = False 
                    masks.append(np.zeros(img_dict[band].shape))
                    flag = True 
            mask1 = np.isin(segmap0, cent) # Pixels belonging to central object
            mask2 = np.isin(segmap0, 0) # Pixels belonging to no object
            mask = ~(mask1 + mask2)
            masks.append(mask)
        masks = np.array(masks)
        seg_mask = np.sum(masks, axis=0, dtype=bool)

        # Gather auto-priors for each band
        prior_dict = {}
        for i, band in enumerate(bands[is_good]):
            # prior = autoprior(img_dict[band], 'sersic', seg_mask, sky_type='none')
            # prior.set_uniform_prior('r_eff', 0, 1.5 * self.a)
            # prior.set_uniform_prior('n', 0.5, 3.0)

            properties = SourceProperties(image=img_dict[band], mask=seg_mask)
            print(properties)
            prior_default = properties.generate_prior('sersic', sky_type='none')
            prior_default.set_uniform_prior('n', 0.5, 3.0)
            # prior = PySersicSourcePrior(profile_type='sersic', sky_type='none')

            print(prior_default)
            prior_dict[band] = prior_default
        
        # Gather a list of single-object pysersic fitter objects
        fitters = []
        for i, band in enumerate(bands[is_good]):
            fitter = FitSingle(img_dict[band], err_dict[band], psf_dict[band], prior_dict[band], seg_mask, student_t_loss)
            fitters.append(fitter)
        
        # Make the multi-band fitter object 
        band_pivots = FILTER_PIVOTS[np.isin(FILTER_NAMES, bands)]
        wavelength_arr = np.linspace(np.min(band_pivots), np.max(band_pivots), num=50)
        MultiFitter = FitMultiBandBSpline(fitter_list=fitters,
                                          wavelengths=band_pivots[is_good], 
                                          band_names=bands[is_good], 
                                          linked_params=['n', 'ellip', 'r_eff'], 
                                          const_params=['xc', 'yc', 'theta'], 
                                          wv_to_save=wavelength_arr, 
                                          linked_params_range={'r_eff':[0, 10*self.a]})
        
        # Run the fit!!
        rkey = PRNGKey(19)
        multires = MultiFitter.estimate_posterior(method='laplace', rkey=rkey)

        return multires, is_good
    

    def run_pysersic_single_band(self, filt, estimate_posterior=True, plot_map_residual=False):
        '''
        
        '''
        n = int(round(12 * self.a)) # side length of square cutout [pix]
        if n % 2 == 0: n += 1 # n needs to be odd for PSF convolution

        data, sigma, __, __ = self.get_mosaic_cut(filt, n=n)

        # Find a threshold for the segmentation map s.t. make_segmap() returns a "False" flag (see functions.py)
        thres = 1.5 # Initial threshold
        x0 = int(6 * self.a)
        a = self.a

        segmap0, cent0, flag = make_segmap(data, sigma, thres, x0, a)

        while flag == False:
            thres += 0.5 
            segmap0, cent0, flag = make_segmap(data, sigma, thres, x0, a)
            if cent0 == 0:
                flag = True 

        # Using the segmap from SEP, mask out any pixels belonging to objects other than the central one from the fitting process
        cent_mask = np.isin(segmap0, cent0) # Pixels belonging to central object
        mask0 = np.isin(segmap0, 0) # Pixels not belonging to any object
        seg_mask = ~(cent_mask + mask0) # Pixels that belong to an object, but not the center one

        psf = get_filter_PSF(filt, n)
        prior = autoprior(data, 'sersic', seg_mask, sky_type='none')
        prior.set_uniform_prior('r_eff', 0, 1.5 * self.a)
        prior.set_uniform_prior('n', 0.5, 3.0)
        prior.set_uniform_prior('xc', x0 - 5, x0 + 5)
        prior.set_uniform_prior('yc', x0 - 5, x0 + 5)
        print(prior)
        fitter = FitSingle(data=data,rms=sigma,mask=seg_mask,psf=psf, prior=prior,loss_func=student_t_loss)

        # Run the fit!!
        if plot_map_residual:
            map_params = fitter.find_MAP(rkey=PRNGKey(1000))
            print(map_params)
            plot_residual(data, map_params['model'], mask=seg_mask, vmin=-0.1, vmax=0.1)

        if estimate_posterior:
            rkey = PRNGKey(19)
            res = fitter.estimate_posterior(method='laplace', rkey=rkey)

            return res
        
        else:
            return fitter
    
    
    def run_bagpipes(self, fit_instructions='delayed_tau', aperture='KRON_S', mask_filters=False, z='phot', error_floor=0.05, 
                     sampler='nautilus'):
        '''
        
        '''
        filt_list = [] # List of filter .par paths to feed into BAGPIPES
        for filt in FILTER_NAMES:
            filt_list.append(FILTER_PAR_PATH + filt + '.par') 

        if aperture == 'KRON_S':
            # Scale small Kron by large/small Kron ratio in F200W to create fiducial photometry
            Fnu, Fnu_err = self.aperture_Fnu('KRON_S') # nJy
            correction = self.Fnu('KRON', 'F200W')[0] / self.Fnu('KRON_S', 'F200W')[0]
            Fnu = Fnu * correction * 1e-3 # mJy 
            Fnu_err = Fnu_err * correction * 1e-3 # mJy
        else:
            Fnu, Fnu_err = self.aperture_Fnu(aperture) # nJy
            Fnu *= 1e-3 # mJy
            Fnu_err *= 1e-3 # mJy

        if error_floor != False:
            snr = Fnu / Fnu_err 
            snr_ceiling = 1 / error_floor 
            Fnu_err[snr > snr_ceiling] = error_floor * Fnu[snr > snr_ceiling]

        # Mask out filters that have "bad" fluxes (i.e. where error is 0)
        if mask_filters:
            is_good = (Fnu_err > 0)
        else:
            is_good = np.full(len(filt_list), True, dtype='bool')      
        filt_list = np.array(filt_list)[is_good]

        def load_data(objid):
            return np.array([Fnu, Fnu_err]).T

        galaxy = bp.galaxy(self.objid, load_data, spectrum_exists=False, filt_list=filt_list, photometry_exists=True, phot_units='mujy', spec_units='mujy') # Initialize BP galaxy class for the object
        
        if z == 'phot':
            z = self.z_a[0]

        if fit_instructions == 'delayed_tau':
            fit_instructions = build_delayed_tau_fit_instructions(z)
        elif fit_instructions == 'dbl_power_law':
            fit_instructions = build_dbl_power_law_fit_instructions(z)

        try:
            remove(f'pipes/posterior/{self.objid}.h5')
        except:
            pass

        fit = bp.fit(galaxy, fit_instructions)

        if sampler == 'nautilus':
            n_eff, n_live = int(1e+3), int(1e+3)
            fit.fit(n_live=n_live, n_eff=n_eff, verbose=True, sampler=sampler)
        
        elif sampler == 'multinest':
            fit.fit(verbose=True, sampler=sampler)
            
        return fit, is_good
    

    def plot_RGB(self, n=100, r_filter='F356W', g_filter='F200W', b_filter='F115W', r_scale=0.80, g_scale=1.00, b_scale=1.00, min_scale=1000):
        '''
        Like get_cut(), but plot the cutout as an RGB from 3 different filters as an RGB image.
            n(=100): Side length of the square cutout [pix] (ODD int)
            r_filter(='F356W'): Filter keyword in MOSAIC_NAMES to serve as the "red" filter (str)
            g_filter(='F200W'): Filter keyword in MOSAIC_NAMES to serve as the "green" filter (str)
            b_filter(='F090W'): Filter keyword in MOSAIC_NAMES to serve as the "blue" filter (str)
            r_scale(=0.80), g_scale(=1.00), b_scale(1.00): Values to scale r, g, and b images by (float, between 0 and 1)
            min_scale(=3e+4): Number to scale the minimum clip of the image by (int/float)
        '''
        # Check if n is odd
        if n % 2 == 0:
            n += 1

        r, g, b = self.get_mosaic_cut(r_filter, n=n)[0], self.get_mosaic_cut(g_filter, n=n)[0], self.get_mosaic_cut(b_filter, n=n)[0]
        r_psf, g_psf, b_psf = get_filter_PSF(r_filter, n), get_filter_PSF(g_filter, n), get_filter_PSF(b_filter, n)

        rgb = np.zeros((n, n, 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        rgb_max = np.max(rgb)
        rgb_min = min_scale * np.min(rgb[rgb > 0])

        clipped_r = np.clip(r, rgb_min, rgb_max)
        clipped_g = np.clip(g, rgb_min, rgb_max)
        clipped_b = np.clip(b, rgb_min, rgb_max)

        rgb_range = rgb_max - rgb_min
        rgb[:, :, 0] = 0.80 * (clipped_r - rgb_min) / rgb_range
        rgb[:, :, 1] = 1.00 * (clipped_g - rgb_min) / rgb_range
        rgb[:, :, 2] = 1.00 * (clipped_b - rgb_min) / rgb_range
        rgb[np.isnan(rgb)] = 0.0

        norm = ImageNormalize(rgb, interval=ZScaleInterval(), stretch=LinearStretch())

        return rgb, norm
    

    def plot_bagpipes_SED(self):
        latex()
        objid = self.objid
        i = (LSBS == objid)
        z = self.z_a[0]
        correction = self.Fnu('KRON', 'F200W')[0] / self.Fnu('KRON_S', 'F200W')[0]
        Fnu, Fnu_err = self.aperture_Fnu('KRON_S') 
        is_good = (Fnu_err > 0)
        temp = LSB_BP_DATA[f'SED Templates/{objid}/Template']
        wave = LSB_BP_DATA[f'SED Templates/{objid}/Wavelengths [microns]'][()] * 1e-4 * (1 + z)

        mod_phot = LSB_BP_DATA['Model Photometry'][i, 1, :]
        print(mod_phot.shape)
        mod_phot_err = np.array([mod_phot - LSB_BP_DATA['Model Photometry'][i, 0, :], LSB_BP_DATA['Model Photometry'][i, 2, :] - mod_phot])

        plt.plot(wave, temp[1, :], color='blue')
        plt.plot(wave, temp[0, :], color='b')
        plt.plot(wave, temp[2, :], color='b')
        plt.errorbar(FILTER_PIVOTS[is_good], Fnu[is_good] * correction, yerr=Fnu_err[is_good]*correction, color='r', fmt='.')
        plt.errorbar(FILTER_PIVOTS[is_good], mod_phot[is_good], yerr=mod_phot_err[:, is_good], color='g', fmt='.')
        plt.xlim(0, 6)
        plt.ylim(0, 100)


    def ew_profile(self, z, n=100, Nrad=20):
        # Check that object has all the medium band data
        if self.Fnu('CIRC1', 'F430M')[0] < 0:
            print('No medium band data for object')
            pass

        f430m, __, __, __ = self.get_mosaic_cut('F430M', n=n)
        f460m, __, __, __ = self.get_mosaic_cut('F460M', n=n)
        f480m, __, __, __ = self.get_mosaic_cut('F480M', n=n)

        ew_map = EW(z, f430m, f460m, f480m)*10000 # Angstroms
        
        R_arr = np.linspace(0.1, 2*self.a, num=Nrad) # pix
        ew_r = np.zeros(Nrad)
        for i in range(Nrad):
            aperture = EllipticalAperture((n/2, n/2), R_arr[i], (self.b / self.a) * R_arr[i], self.pa)
            ew_r[i] = aperture.do_photometry(ew_map)[0][0] / (np.pi * (self.b / self.a) * (R_arr[i] ** 2))
        ew_r[ew_r < 0] = 0.0
        return ew_r

