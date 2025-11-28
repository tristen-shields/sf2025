from .filters import *
from sep import extract
from scipy.special import gamma, gammaincinv 
import matplotlib.pyplot as plt
from scipy.optimize import brentq       # For root-finding


# COSMOLOGY
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315, Tcmb0=2.726)


# COSMOLOGICAL CONSTANTS
HB0 = cosmo.H0 # Hubble constant, km/s/Mpc (z = 0)
OMM = cosmo.Om0 # Omega matter (z = 0)
OMB = cosmo.Ode0 # Omega dark energy (z = 0)


def make_segmap(data, sigma, thres, x0, a):
    '''
    Given a square data image and an associated one-standard deviation error image centered around an object of semi-major axis a, use sep (SourceExtractor python) to make a segmentation map, then determine whether or not there are pixels that match the center object's segmentation that are greater than 5*a from the center in either the x or the y coordinates.
        data (2darray): Data image
        sigma (2darray): One-standard deviation error image
        thres (float): Threshold to feed to SEP
        x0 (float): Center coordinate of the given image [pix]
        a (float): Object semi-major axis [pix]

        returns:
            segmap0 (2darray): Map that assigns segmentation index to every pixel (same shape as data image)
            cent0 (int): Segmentation index at (x0, x0) 
            flag (bool): 'True' if central segmentation index cent0 is less than 5a pixels away from x0 in either the x or y coordinate, 'False' if not
    '''
    # Make sure data and sigma images are compatible with sep.extract()
    data = np.ascontiguousarray(data, dtype='float32')
    sigma = np.ascontiguousarray(sigma, dtype='float32')

    sources = extract(data, thres, err=sigma, segmentation_map=True)
    segmap0 = sources[1] # Source at center of the map
    cent0 = segmap0[x0, x0] # Index of the center object

    # To tell whether or not pixels that are further from 5*a from the center are matching the center object
    flag = True 
    for i, j in np.argwhere(segmap0 == cent0):
        if (i > x0 + a | (i < x0 - a) | (j > x0 + a) | (j < x0 - a)):
            flag = False 
    
    return segmap0, cent0, flag 


# SURFACE BRIGHTNESS DEFINITIONS/SERSIC PROFILES
def mean_eff_surface_brightness(m_tot, R_eff_as, ell, m_tot_err=0, R_eff_as_err=0, ell_err=0):
    '''
    Given the total model magnitude from a surface brightness fitting program (pysersic, pyimfit), as well as the fitted effective radius in arcseconds, and fitted ellipticity (and the errors for each, optionally), calculate the mean effective surface brightness of the Sersic profile. The redshift is also needed to scale the surface brightness accordingly.
        m_tot/m_tot_err (float): Measured total magnitude from a fitted Sersic model
        R_eff_as/R_eff_as_err (float): Fitted effective (half-light) radius in arcseconds
        ell/ell_err (float): Fitted ellipticity
        z (float): Object's redshift
    '''
    mean_eff_sb = m_tot + (2.5 * np.log10(2 * np.pi * (1 - ell) * (R_eff_as ** 2)))

    # Error propagation
    mean_eff_sb_var = (m_tot_err ** 2) + (((2.5 * ell_err) / (np.log(10) * (1 - ell))) ** 2) + (((5 * R_eff_as_err) / (np.log(10) * R_eff_as)) ** 2)
    mean_eff_sb_err = np.sqrt(mean_eff_sb_var)

    return mean_eff_sb, mean_eff_sb_err 


def b(n):
    '''
    Geometric factor for a given Sersic profile of index n.
        n (float): Sersic index
    '''
    return gammaincinv(2 * n, 0.5)


def f(n):
    '''
    Part of the expression for central surface brightness that only depends on Sersic index n, and is too complicated to propagate errors through, so it is written as a separate function here.
        n/n_err (float): Sersic index
    '''
    return (2.5 * np.log10(n * np.exp(b(n)) * gamma(2 * n) / (b(n) ** (2 * n)))) - (2.5 * b(n) / np.log(10))


def central_surface_brightness(mean_eff_sb, n, mean_eff_sb_err=0, n_err=0):
    '''
    Given a mean effective surface brightness and Sersic index n, calculate the central surface brightness of a Sersic profile.
    '''
    f_n = f(n)
    print(f_n)
    u0 = mean_eff_sb + f_n 

    # Error propagation
    f_err = np.max([np.abs(f(n + n_err) - f_n), np.abs(f(n - n_err) - f_n)]) # Estimated error for f(n)
    u0_err = u0 * np.sqrt(((mean_eff_sb_err / mean_eff_sb) ** 2) + ((f_err / f_n) ** 2))

    return u0, u0_err 


# BAGPIPES
def build_dust_law(dustLaw):
    if dustLaw == 'CF00':

        dust = {}

        dust['n'] = (+0.3, +1.5)
        dust['n_prior'] = 'Gaussian'
        dust['n_prior_sigma'] = 0.3
        dust['n_prior_mu'] = 0.7

        dust['Av'] = (0.0, 2.0)
        dust['Av_prior'] = 'Gaussian'
        dust['Av_prior_sigma'] = 1.0
        dust['Av_prior_mu'] = 0.3

        dust['eta'] = (1.0, 3.0)
        dust['eta_prior'] = 'Gaussian'
        dust['eta_prior_sigma'] = 0.3
        dust['eta_prior_mu'] = 1.0

        dust['type'] = dustLaw

    elif dustLaw == 'Salim':

        dust = {}
        dust['B'] = (0.01, 5.)
        dust['Av'] = (0.0, 2.0)
        dust['delta'] = (-0.3, +0.3)
        dust['B_prior'] = 'uniform'
        dust['Av_prior'] = 'uniform'
        dust['delta_prior'] = 'Gaussian'
        dust['delta_prior_sigma'] = 0.1
        dust['delta_prior_mu'] = 0.0
        dust['type'] = dustLaw

    elif dustLaw == 'Calzetti':

        dust = {}
        dust['Av'] = (0.0, 4.0)
        dust['type'] = dustLaw

    elif dustLaw == 'Log':

        dust = {}
        dust['Av'] = (1e-3, 1e-0)
        dust['Av_prior'] = 'log_10'
        dust['type'] = 'Calzetti'
        
    elif dustLaw == 'None':

        dust = {}
        dust['Av'] = 0.0
        dust['type'] = 'Calzetti'

    return dust



def build_delayed_tau_fit_instructions(z, z_fixed=True, zForm=20.0, ageLimit=1e+6, dustLaw='Calzetti'):
    
    # defines fit instructions dictionary

    t = cosmo.age(z).value 

    delayed = {}
    delayed['age'] = (1e-9*ageLimit, 1.0*(t - cosmo.age(zForm).value))
    delayed['tau'] = (0.3, 50)
    # delayed['tau'] = (0.3, 10)
    delayed['tau_prior'] = 'log_10'
    delayed['metallicity_prior'] = 'log_10'
    delayed['metallicity'] = (1e-3, 1e-0)
    delayed['massformed'] = (6.0, 12.0)

    dust = build_dust_law(dustLaw)

    nebular = {}
    nebular['logU'] = (-4.0, -2.0)

    fit_instructions = {}
    fit_instructions['t_bc'] = 0.01
    fit_instructions['dust'] = dust
    fit_instructions['nebular'] = nebular
    fit_instructions['delayed'] = delayed

    fit_instructions['redshift'] = z
    
    # returns fit instructions dictionary

    return fit_instructions


def build_dbl_power_law_fit_instructions(z, dustLaw='Calzetti'):
    # Double power law SFH component
    dblplaw = {}
    dblplaw['alpha'] = (0.01, 1000)
    dblplaw['alpha_prior'] = 'log_10'
    dblplaw['beta'] = (0.01, 1000)
    dblplaw['beta_prior'] = 'log_10'
    dblplaw['tau'] = (0, 14) # Gyr
    dblplaw['metallicity'] = (0, 2.5)
    dblplaw['massformed'] = (1.0, 14.0)

    dust = build_dust_law(dustLaw)
    
    nebular = {}
    nebular['logU'] = (-3.0, -1.0)

    fit_instructions = {}
    fit_instructions['t_bc'] = 0.01
    fit_instructions['dust'] = dust 
    fit_instructions['nebular'] = nebular 
    fit_instructions['dblplaw'] = dblplaw 
    fit_instructions['redshift'] = z 

    return fit_instructions
    

# UNITS
def nJy_to_mag(nJy, band, nJy_err=0):
    '''
    Convert a flux density reading (and optionally an associated error) in nJy to an apparent magnitude.
        nJy: Flux density reading in nJy (float/int)
        band: Band to use for the zeropoint (zp) flux ('B', 'V', or 'AB', where 'AB' covers the u, g, r, i, and z bands)
    '''
    Jy = nJy * 1e-9
    Jy_err = nJy_err * 1e-9 

    # Zero-magnitude fluxes for different bands
    if band == 'B':
        zp = 4063 # in Jy

    elif band == 'V':
        zp = 3636 # in Jy

    elif band == 'AB':
        zp = 3631 # in Jy

    else:
        print('Band must be "B", "V", or "AB"')
        pass

    return (-2.5) * np.log10(Jy / zp), (2.5 / np.log(10)) * (Jy_err / Jy)


def MJy_Sr_to_nJy(Fnu_MJy_Sr, filt):
    '''
    Convert from units of MJy/Str (units of the JWST mosaics) to nJy, using the headers in a given 'MOSAICS' fits file.
        Fnu_MJy_str: Flux density reading in MJy/Str (float)
        filt: Filter keyword (e.g. 'F200W', must be in 'MOSAIC_NAMES' constant) (str)
    '''
    i = np.where(MOSAIC_NAMES == filt)[0][0]
    hdul = fits.open(MOSAICS[i])
    str_pix = hdul[0].header['PIXAR_SR'] # Str/pix
    return Fnu_MJy_Sr * str_pix * 1e+15


# CONVENIENCE
def get_filter_PSF(filt, n):
    '''
    Get the PSF data for a given filter.
        filt: Filter keyword in PSF_NAMES (str)
        n: Side length of the PSF [pix] (ODD int)
    '''

    return fits.getdata(FILTER_PSF_PATHS[np.where(PSF_NAMES == filt)][0])[int(334 - (n / 2)):int(334 + (n / 2)), int(334 - (n / 2)):int(334 + (n / 2))]


def sfr_x_axis(ages, z):
    return cosmo.age(z).value - (ages * 10**-9)


def EW(z, f430m, f460m, f480m, bw_460m=0.228):
    C = np.average([f430m, f480m]) # Estimated continuum
    return ((f460m - C) * bw_460m) / (C * (1 + z))


# Function to find redshift given age (in Gyr) 
def redshift_from_age(age_gyr):
    """
    Given an age in Gyr, return the corresponding redshift z.
    """
    # Define a function whose root corresponds to the correct redshift
    def age_difference(z):
        return cosmo.age(z).value - age_gyr  # .value gives Gyr as float

    # Use brentq root-finder: redshift must be > 0, and age(z=∞) ~ 0
    z = brentq(age_difference, 1e-8, 1000)  # Bracket redshift range
    return z


# PLOTTING FUNCTIONS
def latex():
    plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'Serif',
    'figure.dpi': 250
    })