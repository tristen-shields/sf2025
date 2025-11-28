import numpy as np 
from os.path import join 
from astropy.io import fits
from astropy.table import Table 
import h5py as h5 


# Photometric catalog
VERSION = 'v1.0.0' # Version of photometric catalog
CONV = True # Whether or not the photometry is convolved to F444W resolution

if CONV == True:
    conv_tag = 'conv_'
else:
    conv_tag = ''

phot_catalog_path = '/home/tdshield/sf2025/goods_s/photometric_catalog_' + conv_tag + VERSION + '.hdf5' # File location of photometric catalog
PHOT_CATALOG = h5.File(phot_catalog_path, 'r') # Taken from the original JADES catalog through make_phot_catalog.py
'''
    KEYS:
        Apertures: 'CIRC1', 'CIRC2', 'CIRC3', 'CIRC4', 'CIRC5', 'CIRC6', 'KRON', 'KRON_S'
            The 'CIRC{i}' apertures denote circular apertures where the radius increases with {i}. Of these, only 'CIRC1' is used in the LSB paper, which has radius 0.10". 'KRON' is a Kron aperture of parameter 2.5, while 'KRON_S' ("small Kron") is a Kron aperture of parameter 1.2
            Each aperture represents a group with members being the 20 different filters used in the LSB paper, with keys matching FILTER_NAMES (see filters.py) and each of those groups representing a group of flux densities with extensions 'Fnu [nJy]' and 'Fnu error [nJy]  representing arrays (length Nobj, number of objects in catalog) of the observed flux and error in the original JADES catalog, respectively, in that filter in that aperture
            Hierarchical structure: '{Aperture}/{Filter}/{Flux or Flux Error}'
        'SIZE':
            'A [pix]'/'A ERROR [pix]': Semi-major axis 'a' from original JADES catalog for each object, in pixels (length Nobj)
            'B [pix]'/'B ERROR [pix]': Semi-minor axis 'b' from original JADES catalog for each object, in pixels (length Nobj)
            'N_PIX': Number of pixels belonging to each object in the segmentation map (length Nobj)
            'PA [deg]': Position angle of each object, in degrees
            Other sub-extensions of 'SIZE' are irrelevant/unused
        'OBJIDS': IDs for every object in the catalog 
        'RA': Right ascension of every object in the catalog, in degrees (length Nobj)
        'DEC': Declination of every object in the catalog, in degrees (length Nobj)
        'X': x-coordinate of every object, in pixels
        'Y': y-coordinate of every object, in pixels
'''


# eazy-py photometric redshift catalog
Z_CATALOG_FILE = '/home/tdshield/.cache/redshift_catalog_KRON_S.fits'
Z_CATALOG_FITS = fits.open(Z_CATALOG_FILE)
Z_CATALOG_TABLE = Table.read(Z_CATALOG_FITS[1])

Z_CATALOG_IDS = Z_CATALOG_TABLE['ID'].data # Need to store IDs since not every object in the redshift catalog is in the photometric catalog
Z_A = Z_CATALOG_TABLE['z_a'].data # Definition of redshift used in paper: minimum chi squared redshift from eazy-py, denoted 'z_a'


# JADES mosaics
# MOSAICS_PATH = '/home/marcia/GOODSS_v0.9.5/' # Base path/location of mosaics

# Mosaics convolved to F444W PSF
if CONV == False:
    MOSAICS_PATH = '/fenrirdata1/jades/mosaics/goods_s_v1.0' # Base path/location of mosaics
    F090W = join(MOSAICS_PATH, 'F090W/mosaic_F090W.fits')
    F115W = join(MOSAICS_PATH, 'F115W/mosaic_F115W.fits')
    F150W = join(MOSAICS_PATH, 'F150W/mosaic_F150W.fits')
    F182M = join(MOSAICS_PATH, 'F182M/mosaic_F182M.fits')
    F200W = join(MOSAICS_PATH, 'F200W/mosaic_F200W.fits')
    F210M = join(MOSAICS_PATH, 'F210M/mosaic_F210M.fits')
    F250M = join(MOSAICS_PATH, 'F250M/mosaic_F250M.fits')
    F277W = join(MOSAICS_PATH, 'F277W/mosaic_F277W.fits')
    F300M = join(MOSAICS_PATH, 'F300M/mosaic_F300M.fits')
    F335M = join(MOSAICS_PATH, 'F335M/mosaic_F335M.fits')
    F356W = join(MOSAICS_PATH, 'F356W/mosaic_F356W.fits')
    F410M = join(MOSAICS_PATH, 'F410M/mosaic_F410M.fits')
    F430M = join(MOSAICS_PATH, 'F430M/mosaic_F430M.fits')
    F444W = join(MOSAICS_PATH, 'F444W/mosaic_F444W.fits')
    F460M = join(MOSAICS_PATH, 'F460M/mosaic_F460M.fits')
    F480M = join(MOSAICS_PATH, 'F480M/mosaic_F480M.fits')
    F770W = join(MOSAICS_PATH, 'F770W/mosaic_F770W.fits')
    MOSAICS = np.array([F090W, F115W, F150W, F200W, F210M, F250M, F277W, F300M, F335M, F356W, F410M, F444W, F770W]) # WITHOUT F182M (for now)
    MOSAIC_NAMES = np.array(['F090W', 'F115W', 'F150W', 'F200W', 'F210M', 'F250M', 'F277W', 'F300M', 'F335M', 'F356W', 
                         'F410M', 'F444W', 'F770W']) # WITHOUT F182M (for now)

elif CONV == True:
    MOSAICS_PATH = '/fenrirdata1/jades/mosaics/goods_s_v1.0_conv'
    F090W = join(MOSAICS_PATH, 'mosaic_F090W.conv.fits')
    F115W = join(MOSAICS_PATH, 'mosaic_F115W.conv.fits')
    F150W = join(MOSAICS_PATH, 'mosaic_F150W.conv.fits')
    F182M = join(MOSAICS_PATH, 'mosaic_F182M.conv.fits')
    F200W = join(MOSAICS_PATH, 'mosaic_F200W.conv.fits')
    F210M = join(MOSAICS_PATH, 'mosaic_F210M.conv.fits')
    F250M = join(MOSAICS_PATH, 'mosaic_F250M.conv.fits')
    F277W = join(MOSAICS_PATH, 'mosaic_F277W.conv.fits')
    F300M = join(MOSAICS_PATH, 'mosaic_F300M.conv.fits')
    F335M = join(MOSAICS_PATH, 'mosaic_F335M.conv.fits')
    F356W = join(MOSAICS_PATH, 'mosaic_F356W.conv.fits')
    F410M = join(MOSAICS_PATH, 'mosaic_F410M.conv.fits')
    F430M = join(MOSAICS_PATH, 'mosaic_F430M.conv.fits')
    F444W = join(MOSAICS_PATH, 'mosaic_F444W.conv.fits')
    F460M = join(MOSAICS_PATH, 'mosaic_F460M.conv.fits')
    F480M = join(MOSAICS_PATH, 'mosaic_F480M.conv.fits') 
    MOSAICS = np.array([F090W, F115W, F150W, F182M, F200W, F210M, F250M, F277W, F300M, F335M, F356W, F410M, F444W]) # WITHOUT F182M (for now)
    MOSAIC_NAMES = np.array(['F090W', 'F115W', 'F150W', 'F182M', 'F200W', 'F210M', 'F250M', 'F277W', 'F300M', 'F335M', 'F356W', 
                         'F410M', 'F444W']) # WITHOUT F182M (for now)

# PSFs
PSF_PATH = '/home/tdshield/.cache/filter_PSFs/'
PSF_NAMES = np.array(['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F410M', 'F444W'])
FILTER_PSF_PATHS = np.array([join(PSF_PATH, f'{filt}_PSF.fits') for filt in PSF_NAMES])


# LSB data (Shields et al. 2025)
LSB_DATA_PATH = '/home/tdshield/LSB_data'
LSB_PYIMFIT_DATA = h5.File(join(LSB_DATA_PATH, 'pyimfit_F200W_results.hdf5'), 'r')
BAD_LSBS = np.array([ 51418,  77458, 104231, 143759, 160429, 164273, 172796, 184631, 193760, 198563, 202107, 210324, 217816, 251488,
       284877, 381164, 425140, 461759, 465343, 465395, 479403, 513892, 550103, 638444]) # LSBs thrown out of the sample from visual inspection

LSBS = np.load(join(LSB_DATA_PATH, 'high_z_lsbs.npy'), allow_pickle=True).astype(int)
LOCAL_LSBS = np.load(join(LSB_DATA_PATH, 'local_lsbs.npy'), allow_pickle=True).astype(int)
HSBS = np.load(join(LSB_DATA_PATH, 'HSB_objids.npy'), allow_pickle=True).astype(int)

# LSB_PYSERSIC_DATA = h5.File('/home/tdshield/LSB_data/pysersic_lsb_results.hdf5', 'r')
LSB_PYSERSIC_DATA = h5.File(join(LSB_DATA_PATH, 'pysersic_lsb_results_single_band.hdf5'), 'r')

LSB_BP_DATA = h5.File(join(LSB_DATA_PATH, 'LSB_bagpipes_results.hdf5'), 'r')
LLSB_BP_DATA = h5.File(join(LSB_DATA_PATH, 'local_LSB_bagpipes_results.hdf5'), 'r')
HSB_BP_DATA = h5.File(join(LSB_DATA_PATH, 'HSB_bagpipes_results.hdf5'), 'r')

# Charlotte's Prospector catalog
PROSP_CAT_FILEPATH = '/home/marcia/prospector/prosp_properties_GOODSS.fits'
hdul = fits.open(PROSP_CAT_FILEPATH)
hdu = hdul[1].header
PROSP_OBJIDS = hdul[1].data['ID']
PROSP_Z = hdul[1].data['z']
PROSP_SFR5 = hdul[1].data['SFR5']
PROSP_SFR10 = hdul[1].data['SFR10']
PROSP_SFR50 = hdul[1].data['SFR50']
PROSP_SFR100 = hdul[1].data['SFR100']
PROSP_SFR200 = hdul[1].data['SFR2000']
PROSP_ST_MASS = hdul[1].data['log(Mstar)']
PROSP_T90 = hdul[1].data['t90']
PROSP_T50 = hdul[1].data['t50']
PROSP_NPHOT = hdul[1].data['Nphot(JWST)']
PROSP_SNR_F277W = hdul[1].data['SN(F277W)']
PROSP_SNR_F356W = hdul[1].data['SN(F356W)']
PROSP_SNR_F444W = hdul[1].data['SN(F444W)']


# Bursty objects 2 < z < 5 sample
BURSTY_OBJIDS = np.array([165307, 183825, 175793, 283395, 208368, 191110, 168867, 172091, 168873, 198153, 79116])
BURSTY_OBJIDS_Z = np.array([3.03, 3.04, 3.09, 3.15, 3.27, 3.29, 3.56, 3.6 , 3.75, 4.15, 4.15])