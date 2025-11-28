from .loading import *


FILTER_NAMES = np.array(['F435W', 'F606W', 'F070W', 'F775W', 'F814W', 'F090W', 'F850LP', 'F115W', 'F150W', 'F182M', 'F200W', 
             'F210M', 'F277W', 'F335M', 'F356W', 'F410M', 'F430M', 'F444W', 'F460M', 'F480M'])

# PSFs
PSF_PATH = '/home/tdshield/.cache/filter_PSFs/'
PSF_NAMES = np.array(['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F410M', 'F444W'])
FILTER_PSF_PATHS = np.array([join(PSF_PATH, f'{filt}_PSF.fits') for filt in PSF_NAMES])

# Get the wavelengths from the name of each filter, in microns
FILTER_PIVOTS = np.array([0.4311, 0.5888, 0.704, 0.76651, 0.81153, 0.901, 0.91452, 1.154, 1.501, 1.845, 1.990, 2.093, 2.768, 3.365, 
         3.563, 4.092, 4.280, 4.421, 4.624, 4.834]) # CENTRAL/PIVOT wavelengths for each filter

FILTER_PAR_PATH = '/home/tdshield/.cache/filters/' # File path for filter .par files, to be used in running BAGPIPES

# Other constants
COLORS = ['#377eb8', '#ff7f00', '#4daf4a',
            '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00'] # Colorblind-friendly color cycle

AS_PER_PIX = 0.03 # Arcseconds per pixel conversion