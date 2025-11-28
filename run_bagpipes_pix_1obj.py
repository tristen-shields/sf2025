from goods_s.objects import *
from os import remove
from shutil import rmtree
import time 


'''
Run the double power law SFH BAGPIPES on a pixel-by-pixel basis for a single object.
'''

start_time = time.time()

objid = 170758 
run_name = f'1obj_{objid}'
# z = BURSTY_OBJIDS_Z[BURSTY_OBJIDS == objid][0] # Photometric redshift from Charlotte's catalog
z = 3.0 # Photometric (Kevin's unconvolved CIRC1, within range of Hodge et al. 2025 zphot)
print(f'Redshift used: z = {z}')
o = obj(objid)
Ncut = 50 # Length of pixel cutout
clear_posterior = False # Whether or not to clear the BAGPIPES posterior, plots, and catalog files for this run

# List of filter .par paths to feed into BAGPIPES
# filters = MOSAIC_NAMES 
filters = ['F090W', 'F115W', 'F150W', 'F200W', 'F250M', 'F277W', 'F335M', 'F356W', 'F410M', 'F444W']
Nfilt = len(filters)
filt_list = []
for i in range(Nfilt):
    filt_list.append(FILTER_PAR_PATH + MOSAIC_NAMES[i] + '.par')

# Gather data cutout, error cutout, background, and Sr per pixel for each filter
data = np.zeros((Nfilt, Ncut, Ncut))
sigma = np.zeros((Nfilt, Ncut, Ncut))
bkg = np.zeros(Nfilt)
sr_pix = np.zeros(Nfilt)

for i, filt in enumerate(filters):
    data[i, :, :], sigma[i, :, :], bkg[i], sr_pix[i] = o.get_mosaic_cut(filt, n=Ncut)

snr = data / sigma # For pixel masking
sigma[snr > 20] = 0.05*data[snr > 20] # Impose 5% error floor on all pixels

# Create a mask that ensures good (>3) SNR and that pixels belong to the object in each filter
mask = np.zeros((Ncut, Ncut))
thres = 1.5 # Starting threshold for masking routine
x0 = int(Ncut/2) # Center of cutout
for i in range(Nfilt):
    segmap, cent, flag = make_segmap(data[i, :, :], sigma[i, :, :], thres, x0, int(o.a))

    while flag == False:
        thres += 0.5
        segmap, cent, flag = make_segmap(data[i, :, :], sigma[i, :, :], thres, x0, int(o.a))
        if cent == 0:
            flag = True 
    
    mask += (np.isin(segmap, cent) & (snr[i, :, :] > 3))

pix_mask = (mask > 0) # Any pixel that has SNR > 3 and is belonging to central object in ANY filter
Npix = len(data[0, pix_mask])
print(f'There are {Npix} pixels to fit with BAGPIPES')

# Apply pixel mask to data and error cutouts
masked_data = data[:, pix_mask]
masked_sig = sigma[:, pix_mask]
pixids = np.linspace(0, Npix-1, num=Npix).astype(int) # Label each pixel with an integer as ther "ID"

# Function that loads photometric data for a given pixel to feed into BAGPIPES
def load_data(pixid):
    pixid = int(pixid)
    data_MJy_Sr = masked_data[:, pixid]
    sig_MJy_Sr = masked_sig[:, pixid]

    data_muJy = np.zeros(data_MJy_Sr.shape)
    sig_muJy = np.zeros(sig_MJy_Sr.shape)
    for i in range(Nfilt):
        data_muJy[i] = MJy_Sr_to_nJy(data_MJy_Sr[i], filters[i]) * 1e-3
        sig_muJy[i] = MJy_Sr_to_nJy(sig_MJy_Sr[i], filters[i]) * 1e-3
    
    return np.array([data_muJy, sig_muJy]).T

if clear_posterior:
    # Clear BAGPIPES posterior file
    try:
        rmtree(f'pipes/posterior/{run_name}')
    except:
        pass

    # Clear BAGPIPES catalog file
    try:
        remove(f'pipes/cats/{run_name}.fits')
    except:
        pass

    # Clear BAGPIPES plots directory
    try:
        rmtree(f'pipes/plots/{run_name}')
    except:
        pass

sampler = 'multinest'
fit_instructions = build_dbl_power_law_fit_instructions(z)
fit = bp.fit_catalogue(pixids, fit_instructions, load_data, spectrum_exists=False, photometry_exists=True, cat_filt_list=filt_list,
                    vary_filt_list=False, run=run_name)
fit.fit(verbose=True, sampler=sampler, mpi_serial=True, track_backlog=True)

end_time = time.time()
time_elapsed = end_time - start_time
print(f"Elapsed Time: {time_elapsed} seconds")