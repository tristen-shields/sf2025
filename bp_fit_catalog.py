from goods_s.objects import *


'''
Fit a catalog of pixels of an object with BAGPIPES (standard delayed tau, Calzetti dust model).
'''

# Choose object and initiate object class
objid = 168586
o = obj(objid)

# List of filter .par filepaths to feed into BAGPIPES 
filt_list = []
for i in range(len(MOSAIC_NAMES)):
    filt_list.append(FILTER_PAR_PATH + MOSAIC_NAMES[i] + '.par')

n = 70 # Pixel length of square cutout

# Load data and 1sigma cutouts for the object (square, length n)
filters = MOSAIC_NAMES 
Nf = len(filters)
data = np.zeros((Nf, n, n))
sigma = np.zeros((Nf, n, n))
bkg = np.zeros(Nf)
sr_pix = np.zeros(Nf)

for i in range(Nf):
    data[i, :, :], sigma[i, :, :], bkg[i], sr_pix[i] = o.get_mosaic_cut(filters[i], n=n)

snr = data / sigma # For SNR pixel masking

# Mask pixels belonging to the object in at least one segmap AND having SNR>3 in at least one filter
mask_sum = np.zeros((n, n))
for i in range(Nf):
    segmap, cent, __ = make_segmap(data[i, :, :], sigma[i, :, :], thres=1.5, x0=int(n/2), a=o.a)
    mask_sum += np.isin(segmap, cent) & (snr[i, :, :] > 3)
pix_mask = mask_sum > 0
Npix = len(data[0, pix_mask])
print(f'Npix = {Npix}')
masked_data = data[:, pix_mask]
masked_sig = data[:, pix_mask]
pixids = np.linspace(0, Npix-1, num=Npix).astype(int) # Simply label pixel IDs as integers counting up from 0 to Npix-1
print(pixids.shape, masked_data.shape)


def load_data(pixid):
    pixid = int(pixid)
    data_MJy_Sr = masked_data[:, pixid]
    sig_MJy_Sr = masked_sig[:, pixid]

    data_muJy = np.zeros(data_MJy_Sr.shape)
    sig_muJy = np.zeros(sig_MJy_Sr.shape)
    for i in range(Nf):
        data_muJy[i] = MJy_Sr_to_nJy(data_MJy_Sr[i], filters[i]) * 1e-3
        sig_muJy[i] = MJy_Sr_to_nJy(sig_MJy_Sr[i], filters[i]) * 1e-3
    
    return np.array([data_muJy, sig_muJy]).T 

sampler = 'multinest'
# z = o.z_a[0]
z = 5.737
fit_instructions = build_dbl_power_law_fit_instructions(z)
# n_eff, n_live = 400, 400 # For 'nautilus' sampler only

fit = bp.fit_catalogue(pixids, fit_instructions, load_data, spectrum_exists=False, cat_filt_list=filt_list, vary_filt_list=False)
fit.fit(verbose=True, sampler=sampler, mpi_serial=True, track_backlog=True)
