from goods_s.objects import *
from os import remove
from shutil import rmtree
import time 


'''
Run the double power law SFH BAGPIPES on a pixel-by-pixel basis for multiple objects and store pixel maps of relevant quantities.
'''

start_time = time.time() # Track time that the script takes to run 
objids = BURSTY_OBJIDS
z_arr = BURSTY_OBJIDS_Z
out_filename = 'bursty_objids_bp_data_conv'


for i in range(len(objids)):
    objid = objids[i]
    print(f'Now fitting object ID {objid}')
    run_name = f'{objid}'
    z = z_arr[i] # Photometric redshift from Charlotte's catalog
    print(f'Redshift used: z = {z}')
    o = obj(objid)

    Ncut = int(round(7 * o.a))# Length of pixel cutout

    # List of filter .par paths to feed into BAGPIPES
    filters = MOSAIC_NAMES 
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

    # Clear BAGPIPES plots file for run
    try:
        rmtree(f'pipes/plots/{run_name}')
    except:
        pass

    sampler = 'multinest'
    fit_instructions = build_dbl_power_law_fit_instructions(z)
    fit = bp.fit_catalogue(pixids, fit_instructions, load_data, spectrum_exists=False, photometry_exists=True, cat_filt_list=filt_list,
                        vary_filt_list=False, run=run_name)
    fit.fit(verbose=True, sampler=sampler, mpi_serial=True, track_backlog=True)

    print('Pixel fitting completed, loading data...')

    sfh = np.zeros((Npix, 3, 1654))
    ages = np.zeros((Npix, 1654))
    sfr100 = np.zeros((Npix, 3))
    tMW = np.zeros((Npix, 3))
    Av = np.zeros((Npix, 3))
    ssfr = np.zeros((Npix, 3))
    st_mass = np.zeros((Npix, 3))
    logU = np.zeros((Npix, 3))
    metallicity = np.zeros((Npix, 3))
    burst = np.zeros((Npix, 3))


    for j in range(Npix):
        # Run BAGPIPES on one pixel
        pixel = bp.galaxy(pixids[j], load_data, spectrum_exists=False, filt_list=filt_list, photometry_exists=True)

        fit = bp.fit(pixel, fit_instructions, run=run_name)
        fit.fit(verbose=True, sampler=sampler)

        # fit.posterior.get_advanced_quantities()
        ages[j, :] = fit.posterior.sfh.ages[:1654]
        sfh[j, :, :1654] = np.percentile(fit.posterior.samples['sfh'], [16, 50, 84], axis=0) # Star formation history (SFR [Msun/yr] vs time)
        sfr100[j, :] = np.percentile(fit.posterior.samples['sfr'], [16, 50, 84], axis=0)
        tMW[j, :] = np.percentile(fit.posterior.samples['mass_weighted_age'], [16, 50, 84], axis=0)
        Av[j, :] = np.percentile(fit.posterior.samples['dust:Av'], [16, 50, 84], axis=0)
        ssfr[j, :] = np.percentile(fit.posterior.samples['ssfr'], [16, 50, 84], axis=0)
        st_mass[j, :] = np.percentile(fit.posterior.samples['stellar_mass'], [16, 50, 84], axis=0)
        logU[j, :] = np.percentile(fit.posterior.samples['nebular:logU'], [16, 50, 84], axis=0)
        metallicity[j, :] = np.percentile(fit.posterior.samples['dblplaw:metallicity'], [16, 50, 84], axis=0)

        # x is in DECREASING order
        test_sfh = sfh[j, :, :]
        x = sfr_x_axis(ages[j, :], z) # Age of Universe in Gyr
        x_Myr = x*1e+3
        x_obs = x_Myr[0] # Epoch of observation
        sfr10 = np.average(test_sfh[:, x_Myr > (x_obs - 10)], axis=1)
        sfr90 = np.average(test_sfh[:, (x_Myr < (x_obs - 10)) & (x_Myr > (x_obs - 100))], axis=1)
        burst[i, :] = sfr10 / sfr90

    print(f'Fitting for object {objid} successful!')

    # Save pixel data to a single HDF file
    with h5.File(f'bp_data/{out_filename}.hdf5', 'w') as hdf:
        obj_grp = hdf.create_group(f'{objid}')
        obj_grp.create_dataset('SFHs', shape=(Npix, 3, 1654), data=sfh)
        obj_grp.create_dataset('sSFR', shape=(Npix, 3), data=ssfr)
        obj_grp.create_dataset('Stellar Mass', shape=(Npix, 3), data=st_mass)
        obj_grp.create_dataset('tMW', shape=(Npix, 3), data=tMW)
        obj_grp.create_dataset('logU', shape=(Npix, 3), data=logU)
        obj_grp.create_dataset('SFR10_SFR90', shape=(Npix, 3), data=burst)
        obj_grp.create_dataset('SFH Ages', shape=1654, data=ages[0, :])


end_time = time.time()
time_elapsed = end_time - start_time
print(f"Elapsed Time: {time_elapsed} seconds")