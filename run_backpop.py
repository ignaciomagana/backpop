import numpy as np
import time

from scipy.stats import gaussian_kde

import os
from argparse import ArgumentParser
import glob

from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d

import h5py
from astropy.cosmology import Planck15, FlatLambdaCDM, z_at_value
from astropy import units as u
import astropy.constants as constants
from pesummary.io import read
from backpop import *
from tqdm import tqdm
from nautilus import Prior, Sampler


# split out rv with KDE if you have GW samples
def likelihood(KDE, params_out, qmax, params_in):
    # evolve the binary
    result = evolv2(params_in, params_out)
    # check result
    if result[0] is None:
        return -np.inf, np.full(np.prod(BPP_SHAPE), np.nan, dtype=float), np.full(np.prod(KICK_SHAPE), np.nan, dtype=float)

    # flatten arrays and force dtype
    bpp_flat = np.array(result[1], dtype=float).ravel()
    kick_flat = np.array(result[2], dtype=float).ravel()
    
    # check shapes
    if bpp_flat.size != np.prod(BPP_SHAPE) or kick_flat.size != np.prod(KICK_SHAPE):
        return -np.inf, np.full(np.prod(BPP_SHAPE), np.nan, dtype=float), np.full(np.prod(KICK_SHAPE), np.nan, dtype=float)
    
    m1 = result[0]['mass_1']
    m2 = result[0]['mass_2']
    q = np.where(m2 <= m1, m2/m1, m1/m2)  # Ensure q is defined only when m2 <= m1
    mc = (m1*m2)**(3/5)/(m1 + m2)**(1/5)
    if (q < qmax):
        gw_coord = np.array([mc, q])
        ll = KDE.logpdf(gw_coord)
        return (ll[0], bpp_flat, kick_flat)
    else:
        return -np.inf, np.full(np.prod(BPP_SHAPE), np.nan, dtype=float), np.full(np.prod(KICK_SHAPE), np.nan, dtype=float)

if __name__ == "__main__":
    start = time.time()
    print("Starting timer")

    optp = ArgumentParser()
    optp.add_argument("--samples_path", help="path to event run dir")
    optp.add_argument("--event_name", help="name of event")
    optp.add_argument('--config_name', help="configuration to use")
    optp.add_argument("--weights", type=str_to_bool, nargs='?', const=False, default=True)
    optp.add_argument("--nlive", type=int, default=3000)
    optp.add_argument("--neff", type=int, default=10000)
    optp.add_argument("--resume", type=str_to_bool, nargs='?', const=False, default=False)

    opts = optp.parse_args()

    samples_path = opts.samples_path
    event_name = opts.event_name
    config_name = opts.config_name
    weights = opts.weights
    resume = opts.resume
    print(weights)
    print(resume)

    output_path = "./results/" + event_name + "/" + config_name + "/"
    print(output_path)
    try:
        os.makedirs(output_path, exist_ok=False)
    except:
        print("Output directory already exists. Continuing...")
        pass

    cols_keep = ['tphys', 'mass_1', 'mass_2', 'menv_1', 'menv_2', 'kstar_1', 'kstar_2', 'porb', 'ecc', 'evol_type', 'rad_1', 'rad_2', 'lum_1', 'lum_2']
    KICK_COLUMNS = ['star', 'disrupted', 'natal_kick', 'phi', 'theta', 'mean_anomaly',
                    'delta_vsysx_1', 'delta_vsysy_1', 'delta_vsysz_1', 'vsys_1_total',
                    'delta_vsysx_2', 'delta_vsysy_2', 'delta_vsysz_2', 'vsys_2_total',
                    'theta_euler', 'phi_euler', 'psi_euler', 'randomseed']


    params = labels_dict[config_name]
    print(config_name)

    evolution, lower_bound, upper_bound, params_in = get_backpop_config(config_name)

    KDE, gwsamples, gwsamples_kde, qmin, qmax, mcmin, mcmax = load_data(samples_path, weights)
    qmax = max(qmax, 0.3)
    print(f'qmax={qmax}, qmin={qmin}, mcmax={mcmax}, mcmin={mcmin}')
    # Set up Nautilus prior
    prior = Prior()
    for i in range(len(params_in)):
        prior.add_parameter(params_in[i], dist=(lower_bound[i], upper_bound[i]))

    params_out=['mass_1', 'mass_2']
    num_cores = int(len(os.sched_getaffinity(0)))
    num_threads = int(2*num_cores-2)
    print("using multiprocessing with " + str(num_threads) + " threads")
    
    dtype = [('bpp', float, 25*len(cols_keep)), ('kick_info', float, 2*len(KICK_COLUMNS))]
    n_live = opts.nlive
    n_eff = opts.neff
    sampler = Sampler(
        prior=prior, 
        likelihood=likelihood, 
        n_live=n_live, 
        pool=num_threads,
        blobs_dtype=dtype,
        filepath="./results/" + event_name + "/" + config_name + "/" + "checkpoint.hdf5",
        resume=resume, 
        likelihood_args=(KDE, params_out, qmax)
)
    sampler.run(n_eff=n_eff,verbose=True,discard_exploration=True)
    
    points, log_w, log_l, blobs = sampler.posterior(return_blobs=True)

    log_z = sampler.log_z
    dweights = np.exp(log_w - log_z)
    dweights = dweights/np.sum(dweights)
    
    np.save(output_path + "points.npy", points)
    np.save(output_path + "log_w.npy", log_w)
    np.save(output_path + "log_l.npy", log_l)
    np.save(output_path + "log_z.npy", log_z)
    np.save(output_path + "blobs.npy", blobs)

    end = time.time()
    print("Execution time: " + str(end - start) + " seconds")