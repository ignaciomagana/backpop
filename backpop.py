import numpy as np
from ctypes import *
import pandas as pd

from scipy.stats import gaussian_kde
import arviz as az
from pesummary.io import read
from pesummary.gw.fetch import fetch_open_samples

from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d

from astropy.cosmology import Planck15, FlatLambdaCDM, z_at_value
from astropy import units as u
import astropy.constants as constants
from cosmic import _evolvebin

#     SUBROUTINE evolv2_global(z,zpars,acclim,alphain,qHG,qGB,kick_in)

#libc = cdll.LoadLibrary("/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/src/COSMIC/cosmic/src/evolv2_bhms.so")
np.set_printoptions(suppress=True)

# COSMIC columns
ALL_COLUMNS = ['tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2', 'sep', 'porb',
               'ecc', 'RRLO_1', 'RRLO_2', 'evol_type', 'aj_1', 'aj_2', 'tms_1',
               'tms_2', 'massc_1', 'massc_2', 'rad_1', 'rad_2', 'mass0_1',
               'mass0_2', 'lum_1', 'lum_2', 'teff_1', 'teff_2', 'radc_1',
               'radc_2', 'menv_1', 'menv_2', 'renv_1', 'renv_2', 'omega_spin_1',
               'omega_spin_2', 'B_1', 'B_2', 'bacc_1', 'bacc_2', 'tacc_1',
               'tacc_2', 'epoch_1', 'epoch_2', 'bhspin_1', 'bhspin_2',
               'deltam_1', 'deltam_2', 'SN_1', 'SN_2', 'bin_state', 'merger_type']

INTEGER_COLUMNS = ["bin_state", "bin_num", "kstar_1", "kstar_2", "SN_1", "SN_2", "evol_type"]


BPP_COLUMNS = ['tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2',
               'sep', 'porb', 'ecc', 'RRLO_1', 'RRLO_2', 'evol_type',
               'aj_1', 'aj_2', 'tms_1', 'tms_2',
               'massc_1', 'massc_2', 'rad_1', 'rad_2',
               'mass0_1', 'mass0_2', 'lum_1', 'lum_2', 'teff_1', 'teff_2',
               'radc_1', 'radc_2', 'menv_1', 'menv_2', 'renv_1', 'renv_2',
               'omega_spin_1', 'omega_spin_2', 'B_1', 'B_2', 'bacc_1', 'bacc_2',
               'tacc_1', 'tacc_2', 'epoch_1', 'epoch_2',
               'bhspin_1', 'bhspin_2']

BCM_COLUMNS = ['tphys', 'kstar_1', 'mass0_1', 'mass_1', 'lum_1', 'rad_1',
               'teff_1', 'massc_1', 'radc_1', 'menv_1', 'renv_1', 'epoch_1',
               'omega_spin_1', 'deltam_1', 'RRLO_1', 'kstar_2', 'mass0_2', 'mass_2',
               'lum_2', 'rad_2', 'teff_2', 'massc_2', 'radc_2', 'menv_2',
               'renv_2', 'epoch_2', 'omega_spin_2', 'deltam_2', 'RRLO_2',
               'porb', 'sep', 'ecc', 'B_1', 'B_2',
               'SN_1', 'SN_2', 'bin_state', 'merger_type']

KICK_COLUMNS = ['star', 'disrupted', 'natal_kick', 'phi', 'theta', 'mean_anomaly',
                'delta_vsysx_1', 'delta_vsysy_1', 'delta_vsysz_1', 'vsys_1_total',
                'delta_vsysx_2', 'delta_vsysy_2', 'delta_vsysz_2', 'vsys_2_total',
                'theta_euler', 'phi_euler', 'psi_euler', 'randomseed']

cols_keep = ['tphys', 'mass_1', 'mass_2', 'massc_1', 'massc_2', 'menv_1', 'menv_2', 'kstar_1', 'kstar_2', 'porb', 'ecc', 'evol_type', 'rad_1', 'rad_2', 'lum_1', 'lum_2']

BPP_SHAPE = (25, len(cols_keep))
KICK_SHAPE = (2, len(KICK_COLUMNS))


def set_flags(params_in, fixed_params):
    ''' Set the COSMIC flags based on input parameters.
    If a parameter is not specified in params_in, it is set to a default value.
    
    Parameters
    ----------
    params_in : dict
        Dictionary of input parameters to set. Keys are parameter names, values are parameter values.
    fixed_params : dict
        Dictionary of fixed parameters that should not be sampled but should be fixed. Keys are parameter names, values are parameter values

        
    Returns
    -------
    flags : dict
        Dictionary of all COSMIC flags set.
    '''
    #set the flags to the defaults
    flags = dict()

    flags["neta"] = 0.5
    flags["bwind"] = 0.0
    flags["hewind"] = 0.5
    flags["beta"] = -1
    flags["xi"] = 0.5
    flags["acc2"] = 1.5
    flags["epsnov"] = 0.001
    flags["eddfac"] = 1.0
    flags["alpha1"] = np.array([1.0, 1.0])
    flags["qcrit_array"] = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    flags["lambdaf"] = 0.0
    flags["pts1"] = 0.05
    flags["pts2"] = 0.01
    flags["pts3"] = 0.02
    flags["tflag"] = 1
    flags["ifflag"] = 0
    flags["wdflag"] = 1
    flags["rtmsflag"] = 0
    flags["ceflag"] = 0
    flags["cekickflag"] = 2
    flags["cemergeflag"] = 1
    flags["cehestarflag"] = 0
    flags["bhflag"] = 1
    flags["remnantflag"] = 4
    flags["grflag"] = 1
    flags["bhms_coll_flag"] = 1
    flags["mxns"] = 1.0
    flags["pisn"] = -2
    flags["ecsn"] = 2.5
    flags["ecsn_mlow"] = 1.6
    flags["aic"] = 1
    flags["ussn"] = 1
    flags["sigma"] = 0
    flags["sigmadiv"] = -20.0
    flags["bhsigmafrac"] = 1.0
    flags["polar_kick_angle"] = 90.0
    flags["natal_kick_array"] = np.array([[-100.0,-100.0,-100.0,-100.0,0.0],[-100.0,-100.0,-100.0,-100.0,0.0]])
    flags["kickflag"] = 1
    flags["rembar_massloss"] = 0.5
    flags["bhspinmag"] = 0
    flags["don_lim"] = -1
    flags["acc_lim"] = np.array([-1, -1])
    flags["gamma"] = -2
    flags["bdecayfac"] = 1
    flags["bconst"] = 3000
    flags["ck"] = 1000
    flags["windflag"] = 3
    flags["qcflag"] = 5
    flags["eddlimflag"] = 0
    flags["fprimc_array"] = np.ones(16) * 2.0/21.0 * 0.0
    flags["randomseed"] = 42
    flags["bhspinflag"] = 0
    flags["rejuv_fac"] = 1.0
    flags["rejuvflag"] = 0
    flags["htpmb"] = 1
    flags["ST_cr"] = 1
    flags["ST_tide"] = 1
    flags["zsun"] = 0.014
    flags["using_cmc"] = 0
    natal_kick = np.zeros((2,5))
    qcrit_array = np.zeros(16)
    alpha1 = np.zeros(2)
    acc_lim = np.zeros(2)
    qc_list = ["qMSlo", "qMS", "qHG", "qGB", "qCHeB", "qAGB", "qTPAGB", "qHeMS", "qHeGB", "qHeAGB"]
    
    for param in params_in.keys():
        # handle kicks
        # this is hacky -- think about this more.
        if param in ["vk1", "phi1", "theta1", "omega1", "vk2", "phi2", "theta2", "omega2"]:
            if "1" in param:
                if "vk" in param:
                    natal_kick[0,0] = params_in[param]
                elif "phi" in param:
                    natal_kick[0,1] = params_in[param]
                elif "theta" in param:
                    natal_kick[0,2] = params_in[param]
                elif "omega" in param:
                    natal_kick[0,3] = params_in[param]
                natal_kick[0,4] = 0.0
            elif "2" in param:
                if "vk" in param:
                    natal_kick[1,0] = params_in[param]
                elif "phi" in param:
                    natal_kick[1,1] = params_in[param]
                elif "theta" in param:
                    natal_kick[1,2] = params_in[param]
                elif "omega" in param:
                    natal_kick[1,3] = params_in[param]
                natal_kick[1,4] = 0.0
        elif param in qc_list:
            ind = qc_list.index(param)  # get the index of param in qc_list
            qcrit_array[ind] = params_in[param]
    
        elif param in ["alpha1_1", "alpha1_2"]:
            
            if param == "alpha1_2":
                alpha1[1] = params_in[param]
            elif param == "alpha1_1":
                alpha1[0] = params_in[param]
                
        elif param in ["acc_lim_1", "acc_lim_2"]:
            
            if param == "acc_lim_2":
                acc_lim[1] = params_in[param]
            elif param == "acc_lim_1":
                acc_lim[0] = params_in[param]

        else:
            flags[param] = params_in[param]

    for param in fixed_params.keys():
        # handle kicks
        # this is hacky -- think about this more.
        if param in ["vk1", "phi1", "theta1", "omega1", "vk2", "phi2", "theta2", "omega2"]:
            if "1" in param:
                if "vk" in param:
                    natal_kick[0,0] = fixed_params[param]
                elif "phi" in param:
                    natal_kick[0,1] = fixed_params[param]
                elif "theta" in param:
                    natal_kick[0,2] = fixed_params[param]
                elif "omega" in param:
                    natal_kick[0,3] = fixed_params[param]
                natal_kick[0,4] = 0.0
            elif "2" in param:
                if "vk" in param:
                    natal_kick[1,0] = fixed_params[param]
                elif "phi" in param:
                    natal_kick[1,1] = fixed_params[param]
                elif "theta" in param:
                    natal_kick[1,2] = fixed_params[param]
                elif "omega" in param:
                    natal_kick[1,3] = fixed_params[param]
                natal_kick[1,4] = 0.0

        elif param in qc_list:
            ind = qc_list.index(param)  # get the index of param in qc_list
            qcrit_array[ind] = params_in[param]

        elif param in ["alpha1_1", "alpha1_2"]:

            if param == "alpha1_2":
                alpha1[1] = params_in[param]
            elif param == "alpha1_1":
                alpha1[0] = params_in[param]

        elif param in ["acc_lim_1", "acc_lim_2"]:

            if param == "acc_lim_2":
                acc_lim[1] = params_in[param]
            elif param == "acc_lim_1":
                acc_lim[0] = params_in[param]

        else:
            flags[param] = params_in[param]


    if np.any(qcrit_array != 0.0):
        flags["qcrit_array"] = qcrit_array
    if np.any(natal_kick != 0.0):
        flags["natal_kick_array"] = natal_kick
    if np.any(alpha1 != 0.0):
        flags["alpha1"] = alpha1
    if np.any(acc_lim != 0.0):
        flags["acc_lim"] = acc_lim
    return flags


def set_evolvebin_flags(flags):
    ''' Set the COSMIC flags in the _evolvebin module.

    Parameters
    ----------
    flags : dict
        Dictionary of flags to set in _evolvebin.

    Returns
    -------
    None
    '''
    _evolvebin.windvars.neta = flags["neta"]
    _evolvebin.windvars.bwind = flags["bwind"]
    _evolvebin.windvars.hewind = flags["hewind"]
    _evolvebin.cevars.alpha1 = flags["alpha1"]
    _evolvebin.cevars.lambdaf = flags["lambdaf"]
    _evolvebin.ceflags.ceflag = flags["ceflag"]
    _evolvebin.flags.tflag = flags["tflag"]
    _evolvebin.flags.ifflag = flags["ifflag"]
    _evolvebin.flags.wdflag = flags["wdflag"]
    _evolvebin.flags.rtmsflag = flags["rtmsflag"]
    _evolvebin.snvars.pisn = flags["pisn"]
    _evolvebin.flags.bhflag = flags["bhflag"]
    _evolvebin.flags.remnantflag = flags["remnantflag"]
    _evolvebin.ceflags.cekickflag = flags["cekickflag"]
    _evolvebin.ceflags.cemergeflag = flags["cemergeflag"]
    _evolvebin.ceflags.cehestarflag = flags["cehestarflag"]
    _evolvebin.flags.grflag = flags["grflag"]
    _evolvebin.flags.bhms_coll_flag = flags["bhms_coll_flag"]
    _evolvebin.snvars.mxns = flags["mxns"]
    _evolvebin.points.pts1 = flags["pts1"]
    _evolvebin.points.pts2 = flags["pts2"]
    _evolvebin.points.pts3 = flags["pts3"]
    _evolvebin.snvars.ecsn = flags["ecsn"]
    _evolvebin.snvars.ecsn_mlow = flags["ecsn_mlow"]
    _evolvebin.flags.aic = flags["aic"]
    _evolvebin.ceflags.ussn = flags["ussn"]
    _evolvebin.snvars.sigma = flags["sigma"]
    _evolvebin.snvars.sigmadiv = flags["sigmadiv"]
    _evolvebin.snvars.bhsigmafrac = flags["bhsigmafrac"]
    _evolvebin.snvars.polar_kick_angle = flags["polar_kick_angle"]
    _evolvebin.snvars.natal_kick_array = flags["natal_kick_array"]
    _evolvebin.cevars.qcrit_array = flags["qcrit_array"]
    _evolvebin.mtvars.don_lim = flags["don_lim"]
    _evolvebin.mtvars.acc_lim = flags["acc_lim"]
    _evolvebin.windvars.beta = flags["beta"]
    _evolvebin.windvars.xi = flags["xi"]
    _evolvebin.windvars.acc2 = flags["acc2"]
    _evolvebin.windvars.epsnov = flags["epsnov"]
    _evolvebin.windvars.eddfac = flags["eddfac"]
    _evolvebin.windvars.gamma = flags["gamma"]
    _evolvebin.flags.bdecayfac = flags["bdecayfac"]
    _evolvebin.magvars.bconst = flags["bconst"]
    _evolvebin.magvars.ck = flags["ck"]
    _evolvebin.flags.windflag = flags["windflag"]
    _evolvebin.flags.qcflag = flags["qcflag"]
    _evolvebin.flags.eddlimflag = flags["eddlimflag"]
    _evolvebin.tidalvars.fprimc_array = flags["fprimc_array"]
    _evolvebin.rand1.idum1 = flags["randomseed"]
    _evolvebin.flags.bhspinflag = flags["bhspinflag"]
    _evolvebin.snvars.bhspinmag = flags["bhspinmag"]
    _evolvebin.mixvars.rejuv_fac = flags["rejuv_fac"]
    _evolvebin.flags.rejuvflag = flags["rejuvflag"]
    _evolvebin.flags.htpmb = flags["htpmb"]
    _evolvebin.flags.st_cr = flags["ST_cr"]
    _evolvebin.flags.st_tide = flags["ST_tide"]
    _evolvebin.snvars.rembar_massloss = flags["rembar_massloss"]
    _evolvebin.metvars.zsun = flags["zsun"]
    _evolvebin.snvars.kickflag = flags["kickflag"]
    _evolvebin.se_flags.using_metisse = 0
    _evolvebin.se_flags.using_sse = 1

    return None

def evolv2(params_in, params_out, fixed_params):
    ''' Evolve a binary with COSMIC given initial parameters with
    a direct call to the _evolvebin module.

    Parameters
    ----------
    params_in : dict
        Dictionary of input parameters to evolve. Keys are parameter names, values are parameter values.
    params_out : list   
        List of parameter names to return from the final state of the binary
    fixed_params : dict
        Dictionary of fixed parameters that should not be sampled but should be fixed. Keys are parameter names, values are parameter values
    
    Returns
    -------
    out : tuple
        Tuple of (final_state, bpp, kick_info)
        final_state : pd.DataFrame
            DataFrame of the final state of the binary with columns specified in params_out.
        bpp : np.ndarray
            Array of the binary population parameters (BPP) from COSMIC - format for us to work with Nautilus.
        kick_info : np.ndarray
            Array of the kick information from COSMIC - format for us to work with Nautilus - format for us to work with Nautilus.
    '''
    # handle initial binary parameters first
    m1 = params_in["m1"] 
    q = params_in["q"]
    m2 = q*m1
    tb = 10**params_in["logtb"] 
    if "e" in params_in:
        e = params_in["e"]
    else:
        e = 0.0
    metallicity = 10**params_in['logZ']
    # set the other flags
    flags = set_flags(params_in, fixed_params)
    _ = set_evolvebin_flags(flags)

    bpp_columns = BPP_COLUMNS
    bcm_columns = BCM_COLUMNS
    
    col_inds_bpp = np.zeros(len(ALL_COLUMNS), dtype=int)
    col_inds_bpp[:len(bpp_columns)] = [ALL_COLUMNS.index(col) + 1 for col in bpp_columns]
    n_col_bpp = len(BPP_COLUMNS)    

    col_inds_bcm = np.zeros(len(ALL_COLUMNS), dtype=int)
    col_inds_bcm[:len(bcm_columns)] = [ALL_COLUMNS.index(col) + 1 for col in bcm_columns]
    n_col_bcm = len(BCM_COLUMNS)
    
    _evolvebin.col.n_col_bpp = n_col_bpp
    _evolvebin.col.col_inds_bpp = col_inds_bpp
    _evolvebin.col.n_col_bcm = n_col_bcm
    _evolvebin.col.col_inds_bcm = col_inds_bcm
    
    # setup the inputs for _evolvebin
    zpars = np.zeros(20)
    mass = np.array([m1,m2])
    mass0 = np.array([m1,m2])
    epoch = np.array([0.0,0.0])
    ospin = np.array([0.0,0.0])
    tphysf = 13700.0
    dtp = 0.0
    rad = np.array([0.0,0.0])
    lumin = np.array([0.0,0.0])
    massc = np.array([0.0,0.0])
    radc = np.array([0.0,0.0])
    menv = np.array([0.0,0.0])
    renv = np.array([0.0,0.0])
    B_0 = np.array([0.0,0.0])
    bacc = np.array([0.0,0.0])
    tacc = np.array([0.0,0.0])
    tms = np.array([0.0,0.0])
    bhspin = np.array([0.0,0.0])
    tphys = 0.0
    bkick = np.zeros(20)
    bpp_index_out = 0
    bcm_index_out = 0
    kick_info_out = np.zeros(34)
    kstar = np.array([1,1])
    kick_info = np.zeros((2, 18))


    [_, bpp_index, bcm_index, kick_info_arrays] = _evolvebin.evolv2(kstar,mass,tb,e,metallicity,tphysf,
                                                          dtp,mass0,rad,lumin,massc,radc,
                                                          menv,renv,ospin,B_0,bacc,tacc,epoch,tms,
                                                          bhspin,tphys,zpars,bkick,kick_info)
    
    bpp = _evolvebin.binary.bpp[:25, :n_col_bpp].copy()
    _evolvebin.binary.bpp[:25, :n_col_bpp] = np.zeros(bpp.shape)
    bcm = _evolvebin.binary.bcm[:bcm_index, :n_col_bcm].copy()
    _evolvebin.binary.bcm[:bcm_index, :n_col_bcm] = np.zeros(bcm.shape)
    
    
    bpp = pd.DataFrame(bpp, columns=BPP_COLUMNS)   
    bpp = bpp[cols_keep] 
    kick_info = pd.DataFrame(kick_info_arrays,
                             columns=KICK_COLUMNS,
                             index=kick_info_arrays[:, -1].astype(int))
    
    out = bpp.loc[((bpp.kstar_1 == 14) & (bpp.kstar_2 == 14) & (bpp.evol_type == 3)) |
                  ((bpp.kstar_1 == 14) & (bpp.kstar_2 == 14) & (bpp.evol_type == 3))]
    
    
    
    if len(out) > 0:
        return out[params_out].iloc[0], bpp.to_numpy(), kick_info.to_numpy()
    else:
        return None, None, None



#def evolv2_fixed_kicks(m1, q, logtb, e, alpha_1, alpha_2, acc_lim_1, acc_lim_2, qHG, logZ):
#    vk1 = 0.0
#    theta1 = 0.0
#    phi1 = 0.0
#    omega1 = 0.0
#    vk2 = 0.0
#    theta2 = 0.0
#    phi2 = 0.0
#    omega2 = 0.0
#
#    return evolv2(m1, q, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, logZ)
#
#def evolv2_fixed_kicks_minimal(m1, q, logtb, e, alpha, acc_lim, qHG, logZ):
#    alpha_1 = alpha
#    alpha_2 = alpha
#    acc_lim_1 = acc_lim
#    acc_lim_2 = acc_lim
#    vk1 = 0.0
#    theta1 = 0.0
#    phi1 = 0.0
#    omega1 = 0.0
#    vk2 = 0.0
#    theta2 = 0.0
#    phi2 = 0.0
#    omega2 = 0.0
#    return evolv2(m1, q, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, logZ)
#
def evolv2_lowmass_secondary(m1, q, logtb, e, alpha_1, alpha_2, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, logZ):
    vk1 = 0.0
    theta1 = 0.0
    phi1 = 0.0
    omega1 = 0.0
    
    qHG = 3.0
    return evolv2(m1, q, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, logZ)
#
#def evolv2_lowmass_secondary_minimal(m1, q, logtb, e, alpha, vk2, theta2, phi2, omega2, acc_lim, logZ):
#    vk1 = 0.0
#    alpha_1 = alpha
#    alpha_2 = alpha
#    acc_lim_1 = acc_lim
#    acc_lim_2 = acc_lim
#    theta1 = 0.0
#    phi1 = 0.0
#    omega1 = 0.0
#    e=0.0
#    
#    qHG = 0.0
#
#    return evolv2(m1, q, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, logZ)
#
#
#def evolv2_vk2_alpha_lambda(m1, q, logtb, e, alpha, vk2, theta2, phi2, acc_lim, qHG, logZ):
#    vk1 = 0.0
#    alpha_1 = alpha
#    alpha_2 = alpha
#    acc_lim_1 = acc_lim
#    acc_lim_2 = acc_lim
#    theta1 = 0.0
#    phi1 = 0.0
#    omega1 = 0.0
#    omega2 = 0.0
#    return evolv2(m1, q, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, logZ)
#
#
def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

m1lo = 50.0
m2lo = 10.0
tblo = 1.0
elo = 0.0
alphalo_1 = 0.1
alphalo_2 = 0.1
vklo = 0.0
thetalo = 0.0
philo = -90.0
omegalo = 0.0
acc_limlo_1 = 0.0
acc_limlo_2 = 0.0
qc_kstar2lo = 0.5
Zlo = 0.0001

m1hi = 150.0
m2hi = 60.0
tbhi = 5000.0
ehi = 0.9
alphahi_1 = 20.0
alphahi_2 = 20.0
vkhi = 500.0
thetahi = 360.0
phihi = 90.0
omegahi = 360
acc_limhi_1 = 1.0
acc_limhi_2 = 1.0
qc_kstar2hi = 10.0
Zhi = 0.03

qlo = 0.01
qhi = 1


labels_dict = {"backpop" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha_1$',r'$\alpha_2$',
                            r'$v_1$',r'$\theta_1$',r'$\phi_1$',r'$\omega_1$',r'$v_2$',r'$\theta_2$',
                            r'$\phi_2$',r'$\omega_2$',r'$f_{\rm lim,1}$', r'$f_{\rm lim,2}$',
                            r'$q_{\rm HG}$', r'$\log_{10}Z$'],

               "backpop_fixed_kicks" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha_1$',r'$\alpha_2$',
                                        r'$f_{\rm lim,1}$',r'$f_{\rm lim,2}$',r'$q_{\rm HG}$',r'$\log_{10}Z$'],
               
               "backpop_fixed_kicks_minimal" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha$',
                                                r'$f_{\rm lim}$',r'$q_{\rm HG}$',r'$\log_{10}Z$'],
               
               "backpop_lowmass_secondary" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$\alpha_1$',r'$\alpha_2$',
                                              r'$v_2$',r'$\theta_2$',r'$\phi_2$',r'$\omega_2$',r'$f_{\rm lim,1}$',
                                              r'$f_{\rm lim,2}$',r'$\log_{10}Z$'],

               "backpop_lowmass_secondary_minimal" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha$',
                                              r'$v_2$',r'$\theta_2$',r'$\phi_2$',r'$f_{\rm lim}$',
                                              r'$q_{\rm HG}$', r'$\log_{10}Z$']
              }

def get_backpop_config(config_name):
    if (config_name == "backpop"):
        lower_bound = np.array([m1lo, qlo, np.log10(tblo), alphalo_1, alphalo_2, vklo, thetalo, philo, omegalo, vklo, thetalo, philo, omegalo, acc_limlo_1, acc_limlo_2, np.log10(Zlo)])
        upper_bound = np.array([m1hi, qhi, np.log10(tbhi), alphahi_1, alphahi_2, vkhi, thetahi, phihi, omegahi, vkhi, thetahi, phihi, omegahi, acc_limhi_1, acc_limhi_2, np.log10(Zhi)])
        params_in = ['m1', 'q', 'logtb', 'alpha_1', 'alpha_2', 'vk1', 'theta1', 'phi1', 'omega1', 'vk2', 'theta2', 'phi2', 'omega2', 'acc_lim_1', 'acc_lim_2', 'logZ']
        fixed_params = {}

    if (config_name == "backpop_fixed_kicks"):
        lower_bound = np.array([m1lo, qlo, np.log10(tblo), elo, alphalo_1, alphalo_2, acc_limlo_1, acc_limlo_2, qc_kstar2lo, np.log10(Zlo)])
        upper_bound = np.array([m1hi, qhi, np.log10(tbhi), ehi, alphahi_1, alphahi_2, acc_limhi_1, acc_limhi_2, qc_kstar2hi, np.log10(Zhi)])
        params_in = ['m1', 'q', 'logtb', 'e', 'alpha_1', 'alpha_2', 'acc_lim_1', 'acc_lim_2', 'qHG', 'logZ']
        fixed_params = {'vk1': 0.0, 'theta1': 0.0, 'omega1': 0.0, 'phi1': 0.0, 'vk2': 0.0, 'theta2': 0.0, 'phi2': 0.0, 'omega2': 0.0}
        
    if (config_name == "backpop_fixed_kicks_minimal"):
        ### KB hasn't figured out how to get the alpha_sampling right here
        lower_bound = np.array([m1lo, qlo, np.log10(tblo), elo, alphalo_1, acc_limlo_1, qc_kstar2lo, np.log10(Zlo)])
        upper_bound = np.array([m1hi, qhi, np.log10(tbhi), ehi, alphahi_1, acc_limhi_1, qc_kstar2hi, np.log10(Zhi)])
        params_in = ['m1', 'q', 'logtb', 'e', 'alpha_1', 'acc_lim_1', 'qHG', 'logZ']
        fixed_params = {'vk1': 0.0, 'theta1': 0.0, 'omega1': 0.0, 'phi1': 0.0, 'vk2': 0.0, 'theta2': 0.0, 'phi2': 0.0, 'omega2': 0.0}


    if (config_name == "backpop_lowmass_secondary"):
        lower_bound = np.array([m1lo, qlo, np.log10(tblo), alphalo_1, alphalo_2, vklo, thetalo, philo, omegalo, acc_limlo_1, acc_limlo_2, np.log10(Zlo)])
        upper_bound = np.array([m1hi, qhi, np.log10(tbhi), alphahi_1, alphahi_2, vkhi, thetahi, phihi, omegahi, acc_limhi_1, acc_limhi_2, np.log10(Zhi)])
        params_in = ['m1', 'q', 'logtb', 'alpha_1', 'alpha_2', 'vk2', 'theta2', 'phi2', 'omega2', 'acc_lim_1', 'acc_lim_2', 'logZ']
        fixed_params = {'vk1': 0.0, 'theta1': 0.0, 'omega1': 0.0, 'phi1': 0.0}

        
    if (config_name == "backpop_lowmass_secondary_minimal"):
        ### KB hasn't figured out how to get the alpha_sampling right here
        lower_bound = np.array([m1lo, qlo, np.log10(tblo), alphalo_1, vklo, thetalo, philo, omegalo, acc_limlo_1, np.log10(Zlo)])
        upper_bound = np.array([m1hi, qhi, np.log10(tbhi), alphahi_1, vkhi, thetahi, phihi, omegahi, acc_limhi_1, np.log10(Zhi)])  
        params_in = ['m1', 'q', 'logtb', 'alpha_1', 'vk2', 'theta2', 'phi2', 'omega2', 'acc_lim_1', 'logZ']
        fixed_params = {'vk1': 0.0, 'theta1': 0.0, 'omega1': 0.0, 'phi1': 0.0}


        
    return lower_bound, upper_bound, params_in, fixed_params


# Cosmo routines
Om0Planck = Planck15.Om0
H0Planck = Planck15.H0.value
speed_of_light = constants.c.to('km/s').value

zMax = 100
zgrid = np.expm1(np.linspace(np.log(1), np.log(zMax+1), 10000))
dLgrid = Planck15.luminosity_distance(zgrid).to('Mpc').value
tgrid = Planck15.lookback_time(zgrid).to('Myr').value

tofdL = interp1d(dLgrid,13700-tgrid,bounds_error=False,fill_value=1e100)
dLoft = interp1d(13700-tgrid,dLgrid,bounds_error=False,fill_value=1e100)
ddLdt = interp1d(tgrid,np.gradient(dLgrid,tgrid),bounds_error=False,fill_value=1e100)

zofdL = interp1d(dLgrid,zgrid)
dLofz = interp1d(zgrid,dLgrid,bounds_error=False,fill_value=1e100)
ddLdz = interp1d(zgrid,np.gradient(dLgrid,zgrid),bounds_error=False,fill_value=1e100)

zoft = interp1d(13700-tgrid,zgrid,bounds_error=False,fill_value=1000)
tofz = interp1d(zgrid,13700-tgrid,bounds_error=False,fill_value=1e100)
dtdz = interp1d(zgrid,np.gradient(13700-tgrid,zgrid),bounds_error=False,fill_value=1e100)

def get_190814_data(path, outdir="./"):
    ''' Fetch the GW190814 data from the GWTC-2 catalog using pesummary 
    
    Parameters
    ----------
    path : str
        The path (including filename) to save the data to.
    outdir : str, optional
        The directory to save the file to. Default is the current directory. 
        
    Returns
    -------
    data : pesummary.gw.fileio.GWFile
        The GW190814 data.
    '''

    data = fetch_open_samples(
    "GW190814", outdir=outdir, path=path
    )

    return data


def load_data(samples_path, weights=True):
    ''' Load the GW190814 data from the given path and return the KDE and samples.
    
    Parameters
    ----------
    samples_path : str
        The path to the GW190814 samples file.
    weights : bool, optional
        Whether to use weights for the KDE. Default is True.
        
    Returns
    -------
    KDE : scipy.stats.gaussian_kde
        The KDE of the GW190814 data.
    gwsamples : np.ndarray
        The original GW190814 samples.
    gwsamples_kde : np.ndarray
        The resampled GW190814 samples from the KDE.
    qmin : float
        The minimum mass ratio of the GW190814 samples.
    qmax : float
        The maximum mass ratio of the GW190814 samples.
    mcmin : float
        The minimum chirp mass of the GW190814 samples.
    mcmax : float
        The maximum chirp mass of the GW190814 samples.
    '''
    data = read(samples_path, package="gw")

    m1det = data.samples_dict['C01:Mixed']['mass_1']
    m2det = data.samples_dict['C01:Mixed']['mass_2']
    dL = data.samples_dict['C01:Mixed']['luminosity_distance']

    redshift = zofdL(dL)
    m1s = m1det/(1+redshift)
    m2s = m2det/(1+redshift)
    mcs = (m1s*m2s)**(3/5)/(m1s + m2s)**(1/5)
    Ms = m1s + m2s
    qs = m2s/m1s
    
    # 0.08134597870775964 0.14305660510532858 6.021922663985887 6.203484262510087
    qmin, qmax = az.hdi(qs,0.999)
    mcmin, mcmax = az.hdi(mcs,0.999)
    print(qmin,qmax,mcmin,mcmax)
    
    if weights is True:
        wts = 1/(dL**2*(1+redshift)**2*ddLdz(redshift)*m1s**2/mcs)
    else:
        wts = np.ones(len(m1s))

    wts = wts/np.sum(wts)

    gwsamples = np.column_stack([mcs,qs])
    KDE = gaussian_kde(gwsamples.T, weights=wts)
    
    gwsamples_kde = KDE.resample(len(gwsamples)).T

    return KDE, gwsamples, gwsamples_kde, qmin, qmax, mcmin, mcmax
    
