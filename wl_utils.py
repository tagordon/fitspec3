import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.stats import sigma_clip
import celerite2
import os
import emcee
from multiprocess import Pool
import multiprocess.context as ctx
ctx._force_start_method('spawn')

from distributions import *
from transit import *
from utils import *

ld_data_path = os.environ['EXO_LD_PATH']
os.environ["OMP_NUM_THREADS"] = "1"

pl_param_names = [
    't0', 'radius', 'period',
    'semimajor_axis', 'inclination', 
    'eccentricity', 'periastron'
]

def get_inits(priors):
    '''
    Get the starting values for the transit 
    parameters from the priors dictionary by 
    returning an array of values taken from 
    prior.init for each parameter's prior with 
    the following order: 
    [
        t0, radius, period, 
        semimajor_axis, inclination, 
        eccentricity, periastron
    ]

    Args:
        priors (dict): a dictionary with the above keys 
        containing prior functions (defined in distributions.py) 
        for each parameter. 

    Returns: 
        Initial values for the parameters in the order given 
        above. 
    '''

    order_key = [ 
        't0',
        'radius', 
        'period',
        'semimajor_axis', 
        'inclination', 
        'eccentricity', 
        'periastron'
    ]

    d = priors.copy()
    del d['reference']
    keys = [k for k in d.keys()]
    values = [p.init for p in d.values()]
    if not order_key == keys:
        values = [values[gk] for gk in [order_key.index(k) for k in keys]]
    
    return values

def get_wav_bounds(detector):
    '''
    Get the starting and ending wavelengths 
    for either NRS1 or NRS2. These are currently set to: 
    nrs1: (2.87, 3.81)
    nrs2: (3.92, 5.27)

    Args:
        detector (string): either 'nrs1' or 'nrs2'

    Returns: 
        start, end: starting and ending wavelengths for 
        the given detector.
    '''

    if detector == 'nrs1':
        start_wav = 2.87
        end_wav = 3.81
    elif detector == 'nrs2':
        start_wav = 3.92
        end_wav = 5.27
    else:
        raise ValueError(
            '''
            Invalid detector string. 
            Detector must either be nrs1 or nrs2.
            '''
        )

    return start_wav, end_wav

def get_initial_transit_params(
    time,
    flux,
    param_priors, 
    stellar_params,
    detector,
    estimate_t0=True
):
    '''
    Get the initial set of transit parameters, which 
    are the init values from param_priors, the exception 
    being t0, which will be guessed from 
    the data if estimate_t0=True
    '''

    start_wav, end_wav = get_wav_bounds(detector)
    u, du = get_ld_params(
        start_wav, end_wav, stellar_params
    )

    canon_params = get_inits(param_priors)
    if estimate_t0:
        canon_params[0] = estimate_t0_func(time, flux)

    eastman_params = eastman_from_canon(*canon_params)
    return np.concatenate([[u[0], u[1]], eastman_params])
    

def get_initial_transit_model(
    time,
    flux,
    detector, 
    param_priors,
    stellar_params,
    detrending_vectors=None, 
    polyorder=1, 
    estimate_t0=True
):
    '''
    Returns an initial guess for the transit + systematics model 
    (i.e. the transit curve plus a polynomial of the specified degree plus 
    a generalized least-squares fit to any detrending_vectors provided by 
    the user. The initial transit model is computed with the init values 
    from param_priors, the exception being t0, which will be guessed from 
    the data if estimate_t0=True.

    Args: 
        time (float): the time array corresponding to the flux measurements
        flux (float): the flux measurements 
        detector (string): nrs1 or nrs2
        param_priors (dict): dictionary of priors for the transit parameters 
        stellar_params (dict): dictionary of stellar parameters 
        (must contain metallicity, effective temperature, and log surface gravity)
        detrending_vectors (float, default=None): an array of vectors to detrend on 
        when conducting the inference. Size should be (NxM) where N is the length of 
        the time/flux arrays and M is the number of detrending vectors to be used. 
        polyorder (int, default=1): order of polynomial to add to systematics model. 
        estimate_t0 (bool, default=True): if True, then guess the transit time from 
        the flux. 
    '''
    
    mask = build_mask(flux)
    coeffs, fit = gls_fit(
        time, 
        flux, 
        vectors=detrending_vectors, 
        mask=mask, 
        polyorder=1, 
        return_coeffs=True
    )
    f = coeffs[0]
    p_east = get_initial_transit_params(
        time,
        flux,
        param_priors, 
        stellar_params,
        detector,
        estimate_t0=estimate_t0
    )
    
    return reparam(time, p_east)[0] * f + fit - f 

def build_mask(flux, mask_buffer=50, filter_window=50, out_sigma=3):
    '''
    Build a mask that masks the transit and any outliers. 

    Args:
        mask_buffer (int): the number of points before and after the 
        detected mid ingress/egress times to mask.
        filter_window (int): the window size for the gaussian filter used 
        in sigma clipping.
        out_sigma (int): the number of standard devations beyond 
        which outliers are clipped. 

    Returns: 
        A mask that is True when a point is either in-transit, an outlier, or both 
        and False otherwise. 
    '''

    out_mask = sigma_clip(
        flux - gaussian_filter1d(flux, filter_window), 
        sigma=out_sigma
    ).mask
    
    diffs = np.diff(gaussian_filter1d(flux, filter_window))
    start_mask = np.argmin(diffs) - mask_buffer
    end_mask = np.argmax(diffs) + mask_buffer
    trans_mask = np.zeros_like(flux, dtype=np.bool_)
    trans_mask[start_mask:end_mask] = True
    
    return out_mask | trans_mask

def _unpack_params(p, ncoeffs, gp=False):

    if gp:
        
        err = p[0]
        lsigma, lw0 = p[1:3]
        coeffs = p[3:ncoeffs + 3]
        f = coeffs[0]
        trans_params = p[3 + ncoeffs:]

        return err, lsigma, lw0, coeffs, f, trans_params
        
    else: 
        
        err = p[0]
        coeffs = p[1:ncoeffs + 1]
        f = coeffs[0]
        trans_params = p[1 + ncoeffs:]

        return err, coeffs, f, trans_params

def estimate_t0_func(time, flux):
    '''
    Estimate the center of transit time 
    from the flux. This is accomplished by 
    filtering the flux with a Gaussian filter and 
    a window length of 50 points, computing the finite 
    differences of the filtered timeseries, and 
    returning the midpoint between the maximum 
    and minimum of the finite differences. 

    Caution: This method is not foolproof. It 
    relies on the assumption that non-transit flux 
    variations on scales larger than 50 integrations 
    are small compared to the transit signal. 
    '''

    diffs = np.diff(gaussian_filter1d(flux, 50))
    start_mask = np.argmin(diffs)
    end_mask = np.argmax(diffs)
    return 0.5 * (time[start_mask] + time[end_mask])

def get_initial_params(
    time,
    flux, 
    detector, 
    param_priors,
    stellar_params,
    detrending_vectors=None, 
    polyorder=1,
    estimate_t0=True,
    gp=False
):
    '''
    Returns a set of initial parameters for the transit + systematics model 
    (i.e. the transit curve plus a polynomial of the specified degree plus 
    a generalized least-squares fit to any detrending_vectors provided by 
    the user. The transit parameters are the init values 
    from param_priors, the exception being t0, which will be guessed from 
    the data if estimate_t0=True.
    '''
        
    mask = build_mask(flux)
    coeffs, fit = gls_fit(
        time, 
        flux, 
        vectors=detrending_vectors, 
        mask=mask, 
        polyorder=polyorder, 
        return_coeffs=True
    )
    err_guess = np.std(flux[~mask] - fit[~mask])
    p_east = get_initial_transit_params(
        time,
        flux,
        param_priors, 
        stellar_params,
        detector,
        estimate_t0=estimate_t0
    )

    if gp: 
        lsigma_guess = np.log((err_guess/100)**2)
        lw0_guess = np.log(2 * np.pi * 10 / (time[-1] - time[0]))
        noise_params = [err_guess, lsigma_guess, lw0_guess]
    else:
        noise_params = [err_guess]

    params = np.concatenate([noise_params, coeffs, p_east])
        
    return params

def compute_priors(param_priors, params, u1_prior, u2_prior, ld_prior):
    '''
    Compute the log-value of the prior distribution at the given 
    parameters. 

    Args:
        param_priors (dict): the dictionary of priors 
        params (float): an array of parameter values in the 
        Eastman parameterization.
        u1_prior (distribution): prior distribution for u1
        u2_prior (distrbution): prior distribution for u2

    Returns:
        The log of the prior distribution. 
    '''

    u1, u2 = params[:2]
    pr = eastman_priors(*params[2:])
    canon_params = canon_from_eastman(*params[2:])
    #print(canon_params)
    #print(pl_param_names)
    for p, name in zip(canon_params, pl_param_names):
        #print(name, p, param_priors[name].prior(p))
        pr += param_priors[name].prior(p)

    if ld_prior:
        pr += u1_prior.prior(u1)
        pr += u2_prior.prior(u2)
    else:
        pr += uniform_prior(0, 1).prior(u1)
        pr += uniform_prior(0, 1).prior(u1)

    return pr
        

def build_logp(
    time,
    flux, 
    detector, 
    param_priors,
    stellar_params,
    detrending_vectors=None, 
    polyorder=1, 
    gp=False,
    ld_priors=True
):
    '''
    Builds the log-pobability function, which takes an array of parameters 
    and returns the log-probability (log(prior) + log(likelihood)) for the 
    model. 

    Args:
        time (float): array of times
        flux (float): array of fluxes 
        detector (string): either nrs1 or nrs2
        param_priors (dict): dictionary of priors for the transit parameters 
        stellar_params (dict): dictionary of stellar parameters 
        which must include metallicity (mh), effective temperature (teff), and 
        log-surface gravity (logg) 
        detrending_vectors (float): (float, default=None): an array of vectors to detrend on 
        when conducting the inference. Size should be (NxM) where N is the length of 
        the time/flux arrays and M is the number of detrending vectors to be used. 
        polyorder (int, default=1): order of polynomial to add to systematics model. 
        gp (bool, default=False): whether to include a gp model or not. 

    Returns: 
        A function that computes the log-probability of the model given a vector 
        of parameters. 
    '''

    if detrending_vectors is None:
        ncoeffs = 1 + polyorder
    else:
        ncoeffs = 1 + polyorder + len(detrending_vectors.T)
        
    start_wav, end_wav = get_wav_bounds(detector)
    u1_prior, u2_prior = get_ld_priors(start_wav, end_wav, stellar_params)

    def log_prob_gp(p):

        err, lsigma, lw0, coeffs, f, p = _unpack_params(p, ncoeffs, gp=True)
        trend = get_trend_model(time, detrending_vectors, coeffs[1:], polyorder)
        w0 = np.exp(lw0)

        try:

            term = celerite2.terms.SHOTerm(sigma=np.exp(lsigma), w0=w0, Q=1/np.sqrt(2))
            gp = celerite2.GaussianProcess(term)
            
            mu, jac = reparam(time, p)
            mu = mu * f + trend
            
            gp.compute(time, diag=err**2)
            ll = gp.log_likelihood(flux - mu) + np.log(jac)
    
            pr = compute_priors(
                param_priors, p, u1_prior, u2_prior, ld_priors
            )
            pr += uniform_prior(
                2 * np.pi * 0.5 / (time[-1] - time[0]),
                2 * np.pi * 20 / (time[-1] - time[0])
            ).prior(w0)
    
            if np.isfinite(ll) & (err > 0):
                return ll + pr
            else:
                return -np.inf

        except Exception as e:
            return -np.inf

    def log_prob(p):

        err, coeffs, f, p = _unpack_params(p, ncoeffs, gp=False)
        trend = get_trend_model(time, detrending_vectors, coeffs[1:], polyorder)
        
        try:

            mu, jac = reparam(time, p)
            mu = mu * f + trend
            ll = log_likelihood(flux, mu, err)
            ll += np.log(jac)
    
            pr = compute_priors(
                param_priors, p, u1_prior, u2_prior, ld_priors
            )
    
            if np.isfinite(ll) & (err > 0):
                return ll + pr
            else:
                return -np.inf
            
        except Exception as e:
            return -np.inf

    if gp:
        return log_prob_gp
    else:
        return log_prob

def run(
    time,
    flux, 
    detector, 
    param_priors,
    stellar_params,
    detrending_vectors=None, 
    polyorder=1, 
    samples=10_000,
    progress=True,
    nproc=1,
    gp=False,
    ld_priors=True
):
    '''
    Runs emcee on the model.

    Args:
        time (float): array of times
        flux (float): array of fluxes 
        detector (string): either nrs1 or nrs2
        param_priors (dict): dictionary of priors for the transit parameters 
        stellar_params (dict): dictionary of stellar parameters 
        which must include metallicity (mh), effective temperature (teff), and 
        log-surface gravity (logg) 
        detrending_vectors (float): (float, default=None): an array of vectors to detrend on 
        when conducting the inference. Size should be (NxM) where N is the length of 
        the time/flux arrays and M is the number of detrending vectors to be used. 
        polyorder (int, default=1): order of polynomial to add to systematics model. 
        samples (int, default=10000): the number of MCMC steps to run.
        progress (bool, default=True): whether or not to display a progress bar
        nproc (int, default=1): number of processors to use (set to 1 for no multiprocessing)
        gp (bool, default=False): whether to include a gp model or not

    Returns: 
        An emcee sampler object containing the results of the MCMC sampling. 
    '''

    params = get_initial_params(
        time,
        flux, 
        detector, 
        param_priors,
        stellar_params,
        detrending_vectors=detrending_vectors, 
        polyorder=polyorder, 
        gp=gp
    )

    log_prob = build_logp(
        time,
        flux, 
        detector, 
        param_priors,
        stellar_params,
        detrending_vectors=detrending_vectors, 
        polyorder=polyorder, 
        gp=gp,
        ld_priors=ld_priors
    )

    pos = params + 1e-4 * np.random.randn(len(params)*2, len(params))
    nwalkers, ndim = pos.shape
        
    if nproc == 1:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob
        )
        sampler.run_mcmc(
            pos, 
            samples, 
            progress=progress, 
            skip_initial_state_check=True
        );
        
    elif nproc > 1:
        with Pool(nproc) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_prob, pool=pool
            )
            sampler.run_mcmc(
                pos, 
                samples, 
                progress=progress, 
                skip_initial_state_check=True
            );
            
    return sampler