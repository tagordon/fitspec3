import os
import numpy as np
import decomp
from astropy.stats import sigma_clip
from scipy.ndimage import gaussian_filter1d
from wl_utils import *

def get_log_probs(times, fluxes, detectors, components, priors_dict, st_params_dict, polyorder=1, gp=False):

    lps = []
    for t, f, d, dv in zip(times, fluxes, detectors, components):
        lps.append(
            build_logp(
                t,
                f, 
                d, 
                priors_dict,
                st_params_dict,
                detrending_vectors=dv, 
                polyorder=polyorder, 
                gp=gp,
                ld_priors=True
            )
        )
        
    return lps

def get_joint_initial_params(
    times, 
    fluxes, 
    detectors, 
    param_priors, 
    stellar_params, 
    detrending_vectors, 
    polyorder,
    gp=False,
):

    if detrending_vectors is None:
        n_components = 0
        detrending_vectors = [None] * len(times)
    else:
        n_components = len(detrending_vectors[0].T)

    other_params = []
    for i, (t, f, d, dv) in enumerate(
        zip(times, fluxes, detectors, detrending_vectors)
    ):
        initial_params = get_initial_params(
            t,
            f, 
            d, 
            param_priors,
            stellar_params,
            detrending_vectors=dv, 
            polyorder=polyorder,
            estimate_t0=True,
            gp=gp
        )

        if gp:
            n_other = polyorder + 7 + n_components
        else:
            n_other = polyorder + 5 + n_components
            
        other_params.append(initial_params[:n_other])
        if i == 0:
            transit_params = initial_params[n_other:]

    return np.concatenate([np.concatenate(other_params), transit_params])
    

def get_joint_log_prob(lps, n_components, polyorder=1, gp=False):

    n_lcs = len(lps)

    # (1 + polyorder, err, gp_w0, gp_s0, u1, u2, r(?)) + decomp_coeffs
    if gp:
        n_op = polyorder + 7 + n_components
    else:
        n_op = polyorder + 5 + n_components
        
    n_other_params = n_op * n_lcs

    def log_prob(p):

        transit_params, other_params = p[n_other_params:], p[:n_other_params]

        total_prob = 0
        for i in range(n_lcs):
            op = other_params[:n_op]
            other_params = other_params[n_op:]
            total_prob += lps[i](np.concatenate([op, transit_params]))
        return total_prob

    return log_prob

def run(
    times,
    fluxes, 
    detectors, 
    param_priors,
    stellar_params,
    detrending_vectors=None, 
    polyorder=1, 
    samples=10_000,
    progress=True,
    nproc=1,
    gp=False,
    ld_priors=True,
):
    
    params = get_joint_initial_params(
        times, 
        fluxes, 
        detectors, 
        param_priors, 
        stellar_params, 
        detrending_vectors, 
        polyorder,
        gp=gp,
    )

    if detrending_vectors is None:
        n_components = 0
        detrending_vectors = [None] * len(times)
    else:
        n_components = len(detrending_vectors[0].T)
        
    lps = get_log_probs(
        times, 
        fluxes, 
        detectors, 
        detrending_vectors, 
        param_priors, 
        stellar_params, 
        polyorder=polyorder,
        gp=gp
    )
    log_prob = get_joint_log_prob(
        lps, 
        n_components, 
        polyorder, 
        gp=gp, 
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
