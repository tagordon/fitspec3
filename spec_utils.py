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

def build_mask_canon(time, flux, fixed_params, mask_buffer=50):

    t0, r, per, a, inc, e, w = fixed_params[2:-1]

    b = a * np.cos(inc) * (1 - e**2) / (1 + e * np.sin(w))
    C = np.sqrt((1 + r)**2 - b**2)
    out_mask = sigma_clip(flux - gaussian_filter1d(flux, 50), sigma=3).mask

    dur = per / np.pi * np.arcsin(
        C / (a * np.sin(inc))
    ) * np.sqrt(1 - e**2) / (1 + e * np.sin(w))

    start_mask = np.argmin(np.abs(time - (t0 - 0.5 * dur))) - mask_buffer
    end_mask = np.argmin(np.abs(time - (t0 + 0.5 * dur))) + mask_buffer
    
    trans_mask = np.zeros_like(flux, dtype=np.bool_)
    trans_mask[start_mask:end_mask] = True
    return out_mask | trans_mask

def build_mask(time, flux, fixed_params, mask_buffer=50):

    u1, u2, t0, r, per, a, V, C, Ls, Lc, S, w0 = fixed_params
    t0, r, per, a, inc, e, w = canon_from_eastman(*fixed_params[2:-1])

    out_mask = sigma_clip(flux - gaussian_filter1d(flux, 50), sigma=3).mask

    dur = per / np.pi * np.arcsin(
        C / (a * np.sin(inc))
    ) * np.sqrt(1 - e**2) / (1 + e * np.sin(w))

    #start_mask = np.where(
    #    np.isclose(time, t0 - 0.5 * dur, atol=2 * (time[1]-time[0]))
    #)[0][0] - mask_buffer
    start_mask = np.argmin(np.abs(time - (t0 - 0.5 * dur))) - mask_buffer

    #print(t0 + 0.5 * dur, 2 * (time[1] - time[0]))
    #end_mask = np.where(
    #    np.isclose(time, t0 + 0.5 * dur, atol=2 * (time[1]-time[0]))
    #)[0][-1] + mask_buffer
    end_mask = np.argmin(np.abs(time - (t0 + 0.5 * dur))) + mask_buffer
    
    trans_mask = np.zeros_like(flux, dtype=np.bool_)
    trans_mask[start_mask:end_mask] = True
    return out_mask | trans_mask

def get_initial_params_canon(
    time,
    flux, 
    fixed_params,
    detrending_vectors=None, 
    polyorder=1,
    gp=True,
    fix_gp_timescale=True
):

    u1, u2, t0, r_wlc, per, a, inc, e, w, lw0 = fixed_params
    mask = build_mask_canon(time, flux, fixed_params)
    coeffs, fit = gls_fit(
        time, 
        flux, 
        vectors=detrending_vectors, 
        mask=mask, 
        polyorder=polyorder, 
        return_coeffs=True
    )
    
    err_guess = np.std(flux[~mask] - fit[~mask])
    lsigma_guess = np.log((err_guess/100)**2)

    if not gp:
        noise_params = [err_guess]
    elif gp & fix_gp_timescale:
        noise_params = [err_guess, lsigma_guess]
    else:
        noise_params = [err_guess, lsigma_guess, lw0]

    params = np.concatenate([[r_wlc, u1, u2], noise_params, coeffs])

    return params

def get_initial_params(
    time,
    flux, 
    fixed_params,
    detrending_vectors=None, 
    polyorder=1,
    gp=True,
    fix_gp_timescale=True
):
        
    u1, u2, t0, r_wlc, per, a, V, C, Ls, Lc, S, lw0 = fixed_params
    mask = build_mask(time, flux, fixed_params)
    coeffs, fit = gls_fit(
        time, 
        flux, 
        vectors=detrending_vectors, 
        mask=mask, 
        polyorder=polyorder, 
        return_coeffs=True
    )
    
    err_guess = np.std(flux[~mask] - fit[~mask])
    lsigma_guess = np.log((err_guess/100)**2)

    if not gp:
        noise_params = [err_guess]
    elif gp & fix_gp_timescale:
        noise_params = [err_guess, lsigma_guess]
    else:
        noise_params = [err_guess, lsigma_guess, lw0]

    params = np.concatenate([[r_wlc, u1, u2], noise_params, coeffs])
        
    return params

def build_logp_canon(
    time,
    flux, 
    fixed_params,
    stellar_params,
    start_wav,
    end_wav,
    detrending_vectors=None, 
    polyorder=1, 
    gp=True,
    fix_gp_timescale=True,
):

    if detrending_vectors is None:
        ncoeffs = 1 + polyorder
    else:
        ncoeffs = 1 + polyorder + len(detrending_vectors.T)

    u1_prior, u2_prior = get_ld_priors(start_wav, end_wav, stellar_params)

    #u1_wlc, u2_wlc, t0, r_wlc, per, a, V, C, Ls, Lc, S, lw0 = fixed_params
    #t0, r_wlc, per, a, inc, e, w = canon_from_eastman(*fixed_params[2:-1])
    u1_wlc, u2_wlc, t0, r_wlc, per, a, inc, ecc, w, lw0 = fixed_params
    t0, r_wlc, per, a, inc, ecc, w = fixed_params[2:-1]

    def log_prob_gp(p):

        r = p[0]
        u1, u2 = p[1:3]
        err = p[3]
        lsigma = p[4]
        coeffs = p[5:ncoeffs + 5]
        f = coeffs[0]
        trend = get_trend_model(time, detrending_vectors, coeffs[1:], polyorder)

        try:

            term = celerite2.terms.SHOTerm(sigma=np.exp(lsigma), w0=np.exp(lw0), Q=1/np.sqrt(2))
            gp = celerite2.GaussianProcess(term)
            
            gp.compute(time, diag=err**2)
            #mu, jac = reparam(time, [u1, u2, t0, r, per, a, V, C, Ls, Lc, S])
            mu = 1 + keplerian_transit(
                time, u1, u2, t0, r, per, a, inc, ecc, w
            )

            mu = mu * f + trend
            
            ll = gp.log_likelihood(flux - mu)
            #ll -= np.log(jac)
    
            pr = u1_prior.prior(u1)
            pr += u2_prior.prior(u2)
            pr += uniform_prior(0, 10).prior(lsigma)
            pr += uniform_prior(0, 1).prior(r)
        
            if np.isfinite(ll) & (err > 0):
                return ll + pr
            else:
                return -np.inf

        except Exception as e:
            return -np.inf

    def log_prob_no_gp(p):

        r = p[0]
        u1, u2 = p[1:3]
        err = p[3]
        coeffs = p[4:ncoeffs + 4]
        f = coeffs[0]
        trend = get_trend_model(time, detrending_vectors, coeffs[1:], polyorder)

        try:

        #mu, jac = reparam(time, [u1, u2, t0, r, per, a, V, C, Ls, Lc, S])
            mu = 1 + keplerian_transit(
                time, u1, u2, t0, r, per, a, inc, ecc, w
            )
            mu = mu * f + trend
            
            ll = log_likelihood(flux, mu, err)
            #ll -= np.log(jac)
    
            pr = u1_prior.prior(u1)
            pr += u2_prior.prior(u2)
            pr += uniform_prior(0, 1).prior(r)
    
            if np.isfinite(ll) & (err > 0):
                return ll + pr
            else:
                return -np.inf

        except Exception as e:
            return -np.inf

    if gp:
        return log_prob_gp
    else:
        return log_prob_no_gp

def build_logp(
    time,
    flux, 
    fixed_params,
    stellar_params,
    start_wav,
    end_wav,
    detrending_vectors=None, 
    polyorder=1, 
    gp=True,
    fix_gp_timescale=True,
):

    if detrending_vectors is None:
        ncoeffs = 1 + polyorder
    else:
        ncoeffs = 1 + polyorder + len(detrending_vectors.T)

    u1_prior, u2_prior = get_ld_priors(start_wav, end_wav, stellar_params)

    u1_wlc, u2_wlc, t0, r_wlc, per, a, V, C, Ls, Lc, S, lw0 = fixed_params
    t0, r_wlc, per, a, inc, e, w = canon_from_eastman(*fixed_params[2:-1])

    def log_prob_vary_w0(p):

        r = p[0]
        u1, u2 = p[1:3]
        err = p[3]
        lsigma = p[4]
        lw0 = p[5]
        coeffs = p[6:ncoeffs + 6]
        f = coeffs[0]
        trend = get_trend_model(time, detrending_vectors, coeffs[1:], polyorder)

        try:

            term = celerite2.terms.SHOTerm(sigma=np.exp(lsigma), w0=np.exp(lw0), Q=1/np.sqrt(2))
            gp = celerite2.GaussianProcess(term)
            
            gp.compute(time, diag=err**2)
            mu, jac = reparam(time, [u1, u2, t0, r, per, a, V, C, Ls, Lc, S])
            mu = mu * f + trend
            
            ll = gp.log_likelihood(flux - mu)
            ll -= np.log(jac)
                
            pr = u1_prior.prior(u1)
            pr += u2_prior.prior(u2)
            pr += uniform_prior(lsigma, 0, 6)
            pr += uniform_prior(
                lw0, 
                np.log(2 * np.pi * 5 / (time[-1] - time[0])), 
                np.log(2 * np.pi * 50 / (time[-1] - time[0]))
            )
            pr += uniform_prior(0, 1).prior(r)
    
            if np.isfinite(ll) & (err > 0):
                return ll + pr
            else:
                return -np.inf

        except Exception as e:
            return -np.inf

    def log_prob_gp(p):

        r = p[0]
        u1, u2 = p[1:3]
        err = p[3]
        lsigma = p[4]
        coeffs = p[5:ncoeffs + 5]
        f = coeffs[0]
        trend = get_trend_model(time, detrending_vectors, coeffs[1:], polyorder)

        try:

            term = celerite2.terms.SHOTerm(sigma=np.exp(lsigma), w0=np.exp(lw0), Q=1/np.sqrt(2))
            gp = celerite2.GaussianProcess(term)
            
            gp.compute(time, diag=err**2)
            mu, jac = reparam(time, [u1, u2, t0, r, per, a, V, C, Ls, Lc, S])

            mu = mu * f + trend
            
            ll = gp.log_likelihood(flux - mu)
            ll -= np.log(jac)
    
            pr = u1_prior.prior(u1)
            pr += u2_prior.prior(u2)
            pr += uniform_prior(0, 10).prior(lsigma)
            pr += uniform_prior(0, 1).prior(r)
        
            if np.isfinite(ll) & (err > 0):
                return ll + pr
            else:
                return -np.inf

        except Exception as e:
            return -np.inf

    def log_prob_no_gp(p):

        r = p[0]
        u1, u2 = p[1:3]
        err = p[3]
        coeffs = p[4:ncoeffs + 4]
        f = coeffs[0]
        trend = get_trend_model(time, detrending_vectors, coeffs[1:], polyorder)

        try:

            mu, jac = reparam(time, [u1, u2, t0, r, per, a, V, C, Ls, Lc, S])
            mu = mu * f + trend
            
            ll = log_likelihood(flux, mu, err)
            ll -= np.log(jac)

            pr = u1_prior.prior(u1)
            pr += u2_prior.prior(u2)
            pr += uniform_prior(0, 1).prior(r)
    
            if np.isfinite(ll) & (err > 0):
                return ll + pr
            else:
                return -np.inf

        except Exception as e:
            return -np.inf

    if gp:
        if fix_gp_timescale:
            return log_prob_gp
        else:
            return log_prob_vary_w0
    else:
        return log_prob_no_gp

def run(
    time,
    spec, 
    wl_params,
    stellar_params,
    wav_bin_edges,
    out_mask=None,
    detrending_vectors=None, 
    detrending_cube=None,
    polyorder=1, 
    progress=True,
    nproc=1,
    samples=5000,
    save_posterior=True,
    print_initial_logp=False,
    gp=False,
    fix_gp_timescale=True
):

    fixed_params = np.concatenate([wl_params[-11:], [wl_params[2]]])

    if out_mask is None:
        out_mask = np.zeros_like(time)

    if len(out_mask.shape) == 1:
        out_mask = np.tile(out_mask, (1, spec.shape[1])).T

    def run_single_band(ind):

        mask = np.array(~out_mask[:, ind])

        if detrending_cube is not None:
            if detrending_vectors is not None:
                new_detrending_vectors = np.hstack([detrending_vectors[mask], detrending_cube[ind][mask]])
            else:
                new_detrending_vectors = detrending_cube[ind][mask]
        else: 
            if detrending_vectors is not None:
                new_detrending_vectors = detrending_vectors[mask]
            else:
                new_detrending_vectors = None

        params = get_initial_params(
            time[mask], 
            spec[mask, ind], 
            fixed_params, 
            polyorder=polyorder,
            detrending_vectors=new_detrending_vectors,
            fix_gp_timescale=fix_gp_timescale,
            gp=gp
        )
        
        log_prob = build_logp(
            time[mask], 
            spec[mask, ind], 
            fixed_params, 
            stellar_params,
            wav_bin_edges[ind],
            wav_bin_edges[ind + 1],
            polyorder=polyorder,
            detrending_vectors=new_detrending_vectors,
            gp=gp,
            fix_gp_timescale=fix_gp_timescale
        )

        if print_initial_logp:
            print(log_prob(params))
        
        pos = params + 1e-4 * np.random.randn(len(params)*2, len(params))
        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob
        )
        sampler.run_mcmc(
            pos, 
            samples, 
            progress=progress, 
            skip_initial_state_check=True
        );

        if save_posterior:
            
            return sampler

        else:
            
            chain = sampler.get_chain()
            means = np.mean(chain, axis=(0, 1))
            stds = np.std(chain, axis=(0, 1))
            return means, stds

    posterior = []
    means = []
    stds = []

    if nproc == 1:

        for i in range(spec.shape[1]):

            if save_posterior:
                sampler = run_single_band(i)
                posterior.append(sampler)
            else:
                m, std = run_single_band(i)
                means.append(m)
                stds.append(std)

    else:

        nbands = spec.shape[1]
        nbatches = nbands // nproc
        remainder = np.remainder(nbands, nproc)
        
        if remainder > 0:
            nbatches += 1
        
        band_inds = np.arange(nbands)
        batches = [band_inds[i * nproc:(i + 1) * nproc] for i in range(nbatches)]

        for b in batches:

            print('\r                                                            ', end='')
            print('\rrunning bands {}-{} of {}'.format(b[0] + 1, b[-1] + 1, nbands), end='')
            with Pool(nproc) as pool:
                
                samplers_or_means = pool.map(run_single_band, b)

            if save_posterior:
                posterior.append(samplers_or_means)
            else:
                means.append(samplers_or_means[0])
                stds.append(samplers_or_means[1])

        if save_posterior:
            posterior = np.concatenate(posterior)
        else:
            means = np.concatenate(means)
            stds = np.concatenate(stds)

    if save_posterior:
        return posterior
    else:
        return means, stds