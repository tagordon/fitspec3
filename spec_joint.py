import os
import numpy as np
import decomp
from astropy.stats import sigma_clip
from scipy.ndimage import gaussian_filter1d
from spec_utils import *

def get_log_probs(
    times, 
    fluxes, 
    fixed_params, 
    detrending_vectors, 
    start_wav, 
    end_wav, 
    st_params_dict, 
    polyorder=1,
    gp=False
):

    lps = []
    for t, f, fp, dv in zip(times, fluxes, fixed_params, detrending_vectors):
        
        lps.append(
            build_logp_canon(
                t,
                f, 
                fp,
                st_params_dict,
                start_wav,
                end_wav,
                detrending_vectors=dv, 
                polyorder=polyorder, 
                gp=gp,
                fix_gp_timescale=True,
            )
        )
        
    return lps

def get_joint_initial_params(
    times, 
    fluxes, 
    fixed_params, 
    detrending_vectors, 
    polyorder=1,
    gp=False
):

    other_params = []
    for i, (t, f, fp, dv) in enumerate(
        zip(times, fluxes, fixed_params, detrending_vectors)
    ):
        initial_params = get_initial_params_canon(
            t,
            f, 
            fp, 
            detrending_vectors=dv, 
            polyorder=polyorder,
            gp=gp,
        )
            
        other_params.append(initial_params[3:])
        if i == 0:
            transit_params = initial_params[:3]

    # transit_params are r, u1, u2
    # other_params are [err_guess, (opt: lsigma_guess), (opt: lw0)], [p0, p1, ... pn, c0, c1, ... cn]
    # and then other_params is tiled for each individual transit. 
    return np.concatenate([np.concatenate(other_params), transit_params])
    
def get_joint_log_prob(lps, n_components, polyorder, gp=False, same_rad=False):

    n_lcs = len(lps)

    if gp:
        n_op = polyorder + 3 + n_components
    else:
        n_op = polyorder + 2 + n_components
        
    n_other_params = n_op * n_lcs

    def log_prob(p):

        transit_params, other_params = p[n_other_params:], p[:n_other_params]

        total_prob = 0
        for i in range(n_lcs):
            op = other_params[:n_op]
            other_params = other_params[n_op:]
            total_prob += lps[i](np.concatenate([transit_params, op]))
            
        return total_prob

    return log_prob

def run(
    times,
    specs, 
    wl_params,
    stellar_params,
    wav_bin_edges,
    out_masks=None,
    detrending_vectors=None, 
    polyorder=1, 
    progress=True,
    nproc=1,
    samples=5000,
    save_posterior=True,
    print_initial_logp=False,
    gp=False,
    fix_gp_timescale=True
):

    if detrending_vectors is None:
        detrending_vectors = [None for t in times]

    fixed_params = [np.concatenate([wlp[-9:], [wlp[2]]]) for wlp in wl_params]

    for i in range(len(out_masks)):
        if out_masks[i] is None:
            out_masks[i] = np.zeros_like(times[i])

        if len(out_masks[i].shape) == 1:
            out_masks[i] = np.tile(out_masks[i], (1, specs[i].shape[1])).T

    def run_single_band(ind):

        masks = [np.array(~om[:, ind]) for om in out_masks]

        new_detrending_vectors = []
        for dv, mask in zip(detrending_vectors, masks):
            if dv is not None:
                new_detrending_vectors.append(dv[mask])
            else:
                new_detrending_vectors.append(None)

        params = get_joint_initial_params(
            [t[m] for t, m in zip(times, masks)], 
            [s[m, ind] for s, m in zip(specs, masks)], 
            fixed_params, 
            polyorder=polyorder,
            detrending_vectors=new_detrending_vectors,
            gp=gp
        )

        if detrending_vectors[0] is not None:
            n_components = len(detrending_vectors[0].T)
        else:
            n_components = 0

        lps = get_log_probs(
            [t[m] for t, m in zip(times, masks)], 
            [s[m, ind] for s, m in zip(specs, masks)], 
            fixed_params, 
            new_detrending_vectors, 
            wav_bin_edges[ind],
            wav_bin_edges[ind + 1], 
            stellar_params, 
            polyorder=polyorder,
            gp=gp
        )
        
        log_prob = get_joint_log_prob(
            lps, 
            n_components, 
            polyorder, 
            gp=gp
        )
        
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

        for i in range(specs[0].shape[1]):

            if save_posterior:
                sampler = run_single_band(i)
                posterior.append(sampler)
            else:
                m, std = run_single_band(i)
                means.append(m)
                stds.append(std)

    else:

        nbands = specs[0].shape[1]
        nbatches = nbands // nproc
        remainder = np.remainder(nbands, nproc)
        
        if remainder > 0:
            nbatches += 1
        
        band_inds = np.arange(nbands)
        batches = [band_inds[i * nproc:(i + 1) * nproc] for i in range(nbatches)]

        for b in batches:

            if progress:
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