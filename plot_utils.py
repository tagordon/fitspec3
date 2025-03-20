import numpy as np
import batman
import exotic_ld
from exotic_ld import StellarLimbDarkening
from scipy.ndimage import gaussian_filter1d
from astropy.stats import sigma_clip
import celerite2
import corner 
import matplotlib.pyplot as plt

from distributions import *
from transit import *
from wl_utils import _unpack_params, get_trend_model

def plot_fit(
    result, 
    data_color=plt.cm.terrain(0.1), 
    transit_color=plt.cm.terrain(0.7),
    systematics_color=plt.cm.terrain(0.3),
    gp_color=plt.cm.terrain(0.9),
    combined_color=plt.cm.terrain(0.1),
    axs=None
):

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
    time = result['time']
    if result['gp']:
        trans, sys, pred, f = get_wl_models(result, nsamples=100)
    else:
        trans, sys, f = get_wl_models(result, nsamples=100)
    
    flux = np.sum(result['spec'], axis=1)[~result['mask']]
    masked_time = time[~result['mask']]
    
    axs[0].plot(masked_time, flux, '.', color=data_color, alpha=0.5)
    axs[0].plot(masked_time, trans.T + sys.T[0], color=transit_color, alpha=0.1);
    axs[0].plot(masked_time, sys.T + f, color=systematics_color, alpha=0.1);
    
    if result['gp']:
        axs[0].plot(masked_time, pred.T + sys.T[0] + f, color=gp_color, alpha=0.1);
    
    axs[1].plot(masked_time, flux, '.', color=data_color, alpha=0.5)
    
    if result['gp']:
        axs[1].plot(masked_time, trans.T + sys.T + pred.T, color=combined_color, alpha=0.1);
    else:
        axs[1].plot(masked_time, trans.T + sys.T, color=combined_color, alpha=0.1);
    
    [ax.grid(alpha=0.5) for ax in axs];
    [ax.set_xlabel('time (days)', fontsize=15) for ax in axs];
    [ax.set_ylabel('flux (e-/s)', fontsize=15) for ax in axs];

    return axs

def plot_fit_spec(
    result,
    data_color=plt.cm.terrain(0.1), 
    transit_color=plt.cm.terrain(0.7),
    systematics_color=plt.cm.terrain(0.3),
    gp_color=plt.cm.terrain(0.9),
    combined_color=plt.cm.terrain(0.1),
    axs=None
):

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
    time = result['time']

    
def plot_corner(
    result, 
    color=None, 
    fig=None, 
    burnin=0, 
    vars_to_plot=None, 
    ignore_chains=None, 
    label=None,
):

    if fig is None:
        fig = plt.figure(figsize=(8, 8))

    pnames = [
        '$t_0$ (days)', 'radius', 'period', 
        'semimajor-axis (a/R$^*$)', 'inclination (degrees)', 
        'eccentricity', 'periastron'
    ]
    
    if vars_to_plot is None:
        inds = np.arange(len(pnames))
    else:
        inds = [pnames.index(v) for v in vars_to_plot]

    if color is None:
        color = plt.cm.rainbow(0.1)
    
    p = get_canonical_params(result, 1)

    if ignore_chains is None:
        chains = np.arange(0, p.shape[1])
    else:
        chains = np.delete(np.arange(0, p.shape[1]), ignore_chains)
    
    p = p[:, chains, burnin:]
    p[4] = p[4] * 180 / np.pi
    p[6] = p[6] * 180 / np.pi
    
    fig = corner.corner(
        np.vstack([p[i].flatten() for i in inds]).T,
        color=color, 
        fig=fig, 
        labels=[pnames[i] for i in inds], 
        label_kwargs={'fontsize': 12},
        hist_kwargs={
            'label': label, 
            'density': True, 
            'linewidth': 2
        },
    );
            
    return fig

def get_canonical_params(result, polyorder):

    detrending_vectors = result['detrending_vectors']
    polyorder = result['polyorder']
    gp = result['gp']

    if detrending_vectors is None:
        ncoeffs = 1 + polyorder
    else:
        ncoeffs = 1 + polyorder + len(detrending_vectors.T)

    if gp:
        p = result['chains'][:, :, 5 + ncoeffs:]
    else:
        p = result['chains'][:, :, 3 + ncoeffs:]

    return np.array(np.vectorize(canon_from_eastman)(*p.T))

def _get_model_from_sample(time, p, detrending_vectors=None, polyorder=1, gp=False, flux=None):

    if detrending_vectors is None:
        len_det = 0
    else:
        len_det = len(detrending_vectors.T)

    ncoeffs = 1 + polyorder + len_det

    if gp:

        err, lsigma, lw0, coeffs, f, p = _unpack_params(p, ncoeffs, gp=True)
        trend = get_trend_model(time, detrending_vectors, coeffs[1:], polyorder)
        term = celerite2.terms.SHOTerm(sigma=np.exp(lsigma), w0=np.exp(lw0), Q=1/np.sqrt(2))
        gp = celerite2.GaussianProcess(term)
        mu, jac = reparam(time, p)

        if flux is not None:
            gp.compute(time, diag=err**2)
            pred = gp.predict(flux - mu * f - trend)
            return mu * f, trend, pred
            
        if flux is None:
            return mu * f, trend
        
    else:

        err, coeffs, f, p = _unpack_params(p, ncoeffs, gp=False)
        trend = get_trend_model(time, detrending_vectors, coeffs[1:], polyorder)
        mu, jac = reparam(time, p)
        
    return mu * f, trend

def get_spec_models(spec_result, ind, transit_ind, nsamples=100):
    '''
    Get posterior samples for the the transit model, 
    systematics model, and GP model (if applicable) for 
    the results of a spectral lightcurve fit.

    Args:
        spec_results (dict): results dictionary returned from fit_spec_joint
        nsamples (int, default=100): number of posterior samples 
        to return. 
        ind (int): the index of the spectral lightcurve to model 

    Returns: 
        transit models (2D array): the transit model
        systematics models (2D array): polynomial + linear combination 
        detrending vectors and PCA vectors as specified in fit_wlc
        GP prediction (2D array): only returned if gp=True in fit_wlc
        f0 (1D array): the constant term of the polynomial in the systematics 
        model, which is useful for plotting the systematics model and GP 
        prediction.

    Note: the systematics model and GP prediction vectors will have zero flux 
    offset, so f0 should be added to these if they are to be plotted over the 
    observations. 
    '''

    wl_results = spec_result['wl_results'][transit_ind]
    time = wl_results['time']
    spec = wl_results['spec']
    stellar_params = wl_results['stellar_params']
    cube = wl_results['cube']
    detector = wl_results['detector']
    gp = wl_results['gp']
    wl_params = wl_results['wl_params']
    wavs = wl_results['wavs']
    mask = wl_results['mask']
    detrending_vectors = wl_results['detrending_vectors']
    polyorder = wl_results['polyorder']

    if spec_result['chains'] is None:
        raise AttributeError(
            '''
            Result dictionary does not contain MCMC chains. 
            Run fit_wlc with return_chains=True.
            '''
        )

    if detrending_vectors is None:
        len_det = 0
    else:
        len_det = len(detrending_vectors.T)

    ncoeffs = 1 + polyorder + len_det

    flux = spec_result['lightcurves'][transit_ind][:, ind]
    chain = spec_result['chains'][ind].get_chain()
    other_params = np.concatenate(
        chain, axis=0
    )[:, :-3][:, transit_ind * (ncoeffs + 1):(ncoeffs + 1) * (transit_ind + 1)].T
    r, u1, u2 = np.concatenate(chain, axis=0)[:, -3:].T
    err, coeffs, f, p = _unpack_params(spec_result['wl_params_eastman'][transit_ind], ncoeffs=ncoeffs, gp=gp)
    p_tiled = np.vstack([np.tile(pi, r.shape) for pi in p])
    p_tiled[3] = r
    p_tiled[:2] = [u1, u2]
    flat_samples = np.concatenate([other_params, p_tiled], axis=0).T

    ret = _get_models(
        time[~mask], 
        flat_samples, 
        nsamples=100, 
        detrending_vectors=detrending_vectors, 
        gp=gp, 
        polyorder=polyorder,
        flux=flux[~mask], 
        return_idx=True
    )

    idx = ret[-1]
    if gp:
        f = flat_samples[idx, 3]
    else:
        f = flat_samples[idx, 1]

    return *ret[:-1], f

def get_wl_models(wl_results, nsamples=100):
    '''
    Get posterior samples for the the transit model, 
    systematics model, and GP model (if applicable).

    Args:
        wl_results (dict): results dictionary returned from fit_wlc
        nsamples (int, default=100): number of posterior samples 
        to return. 

    Returns: 
        transit models (2D array): the transit model
        systematics models (2D array): polynomial + linear combination 
        detrending vectors and PCA vectors as specified in fit_wlc
        GP prediction (2D array): only returned if gp=True in fit_wlc
        f0 (1D array): the constant term of the polynomial in the systematics 
        model, which is useful for plotting the systematics model and GP 
        prediction.

    Note: the systematics model and GP prediction vectors will have zero flux 
    offset, so f0 should be added to these if they are to be plotted over the 
    observations. 
    '''

    time = wl_results['time']
    spec = wl_results['spec']
    stellar_params = wl_results['stellar_params']
    cube = wl_results['cube']
    detector = wl_results['detector']
    gp = wl_results['gp']
    wl_params = wl_results['wl_params']
    wavs = wl_results['wavs']
    mask = wl_results['mask']
    detrending_vectors = wl_results['detrending_vectors']
    #detrending_vectors = wl_results['pca_components']
    polyorder = wl_results['polyorder']

    if wl_results['chains'] is None:
        raise AttributeError(
            '''
            Result dictionary does not contain MCMC chains. 
            Run fit_wlc with return_chains=True.
            '''
        )

    if len(spec.shape) > 1:
        flux = np.sum(spec, axis=1)
    else:
        flux = spec
    flat_samples = np.concatenate(wl_results['chains'], axis=0)

    ret = _get_models(
        time[~mask], 
        flat_samples, 
        nsamples=100, 
        detrending_vectors=detrending_vectors, 
        gp=gp, 
        polyorder=polyorder,
        flux=flux[~mask], 
        return_idx=True
    )

    idx = ret[-1]
    if gp:
        f = flat_samples[idx, 3]
    else:
        f = flat_samples[idx, 1]

    return *ret[:-1], f

def _get_models(time, flat_samples, polyorder=1, nsamples=100, detrending_vectors=None, gp=False, flux=None, return_idx=False):

    idx = np.random.randint(len(flat_samples), size=nsamples)
    if gp & (flux is not None):
        trans = np.zeros((nsamples, len(time)))
        sys = np.zeros((nsamples, len(time)))
        pred = np.zeros((nsamples, len(time)))
        for i, j in enumerate(idx):
            trans[i], sys[i], pred[i] = _get_model_from_sample(
                time, 
                flat_samples[j], 
                detrending_vectors=detrending_vectors, 
                gp=gp, 
                polyorder=polyorder,
                flux=flux
            )
        if return_idx:
            return trans, sys, pred, idx
        else:
            return trans, sys, pred

    else:
        trans = np.zeros((nsamples, len(time)))
        sys = np.zeros((nsamples, len(time)))
        for i, j in enumerate(idx):
            trans[i], sys[i] = _get_model_from_sample(
                time, 
                flat_samples[j], 
                detrending_vectors=detrending_vectors, 
                gp=gp, 
                polyorder=polyorder,
                flux=flux
            )
        if return_idx:
            return trans, sys, idx
        else:
            return trans, sys