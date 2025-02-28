import numpy as np
import os
import exotic_ld
from exotic_ld import StellarLimbDarkening
from distributions import *

ld_data_path = os.environ['EXO_LD_PATH']

def get_ld_params(start_wav, end_wav, stellar_params):
    
    sld = StellarLimbDarkening(
        M_H=stellar_params['mh'], 
        Teff=stellar_params['teff'], 
        logg=stellar_params['logg'],
        ld_model="mps1",
        ld_data_path=ld_data_path
    )
    
    u, du = sld.compute_quadratic_ld_coeffs(
        wavelength_range=[start_wav * 1e4, end_wav * 1e4],
        mode="JWST_NIRSpec_G395H",
        return_sigmas=True
    )
    return u, du

def get_ld_priors(start_wav, end_wav, stellar_params):

    u, du = get_ld_params(start_wav, end_wav, stellar_params)
    u1_prior = trunc_normal_prior(u[0], du[0], 0, 1)
    u2_prior = trunc_normal_prior(u[1], du[1], 0, 1)
    return u1_prior, u2_prior

def gls_fit(time, flux, vectors=None, mask=None, polyorder=1, return_coeffs=False):

    if mask is None:
        mask = np.zeros_like(time, dtype=np.bool_)
    
    time_terms_masked = np.array(
        [time[~mask]**i for i in np.arange(polyorder + 1)]
    )
    time_terms = np.array(
        [time**i for i in np.arange(polyorder + 1)]
    )

    if vectors is None:
        P = np.concatenate([
            time_terms_masked.T,
        ], axis=1)

        coeffs = np.linalg.inv(P.T @ P) @ (P.T @ flux[~mask])
    
        P = np.concatenate([
            time_terms.T,
        ], axis=1)

        if return_coeffs:
            return coeffs, P @ coeffs
        else:
            return P @ coeffs
    else:
        P = np.concatenate([
            time_terms_masked.T,
            vectors[~mask, :],
        ], axis=1)
        
        coeffs = np.linalg.inv(P.T @ P) @ (P.T @ flux[~mask])

        P = np.concatenate([
            time_terms.T,
            vectors,
        ], axis=1)

        if return_coeffs:
            return coeffs, P @ coeffs
        else:
            return P @ coeffs

def get_trend_model(time, vectors, coeffs, polyorder):

    time_terms = np.array(
        [time**i for i in np.arange(1, polyorder + 1)]
    )

    if vectors is None:
        P = np.concatenate([
            time_terms.T,
        ], axis=1)
    else:
        P = np.concatenate([
            time_terms.T,
            vectors,
        ], axis=1)
        
    return P @ coeffs