import numpy as np
import batman
from distributions import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

epos = lambda w, V: -(
        np.sin(w) * V**2 + np.sqrt(1 + (np.sin(w)**2 - 1) * V**2)
    ) / (1 + (np.sin(w) * V)**2)
    
eneg = lambda w, V: -(
        np.sin(w) * V**2 - np.sqrt(1 + (np.sin(w)**2 - 1) * V**2)
    ) / (1 + (np.sin(w) * V)**2)

def keplerian_transit(t, u1, u2, t0, r, p, a, i, e, w):

    params = batman.TransitParams()
    params.t0 = t0                     #time of inferior conjunction
    params.per = p                     #orbital period
    params.rp = r                      #planet radius (in units of stellar radii)
    params.a = a                       #semi-major axis (in units of stellar radii)
    params.inc = i * 180 / np.pi       #orbital inclination (in degrees)
    params.ecc = e                     #eccentricity
    params.w = w * 180 / np.pi         #longitude of periastron (in degrees)
    params.u = [u1, u2]                #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"     #limb darkening model

    m = batman.TransitModel(params, t, fac=0.01)    
    return m.light_curve(params) - 1

def eastman_from_canon(t0, r, per, a, i, e, w):

    V = np.sqrt(1 - e**2) / (1 + e * np.sin(w))
    b = a * np.cos(i) * (1 - e**2) / (1 + e * np.sin(w))
    C = np.sqrt((1 + r)**2 - b**2)
    L = 0.5
    Ls = L * np.sin(w)
    Lc = L * np.cos(w)
    S = 0.5

    if not np.isclose(epos(w, V), e):
        S = 1.5

    return t0, r, per, a, V, C, Ls, Lc, S

# obtain the standard parameterization from the Eastman 
# parameters (inverse of eastman_from_canon)
def canon_from_eastman(t0, r, per, a, V, C, Ls, Lc, S):

    L2 = Ls**2 + Lc**2
    w = np.arctan2(Ls, Lc)

    if S < 1:
        e = epos(w, V)
    else:
        e = eneg(w, V)
        
    inc = np.arccos(
        np.sqrt((1 + r)**2 - C**2) * (1 + e * np.sin(w)) / ((1 - e**2) * a)
    )

    return t0, r, per, a, inc, e, w

# Computes the jacobian of the reparameterization, which is needed to 
# correctly account for implicit priors, and also returns the transit 
# model plus quadratic trend. 
def reparam(t, p):
    
    C = p[-4]
    u1, u2 = p[:2]
    t0, r, per, a, inc, e, w = canon_from_eastman(*p[2:])

    # jacobian of the reparameterization transform 
    b = a * np.cos(inc) * (1 - e**2) / (1 + e * np.sin(w))
    jac = np.abs(
        (e + np.sin(w)) * b**2 / (
            np.sqrt(1 - e**2) * (1 + e * np.sin(w))**2 * np.cos(inc) * C
        )
    )

    transit = keplerian_transit(
        t, u1, u2, t0, r, per, a, inc, e, w
    )
    mu = 1 + transit

    return mu, jac

# priors for the Eastman parameterization from: https://arxiv.org/pdf/2309.14410 -- figure 4
def eastman_priors(t0, r, per, a, V, C, Ls, Lc, S):

    L2 = Ls**2 + Lc**2

    priors = uniform_prior(Lc**2 - 1, 1 - Lc**2).prior(Ls)
    priors += uniform_prior(Ls**2 - 1, 1 - Ls**2).prior(Lc)
    priors += uniform_prior(0, 1 + r).prior(C)
    priors += uniform_prior(0, 1).prior(L2)
    priors += uniform_prior(0, 2 * a / (1 + r) - 1).prior(V**2)
    priors += uniform_prior(0, np.inf).prior(V)
    priors += uniform_prior(0, 2).prior(S)
    
    return priors