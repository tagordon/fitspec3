import numpy as np
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from astropy.stats import sigma_clip

def get_fa_components(cube, n_components=3):

    nt, ny, nl = cube.shape
    
    fa = FactorAnalysis(n_components=n_components)
    
    fluxes = np.concatenate(cube.T, axis=0).T
    cube = np.nan_to_num(cube, 0)
    relflux = (
        fluxes
        / np.tile(
            np.nansum(
                fluxes, axis=1
            ), (ny * nl, 1)
        ).T
    )
    
    return fa.fit_transform(relflux)

def get_ica_components(cube, n_components=3):

    nt, ny, nl = cube.shape
    
    ica = FastICA(n_components=n_components)
    
    fluxes = np.concatenate(cube.T, axis=0).T
    cube = np.nan_to_num(cube, 0)
    relflux = (
        fluxes
        / np.tile(
            np.nansum(
                fluxes, axis=1
            ), (ny * nl, 1)
        ).T
    )
    
    return ica.fit_transform(relflux)

def get_pca_components(cube, n_components=3):

    nt, ny, nl = cube.shape
    
    pca = PCA(n_components=n_components)
    
    fluxes = np.concatenate(cube.T, axis=0).T
    cube = np.nan_to_num(cube, 0)
    relflux = (
        fluxes
        / np.tile(
            np.nansum(
                fluxes, axis=1
            ), (ny * nl, 1)
        ).T
    )
    
    return pca.fit_transform(relflux)

def get_cube(wavs, wav_bin_edges, cube, ind, n_components=5, trim_edges=2, decomposer=get_pca_components):

    binned_wav_inds = np.array(
        [
            np.where(
                np.isclose(wavs, edge, atol=np.mean(np.diff(wavs))/2)
            )[0][0] for edge in wav_bin_edges
        ]
    )

    if ind <= 1:
        components = decomposer(
            cube[:, trim_edges:-trim_edges, binned_wav_inds[0]:binned_wav_inds[ind+2]], 
            n_components=n_components
        )
    elif ind >= len(wav_bin_edges - 2):
        components = decomposer(
            cube[:, trim_edges:-trim_edges, binned_wav_inds[ind-2]:binned_wav_inds[ind+2]], 
            n_components=n_components
        )
    else:
        components = decomposer(
            cube[:, trim_edges:-trim_edges, binned_wav_inds[ind]:binned_wav_inds[-1]], 
            n_components=n_components
        )

    return components

def gls_fit(time, data, components=None, mask=None, polyorder=2, return_coeffs=False):

    if mask is None:
        mask = np.zeros_like(time, dtype=bool)
    
    time_terms_masked = np.array(
        [time[~mask]**i for i in np.arange(polyorder + 1)]
    )
    time_terms = np.array(
        [time**i for i in np.arange(polyorder + 1)]
    )

    if components is None:
        P = np.concatenate([
            time_terms_masked.T,
        ], axis=1)

        coeffs = np.linalg.inv(P.T @ P) @ (P.T @ data[~mask])
    
        P = np.concatenate([
            time_terms.T,
        ], axis=1)

        if return_coeffs:
            return coeffs, P @ coeffs
        else:
            return P @ coeffs
    else:
        P = np.concatenate([
            components[~mask, :],
            time_terms_masked.T,
        ], axis=1)
        
        coeffs = np.linalg.inv(P.T @ P) @ (P.T @ data[~mask])

        P = np.concatenate([
            components,
            time_terms.T,
        ], axis=1)

        if return_coeffs:
            return coeffs, P @ coeffs
        else:
            return P @ coeffs

def replace_outliers(x, sigma, radius=1):

    mask = sigma_clip(x, sigma=sigma).mask
    
    y = x.copy()

    if np.any(mask[:radius] == True):
        y[:radius] = np.mean(y) + np.random.randn(radius) * np.std(y)
        mask[:radius] = False
    if np.any(mask[-radius:] == True):
        y[-radius:] = np.mean(y) + np.random.randn(radius) * np.std(y)
        mask[-radius:] = False
    
    y[mask] = 0.5 * (x[np.where(mask)[0] - radius] + x[np.where(mask)[0] + radius])
    return y

def replace_outliers_all_components(components, sigma, radius=1):

    new_components = np.zeros_like(components)
    for i, c in enumerate(components.T):
        new_components[:, i] = replace_outliers(c, sigma, radius=radius)
    return new_components

def mask(components, mask):

    new_components = np.zeros((np.sum(mask), components.shape[1]))
    for i, c in enumerate(components.T):
        new_components[:, i] = c[mask]
    return new_components