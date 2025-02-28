import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from astropy.stats import sigma_clip
from wl_utils import run as wl_run
from wl_joint import run as wl_run_joint
from spec_utils import run as spec_run
from spec_joint import run as spec_run_joint
import decomp
from plot_utils import get_canonical_params
from transit import canon_from_eastman

import warnings
warnings.filterwarnings("ignore")

decomposer = decomp.get_pca_components

pl_param_names = [
    'period', 'radius', 't0', 
    'semimajor_axis', 'inclination', 
    'eccentricity', 'periastron'
]
st_param_names = ['mh', 'logg', 'teff']

def validate_priors(priors_dict):

    return all(item in priors_dict.keys() for item in pl_param_names)

def validate_st_params(stellar_params_dict):
    
    return  all(item in stellar_params_dict.keys() for item in st_param_names)

def validate_options(wl_options_dir, options_dir):

    return all(wl_options_dir[key] == options_dir[key] for key in ['gp', 'detector', 'n_components'])

def fit_joint_wlc(
    times,
    specs,
    wavs, 
    priors,
    stellar_params,
    detectors,
    polyorder=1,
    cubes=None,
    out_dir='./',
    samples=30000, 
    burnin=10000, 
    thin=1,
    nproc=1, 
    gp=False,
    out_filter_width=50,
    out_sigma=4,
    n_components=0,
    detrending_vectors=None,
    save_chains=True,
    return_chains=False,
    ld_priors=True,
    progress=True,
):
    fluxes = [np.sum(s, axis=1) for s in specs]

    n = len(times)
    if np.any([len(x) != n for x in [times, specs, detectors]]):

        raise ValueError(
            '''
                times, specs, and detectors should each be lists of the 
                same length. 
            '''
        )

    if gp:
        n_op = polyorder + 7 + n_components
    else:
        n_op = polyorder + 5 + n_components
        
    n_other_params = n_op * len(detectors)

    if not validate_priors(priors):
        
        raise ValueError(
            '''
                priors Dictionary must contain the following keys: 
                period, radius, t0, semimajor_axis, inclination, 
                eccentricity, periastron.'''
            ) 
        
    if not validate_st_params(stellar_params):
        
        raise ValueError(
            '''
                stellar_params Dictionary must contain the following keys: 
                mh, logg, teff.'''
            )
        
    to = times[0][0]
    for i in range(len(times)):
        times[i] -= to

    masks = []
    for flux in fluxes:
        masks.append(
            sigma_clip(
                flux - gaussian_filter1d(flux, out_filter_width), 
                sigma=out_sigma
            ).mask
        )

    if (cubes is not None) & (n_components > 0):

        if cubes[0].shape[0] != len(times[0]):
            raise ValueError(
                '''
                First dimension of data cube should 
                be the same length as time array.
                '''
            )

        components = []
        components_unmasked = []

        for flux, cube, mask in zip(fluxes, cubes, masks):

            p = decomposer(cube, n_components=n_components)
            p = decomp.replace_outliers_all_components(p, 3, radius=5)
            components_unmasked.append(gaussian_filter1d(p, 5, axis=0))
            
            p = decomp.mask(p, ~mask)
            p = gaussian_filter1d(p, 5, axis=0)
            components.append(p)
    else:
        components = None

    sampler = wl_run_joint(
        [t[~m] for t, m in zip(times, masks)],
        [f[~m] for f, m in zip(fluxes, masks)], 
        detectors, 
        priors,
        stellar_params,
        detrending_vectors=components, 
        polyorder=polyorder, 
        samples=samples,
        progress=progress,
        nproc=nproc,
        gp=gp,
        ld_priors=ld_priors,
    )

    chains = sampler.get_chain()[burnin::thin, :, :]

    result_dirs = []
    transit_params, other_params = chains[:, :, n_other_params:], chains[:, :, :n_other_params]
    
    for i in range(n):

        op = other_params[:, :, :n_op]
        other_params = other_params[:, :, n_op:]
        chain = np.concatenate([op, transit_params], axis=2)

        if components is None:
            ncoeffs = 1 + polyorder
        else:
            if components[0] is None:
                ncoeffs = 1 + polyorder
            else:
                ncoeffs = 1 + polyorder + len(components[i].T)
    
        if gp:
            op, p = chain[:, :, :5 + ncoeffs], chain[:, :, 5 + ncoeffs:]
        else:
            op, p = chain[:, :, :3 + ncoeffs], chain[:, :, 3 + ncoeffs:]
    
        canon_params = np.array(np.vectorize(canon_from_eastman)(*p.T)).T
        canon_params = np.median(canon_params, axis=(0, 1))
        canon_params = np.concatenate([np.median(op, axis=(0, 1)), canon_params])

        if components is None:
            components = [None] * len(times)
            components_unmasked = [None] * len(times)

        result_dirs.append(
            {
                'components': components[i],
                'n_components': n_components,
                'gp': gp,
                'detector': detectors[i],
                'chains': [chain if return_chains else None][0],
                'wl_params': np.median(chain, axis=(0, 1)),
                'wl_params_canon': canon_params,
                'mask': masks[i],
                'time': times[i],
                'spec': specs[i],
                'wavs': wavs[i],
                'stellar_params': stellar_params,
                'cube': cubes[i],
                'detrending_vectors': components[i],
                'detrending_vectors_unmasked': components_unmasked[i],
                'polyorder': polyorder
            }
        )

    if save_chains:
        np.save(out_dir + 'wl_joint_mcmc_chains', sampler.get_chain())

    return result_dirs

#def fit_wlc(
#    time, 
#    spec, 
#    wavs,
#    priors,
#    stellar_params,
#    detector,
#    polyorder=1,
#    cube=None, 
#    out_dir='./', 
#    samples=30000, 
#    burnin=10000, 
#    thin=1,
#    nproc=1, 
#    gp=False,
#    out_filter_width=50,
#    out_sigma=4,
#    n_components=0,
#    detrending_vectors=None,
#    save_chains=True,
#    return_chains=False,
#    ld_priors=True
#):
#
#    if not validate_priors(priors):
#        
#        raise ValueError(
#            '''
#                priors Dictionary must contain the following keys: 
#                period, radius, t0, semimajor_axis, inclination, 
#                eccentricity, periastron.'''
#            ) 
#        
#    if not validate_st_params(stellar_params):
#        
#        raise ValueError(
#            '''
#                stellar_params Dictionary must contain the following keys: 
#                mh, logg, teff.'''
#            )
#
#    time = np.array(time, dtype=np.float64)
#
#    if len(spec.shape) == 2:
#        flux = np.sum(spec, axis=1)
#    elif len(spec.shape) == 1:
#        flux = spec
#    else:
#        raise ValueError('Flux or spectrum has wrong shape.')
#
#    if len(time) != len(flux):
#        raise ValueError('Time array should have same length as flux or spectrum.')
#        
#    out_mask = sigma_clip(flux - gaussian_filter1d(flux, out_filter_width), sigma=out_sigma).mask
#
#    if (cube is not None) & (n_components > 0):
#
#        if cube.shape[0] != len(time):
#            raise ValueError(
#                'First dimension of data cube should be the same length as time array.'
#            )
#        
#        components = decomposer(cube, n_components=n_components)
#        components = decomp.replace_outliers_all_components(components, 3, radius=5)
#        components = decomp.mask(components, ~out_mask)
#        components = gaussian_filter1d(components, 5, axis=0)
#
#        if detrending_vectors is not None: 
#            detrending_vectors = decomp.mask(detrending_vectors, ~out_mask)
#            detrending_vectors = np.hstack([detrending_vectors, components])
#        else:
#            detrending_vectors = components
#
#    elif detrending_vectors is not None:
#        detrending_vectors = decomp.mask(detrending_vectors, ~out_mask)
#        components = None
#    else:
#        components = None
#
#    sampler = wl_run(
#        time[~out_mask],
#        flux[~out_mask], 
#        detector, 
#        priors,
#        stellar_params,
#        detrending_vectors=detrending_vectors, 
#        polyorder=polyorder, 
#        samples=samples,
#        progress=True,
#        nproc=nproc,
#        gp=gp,
#        ld_priors=ld_priors
#    )
#
#    chains = sampler.get_chain()[burnin::thin, :, :]
#
#    ncoeffs = 1 + polyorder + len(components.T)
#    
#    if gp:
#        p = chains[:, :, 5 + ncoeffs:]
#    else:
#        p = chains[:, :, 3 + ncoeffs:]
#
#    canon_params = np.array(np.vectorize(canon_from_eastman)(*p.T))
#
#    result_dir = {
#        'components': components,
#        'n_components': n_components,
#        'gp': gp,
#        'detector': detector,
#        'chains': [chains if return_chains else None][0],
#        'wl_params': np.median(chains, axis=(0, 1)),
#        'wl_params_canon': np.median(canon_params.T, axis=(0, 1)),
#        'mask': out_mask,
#        'time': time,
#        'spec': spec,
#        'wavs': wavs,
#        'stellar_params': stellar_params,
#        'cube': cube,
#        'detrending_vectors': detrending_vectors,
#        'polyorder': polyorder,
#    }
#
#    if save_chains:
#        np.save(out_dir + 'wl_mcmc_chains', sampler.get_chain())
#
#    return result_dir

def fit_joint_spec(
    wl_results,
    wav_per_bin=None,
    pix_per_bin=None,
    polyorder=None, 
    out_dir='./', 
    samples=10000, 
    burnin=5000, 
    nproc=1, 
    out_filter_width=50,
    out_sigma=4,
    n_components_spec=0,
    n_components=None,
    detrending_vectors=None,
    save_chains=True,
    return_chains=False,
    progress=False,
    gp=None,
    start_wav=2.87,
    end_wav=5.17692
):

    if detrending_vectors is None:
        detrending_vectors = [res['detrending_vectors_unmasked'] for res in wl_results]
    
    if polyorder is None:
        polyorder = wl_results[0]['polyorder']

    if (not wl_results[0]['gp']) & gp:
        raise AttributeError(
            '''
            Cannot use GP model for spectral 
            fitting if a GP was not used for 
            white light fitting.
            '''
        )
        
    if gp is None:
        gp = wl_results[0]['gp']

    times = [res['time'] for res in wl_results]
    specs = [res['spec'] for res in wl_results]
    stellar_params = wl_results[0]['stellar_params']
    cubes = [res['cube'] for res in wl_results]
    detector = wl_results[0]['detector']
    wl_params = [res['wl_params_canon'] for res in wl_results]
    wavs = wl_results[0]['wavs']

    if n_components is not None:
        n_components = len(wl_results[0]['components_unmasked'].T)
    else:
        n_components = 0

    times = [np.array(t, dtype=np.float64) for t in times]

    if detector == 'nrs1':

        specs = [s[:, wavs > start_wav] for s in specs]
        wavs = wavs[wavs > start_wav]

    if detector == 'nrs2':
        
        specs = [s[:, wavs < end_wav] for s in specs]
        wavs = wavs[wavs < end_wav]

    if (wav_per_bin is None) & (pix_per_bin is None):
        binned_wavs = wavs
        binned_specs = specs
    elif wav_per_bin is None:
        wavs_unbinned = result_nrs1[0]['wavs']
        binned_wavs = np.array(
            [np.mean(
                wavs_unbinned[pix_per_bin * i: pix_per_bin * (i + 1)]
            ) for i in range(len(wavs_unbinned) // pix_per_bin)]
        )
        wav_bin_edges = np.array(
            [wavs_unbinned[30 * i] for i in range(len(wavs_unbinned) // 30 + 1)]
        )
        spec_unbinned = result_nrs1[0]['spec']
        binned_spec = np.array(
            [np.mean(
                spec_unbinned[:, pix_per_bin * i: pix_per_bin * (i + 1)], axis=1
            ) for i in range(len(wavs_unbinned) // pix_per_bin)]
        ).T
    else:
        nbands = np.int64((wavs[-1] - wavs[0]) // wav_per_bin)
        wav_bin_edges = np.linspace(wavs[0], wavs[-1], nbands + 1)
        binned_wavs = wav_bin_edges[:-1] + 0.5 * np.diff(wav_bin_edges)
    
        binned_specs = []
        for spec in specs:
            binned_specs.append(
                np.array([
                    np.sum(
                        spec[:, np.where(
                            (wavs >= wav_bin_edges[i]) & 
                            (wavs <= wav_bin_edges[i+1])
                        )[0]],
                        axis=1
                    )
                    for i in range(nbands)
                ]).T
            )

    filts = [gaussian_filter1d(bs, out_filter_width, axis=0) for bs in binned_specs]
    masks = [sigma_clip(bs - f, sigma=out_sigma).mask for bs, f in zip(binned_specs, filts)]

    detrending_cubes = []
    for i, (cube, bs) in enumerate(zip(cubes, binned_specs)):
        if (cube is not None) & (n_components_spec > 0):

            detrending_cubes.append(
                np.zeros(
                    (bs.shape[1], bs.shape[0], n_components_spec)
                )
            )
        
            for i in range(binned_spec.shape[1]):
                detrending_cube[i, :, :] = decomp.get_cube(
                    wavs, wav_bin_edges, cube, i, n_components=n_components_spec
                )
    
            if n_components > 0:
                components = decomposer(cube, n_components=n_components)
                components = decomp.replace_outliers_all_components(components, 3, radius=5)
                components = gaussian_filter1d(components, 5, axis=0)
    
                if detrending_vectors[i] is not None: 
                    detrending_vectors[i] = np.hstack([detrending_vectors[i], components])
                else:
                    detrending_vectors[i] = components
    
        else:
            detrending_cubes.append(None)

    post = spec_run_joint(
        times,
        binned_specs, 
        wl_params,
        stellar_params,
        wav_bin_edges,
        out_masks=masks,
        detrending_cubes=detrending_cubes,
        detrending_vectors=detrending_vectors,
        nproc=nproc,
        samples=samples,
        polyorder=polyorder,
        progress=progress,
        gp=gp
    )

    if save_chains:
        for i in range(len(post)):
            np.save(out_dir + 'spec_mcmc_chains_{0}'.format(i), post[i].get_chain()[burnin:, :, :])
            np.save(out_dir + 'wavs_{0}'.format(i), binned_wavs)
    else:
        return post, binned_wavs

#def fit_spec(
#    wl_results,
#    wav_per_bin=0.02,
#    polyorder=None, 
#    out_dir='./', 
#    samples=10000, 
#    burnin=5000, 
#    nproc=1, 
#    out_filter_width=50,
#    out_sigma=4,
#    n_components_spec=0,
#    n_components=None,
#    detrending_vectors=None,
#    save_chains=True,
#    return_chains=False,
#    progress=False,
#    gp=None
#):
#
#    if polyorder is None:
#        polyorder = wl_results['polyorder']
#
#    if (not wl_results['gp']) & gp:
#        raise AttributeError(
#            '''
#            Cannot use GP model for spectral 
#            fitting if a GP was not used for 
#            white light fitting.
#            '''
#        )
#    if gp is None:
#        gp = wl_results['gp']
#
#    time = wl_results['time']
#    spec = wl_results['spec']
#    stellar_params = wl_results['stellar_params']
#    cube = wl_results['cube']
#    detector = wl_results['detector']
#    wl_params = wl_results['wl_params']
#    wavs = wl_results['wavs']
#
#    if n_components is not None:
#        n_components = len(wl_results['components'].T)
#    else:
#        n_components = 0
#
#    time = np.array(time, dtype=np.float64)
#
#    if detector == 'nrs1':
#
#        spec = spec[:, wavs > 2.87]
#        wavs = wavs[wavs > 2.87]
#
#    if detector == 'nrs2':
#        
#        spec = spec[:, wavs < 5.17692]
#        wavs = wavs[wavs < 5.17692]
#
#    nbands = np.int64((wavs[-1] - wavs[0]) // wav_per_bin)
#    wav_bin_edges = np.linspace(wavs[0], wavs[-1], nbands + 1)
#    binned_wavs = wav_bin_edges[:-1] + 0.5 * np.diff(wav_bin_edges)
#    
#    binned_spec = np.array([
#        np.sum(
#            spec[:, np.where(
#                (wavs >= wav_bin_edges[i]) & 
#                (wavs <= wav_bin_edges[i+1])
#            )[0]],
#            axis=1
#        )
#        for i in range(nbands)
#    ])
#    binned_spec = binned_spec.T
#
#    filt = gaussian_filter1d(binned_spec, out_filter_width, axis=0)
#    mask = sigma_clip(binned_spec - filt, sigma=out_sigma).mask
#
#    if (cube is not None) & (n_components_spec > 0):
#
#        detrending_cube = np.zeros(
#            (binned_spec.shape[1], binned_spec.shape[0], n_components_spec)
#        )
#    
#        for i in range(binned_spec.shape[1]):
#            detrending_cube[i, :, :] = decomp.get_cube(
#                wavs, wav_bin_edges, cube, i, n_components=n_components_spec
#            )
#
#        if n_components > 0:
#            components = decomposer(cube, n_components=n_components)
#            components = decomp.replace_outliers_all_components(components, 3, radius=5)
#            components = gaussian_filter1d(components, 5, axis=0)
#
#            if detrending_vectors is not None: 
#                detrending_vectors = np.hstack([detrending_vectors, components])
#            else:
#                detrending_vectors = components
#
#    else:
#        detrending_cube = None
#
#    post = spec_run(
#        time,
#        binned_spec, 
#        wl_params,
#        stellar_params,
#        wav_bin_edges,
#        out_mask=mask,
#        detrending_cube=detrending_cube,
#        detrending_vectors=detrending_vectors,
#        nproc=nproc,
#        samples=samples,
#        polyorder=polyorder,
#        progress=progress,
#        gp=gp
#    )
#
#    if save_chains:
#        for i in range(len(post)):
#            np.save(out_dir + 'spec_mcmc_chains_{0}'.format(i), post[i].get_chain()[burnin:, :, :])
#            np.save(out_dir + 'wavs_{0}'.format(i), binned_wavs)
#    else:
#        return post, binned_wavs

    