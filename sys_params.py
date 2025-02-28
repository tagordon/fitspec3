from distributions import *

stellar_params_dict = {

    
    '134.01' : {
        'mh': -0.03,
        'logg': 4.84,
        'teff': 3842,
        'radius': 0.604,
        'reference': 'https://ui.adsabs.harvard.edu/abs/2024A%26A...688A.216H/abstract'
    },

    '175.01': {
        'mh': -0.46,
        'logg': 4.45,
        'teff': 3415,
        'radius': 0.303,
        'reference': 'https://ui.adsabs.harvard.edu/abs/2021A%26A...653A..41D/abstract'  
    },

    '260.01': {
        'mh': -0.47,
        'logg': 4.7,
        'teff': 4026,
        'radius': 0.607,
        'reference': 'https://ui.adsabs.harvard.edu/abs/2024A%26A...688A.216H/abstract'  
    },

    '402.01': {
        'mh': 0.03,
        'logg': 4.48,
        'teff': 5131,
        'radius': 0.855,
        'reference': 'https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.282R/abstract'
    },

    '402.02': {
        'mh': 0.03,
        'logg': 4.48,
        'teff': 5131,
        'radius': 0.855,
        'reference': 'https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.282R/abstract'
    },

    '562.01': {
        'mh': -0.12,
        'logg': 4.94,
        'teff': 3505,
        'radius': 0.337,
        'reference': 'https://ui.adsabs.harvard.edu/abs/2019A%26A...628A..39L/abstract'
    },

    '776.01': {
        'mh': -0.21,
        'logg': 4.8,
        'teff': 3725,
        'radius': 0.547,
        'reference': 'https://ui.adsabs.harvard.edu/abs/2024A&A...684A..12F/abstract'
    },

    '776.02': {
        'mh': -0.21,
        'logg': 4.8,
        'teff': 3725,
        'radius': 0.547,
        'reference': 'https://ui.adsabs.harvard.edu/abs/2024A&A...684A..12F/abstract'
    },

    '836.01': {
        'mh': -0.284,
        'logg': 4.743,
        'teff': 4552,
        'radius': 0.665
    },

    '836.02': {
        'mh': -0.284,
        'logg': 4.743,
        'teff': 4552,
        'radius': 0.665
    },

}

planet_params_dict = {
    '134.01': {
            'mass': 4.07,
            'equilib_temp': 998.0,
            'rade': 1.63,
            'ref': 'https://ui.adsabs.harvard.edu/abs/2024A%26A...688A.216H/abstract'
    },

    '175.01': {
        'mass': 2.22,
        'equilib_temp': 553.0,
        'rade': 1.385,
        'reference': 'https://ui.adsabs.harvard.edu/abs/2021A%26A...653A..41D/abstract'  
    },

    '260.01': {
        'mass': 4.23, 
        'equilib_temp': 493.0,
        'rade': 1.71,
        'reference': 'https://ui.adsabs.harvard.edu/abs/2024A%26A...688A.216H/abstract'
    },

    '402.01': {
        'mass': 6.519,
        'equilib_temp': 1006,
        'rade': 1.770,
        'reference': 'https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.282R/abstract'
    },

    '402.02': {
        'mass': 6.792,
        'equilib_temp': 656, # Dumusque 
        'rade': 2.526,
        'reference': 'https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.282R/abstract'
    },

    '562.01': {
        'mass': 1.84,
        'equilib_temp': 525,
        'rade': 1.20, # Oddo -- additional transits with CHEOPS so should be better than Luque 
        'reference': 'https://ui.adsabs.harvard.edu/abs/2019A%26A...628A..39L/abstract'
    },

    '776.02': {
        'mass': 5.0,
        'equilib_temp': 520,
        'rade': 1.798
    },

    '776.01': {
        'mass': 6.9,
        'equilib_temp': 420,
        'rade': 2.047
    },

    '836.02': {
        'mass': 4.53,
        'equilib_temp': 871,
        'rade': 1.704
    },

    '836.01': {
        'mass': 9.60,
        'equilib_temp': 665,
        'rade': 2.587
    },
}

priors_dict = {

    '134.01': {
        'period': normal_prior(1.40152604, 0.00000082),
        'radius': uniform_prior(0.0, 1.0, init=0.0247),
        't0': uniform_prior(0.0, 100, init=0.085),
        'semimajor_axis': uniform_prior(4.0, 10.0, init=7.61), # not from ref. 
        'inclination': normal_prior(84.27 * np.pi / 180, 1.01 * np.pi / 180, init=85.8 * np.pi / 180),
        'eccentricity': uniform_prior(0, 1e-4, init=0.5e-3),
        'periastron': uniform_prior(-np.pi, np.pi),
        'reference': (
            'https://ui.adsabs.harvard.edu/abs/2024A%26A...688A.216H/abstract',
        ),
    },

    '175.01': {
        'period': normal_prior(3.6906777, 0.0000026),
        'radius': uniform_prior(0.0, 1.0, init=0.04088),
        't0': uniform_prior(0.0, 100, init=1.0),
        'semimajor_axis': uniform_prior(10, 30, init=19),
        'inclination': normal_prior(88.11 * np.pi / 180, 0.36 * np.pi / 180, init = 88.11 * np.pi / 180),
        'eccentricity': trunc_normal_prior(0.103, 0.058, 0, 1),
        'periastron': normal_prior(-99 * np.pi / 180, 20 * np.pi / 180),
        'reference': 'https://ui.adsabs.harvard.edu/abs/2021A%26A...653A..41D/abstract',
    },

    '260.01': {
        'period': normal_prior(13.475853, 0.000013),
        'radius': uniform_prior(0, 1, init=0.0258),
        't0': uniform_prior(0.0, 100, init=1.0),
        'semimajor_axis': uniform_prior(15, 45, init=35.32),
        'inclination': normal_prior(88.84 * np.pi / 180, 0.17 * np.pi / 180, init=88.7 * np.pi / 180),
        'eccentricity': uniform_prior(0.0, 1e-4), # according to Hobson et al 2024, circular model is preferred -- non-circular best-fit eccentricity seems inconsistent with previous fits.
        'periastron': uniform_prior(-np.pi, np.pi),
        'reference': (
            'https://ui.adsabs.harvard.edu/abs/2024A%26A...688A.216H/abstract'
        ),
    },

    '402.01': {
        'period': normal_prior(4.7559804, 0.0000062),
        'radius': uniform_prior(0.0, 1.0, init=0.01898), 
        't0': uniform_prior(0.0, 100, init=1.0),
        'semimajor_axis': uniform_prior(10, 20, init=13),
        'inclination': normal_prior(89.30 * np.pi / 180, 0.53 * np.pi / 180, init=89 * np.pi / 180),
        'eccentricity': trunc_normal_prior(0.058, 0.022, 0, 1),
        'periastron': normal_prior(71.62 * np.pi / 180, 52.51 * np.pi / 180),
        'reference': (
            'https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.282R/abstract',
        ),
    },

    '402.02': {
        'period': normal_prior(17.1784, 0.00016),
        'radius': uniform_prior(0.0, 1.0, init=0.02707),
        't0': uniform_prior(0.0, 100, init=1.0),
        'semimajor_axis': uniform_prior(25, 40, init=31.87),
        'inclination': normal_prior(88.41 * np.pi / 180, 0.07 * np.pi / 180, init=88.35 * np.pi / 180),
        'eccentricity': trunc_normal_prior(0.0960, 0.059, 0, 1),
        'periastron': normal_prior(56.63 * np.pi / 180, 77.89 * np.pi / 180),
        'reference': 'https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.282R/abstract',
    },

    '562.01': {
        'period': normal_prior(3.930600, 0.000002),
        'radius': uniform_prior(0.0, 1.0, init=0.0309),
        't0': uniform_prior(0.0, 100, init=1.0),
        'semimajor_axis': uniform_prior(10, 30, init=22.89),
        'inclination': normal_prior(89.228 * np.pi / 180, 0.483 * np.pi / 180, init=89.228 * np.pi / 180),
        'eccentricity': trunc_normal_prior(0.0, 1e-6, 0, 1), # Luque fit for circular orbit 
        'periastron': uniform_prior(-np.pi, np.pi),
        'reference': 'https://ui.adsabs.harvard.edu/abs/2023AJ....165..134O/abstract',
    },

    '776.02': {
        'period': normal_prior(8.246620, 0.000031),
        'radius': normal_prior(0.0, 1.0, init=0.03),
        't0': uniform_prior(0.0, 100, init=1.0),
        'semimajor_axis': uniform_prior(20.0, 35.0, init=27.87),
        'inclination': normal_prior(89.41 * np.pi / 180, 0.39 * np.pi / 180, init = 89.65 * np.pi / 180),
        'eccentricity': trunc_normal_prior(0.052, 0.037, 0, 1),
        'periastron': normal_prior(-67 * np.pi / 180, 117 * np.pi / 180),
        'reference': [
            'https://ui.adsabs.harvard.edu/abs/2024A&A...684A..12F/abstract',
            'https://ui.adsabs.harvard.edu/abs/2023ApJS..265....4K/abstract'
        ],
    },

    '776.01': {
        'period': normal_prior(15.665323, 0.000075),
        'radius': normal_prior(0.0, 1.0, init=0.0344),
        't0': uniform_prior(0.0, 100, init=1.0),
        'semimajor_axis': uniform_prior(30, 50, init=39.4),
        'inclination': normal_prior(89.49 * np.pi / 180, 0.3 * np.pi / 180, init=89.51 * np.pi / 180),
        'eccentricity': trunc_normal_prior(0.0890, 0.0540, 0, 1),
        'periastron': normal_prior(45 * np.pi / 180, 110 * np.pi / 180),
        'reference': [
            'https://ui.adsabs.harvard.edu/abs/2024A&A...684A..12F/abstract',
            'https://ui.adsabs.harvard.edu/abs/2023ApJS..265....4K/abstract'
        ],
    },

    '836.02': {
        'period': normal_prior(3.81673, 0.00001),
        'radius': uniform_prior(0.0, 1.0, init=0.0235),
        't0': uniform_prior(0.0, 100, init=1.0),
        'semimajor_axis': uniform_prior(10, 20, init=14),
        'inclination': normal_prior(87.57 * np.pi / 180, 0.44 * np.pi / 180, init = 87.57 * np.pi / 180),
        'eccentricity': trunc_normal_prior(0.053, 0.042, 0, 1),
        'periastron': normal_prior(9.0 * np.pi / 180, 92.0 * np.pi / 180),
        'reference': 'https://ui.adsabs.harvard.edu/abs/2023MNRAS.tmp..458H/abstract',
    },

    '836.01': {
        'period': normal_prior(8.59545, 0.00001),
        'radius': normal_prior(0.0, 1.0, init=0.0357),
        't0': uniform_prior(0.0, 100, init=1.0),
        'semimajor_axis': uniform_prior(15, 35, init=22),
        'inclination': normal_prior(88.7 * np.pi / 180, 1.5 * np.pi / 180, init = 88.7 * np.pi / 180),
        'eccentricity': trunc_normal_prior(0.0780, 0.056, 0, 1),
        'periastron': uniform_prior(-np.pi, np.pi),
        'reference': 'https://ui.adsabs.harvard.edu/abs/2023MNRAS.tmp..458H/abstract',
    },
}