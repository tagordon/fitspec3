import numpy as np

log_likelihood = lambda y, mu, err: -0.5 * np.sum((y - mu)**2 / (err**2) + np.log(err**2))

class uniform_prior:
    '''
    A uniform prior. 

    Args:
        low (float): lower bound
        high (float): upper bound
        init (float, optional): where to 
        initialize the sampler, otherwise 
        set to the mid-point of the prior.
    '''

    def __init__(self, low, high, init=None):

        self.low = low
        self.high = high

        if init is None:
            self.init = 0.5 * (high + low)
        else:
            self.init = init

    def prior(self, x):
        '''
        Evaluate the distribution at a point x, and 
        return its log-probability.
        '''

        if (x >= self.low) & (x <= self.high):
            return 0.0
        else:
            return -np.inf

class normal_prior:

    '''
    A normal prior

    Args: 
        mu (float): the mean of the normal distribution 
        sig (float): the standard deviation of the 
        normal distribution 
        init (float, optional): where to initialize the 
        sampler, otherwise set to mu
    '''

    def __init__(self, mu, sig, init=None):

        self.mu = mu
        self.sig = sig

        if init is None:
            self.init = mu
        else:
            self.init = init

    def prior(self, x):
        '''
        Evaluate the distribution at a point x, and 
        return its log-probability.
        '''
        return -0.5 * (x - self.mu)**2 / self.sig**2

class trunc_normal_prior:
    '''
    A truncated normal distribution

    Args: 
        mu (float): the mean of the normal distribution 
        sig (float): the standard deviation of the normal distribution
        low (float): lower bound
        high (float): upper bound
        init (float, optional): where to initialize the sampler, 
        otherwise set to mu
    '''

    def __init__(self, mu, sig, low, high, init=None):

        self.mu = mu
        self.sig = sig
        self.low = low
        self.high = high

        if init is None:
            self.init = mu
        else:
            self.init = init

    def prior(self, x):
        '''
        Evaluate the distribution at a point x, and 
        return its log-probability.
        '''

        if (x >= self.low) & (x <= self.high):
            return -0.5 * (x - self.mu)**2 / self.sig**2
        else:
            return -np.inf