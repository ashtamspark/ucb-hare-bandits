import numpy as np
from numba import njit

@njit    
def sample_reward(mean, sigma2, env_type):
    """
    JIT-compiled reward sampler.
    Parameters:
    - mean (float): The expected reward of the arm.
    - sigma2 (float): The variance proxy.
    - env_type (int): Determines the distribution family.
        0: Bernoulli
        1: Gaussian
        2: Mixed Sub-Gaussian (Uniform, Gaussian, or Triangular based on mean)
    """
    u = np.random.rand()  
    if env_type == 0:
        return 1.0 if u < mean else 0.0
    elif env_type == 1:
        return np.random.normal(mean, np.sqrt(sigma2))
    else:
        # Example sub-gaussian mixture for robust testing
        if 10 <= mean < 40:
            return np.random.uniform(mean - np.sqrt(sigma2), mean + np.sqrt(sigma2))
        elif 40 <= mean < 70:
            return np.random.normal(mean, np.sqrt(sigma2))
        else: 
            d = np.sqrt(6.0 * sigma2)     
            return np.random.triangular(mean - d, mean, mean + d)