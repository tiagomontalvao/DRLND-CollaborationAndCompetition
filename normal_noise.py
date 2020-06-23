"""Random noise from a normal distribution."""

import copy
import numpy as np

class NormalNoise:
    """Random noise from a normal distribution."""
    def __init__(self, size, seed, sigma=0.2):
        """Initialize parameters and noise process."""
        np.random.seed(seed)
        self.size = size
        self.sigma = sigma

    def sample(self):
        """Return noise sample."""
        return np.random.normal(scale=self.sigma, size=self.size)