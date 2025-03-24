import numpy as np

class MyRNG:
    def __init__(self):
        self.rng = np.random.default_rng(1234)  # Random number generator with a seed
