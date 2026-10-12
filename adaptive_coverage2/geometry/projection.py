import numpy as np

def project(a_hat, a_min=0.01):
    return np.maximum(a_hat, a_min)
