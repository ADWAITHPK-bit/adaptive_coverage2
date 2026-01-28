import numpy as np
from adaptive_coverage2.geometry.basis_function import basis_function

def estimated_density(x, y, a_hat):
    phi = basis_function(x, y)
    return float(phi.T @ a_hat), phi
