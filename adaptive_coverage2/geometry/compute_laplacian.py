import numpy as np

def laplacian_consensus(a_hat, neighbor_a_hat, l_ij, zeta=0.1):
    
    
    a_dot = np.zeros_like(a_hat)

    for name, a_j in neighbor_a_hat.items():
        w = l_ij.get(name, 0.0)
        a_dot -= zeta * w * (a_hat - a_j)

    return a_dot