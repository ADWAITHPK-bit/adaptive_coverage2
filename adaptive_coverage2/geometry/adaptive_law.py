import numpy as np

class AdaptiveEstimator:
   

    def __init__(self, n_basis, gamma=0.5, a_min=0.01):
        self.gamma = gamma
        self.a_min = a_min
        self.Lambda = np.zeros((n_basis, n_basis))
        self.lambda_vec = np.zeros((n_basis, 1))

    def update_memory(self, phi, lambda_measured, dt):
        self.Lambda += phi @ phi.T * dt
        self.lambda_vec += phi * lambda_measured * dt

    def projection(self, a_hat):
        return np.maximum(a_hat, self.a_min)