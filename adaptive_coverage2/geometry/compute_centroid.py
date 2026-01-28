import numpy as np
from adaptive_coverage2.geometry.compute_density import estimated_density

def in_voronoi_cell(q, p_i, neighbors):
  
    for p_j in neighbors.values():
        p_j = np.array(p_j).reshape(2,1)
        if np.linalg.norm(q - p_i) > np.linalg.norm(q - p_j):
            return False
    return True


def compute_centroid(position, a_hat, neighbors):
    x_i, y_i = position.flatten()

    grid = 0.6
    res = 0.05

    xs = np.arange(x_i - grid, x_i + grid, res)
    ys = np.arange(y_i - grid, y_i + grid, res)

    M_i = 0.0
    c_num = np.zeros((2,1))

    for x in xs:
        for y in ys:
            q = np.array([[x],[y]])

            if neighbors and not in_voronoi_cell(q, position, neighbors):
                continue

            lam, _ = estimated_density(x, y, a_hat)
            if lam <= 0:
                continue

            dA = res**2
            M_i += lam * dA
            c_num += lam * q * dA

    if M_i < 1e-6:
        return position, None, 0.0

    c_i = c_num / M_i
    _, phi_i = estimated_density(x_i, y_i, a_hat)

    return c_i, phi_i, M_i