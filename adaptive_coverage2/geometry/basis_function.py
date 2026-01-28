import numpy as np

def basis_function(x: float, y: float):
   
    return np.array([
        1.0,
        x,
        y,
        x**2,
        y**2,
        x*y,
        x**3,
        y**3,
        x**2 * y
    ]).reshape(-1, 1)
