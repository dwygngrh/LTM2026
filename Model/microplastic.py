import numpy as np
from numba import njit

@njit
def get_polymer_density(polymer_type):
    """Specific density mapping from Pilechi et al. (2022) Table 1[cite: 2]."""
    if polymer_type == "PE": return 910.0 
    if polymer_type == "HDPE": return 950.0
    return 1025.0

@njit
def calculate_3d_ws(rho_p, d_mm, rho_bf, bt_mm, shape):
    """3D sinking based on Jalon-Rojas et al. (2019) modular framework[cite: 1]."""
    g, rho_w, nu = 9.81, 1025.0, 1e-6
    R0, BT = (d_mm/2.0)/1000.0, bt_mm/1000.0
    vol_ratio = (R0**3) / ((R0 + BT)**3)
    rho_fouled = (rho_p * vol_ratio) + (rho_bf * (1.0 - vol_ratio))
    d_star = 2 * (R0 + BT) * (g * (rho_fouled - rho_w) / (rho_w * nu**2))**(1/3)
    
    if shape == "sphere":
        return (nu / (2 * (R0 + BT))) * d_star**3 * (38.1 + 0.93 * d_star**(12/7))**-0.875
    return (np.pi / 2) * (1/nu) * g * ((rho_fouled - rho_w)/rho_w) * (2*R0*R0 / 55.0)
