from numba import njit

@njit
def get_leeway(debris_type):
    """Windage coefficients based on debris type and buoyancy[cite: 1, 2]."""
    if debris_type == "macro_plastic": return 0.03
    elif debris_type == "human": return 0.04
    elif debris_type == "wood": return 0.02
    elif debris_type == "styrofoam": return 0.05
    return 0.01
