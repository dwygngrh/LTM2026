import numpy as np
from numba import njit, prange

@njit
def interp(x, y, xg, yg, data):
    """
    Bilinear interpolation for environmental grids[cite: 9].
    """
    if x < xg[0] or x > xg[-1] or y < yg[0] or y > yg[-1]: 
        return 9.96921e+36
    
    # Find grid cell indices[cite: 9]
    i = np.searchsorted(xg, x) - 1
    j = np.searchsorted(yg, y) - 1
    
    if i < 0 or j < 0 or i >= len(xg)-1 or j >= len(yg)-1: 
        return 9.96921e+36
    
    # Calculate fractional distance within the cell[cite: 9]
    dx = (x - xg[i]) / (xg[i+1] - xg[i])
    dy = (y - yg[j]) / (yg[j+1] - yg[j])
    
    # Note: data indexing follows (y_lat, x_lon) order[cite: 9]
    v00 = data[j, i]
    v10 = data[j, i+1]
    v01 = data[j+1, i]
    v11 = data[j+1, i+1]
    
    return (1-dx)*(1-dy)*v00 + dx*(1-dy)*v10 + (1-dx)*dy*v01 + dx*dy*v11

@njit(parallel=True)
def update_3d_safe(lons, lats, depths, active_mask, u, v, uw, vw, l_c, a_c, l_w, a_w, dt, leeway, ws, kh, kz):
    """
    Lagrangian transport kernel with 17 arguments including diffusion.
    Implements Advection + Random Walk.
    """
    LAND = 9.96921e+36
    DEG_TO_M = 111320.0 # Approximate meters per degree
    
    for i in prange(len(lons)):
        if active_mask[i] == 0: 
            continue
        
        # 1. Fetch Advection Components[cite: 9]
        uc = interp(lons[i], lats[i], l_c, a_c, u)
        vc = interp(lons[i], lats[i], l_c, a_c, v)
        
        # Land check[cite: 9]
        if abs(uc - LAND) < 1e+30:
            active_mask[i] = 0
            continue

        u_total, v_total = uc, vc
        
        # Apply windage at surface[cite: 1, 9]
        if depths[i] <= 0.05:  
            u_total += leeway * interp(lons[i], lats[i], l_w, a_w, uw)
            v_total += leeway * interp(lons[i], lats[i], l_w, a_w, vw)

        # 2. Stochastic Random Walk (Diffusion)
        # Calculate unique displacement for each particle[cite: 1, 8]
        r_lon = np.random.normal(0, 1)
        r_lat = np.random.normal(0, 1)
        
        # dx = sqrt(2 * Kh * dt) * R[cite: 8]
        diff_x = (np.sqrt(2 * kh * dt) * r_lon) / (DEG_TO_M * np.cos(np.radians(lats[i])))
        diff_y = (np.sqrt(2 * kh * dt) * r_lat) / DEG_TO_M

        # 3. Predict New Position[cite: 2, 8]
        adv_x = (u_total * dt) / (DEG_TO_M * np.cos(np.radians(lats[i])))
        adv_y = (v_total * dt) / DEG_TO_M
        
        new_lon = lons[i] + adv_x + diff_x
        new_lat = lats[i] + adv_y + diff_y
        
        # Collision Detection[cite: 2, 9]
        if abs(interp(new_lon, new_lat, l_c, a_c, u) - LAND) < 1e+30:
            active_mask[i] = 0
        else:
            lons[i], lats[i] = new_lon, new_lat
            
            # Vertical transport: Settling + Vertical Random Walk[cite: 8]
            if kz > 0:
                r_z = np.random.normal(0, 1)
                depths[i] += (ws * dt) + (np.sqrt(2 * kz * dt) * r_z)
            else:
                depths[i] += ws * dt
                
            # Boundary control for depth[cite: 9]
            if depths[i] < 0: depths[i] = 0.0
