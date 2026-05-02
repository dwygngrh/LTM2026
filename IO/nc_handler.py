import xarray as xr
import pandas as pd
import numpy as np
import glob
import os

class DataProvider:
    def __init__(self, c_dir, e_dir):
        c_files = sorted(glob.glob(os.path.join(c_dir, "*.nc")))
        u_files = sorted(glob.glob(os.path.join(e_dir, "U10_*.nc")))
        v_files = sorted(glob.glob(os.path.join(e_dir, "V10_*.nc")))

        def rename_and_clean_time(ds):
            t_name = next((d for d in ['time', 'valid_time', 'ocean_time'] if d in ds.dims or d in ds.coords), None)
            if t_name and t_name != 'time':
                ds = ds.rename({t_name: 'time'})
            if 'time' in ds.coords:
                ds = ds.drop_duplicates('time', keep='first')
            return ds

        def process_dataset(files):
            # Performance Fix: Larger chunks for better I/O throughput[cite: 1]
            ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', 
                                   preprocess=rename_and_clean_time, coords='minimal', 
                                   compat='override', chunks={'time': 10}) 
            return ds.drop_duplicates('time', keep='first').sortby('time')

        self.cmems = process_dataset(c_files)
        self.u_wind = process_dataset(u_files)
        self.v_wind = process_dataset(v_files)

        self.lon_c, self.lat_c = self.cmems.longitude.values, self.cmems.latitude.values
        self.lon_w, self.lat_w = self.u_wind.longitude.values, self.u_wind.latitude.values

    def get_step_data(self, dt_obj, mode='microplastic'):
        """
        Fetches environmental data. If mode is marine_debris, it strictly 
        slices the surface layer to optimize performance.
        """
        if mode == 'marine_debris':
            # Efficiency Fix: Slice depth first to avoid 3D interpolation
            c_ds = self.cmems.sel(depth=0, method='nearest').interp(time=dt_obj)
        else:
            # For microplastics, keep depth for 3D transport[cite: 5, 6]
            c_ds = self.cmems.interp(time=dt_obj)

        # Extract values as numpy arrays for the Numba kernel
        u = np.nan_to_num(c_ds.uo.values, nan=9.96921e+36)
        v = np.nan_to_num(c_ds.vo.values, nan=9.96921e+36)
        
        u_var = 'u10' if 'u10' in self.u_wind.data_vars else 'U10'
        v_var = 'v10' if 'v10' in self.v_wind.data_vars else 'V10'
        
        uw = self.u_wind[u_var].sel(time=dt_obj, method='nearest').values
        vw = self.v_wind[v_var].sel(time=dt_obj, method='nearest').values
        return u, v, uw, vw
