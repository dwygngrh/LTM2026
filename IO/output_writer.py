import xarray as xr
import os
import numpy as np
import pandas as pd
import netCDF4 as nc

class TrajectoryWriter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)

    def create_file(self, name, start_sim, end_sim, dt, total_particles, init_coords):
        file_path = os.path.join(self.output_dir, f"{name}.nc")
        times = pd.date_range(start=start_sim, end=end_sim, freq=f"{dt}s")
        lon0, lat0, z0 = init_coords

        ds = xr.Dataset(
            {
                "lon": (["time", "particle"], np.full((len(times), total_particles), lon0, dtype="f4")),
                "lat": (["time", "particle"], np.full((len(times), total_particles), lat0, dtype="f4")),
                "z": (["time", "particle"], np.full((len(times), total_particles), z0, dtype="f4"))
            },
            coords={"time": times, "particle": np.arange(total_particles)}
        )
        ds.to_netcdf(file_path, engine='netcdf4')
        ds.close()

    def write_step(self, loc_name, time, lons, lats, depths):
        file_path = os.path.join(self.output_dir, f"{loc_name}.nc")
        n_active = len(lons)
        if n_active == 0: return

        with nc.Dataset(file_path, 'r+') as root:
            time_var = root.variables['time']
            target_time = nc.date2num(time, units=time_var.units, calendar=time_var.calendar)
            # Find exact index for the current simulation time
            time_idx = np.where(time_var[:] == target_time)[0][0]

            root.variables['lon'][time_idx, 0:n_active] = lons
            root.variables['lat'][time_idx, 0:n_active] = lats
            root.variables['z'][time_idx, 0:n_active] = depths
