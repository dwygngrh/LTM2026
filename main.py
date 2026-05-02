import pandas as pd
import numpy as np
from tqdm import tqdm
from Namelist.config_parser import LTMConfig
from IO.nc_handler import DataProvider
from IO.output_writer import TrajectoryWriter
from Initial.seeding import spawn
from Hydrodynamic.kernels import update_3d_safe

def run():
    cfg = LTMConfig()
    io = DataProvider(cfg.config['Files']['cmems_dir'], cfg.config['Files']['era5_dir'])
    writer = TrajectoryWriter(cfg.config['Files']['output_dir'])
    sources = cfg.get_sources()
    start_sim, end_sim = cfg.get_simulation_times()
    dt = int(cfg.config['Simulation']['dt'])
    write_freq = int(cfg.config['Simulation'].get('write_frequency', 3600))
    mode = cfg.config['Simulation']['mode']

    # --- Diffusion & Physics Setup ---
    kh = float(cfg.config['Diffusion'].get('Kh', 1.0))
    kz = float(cfg.config['Diffusion'].get('Kz', 0.001))
    
    if mode == 'microplastic':
        # (Assuming Microplastic functions exist in your Model/microplastic.py)
        leeway = 0.0
        ws = 0.001 # Example settling velocity
    else:
        debris_type = cfg.config['MarineDebris']['type']
        leeway = float(cfg.config['MarineDebris'].get(f'leeway_{debris_type}', 0.03))
        ws = 0.0

    # --- Pre-allocation ---
    for s in sources:
        duration_hours = (s['end_rel'] - s['start_rel']).total_seconds() / 3600.0
        total_particles = int(np.ceil(s['rate'] * duration_hours))
        writer.create_file(s['name'], start_sim, end_sim, dt, total_particles, (s['lon'], s['lat'], s['depth']))

    p_data = {s['name']: {'lon': [], 'lat': [], 'z': [], 'active': []} for s in sources}
    cur = start_sim
    total_steps = int((end_sim - start_sim).total_seconds() / dt) + 1
    pbar = tqdm(total=total_steps, desc="LTM Simulation")

    while cur <= end_sim:
        u, v, uw, vw = io.get_step_data(cur, mode=mode)
        
        for s in sources:
            name = s['name']
            # Seed strictly once per hour to avoid exceeding pre-allocated NetCDF slots[cite: 5]
            if cur.minute == 0 and cur.second == 0 and (s['start_rel'] <= cur < s['end_rel']):
                nl, na, nz = spawn(s, cur)
                p_data[name]['lon'].extend(nl)
                p_data[name]['lat'].extend(na)
                p_data[name]['z'].extend(nz)
                p_data[name]['active'].extend([1] * len(nl))
            
            if p_data[name]['lon']:
                lo, la, zz, act = np.array(p_data[name]['lon']), np.array(p_data[name]['lat']), \
                                  np.array(p_data[name]['z']), np.array(p_data[name]['active'])
                
                # kernel update including Kh and Kz for the random walk
                update_3d_safe(lo, la, zz, act, u, v, uw, vw, io.lon_c, io.lat_c, 
                               io.lon_w, io.lat_w, dt, leeway, ws, kh, kz)
                
                p_data[name]['lon'], p_data[name]['lat'] = lo.tolist(), la.tolist()
                p_data[name]['z'], p_data[name]['active'] = zz.tolist(), act.tolist()
                
                if (cur - start_sim).total_seconds() % write_freq == 0:
                    writer.write_step(name, cur, lo.tolist(), la.tolist(), zz.tolist())
        
        cur += pd.Timedelta(seconds=dt)
        pbar.update(1)
    pbar.close()

if __name__ == "__main__": run()
