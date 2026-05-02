import numpy as np

def spawn(src, now):
    # Only release if current time is within the release window[cite: 1]
    if not (src['start_rel'] <= now <= src['end_rel']):
        return [], [], []
    
    n = src['rate'] if src['type'] == 'active' else 0
    if n <= 0: return [], [], []
    
    r_deg = (src['diameter'] / 2.0) / 111000.0
    r = r_deg * np.sqrt(np.random.rand(n))
    theta = np.random.rand(n) * 2 * np.pi
    
    return (src['lon'] + r * np.cos(theta)).tolist(), \
           (src['lat'] + r * np.sin(theta)).tolist(), \
           [src['depth']] * n
