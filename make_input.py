import h5py
import numpy as np
import scipy.constants as sc

"""
(base) :~/Documents/git_repos/2D-EMC$ h5ls -r ~/Documents/2023/P3004-take-2/gold/hits_r0087.cxi 
/                        Group
/entry_1                 Group
/entry_1/cellId          Dataset {59087}
/entry_1/data_1          Group
/entry_1/data_1/data     Soft Link {/entry_1/instrument_1/detector_1/data}
/entry_1/experiment_identifier Dataset {59087}
/entry_1/instrument_1    Group
/entry_1/instrument_1/data_1 Group
/entry_1/instrument_1/detector_1 Group
/entry_1/instrument_1/detector_1/data Dataset {59087, 16, 128, 512}
/entry_1/instrument_1/detector_1/good_pixels Dataset {16, 128, 512}
/entry_1/instrument_1/detector_1/xyz_map Dataset {3, 16, 128, 512}
/entry_1/instrument_1/name Dataset {SCALAR}
/entry_1/pulseId         Dataset {59087}
/entry_1/sample_1        Group
/entry_1/sample_1/name   Dataset {SCALAR}
/entry_1/trainId         Dataset {59087}
/misc                    Group
/static_emc              Group
/static_emc/class        Dataset {59087}
/static_emc/good_classes Dataset {219}
/static_emc/good_hit     Dataset {59087}
"""

frames = 1000

with h5py.File('/home/andyofmelbourne/Documents/2023/P3004-take-2/gold/hits_r0087.cxi', 'r') as f:
    tags = f['/static_emc/good_hit'][()]
    data = f['entry_1/data_1/data'][np.where(tags)[0][:frames]]
    xyz  = f['/entry_1/instrument_1/detector_1/xyz_map'][()]
    x_pixel_size = np.abs(xyz[0, 0, 0, 1] - xyz[0, 0, 0, 0])
    y_pixel_size = np.abs(xyz[0, 0, 1, 0] - xyz[0, 0, 0, 0])
    pixel_area   = 3 * 3**0.5 / 2 * 136e-6**2 # for hexaginal pixels
    
    with h5py.File('data.cxi', 'w') as g:
        g['entry_1/data_1/data'] = data
        g['/entry_1/instrument_1/detector_1/mask'] = f['/entry_1/instrument_1/detector_1/good_pixels'][()]
        g['/entry_1/instrument_1/detector_1/xyz_map'] = xyz
        g['/entry_1/instrument_1/detector_1/x_pixel_size'] = x_pixel_size
        g['/entry_1/instrument_1/detector_1/y_pixel_size'] = y_pixel_size
        g['/entry_1/instrument_1/detector_1/distance'] = 0.552
        g['/entry_1/instrument_1/detector_1/pixel_area'] = pixel_area
        g['/entry_1/instrument_1/source_1/energy'] = 3e3 * sc.e * np.ones((frames,), dtype=float)
        g['/entry_1/instrument_1/source_1/wavelength'] = sc.h * sc.e / (3e3 * sc.e) * np.ones((frames,), dtype=float)
    

