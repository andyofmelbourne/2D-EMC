import h5py
import numpy as np
import scipy.constants as sc
from tqdm import tqdm

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
    shape = f['entry_1/data_1/data'].shape
    dtype = f['entry_1/data_1/data'].dtype
    data = np.empty((frames,) + shape[1:], dtype=dtype)
    for i, d in tqdm(enumerate(np.where(tags)[0][:frames]), total = frames, desc = 'loading frames'):
        data[i] = f['entry_1/data_1/data'][d]
    
    datasets = ['/entry_1/instrument_1/detector_1/mask',
                '/entry_1/instrument_1/detector_1/xyz_map',
                '/entry_1/instrument_1/detector_1/x_pixel_size',
                '/entry_1/instrument_1/detector_1/y_pixel_size',
                '/entry_1/instrument_1/detector_1/pixel_area',
                '/entry_1/instrument_1/detector_1/distance',
                '/entry_1/instrument_1/detector_1/background',
                '/entry_1/instrument_1/source_1/energy',
                '/static_emc/background_weights',
                '/entry_1/data_1/data']
    
    with h5py.File('data.cxi', 'w') as g:
        for d in datasets:
            print(d)
            if 'data' in d :
                v = data
            else :
                v = f[d][()]
            
            if d in g :
                g[d][...] = v
            else :
                g[d] = v
