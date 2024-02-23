import numpy as np
import h5py
import utils
from tqdm import tqdm

# parameters
# ----------
frames    = D = 1000
rotations = M = 50
classes   = C = 10
J         = 64
s         = (0, slice(None, 64, None), slice(None, 64, None))
shape_2d  = (64, 64)
minval    = 1e-15
beta      = 0.001
iters     = 4




# get input
# ---------
with h5py.File('data.cxi', 'r') as f:
    # inner pixels
    photon_energy  = f['/entry_1/instrument_1/source_1/energy'][0]
    m    = f['/entry_1/instrument_1/detector_1/mask'][()]
    mask = np.zeros_like(m)
    mask[s] = m[s]
    inds = np.where(mask)[0]
    B    = f['/entry_1/instrument_1/detector_1/background'][()][mask]
    b    = f['/static_emc/background_weights'][:frames, 0]
    xyz  = f['/entry_1/instrument_1/detector_1/xyz_map'][()][:, mask]
    z    = f['/entry_1/instrument_1/detector_1/distance'][()]
    K    = f['entry_1/data_1/data'][()][:frames, mask]
    pixel_area = f['/entry_1/instrument_1/detector_1/pixel_area'][()]

pixels = K.shape[1]

xyz[2] = z
assert(np.allclose(xyz[2], z))





# initialise
# ----------
I    = np.random.random((classes, J, J))
W    = np.empty((classes, rotations, pixels))
w    = np.ones((frames,))
logR = np.zeros((frames, classes, rotations))
P    = np.zeros((frames, classes, rotations))

# location of zero pixel in merge
i0 = np.float32(J // 2)
r  = np.sum(xyz[:2]**2, axis=0)**0.5
rmax_merge = r.max()
if (J % 2) == 0 :
    dx = rmax_merge / (J / 2 - 1)
else :
    dx = 2 * rmax_merge / (J - 1)
x  = dx * (np.arange(J) - i0)
points_I = (x.copy(), x.copy())

# solid angle and polarisation correction
C = utils.solid_angle_polarisation_factor(xyz, pixel_area, polarisation_axis = 'x')

# rotation matrices
R = utils.calculate_rotation_matrices(M)


for i in range(1): 
    # Expand 
    # ------
    utils.expand(points_I, I, W, xyz, R, C, minval)


    # Probability matrix
    # ------------------
    utils.calculate_probability_matrix(w, W, b, B, K, logR, P, beta)


    # Maximise
    # --------
    utils.update_W(w, W, b, B, P, K, minval, iters)
    utils.update_w(w, W, b, B, P, K, minval, iters)
    #utils.update_b(w, W, b, B, P, K, minval, iters + 2)


    # Compress
    # --------
    utils.compress(W, R, xyz, i0, dx, I)

# show W
# --------
ims = utils.make_W_ims(W, mask, s, shape_2d)
