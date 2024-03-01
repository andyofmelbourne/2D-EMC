import numpy as np
import h5py
from emc_2d import utils_cl
from emc_2d import utils
from tqdm import tqdm
import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


"""
Assume W is too big to fit in memory
"""

# load configuration file
config = utils.load_config(sys.argv[1] + '/config.py')

# take the reconstruction directory as the argument
with h5py.File(sys.argv[1] + '/recon.h5', 'r') as f:
    I = f['models'][()]
    w = f['fluence'][()]
    logR = f['logR'][()]
    P = f['probability_matrix'][()]
    B = f['background'][()]
    b = f['background_weights'][()]
    points_I = f['model_xy_map'][()]
    C = f['solid_angle_polarisation_factor'][()]
    R = f['rotation_matrices'][()]
    iteration = f['iterations/iters'][()]
    dx = f['model_voxel_size'][()]

frames, classes, rotations = P.shape
pixels                     = B.shape[-1]
J                          = I.shape[1]

# load data
if rank == 0 :
    with h5py.File(sys.argv[1] + '/data.cxi') as f:
        xyz  = f['/entry_1/instrument_1/detector_1/xyz_map'][()]
        data = f['entry_1/data_1/data']
        indices = f['entry_1/instrument_1/detector_1/pixel_indices'][()]
        frame_shape = f['entry_1/instrument_1/detector_1/frame_shape'][()]
        K = np.zeros((frames, pixels), dtype = data.dtype)
        for d in tqdm(range(1), desc = 'loading data'):
            data.read_direct(K, np.s_[:], np.s_[:])
else :
    K = xyz = None

K   = comm.bcast(K, root=0)
xyz = comm.bcast(xyz, root=0)

Ksums = np.sum(K, axis=1)

minval = 1e-10
iters  = 4
i0     = J // 2

# split classes by rank
my_classes = list(range(rank, classes, size))

# split frames by rank
my_frames = list(range(rank, frames, size))

W    = np.empty((classes, rotations, pixels))

for i in range(iteration, iteration + config['iters']): 
    beta = config['betas'][min(config['iters']-1, i)]
    
    # Probability matrix
    # ------------------
    c = utils_cl.Prob(C, R, K, w, I, b, B, logR, P, xyz, dx, beta)
    expectation_value, log_likihood = c.calculate()
    del c
    
    # Maximise + Compress
    # -------------------
    cW = utils_cl.Update_W(w, I, b, B, P, K, C, R, xyz, dx, pixels, minval = 1e-10, iters = iters)
    cW.update()
    Wsums = cW.Wsums.copy()
    del cW
    
    # this will only help next iteration
    cw = utils_cl.Update_w(Ksums, Wsums, P, w, I, b, B, K, C, R, dx, xyz, frames, iters)
    cw.update()
    
    cb = utils_cl.Update_b(B, Ksums, cw)
    cb.update()
    del cb; del cw

    # Save
    # ----
    if rank == 0 : 
        utils.save(sys.argv[1], w, b, P, logR, I, beta, expectation_value, log_likihood, i)
        utils.plot_iter(sys.argv[1])
    
