import numpy as np
import h5py
from emc_2d import utils
from tqdm import tqdm
import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
with h5py.File(sys.argv[1] + '/data.cxi') as f:
    xyz  = f['/entry_1/instrument_1/detector_1/xyz_map'][()]
    data = f['entry_1/data_1/data']
    K = np.zeros((frames, pixels), dtype = data.dtype)
    for d in tqdm(range(frames), desc = 'loading data'):
        K[d] = f['entry_1/data_1/data'][d]

minval = 1e-10
iters  = 4
i0     = J // 2

# initialise
# ----------
W    = np.empty((classes, rotations, pixels))

# split classes by rank
my_classes = list(range(rank, classes, size))

# split frames by rank
my_frames = list(range(rank, frames, size))


for i in range(iteration, config['iters']): 
    # Expand 
    # ------
    utils.expand(my_classes, points_I, I, W, xyz, R, C, minval)
    utils.allgather(W, axis=0)
    
    
    # Probability matrix
    # ------------------
    beta = config['betas'][i]
    utils.calculate_probability_matrix(my_frames, w, W, b, B, K, logR, P, beta)
    utils.allgather(P, axis=0)
    utils.allgather(logR, axis=0)
    if rank == 0 :
        expectation_value, log_likihood = utils.calculate_P_stuff(P, logR, beta)
    
    
    # Maximise
    # --------
    utils.update_W(my_classes, w, W, b, B, P, K, minval, iters)
    utils.allgather(W, axis=0)

    # check
    #utils.calculate_probability_matrix(my_frames, w, W, b, B, K, logR, P.copy(), beta)
    #utils.allgather(logR, axis=0)
    #if rank == 0 :
    #    expectation_value, log_likihood = utils.calculate_P_stuff(P, logR)
    
    utils.update_w(my_frames, w, W, b, B, P, K, minval, iters)
    utils.allgather(w, axis=0)
    
    utils.update_b(my_frames, w, W, b, B, P, K, minval, iters)
    utils.allgather(b, axis=0)
    
    
    # Compress
    # --------
    utils.compress(my_classes, W, R, xyz, i0, dx, I)
    utils.allgather(I, axis=0)
    
    
    # Save
    # ----
    if rank == 0 : 
        utils.save(sys.argv[1], w, b, P, logR, I, beta, expectation_value, log_likihood, i)
        utils.plot_iter(sys.argv[1])

# show W
# --------
#ims = utils.make_W_ims(W, mask, s, shape_2d)
