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
    for d in tqdm(range(1), desc = 'loading data'):
        with h5py.File(sys.argv[1] + '/data.cxi') as f:
            xyz  = f['/entry_1/instrument_1/detector_1/xyz_map'][()].astype(np.float32)
            litpix = f['entry_1/data_1/litpix'][()].astype(np.int32)
            start_inds = np.concatenate(([0,], np.cumsum(litpix))).astype(np.int32)
            #K      = f['entry_1/data_1/data'][()]
            #inds   = f['entry_1/data_1/inds'][()].astype(np.int32)
            K      = np.split(f['entry_1/data_1/data'][()], start_inds[1:-1])
            inds   = np.split(f['entry_1/data_1/inds'][()].astype(np.int32), start_inds[1:-1])
            Ksums  = f['entry_1/data_1/photons'][()].astype(np.int32)
            indices = f['entry_1/instrument_1/detector_1/pixel_indices'][()]
            frame_shape = f['entry_1/instrument_1/detector_1/frame_shape'][()]
            #K = np.zeros((frames, pixels), dtype = data.dtype)
            #for d in tqdm(range(1), desc = 'loading data'):
            #    data.read_direct(K, np.s_[:], np.s_[:])
else :
    K = xyz = inds = Ksums = litpix = start_inds = None

K     = comm.bcast(K, root=0)
inds  = comm.bcast(inds, root=0)
Ksums = comm.bcast(Ksums, root=0)
xyz        = comm.bcast(xyz, root=0)
litpix     = comm.bcast(litpix, root=0)
start_inds = comm.bcast(start_inds, root=0)

# make dense K for testing
K_dense = np.zeros((frames, pixels), dtype = K[0].dtype)
for d in range(frames):
    K_dense[d, inds[d]] = K[d]

minval = 1e-10
iters  = 6

Wsums = utils_cl.calculate_Wsums(C, R, I, xyz, dx)

for i in range(iteration, iteration + config['iters']): 
    beta     = config['betas'][min(config['iters']-1, i)]
    update_b = config['update_b'][min(config['iters']-1, i)]

    # Probability matrix
    # ------------------
    c = utils_cl.Prob(C, R, inds, K, w, I, b, B, logR, P, xyz, dx, beta)
    expectation_value, log_likihood = c.calculate()
    if rank == 0 : print('expectation value: {:.6e}'.format(np.sum(P * logR) / beta))
    
    # Maximise + Compress
    # -------------------
    cW = utils_cl.Update_W(w, I, b, B, P, inds, K, C, R, xyz, dx, pixels, minval = 1e-10, iters = iters)
    cW.update()
    Wsums = cW.Wsums.copy()
    del cW
    
    break
    #c = utils_cl.Prob(C, R, K, w, I, b, B, logR, P.copy(), xyz, dx, beta)
    #expectation_value, log_likihood = c.calculate()
    #del c
    #if rank == 0 : print('expectation value: {:.6e}'.format(np.sum(P * logR) / beta))
    
    cw = utils_cl.Update_w(Ksums, Wsums, P, w, I, b, B, K_dense, C, R, dx, xyz, frames, iters)
    cw.update()
    
    if update_b :
        cb = utils_cl.Update_b(B, Ksums, cw)
        cb.update()
        del cb

    del cw

    # Save
    # ----
    if rank == 0 : 
        utils.save(sys.argv[1], w, b, P, logR, I, beta, expectation_value, log_likihood, i)
        utils.plot_iter(sys.argv[1])
    

#ims = utils.make_W_ims(cw.W, indices, frame_shape)
