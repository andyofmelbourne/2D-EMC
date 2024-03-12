import numpy as np
import pyopencl as cl
import pyopencl.array 
from tqdm import tqdm
import sys
import pathlib
import math
import time

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0 :
    silent = False
else :
    silent = True

# find an opencl device (preferably a GPU) in one of the available platforms
done = False
for p in cl.get_platforms():
    devices = p.get_devices(cl.device_type.GPU)
    if (len(devices) > 0) and ('NVIDIA' in p.name):
        done = True
        break

if not done :
    for p in cl.get_platforms():
        devices = p.get_devices(cl.device_type.GPU)
        if (len(devices) > 0) :
            break
    
if len(devices) == 0 :
    for p in cl.get_platforms():
        devices = p.get_devices()
        if len(devices) > 0:
            break

print('number of devices:', len(devices))
print(rank, 'my device:', devices[rank % len(devices)])
sys.stdout.flush()
context = cl.Context([devices[rank % len(devices)]])
queue   = cl.CommandQueue(context)

# need an out-of-order queue
queue2 = cl.CommandQueue(context, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)

code = pathlib.Path(__file__).resolve().parent.joinpath('utils.c')
cl_code = cl.Program(context, open(code, 'r').read()).build()

def calculate_Wsums(C, R, I, xyz, dx):
    """
    Wsums[t, r] = sum_i C[i] I[u[i, r], v[i, r]] 
    """
    if rank == 0 :
        i0 = np.float32(I.shape[-1] // 2)
        dx = np.float32(dx)

        classes   = np.int32(I.shape[0])
        rotations = np.int32(R.shape[0])
        pixels    = np.int32(C.shape[-1])
        
        C_cl     = cl.array.empty(queue, C.shape      , dtype = np.float32)
        R_cl     = cl.array.empty(queue, R.shape      , dtype = np.float32)
        rx_cl    = cl.array.empty(queue, xyz[0].shape , dtype = np.float32)
        ry_cl    = cl.array.empty(queue, xyz[1].shape , dtype = np.float32)
        Wsums_cl = cl.array.empty(queue, (classes, rotations), dtype = np.float32)

        Wsums = np.empty((classes, rotations), dtype = np.float32)
        
        cl.enqueue_copy(queue, C_cl.data, C)
        cl.enqueue_copy(queue, R_cl.data, R)
        cl.enqueue_copy(queue, rx_cl.data, np.ascontiguousarray(xyz[0].astype(np.float32)))
        cl.enqueue_copy(queue, ry_cl.data, np.ascontiguousarray(xyz[1].astype(np.float32)))

        # copy I as an opencl "image" for bilinear sampling
        shape        = I.shape
        image_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
        flags        = cl.mem_flags.READ_ONLY
        I_cl         = cl.Image(context, flags, image_format, 
                                shape = shape[::-1], is_array = True)
        
        cl.enqueue_copy(queue, dest = I_cl, src = I, 
                        origin = (0, 0, 0), region = shape[::-1])
        
        for i in tqdm(range(1), desc = 'calculating tomogram sums', disable = silent) :
            cl_code.calculate_tomogram_sums(queue, (classes, rotations), None, 
                    Wsums_cl.data, I_cl, C_cl.data, R_cl.data, 
                    rx_cl.data, ry_cl.data,
                    i0, dx, pixels)
            queue.finish()
        
        cl.enqueue_copy(queue, src = Wsums_cl.data, dest = Wsums)
    else :
        Wsums = None
    
    Wsums = comm.bcast(Wsums, root=0)
    return Wsums
    

class Prob():
    def __init__(self, C, R, inds, K, w, I, b, B, logR, P, xyz, dx, beta):
        """
        keep i on the slow axis to speed up the sum
        
        T[i, t, r] = w[d] * W[i,t,r] + np.dot(b[d], B)[i]
        logR[t, r] = beta * sum_i K[i] log T[i, t, r] - T[i, t, r]
    
        but if we do it this way we have to calcuate the entire W for every frame (~5e5)
        seems to be pretty fast anyway...
        """
        # split frames by MPI rank
        self.dchunk = 2048
        self.d_list, dstart, dstop = self.my_frames(rank, P.shape[0], self.dchunk)
        self.dstart = dstart
        self.dstop  = dstop
        
        self.frames    = frames    = np.int32(dstop - dstart)
        self.classes   = classes   = np.int32(P.shape[1])
        self.rotations = rotations = np.int32(P.shape[2])
        self.pixels    = pixels    = np.int32(B.shape[-1])
         
        self.beta = np.float32(beta)
        self.dx   = np.float32(dx)
        
        self.i0 = np.float32(I.shape[-1] // 2)
        
        self.P = P
        self.logR = logR
        self.inds = inds
        self.K    = K
        
        self.LR_cl = cl.array.zeros(queue, (self.dchunk, classes, rotations), dtype = np.float32)
        self.w_cl  = cl.array.empty(queue, (frames,)   , dtype = np.float32)
        self.I_cl  = cl.array.empty(queue, I.shape   , dtype = np.float32)
        self.b_cl  = cl.array.empty(queue, (frames,)   , dtype = np.float32)
        self.B_cl  = cl.array.empty(queue, (pixels,) , dtype = np.float32)
        self.C_cl  = cl.array.empty(queue, C.shape   , dtype = np.float32)
        self.R_cl  = cl.array.empty(queue, R.shape   , dtype = np.float32)
        self.rx_cl  = cl.array.empty(queue, xyz[0].shape   , dtype = np.float32)
        self.ry_cl  = cl.array.empty(queue, xyz[1].shape   , dtype = np.float32)
        
        # load arrays to gpu
        cl.enqueue_copy(queue, self.w_cl.data, w[dstart: dstop])
        cl.enqueue_copy(queue, self.b_cl.data, b[dstart: dstop])
        cl.enqueue_copy(queue, self.B_cl.data, B)
        cl.enqueue_copy(queue, self.C_cl.data, C)
        cl.enqueue_copy(queue, self.R_cl.data, R)
        cl.enqueue_copy(queue, self.rx_cl.data, np.ascontiguousarray(xyz[0].astype(np.float32)))
        cl.enqueue_copy(queue, self.ry_cl.data, np.ascontiguousarray(xyz[1].astype(np.float32)))
        
        # copy I as an opencl "image" for bilinear sampling
        shape        = I.shape
        image_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
        flags        = cl.mem_flags.READ_ONLY
        self.I_cl    = cl.Image(context, flags, image_format, 
                                shape = shape[::-1], is_array = True)
        
        cl.enqueue_copy(queue, dest = self.I_cl, src = I, 
                        origin = (0, 0, 0), region = shape[::-1])

    def my_frames(self, r, frames, chunk = 64):
        ds = np.linspace(0, frames, size + 1).astype(int)
        dstart = ds[:-1:][r]
        dstop  = ds[1::][r]
        my_ds = np.arange(dstart, dstop, chunk, dtype=np.int32)
        return my_ds, dstart, dstop
    
    def calculate(self): 
        if not silent :
            print()
        
        self.expectation_value = 0.
        self.log_likihood      = 0.
        
        logR = self.logR
        P    = self.P
        dchunk = 64
        self.K_cl  = cl.array.empty(queue, (dchunk, self.pixels,) , dtype = np.uint8)
        K          = np.empty((dchunk, self.pixels,), dtype = np.uint8)
        
        for i_d, d in tqdm(enumerate(self.d_list), total = len(self.d_list), 
                         desc = 'calculating logR matrix', disable = silent):
            d1 = min(d+dchunk, self.dstop)
            dd = d1 - d 
            
            # make dense K over frames chunk size
            K.fill(0)
            for i, di in enumerate(range(d, d1)):
                K[i, self.inds[di]] = self.K[di]
            cl.enqueue_copy(queue, self.K_cl.data, K)
            
            cl_code.calculate_LR(queue, (self.rotations, self.classes, np.int32(dd)), None,
                    self.I_cl, self.LR_cl.data, self.K_cl.data, self.w_cl.data, 
                    self.b_cl.data, self.B_cl.data, self.C_cl.data, self.R_cl.data, 
                    self.rx_cl.data, self.ry_cl.data,
                    self.beta, self.i0, self.dx, self.pixels, np.int32(d - self.dstart))
             
            # offset in bytes for LR_cl
            cl.enqueue_copy(queue, dest = logR[d: d1], src = self.LR_cl.data)

            # test offset
            #assert(np.allclose(logR[self.dstart: d1], self.LR_cl.get()[:d1-self.dstart]))
             
            for d in range(d, d1):
                m        = np.max(logR[d])
                P[d]     = logR[d] - m
                P[d]     = np.exp(P[d])
                P[d]    /= np.sum(P[d])
                
                self.expectation_value += np.sum(P[d] * logR[d]) / self.beta
                self.log_likihood      += np.sum(logR[d])        / self.beta

            # allgather here to reduce size
            for r in range(size):
                r_ds, dstart, dstop = self.my_frames(r, self.P.shape[0], self.dchunk)
                d = r_ds[i_d]
                d1 = min(d+dchunk, dstop)
                dd = d1 - d 
                self.logR[d:d1] = comm.bcast(self.logR[d:d1], root=r)
                self.P[d:d1]    = comm.bcast(self.P[d:d1], root=r)
        
        self.expectation_value = comm.allreduce(self.expectation_value)
        self.log_likihood      = comm.allreduce(self.log_likihood)
        return self.expectation_value, self.log_likihood
    
    def allgather(self):
        for r in range(size):
            dstart = self.ds[:-1:][r]
            dstop  = self.ds[1::][r]
            self.logR[dstart:dstop] = comm.bcast(self.logR[dstart:dstop], root=r)
            self.P[dstart:dstop]    = comm.bcast(self.P[dstart:dstop], root=r)
        
        self.expectation_value = comm.allreduce(self.expectation_value)
        self.log_likihood      = comm.allreduce(self.log_likihood)

class Update_W():
    """
    #- gW[t, r, i] = sum_d w[d] P[d, t, r] (K[d, i] / T[d, t, r, i] - 1)
    #    = sum_d P[d, t, r] K[d, i] w[d] / T[d, t, r, i]  - \sum_d w[d] P[d, t, r] 
    
    # xmax[t, r, i] = sum_d P[d, t, r] K[d, i] / \sum_d w[d] P[d, t, r] 
    
    c[t, r]       = sum_d w[d] P[d, t, r]
    xmax[t, r, i] = sum_d P[d, t, r] K[d, i] / c[t, r]
    
    loop iters:
        T[d, t, r, i] = W[t, r, i] + b[d] B[i] / w[d]
        PK[d, t, r]   = P[d, t, r] K[d, i]
        f[t, r, i]    = sum_d PK[d, t, r] / T[d, t, r, i]
        g[t, r, i]    =-sum_d PK[d, t, r] / T[d, t, r, i]^2
        
        step[t, r, i] = f[t, r, i] / g[t, r, i] * (1 - f[t, r, i] / c[t, r])
            
        W[t, r, i] += step[t, r, i]
    
    then W --> I, overlap
    
    sum over d is good for gpu
    if every worker has a pixel then we need to loop over all t, r
        this makes it easy to make use of P-sparsity
        we would need to store all K values unless chunking is employed
    
    I wonder if a dot product approach is better...
    
    Non-sparse:
    chunk over pixels
    load B
    load I
    load all w
    load all b
    load all K  # big ~300 GB x 10000 at worst ~300 GB at best
    loop over (t, r) with pixel-chunking 
        - sparsity: find frames with low P value
        load P[:, t, r] # small
        c       = sum_d w[d] P[d]
        xmax[i] = sum_d P[d] K[d, i] / c
        
        loop iters:
            calculate W[i] <-- I
            loop d :
                T[i]    = W[i] + b[d] B[i] / w[d]
                PK      = P[d] K[d, i] 
                f[i]   += PK / T[i]
                g[i]   -= PK / T[i]^2
            
            step[i] = f[i] / g[i] * (1 - f[i] / c)
                
            W[i] += step[i]
            
            merge W[i] --> I, overlap
    
    we could reduce global memory reads of P[d] with local mem.
    this approach has the disadvantage that we need to load all of K
    at for every t, r combo.
        if we loop over pixels and keep (t, r) as workers then we only
        need to load K once, but then we cannot take advantage of 
        P-sparsity...
    """
    
    def __init__(self, w, I, b, B, P, inds, K, C, R, xyz, dx, pixel_chunk_size, minval = 1e-10, iters = 4):
        if not silent :
            print()
            print('initialising update_W routine')
        
        # split classes by MPI rank
        self.my_classes = np.arange(rank, I.shape[0], size)

        # chunk over pixels
        pixels    = B.shape[-1]
        pc = 2**12
        #pc = 25058//2
        i = np.arange(0, pixels+pc, pc, dtype=np.int32)
        self.istart = i[:-1]
        self.istop  = np.clip(i[1:], 0, pixels)
        
        self.pixels    = pixels    = np.int32(pc)
        self.frames    = frames    = np.int32(P.shape[0])
        self.classes   = classes   = np.int32(P.shape[1])
        self.rotations = rotations = np.int32(P.shape[2])
        self.iters     = np.int32(iters)
        self.R         = R
        self.i0        = I.shape[-1] // 2
        self.dx        = dx
        self.minval    = minval
        self.P         = P
        
        self.rx = np.ascontiguousarray(xyz[0].astype(np.float32))
        self.ry = np.ascontiguousarray(xyz[1].astype(np.float32))
        
        # for merging
        self.I = I
        self.I.fill(0)
        self.overlap2 = np.zeros(I.shape, dtype=float)
        self.C        = C
        self.B        = B
        self.K        = K
        self.inds     = inds
          
        self.points = np.empty((2, pixels))
        self.mi     = np.empty((pixels,), dtype=int)
        self.mj     = np.empty((pixels,), dtype=int)
        
        # initialise gpu arrays
        self.B_cl  = cl.array.empty(queue, (pixels,)              , dtype = np.float32)
        self.K_cl  = cl.array.empty(queue, (self.frames, pixels)  , dtype = np.uint8)
        self.W_cl  = cl.array.empty(queue, (pixels,)              , dtype = np.float32)
        self.w_cl  = cl.array.empty(queue, w.shape   , dtype = np.float32)
        self.b_cl  = cl.array.empty(queue, b.shape   , dtype = np.float32)
        self.P_cl  = cl.array.empty(queue, (self.frames,)   , dtype = np.float32)
        
        self.Wbuf  = np.empty((pixels,), dtype=np.float32)
        
        cl.enqueue_copy(queue, dest = self.w_cl.data, src = w)
        cl.enqueue_copy(queue, dest = self.b_cl.data, src = b)

        # calculate c[t, r] = sum_d w[d] P[d, t, r]
        self.c = np.tensordot(w, P, axes=1)
        
        # for w update 
        self.Wsums = np.zeros((self.classes, self.rotations), dtype=np.float32)

        # skip unpopular rotations for a given class
        # but not unpopular classes, since we want to keep updating them
        for t in tqdm(range(1), desc = 'calculating per-frame favour', disable = silent) :
            favour_rotations = np.sum(P, axis = (0,))
        
        self.mask_rotations   = {}
        skipped_rots = 0
        for t in tqdm(self.my_classes, desc = 'masking rotations based on low P-value', disable = silent) :
            self.mask_rotations[t] = np.where(favour_rotations[t] > (1e-3 * np.max(favour_rotations[t])))[0].astype(np.int32)
            skipped_rots += rotations - self.mask_rotations[t].shape[0]
        
        self.weights = np.zeros((classes, rotations))#np.sum(P, axis=0)
        
        # skip unpopular frames for a given class and rotation
        favour_frames = np.sum(P, axis = (0, 2))
        self.mask_frames   = {}
        skipped_ds = 0
        for t in tqdm(self.my_classes, desc = 'masking frames for a given (class, rotation) based on low P-value', disable = silent) :
            for r in range(rotations):
                p = P[:, t, r].copy()
                self.mask_frames[(t, r)] = np.where(p > (1e-3 * np.max(p)))[0].astype(np.int32)
                self.weights[t, r] += np.sum(p[self.mask_frames[(t, r)]])
                skipped_ds += frames - self.mask_frames[(t, r)].shape[0]
        
        self.frame_list_cl  = cl.array.empty(queue, (frames,)   , dtype = np.float32)

        if not silent : 
            print()
            print('skipping', round(100 * skipped_rots / (classes * rotations), 2) ,'% of classes/rotations due to small P-value') 
            print('skipping', round(100 * skipped_ds / P.size, 2) ,'% of frames due to small P-value') 
        
        # test
        #self.W = np.zeros((classes, rotations, B.shape[-1]), dtype=np.float32)
        

    def merge_pixels(self, t, r, istart, istop):
        di = istop - istart
        self.points[0, :di] = self.R[r, 0, 0] * self.rx[istart: istop] + self.R[r, 0, 1] * self.ry[istart: istop]
        self.points[1, :di] = self.R[r, 1, 0] * self.rx[istart: istop] + self.R[r, 1, 1] * self.ry[istart: istop]
        self.mi[:di] = np.round(self.i0 + self.points[0, :di]/self.dx)
        self.mj[:di] = np.round(self.i0 + self.points[1, :di]/self.dx)
        
        np.add.at(self.I[t], (self.mi[:di], self.mj[:di]), self.Wbuf[:di] / self.C[istart: istop] * self.weights[t, r])
        np.add.at(self.overlap2[t], (self.mi[:di], self.mj[:di]), self.weights[t, r])
        
        # keep track of Wsums
        self.Wsums[t, r] += np.sum(self.Wbuf[:di])

    def update(self):
        K2 = np.zeros((self.frames, self.pixels), dtype=np.uint8)
        k  = np.zeros((self.B.shape[-1],), dtype=np.uint8)
        
        for i, istart in tqdm(enumerate(self.istart), total = len(self.istart), desc='updating classes over pixel chunks', disable = silent):
            istop = self.istop[i]
            di    = istop - istart
            
            # fill partial dense K-array
            for d in tqdm(range(self.frames), desc = 'loading partial photons', leave = False):
                k.fill(0)
                k[self.inds[d]] = self.K[d]
                K2[d, :di] = k[istart: istop]
            
            for j in tqdm(range(1), desc = 'loading to gpu', leave = False):
                cl.enqueue_copy(queue, dest = self.K_cl.data, src = K2)
                cl.enqueue_copy(queue, dest = self.B_cl.data, src = self.B[0, istart:istop])
             
            for t in tqdm(self.my_classes, desc='updating classes', leave = False, disable = silent):
                for r in tqdm(self.mask_rotations[t], desc='looping over rotations', leave = False, disable = silent):
                    cl.enqueue_copy(queue, dest = self.P_cl.data, src = np.ascontiguousarray(self.P[:, t, r]))
                     
                    cl.enqueue_copy(queue, dest = self.frame_list_cl.data, src = self.mask_frames[(t, r)])
                    
                    cl_code.update_W(queue, (di,), None, 
                                     self.W_cl.data, self.B_cl.data, 
                                     self.w_cl.data, self.b_cl.data,
                                     self.K_cl.data, self.P_cl.data,
                                     self.frame_list_cl.data,
                                     self.c[t, r], self.iters, 
                                     np.int32(self.mask_frames[(t, r)].shape[0]), 
                                     self.pixels)
                                 
                    cl.enqueue_copy(queue, dest = self.Wbuf, src = self.W_cl.data)
    
                    # test
                    #self.W[t, r, istart: istop] = self.Wbuf[:di]
                    
                    self.merge_pixels(t, r, istart, istop)
        self.allgather()
        

    def allgather(self):
        self.Wsums    = comm.allreduce(self.Wsums)
        self.I[:]     = comm.allreduce(self.I)
        self.overlap2 = comm.allreduce(self.overlap2)

        self.test1 = self.I.copy()
        self.test2 = self.overlap2.copy()
        
        self.overlap2[self.overlap2 <= 1e-20] = 1
        self.I /= self.overlap2
        np.clip(self.I, self.minval, None, self.I)
        
class Update_w():
    def __init__(self, Ksums, Wsums, P, w, I, b, B, inds, K, C, R, dx, xyz, frame_chunk_size, iters):
        # split frames by MPI rank
        frames = P.shape[0]
        self.ds = ds = np.linspace(0, frames, size + 1).astype(int)
        self.dstart = dstart = ds[:-1:][rank]
        self.dstop  = dstop  = ds[1::][rank]
        
        # calculate c[d] = sum_tr P[d, t, r] sum_i W[t, r, i]
        self.c = np.sum(P[dstart:dstop] * Wsums, axis=(1,2))
        
        # calculate xmax[d] = sum_i K[d, i] / c[d]
        self.xmax = Ksums[dstart: dstop].astype(np.float32) / self.c
        
        self.i0 = np.float32(I.shape[-1] // 2)
        self.dx = np.float32(dx)
        self.w = w
        self.b = b
        self.P = np.ascontiguousarray(P[dstart: dstop])
        self.inds = inds[dstart: dstop]
        self.K = K[dstart: dstop]
        self.B = B
        self.C = C
        self.rx = np.ascontiguousarray(xyz[0].astype(np.float32))
        self.ry = np.ascontiguousarray(xyz[1].astype(np.float32))
        
        
        self.iters  = np.int32(iters)
        self.frames = frames         = np.int32(self.dstop - self.dstart)
        classes     = self.classes   = np.int32(P.shape[1])
        rotations   = self.rotations = np.int32(P.shape[2])
        pixels      = self.pixels    = np.int32(B.shape[-1])

        pixels_max = np.max(Ksums)
        
        # initialise gpu arrays
        self.w_cl  = cl.array.empty(queue, (frames,)                   , dtype = np.float32)
        self.I_cl  = cl.array.empty(queue, I.shape                     , dtype = np.float32)
        self.b_cl  = cl.array.empty(queue, (frames,)                   , dtype = np.float32)
        self.C_cl  = cl.array.empty(queue, C.shape                     , dtype = np.float32)
        self.R_cl  = cl.array.empty(queue, R.shape                     , dtype = np.float32)
        self.c_cl     = cl.array.empty(queue, (frames,)                 , dtype = np.float32)
        self.xmax_cl  = cl.array.empty(queue, (frames,)                 , dtype = np.float32)
        
        
        self.groups = 8
        self.B_cl = []
        self.K_cl = [] 
        self.W_sub_cl = []
        self.rx_cl = []
        self.ry_cl = []
        self.C_cl = []
        self.tr_list_cl = []
        self.P_cl = []
        for g in range(self.groups):
            self.P_cl.append(cl.array.empty(queue, (classes, rotations), dtype = np.float32))
            self.B_cl.append(cl.array.empty(queue, (pixels_max,), dtype = np.float32))
            self.K_cl.append(cl.array.empty(queue, (pixels_max,), dtype = np.uint8))
            self.rx_cl.append(cl.array.empty(queue, (pixels_max,), dtype = np.float32))
            self.ry_cl.append(cl.array.empty(queue, (pixels_max,), dtype = np.float32))
            self.C_cl.append(cl.array.empty(queue, (pixels_max,), dtype = np.float32))
            self.tr_list_cl.append(cl.array.empty(queue, (classes*rotations, 2), dtype = np.float32))
        
        # load arrays to gpu
        cl.enqueue_copy(queue, self.R_cl.data, R)
        
        cl.enqueue_copy(queue, dest = self.w_cl.data, src = self.w[dstart:dstop])
        cl.enqueue_copy(queue, dest = self.b_cl.data, src = self.b[dstart:dstop])
        cl.enqueue_copy(queue, dest = self.c_cl.data, src = self.c)
        cl.enqueue_copy(queue, dest = self.xmax_cl.data, src = self.xmax)
        
        # copy I as an opencl "image" for bilinear sampling
        shape        = I.shape
        image_format = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
        flags        = cl.mem_flags.READ_ONLY
        self.I_cl    = cl.Image(context, flags, image_format, 
                                shape = shape[::-1], is_array = True)
        
        cl.enqueue_copy(queue, dest = self.I_cl, src = I, 
                        origin = (0, 0, 0), region = shape[::-1])
        
        # skip unpopular rotations and classes for a given frame
        self.classes_rotations_list = {}
        skipped = 0
        for d in range(frames) :
            p       = self.P[d]
            pthresh = 1e-3 * p.max()
            ts, rs  = np.where(p >= pthresh)
            self.classes_rotations_list[d] = np.empty((len(rs), 2), dtype=np.int32)
            self.classes_rotations_list[d][:, 0] = ts
            self.classes_rotations_list[d][:, 1] = rs
            skipped += classes * rotations - len(ts)
            
        if not silent : 
            print()
            print('skipping', round(100 * skipped / self.P.size, 2) ,'% of classes/rotations due to small P-value') 

    def update(self):
        events      = self.groups * [1,]
        copy_events = 7 * [1,]
        
        
        for d in tqdm(np.arange(self.frames, dtype=np.int32), desc=f'updating fluence estimates for {self.frames} frames', disable = silent):
            inds = self.inds[d]
            K    = self.K[d]
            pixels = np.int32(len(inds))
            
            index = d % self.groups
            
            # wait for last execution of index to finish
            if d > self.groups : events[index].wait()
            
            # send W, K, B to gpu
            copy_events[0] = cl.enqueue_copy(queue2, dest = self.K_cl[index].data, src = np.ascontiguousarray(K), is_blocking = False)
            copy_events[1] = cl.enqueue_copy(queue2, dest = self.B_cl[index].data, src = np.ascontiguousarray(self.B[0, inds]), is_blocking = False)
            copy_events[2] = cl.enqueue_copy(queue2, dest = self.rx_cl[index].data, src = np.ascontiguousarray(self.rx[inds]), is_blocking = False)
            copy_events[3] = cl.enqueue_copy(queue2, dest = self.ry_cl[index].data, src = np.ascontiguousarray(self.ry[inds]), is_blocking = False)
            copy_events[4] = cl.enqueue_copy(queue2, dest = self.C_cl[index].data, src = np.ascontiguousarray(self.C[inds]), is_blocking = False)
            copy_events[5] = cl.enqueue_copy(queue2, dest = self.P_cl[index].data, src = self.P[d], is_blocking = False)
            
            # send list of classes and rotations to consider
            copy_events[6] = cl.enqueue_copy(queue2, dest = self.tr_list_cl[index].data, src = self.classes_rotations_list[d], is_blocking = False)
            
            # one work group per frame
            events[index] = cl_code.update_w(queue2, (256,), (256,), 
                             self.I_cl, self.B_cl[index].data, 
                             self.w_cl.data, self.b_cl.data,
                             self.K_cl[index].data, self.P_cl[index].data,
                             self.c_cl.data, self.xmax_cl.data, 
                             self.iters, self.frames, self.classes,  
                             self.rotations, pixels, d, 
                             self.C_cl[index].data, self.R_cl.data, 
                             self.rx_cl[index].data, self.ry_cl[index].data, 
                             self.tr_list_cl[index].data, np.int32(self.classes_rotations_list[d].shape[0]),
                             self.i0, self.dx, 
                             wait_for = copy_events)
             
        queue2.finish()

        cl.enqueue_copy(queue, dest = self.w[self.dstart: self.dstop], src = self.w_cl.data)
        
        self.allgather()

    def allgather(self):
        for r in range(size):
            dstart = self.ds[:-1:][r]
            dstop  = self.ds[1::][r]
            self.w[dstart: dstop] = comm.bcast(self.w[dstart: dstop], root=r)

class Update_b():
    """
    c       = sum_i B[i]
    
    xmax[d] = sum_i K[d, i] / c
    
    loop iters:
        T[d, t, r, i] = b[d] + w[d] W[t, r, i] / B[i]
        PK[d, t, r]   = P[d, t, r] K[d, i]
        f[d]          = sum_tri PK[d, t, r] / T[d, t, r, i]
        g[d]          =-sum_tri PK[d, t, r] / T[d, t, r, i]^2
        
        step[d] = f[d] / g[d] * (1 - f[d] / c[d])
            
        b[d] += step[d]

    We should transpose to keep d on the fast axis (coallesed read + local sum)
    We should have frames as the worker index 
    
    chunk over frames
    worker index = ds
    load K[ds] and transpose --> K[i, d]
    load P[ds] and transpose --> P[t, r, d]
    load w[ds]
    load b[ds]
    load B[i]
    load I, R, rx, ry, dx, i0
    
    each worker has a d-index
    w = w[d]
    b = b[d]
    loop iters:
        loop t,r,i :
            W[t, r, i] <-- I, C 
            T    = b + w W[t, r, i] / B[i] 
            PK   = P[t, r, d] K[i, d]
            f   += PK / T
            g   -= PK / T^2
    
        step[d] = f / g * (1 - f / c)
            
        w += step[d]
    """
    def __init__(self, B, Ksums, cw):
        # calculate c
        cw.c   = np.float32(np.sum(B))
        
        # calculate xmax[d]
        xmax = Ksums[cw.dstart: cw.dstop].astype(np.float32) / cw.c
        
        # replace xmax
        cw.xmax = xmax
        cl.enqueue_copy(queue, dest = cw.xmax_cl.data, src = xmax)
        
        self.cw = cw
    
    def update(s):
        self = s.cw   
        if not silent : print()
        events      = self.groups * [1,]
        copy_events = 7 * [1,]
        
        # need an out-of-order queue
        queue2 = cl.CommandQueue(context, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
        
        for d in tqdm(np.arange(self.frames, dtype=np.int32), desc=f'updating background weights for {self.frames} frames', disable = silent):
            inds = self.inds[d]
            K    = self.K[d]
            pixels = np.int32(len(inds))
            
            index = d % self.groups
            
            # wait for last execution of index to finish
            if d > self.groups : events[index].wait()
            
            # send W, K, B to gpu
            copy_events[0] = cl.enqueue_copy(queue2, dest = self.K_cl[index].data, src = np.ascontiguousarray(K), is_blocking = False)
            copy_events[1] = cl.enqueue_copy(queue2, dest = self.B_cl[index].data, src = np.ascontiguousarray(self.B[0, inds]), is_blocking = False)
            copy_events[2] = cl.enqueue_copy(queue2, dest = self.rx_cl[index].data, src = np.ascontiguousarray(self.rx[inds]), is_blocking = False)
            copy_events[3] = cl.enqueue_copy(queue2, dest = self.ry_cl[index].data, src = np.ascontiguousarray(self.ry[inds]), is_blocking = False)
            copy_events[4] = cl.enqueue_copy(queue2, dest = self.C_cl[index].data, src = np.ascontiguousarray(self.C[inds]), is_blocking = False)
            copy_events[5] = cl.enqueue_copy(queue2, dest = self.P_cl[index].data, src = self.P[d], is_blocking = False)
            
            # send list of classes and rotations to consider
            copy_events[6] = cl.enqueue_copy(queue2, dest = self.tr_list_cl[index].data, src = self.classes_rotations_list[d], is_blocking = False)
            
            events[index] = cl_code.update_b2(queue2, (256,), (256,), 
                             self.I_cl, self.B_cl[index].data, 
                             self.w_cl.data, self.b_cl.data,
                             self.K_cl[index].data, self.P_cl[index].data,
                             self.c, self.xmax_cl.data, 
                             self.iters, self.frames, self.classes,  
                             self.rotations, pixels, d, 
                             self.C_cl[index].data, self.R_cl.data, 
                             self.rx_cl[index].data, self.ry_cl[index].data, 
                             self.tr_list_cl[index].data, np.int32(self.classes_rotations_list[d].shape[0]),
                             self.i0, self.dx, 
                             wait_for = copy_events)

        queue2.finish()

        cl.enqueue_copy(queue, dest = self.b[self.dstart: self.dstop], src = self.b_cl.data)
        
        s.allgather()
         
    def allgather(self):
        cw = self.cw
        for r in range(size):
            dstart = cw.ds[:-1:][r]
            dstop  = cw.ds[1::][r]
            cw.b[dstart: dstop] = comm.bcast(cw.b[dstart: dstop], root=r)
