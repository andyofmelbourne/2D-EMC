import numpy as np
import pyopencl as cl
import pyopencl.array 
from tqdm import tqdm
import sys
import pathlib

rank = 0

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

code = pathlib.Path(__file__).resolve().parent.joinpath('utils.c')
cl_code = cl.Program(context, open(code, 'r').read()).build()


class Prob():
    def __init__(self, C, R, K, w, I, b, B, logR, P, xyz, dx, beta):
        """
        keep i on the slow axis to speed up the sum
        
        T[i, t, r] = w[d] * W[i,t,r] + np.dot(b[d], B)[i]
        logR[t, r] = beta * sum_i K[i] log T[i, t, r] - T[i, t, r]
    
        but if we do it this way we have to calcuate the entire W for every frame (~5e5)
        seems to be pretty fast anyway...
        """
        pixels      = np.int32(B.shape[-1])
        frames, classes, rotations = logR.shape
        self.frames    = frames
        self.classes   = classes
        self.rotations = rotations
        self.beta = np.float32(beta)
        self.dx   = np.float32(dx)
        self.pixels = pixels
        
        self.i0 = np.float32(I.shape[-1] // 2)
        
        self.LR_cl = cl.array.zeros(queue, logR.shape, dtype = np.float32)
        self.w_cl  = cl.array.empty(queue, w.shape   , dtype = np.float32)
        self.I_cl  = cl.array.empty(queue, I.shape   , dtype = np.float32)
        self.b_cl  = cl.array.empty(queue, b.shape   , dtype = np.float32)
        self.B_cl  = cl.array.empty(queue, B.T.shape , dtype = np.float32)
        self.K_cl  = cl.array.empty(queue, K.T.shape , dtype = np.uint8)
        self.C_cl  = cl.array.empty(queue, C.shape   , dtype = np.float32)
        self.R_cl  = cl.array.empty(queue, R.shape   , dtype = np.float32)
        self.rx_cl  = cl.array.empty(queue, xyz[0].shape   , dtype = np.float32)
        self.ry_cl  = cl.array.empty(queue, xyz[1].shape   , dtype = np.float32)
        
        # load arrays to gpu
        cl.enqueue_copy(queue, self.w_cl.data, w)
        cl.enqueue_copy(queue, self.b_cl.data, b)
        cl.enqueue_copy(queue, self.B_cl.data, B)
        cl.enqueue_copy(queue, self.K_cl.data, np.ascontiguousarray(K.T))
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

        
    def calculate(self, logR, P): 
        for i in tqdm(range(1)) :
            cl_code.calculate_LR(queue, (self.frames, self.classes, self.rotations), None, 
                    self.I_cl, self.LR_cl.data, self.K_cl.data, self.w_cl.data, 
                    self.b_cl.data, self.B_cl.data, self.C_cl.data, self.R_cl.data, 
                    self.rx_cl.data, self.ry_cl.data,
                    self.beta, self.i0, self.dx, self.pixels)
            cl.enqueue_copy(queue, dest = logR, src=self.LR_cl.data)
        
        expectation_value = 0.
        log_likihood      = 0.
        for d in range(self.frames):
            m = np.max(logR[d])
            P[d]     = logR[d] - m
            P[d]     = np.exp(P[d])
            P[d]    /= np.sum(P[d])

            expectation_value += np.sum(P[d] * logR[d]) / self.beta
            log_likihood      += np.sum(logR[d])        / self.beta
        return expectation_value, log_likihood
        
        
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
    
    def __init__(self, w, I, b, B, P, K, C, R, xyz, dx, pixel_chunk_size, minval = 1e-15, iters = 4):
        self.B = B
        self.P = P
        self.K = K
        self.I = I
        self.C = C
        self.R = R
        self.xyz = xyz
        self.iters  = np.int32(iters)
        self.dx    = np.float32(dx)
        self.i0    = np.float32(I.shape[-1]//2)
        self.pixel_chunk = pixel_chunk_size
        self.frames, self.classes, self.rotations = (np.int32(s) for s in P.shape)
        
        self.B = B
        self.K = K
        self.xyz = xyz
        
        # generate list of start stop values
        pixels = B.shape[-1]
        self.istart = list(range(0, pixels, pixel_chunk_size))
        self.istop  = list(range(pixel_chunk_size, pixels, pixel_chunk_size)) + [pixels]
        
        # initialise pixel based arrays
        self.B_cl  = cl.array.empty(queue, (pixel_chunk_size,)         , dtype = np.float32)
        self.rx_cl = cl.array.empty(queue, (pixel_chunk_size,)        , dtype = np.float32)
        self.ry_cl = cl.array.empty(queue, (pixel_chunk_size,)        , dtype = np.float32)
        self.K_cl  = cl.array.empty(queue, (self.frames, pixel_chunk_size,) , dtype = np.uint8)
    
        # initialise other arrays
        self.W_cl = cl.array.empty(queue, (pixel_chunk_size,), dtype = np.float32)
        #self.W    = np.empty((self.classes, self.rotations, pixels), dtype=np.float32)
        self.Wbuf = np.empty((pixel_chunk_size,), dtype=np.float32)

        # load to gpu
        self.P_cl  = cl.array.empty(queue, P.shape   , dtype = np.float32)
        self.w_cl  = cl.array.empty(queue, w.shape   , dtype = np.float32)
        self.b_cl  = cl.array.empty(queue, b.shape   , dtype = np.float32)
        
        cl.enqueue_copy(queue, dest = self.P_cl.data, src = P)
        cl.enqueue_copy(queue, dest = self.w_cl.data, src = w)
        cl.enqueue_copy(queue, dest = self.b_cl.data, src = b)
        
        # calculate c[t, r] = sum_d w[d] P[d, t, r]
        self.c = np.tensordot(w, P, axes=1)

        # for w update 
        self.Wsums = np.zeros((self.classes, self.rotations), dtype=np.float32)
        
        # for merging
        self.rx = np.empty((pixel_chunk_size,), dtype=np.float32)
        self.ry = np.empty((pixel_chunk_size,), dtype=np.float32)
        self.I.fill(0)
        self.overlap1 = np.zeros(I.shape, dtype=int)
        self.overlap2 = np.zeros(I.shape, dtype=float)
        self.weights = np.sum(P, axis=0)
          
        self.points = np.empty((2, pixel_chunk_size))
        self.mi     = np.empty((pixel_chunk_size,), dtype=int)
        self.mj     = np.empty((pixel_chunk_size,), dtype=int)

    def merge_pixels(self, t, r, istart, istop):
        pixels = istop - istart
        self.points[0] = self.R[r, 0, 0] * self.rx + self.R[r, 0, 1] * self.ry
        self.points[1] = self.R[r, 1, 0] * self.rx + self.R[r, 1, 1] * self.ry
        self.mi[:] = np.round(self.i0 + self.points[0]/self.dx)
        self.mj[:] = np.round(self.i0 + self.points[1]/self.dx)
        
        np.add.at(self.I[t], (self.mi[:pixels], self.mj[:pixels]), self.Wbuf[:pixels] / self.C[istart: istop] * self.weights[t, r])
        np.add.at(self.overlap1[t], (self.mi[:pixels], self.mj[:pixels]), 1)
        np.add.at(self.overlap2[t], (self.mi[:pixels], self.mj[:pixels]), self.weights[t, r])
        
        # keep track of Wsums
        self.Wsums[t, r] += np.sum(self.Wbuf[:pixels])
    
        
    def load_pixel_chunk(self, istart, istop):
        pixels = istop-istart
        self.rx[:pixels] = np.ascontiguousarray(self.xyz[0, istart:istop].astype(np.float32))
        self.ry[:pixels] = np.ascontiguousarray(self.xyz[1, istart:istop].astype(np.float32))
        
        cl.enqueue_copy(queue, dest = self.B_cl.data, src = self.B[0, istart:istop])
        cl.enqueue_copy(queue, dest = self.K_cl.data, src = np.ascontiguousarray(self.K[:, istart:istop]))
            
        cl.enqueue_copy(queue, self.rx_cl.data, self.rx[:pixels])
        cl.enqueue_copy(queue, self.ry_cl.data, self.ry[:pixels])
        
    def update(self):
        for istart, istop in tqdm(zip(self.istart, self.istop), total = len(self.istart), desc='updating classes'):
            pixels = np.int32(istop-istart)
            self.load_pixel_chunk(istart, istop)

            for t in tqdm(np.arange(self.classes, dtype=np.int32), desc='looping over classes', leave = False):
                for r in tqdm(np.arange(self.rotations, dtype=np.int32), desc='looping over rotations', leave = False):
                    P_offset = t * self.rotations + r
                    P_stride = self.classes * self.rotations
                    
                    cl_code.update_W(queue, (pixels,), None, 
                                     self.W_cl.data, self.B_cl.data, 
                                     self.w_cl.data, self.b_cl.data,
                                     self.K_cl.data, self.P_cl.data,
                                     self.c[t, r], self.iters, P_offset, P_stride,
                                     self.frames, pixels)
                                 
                    cl.enqueue_copy(queue, dest = self.Wbuf[:pixels], src = self.W_cl.data)
                    #self.W[t, r, istart:istop] = self.Wbuf[:pixels]

                    self.merge_pixels(t, r, istart, istop)
        
        self.overlap2[self.overlap1 == 0] = 1
        self.I /= self.overlap2

class Update_w():
    """
    c[d]    = sum_tr P[d, t, r] sum_i W[t, r, i]

    xmax[d] = sum_i K[d, i] / c[d]

    loop iters:
        T[d, t, r, i] = w[d] + b[d] B[i] / W[t, r, i]
        PK[d, t, r]   = P[d, t, r] K[d, i]
        f[d]          = sum_tri PK[d, t, r] / T[d, t, r, i]
        g[d]          =-sum_tri PK[d, t, r] / T[d, t, r, i]^2
        
        step[d] = f[d] / g[d] * (1 - f[d] / c[d])
            
        w[d] += step[d]

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
            T     = w + b B[i] / W[t, r, i]
            PK    = P[t, r, d] K[i, d]
            f[d] += PK / T
            g[d] -= PK / T^2
    """
    def __init__(self, Ksums, Wsums, P, w, I, b, B, K, C, R, dx, xyz, frame_chunk_size, iters):
        # get frame chunks
        frames = P.shape[0]
        self.dstart = list(range(0, frames, frame_chunk_size))
        self.dstop  = list(range(frame_chunk_size, frames, frame_chunk_size)) + [frames]

        # calculate c[d] = sum_tr P[d, t, r] sum_i W[t, r, i]
        self.c = np.sum(P * Wsums, axis=(1,2))

        # calculate xmax[d] = sum_i K[d, i] / c[d]
        self.xmax = Ksums.astype(np.float32) / self.c

        self.i0 = np.float32(I.shape[-1] // 2)
        self.dx = np.float32(dx)
        self.w = w
        self.b = b
        self.P = P
        self.K = K
        
        self.iters  = np.int32(iters)
        frames      = self.frames    = np.int32(P.shape[0])
        classes     = self.classes   = np.int32(P.shape[1])
        rotations   = self.rotations = np.int32(P.shape[2])
        pixels      = self.pixels    = np.int32(B.shape[-1])
        
        # initialise gpu arrays
        self.P_cl  = cl.array.zeros(queue, (classes, rotations, frame_chunk_size), dtype = np.float32)
        self.w_cl  = cl.array.empty(queue, (frame_chunk_size,)                   , dtype = np.float32)
        self.I_cl  = cl.array.empty(queue, I.shape                               , dtype = np.float32)
        self.b_cl  = cl.array.empty(queue, (frame_chunk_size,)                   , dtype = np.float32)
        self.B_cl  = cl.array.empty(queue, B.shape                               , dtype = np.float32)
        self.K_cl  = cl.array.empty(queue, (pixels, frame_chunk_size)            , dtype = np.uint8)
        self.C_cl  = cl.array.empty(queue, C.shape                               , dtype = np.float32)
        self.R_cl  = cl.array.empty(queue, R.shape                               , dtype = np.float32)
        self.rx_cl   = cl.array.empty(queue, xyz[0].shape                        , dtype = np.float32)
        self.ry_cl   = cl.array.empty(queue, xyz[1].shape                        , dtype = np.float32)
        self.c_cl    = cl.array.empty(queue, (frame_chunk_size,)                 , dtype = np.float32)
        self.xmax_cl = cl.array.empty(queue, (frame_chunk_size,)                 , dtype = np.float32)
        
        # load arrays to gpu
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

    def load_frame_chunk(self, dstart, dstop):
        K_in = np.ascontiguousarray(self.K[dstart:dstop, :].T)
        P_in = np.ascontiguousarray(np.transpose(self.P[dstart:dstop], (1, 2, 0)))
        
        cl.enqueue_copy(queue, dest = self.w_cl.data, src = self.w[dstart:dstop])
        cl.enqueue_copy(queue, dest = self.b_cl.data, src = self.b[dstart:dstop])
        cl.enqueue_copy(queue, dest = self.K_cl.data, src = K_in)
        cl.enqueue_copy(queue, dest = self.P_cl.data, src = P_in)
        cl.enqueue_copy(queue, dest = self.c_cl.data, src = self.c[dstart:dstop])
        cl.enqueue_copy(queue, dest = self.xmax_cl.data, src = self.xmax[dstart:dstop])
        
 
    def update(self):
        for dstart, dstop in tqdm(zip(self.dstart, self.dstop), total = len(self.dstart), desc='updating fluence estimates'):
            frames = np.int32(dstop-dstart)
            self.load_frame_chunk(dstart, dstop)
            
            cl_code.update_w(queue, (frames,), None, 
                             self.I_cl, self.B_cl.data, 
                             self.w_cl.data, self.b_cl.data,
                             self.K_cl.data, self.P_cl.data,
                             self.c_cl.data, self.xmax_cl.data, 
                             self.C_cl.data, self.R_cl.data, 
                             self.rx_cl.data, self.ry_cl.data,
                             self.i0, self.dx, self.iters, 
                             frames, self.classes,  
                             self.rotations, self.pixels)
                                 
            cl.enqueue_copy(queue, dest = self.w[dstart:dstop], src = self.w_cl.data)


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
        self.c = np.float32(np.sum(B))
        
        # calculate xmax[d]
        xmax = Ksums.astype(np.float32) / self.c

        # replace xmax
        cw.xmax = xmax
        
        self.cw = cw
    
    def update(self):
        cw = self.cw
        for dstart, dstop in tqdm(zip(cw.dstart, cw.dstop), total = len(cw.dstart), desc='updating background weights'):
            frames = np.int32(dstop-dstart)
            cw.load_frame_chunk(dstart, dstop)
            
            cl_code.update_b(queue, (frames,), None, 
                             cw.I_cl, cw.B_cl.data, 
                             cw.w_cl.data, cw.b_cl.data,
                             cw.K_cl.data, cw.P_cl.data,
                             self.c, cw.xmax_cl.data, 
                             cw.C_cl.data, cw.R_cl.data, 
                             cw.rx_cl.data, cw.ry_cl.data,
                             cw.i0, cw.dx, cw.iters, 
                             frames, cw.classes,  
                             cw.rotations, cw.pixels)
                                 
            # will this update the original b?
            cl.enqueue_copy(queue, dest = cw.b[dstart:dstop], src = cw.b_cl.data)
