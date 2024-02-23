import numpy as np
import scipy.constants as sc
#wav = sc.h * sc.c / photon_energy
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

def solid_angle_polarisation_factor(xyz, pixel_area, polarisation_axis = 'x'):
    
    C      = np.zeros(xyz.shape[1:], dtype = float)
    radius = np.sum(xyz**2, axis = 0)**0.5

    # polarisation correction
    if polarisation_axis == 'x' :
        C[:] = 1 - (xyz[0] / radius)**2 
    elif polarisation_axis == 'y' :
        C[:] = 1 - (xyz[1] / radius)**2 
    else :
        raise ValueError("polarisation axis must be one of 'x' or 'y'")

    # solid angle correction
    C[:] *= pixel_area * xyz[2] / radius**3 

    # rescale
    C /= C.max()
    return C


def calculate_rotation_matrices(M):
    # theta[r] = 2 pi r / M_in_plane
    # R[r]     = [cos -sin]
    #            |sin  cos|
    t = 2 * np.pi * np.arange(M) / M
    R = np.empty((M, 2, 2), dtype = np.float32)
    R[:, 0, 0] =  np.cos(t)
    R[:, 0, 1] = -np.sin(t)
    R[:, 1, 0] =  np.sin(t)
    R[:, 1, 1] =  np.cos(t)
    return R


def expand(points_I, I, W, xyz, R, C, minval = 1e-8):
    classes, rotations, pixels = W.shape
    
    interp = RegularGridInterpolator(points_I, I[0], fill_value = 0.)
    points = np.empty((pixels, 2))
    for c in tqdm(range(classes), desc = 'Expand'):
        interp.values[:] = I[c]
        for r in range(rotations):
            points[:, 0] = R[r, 0, 0] * xyz[0] + R[r, 0, 1] * xyz[1]
            points[:, 1] = R[r, 1, 0] * xyz[0] + R[r, 1, 1] * xyz[1]
            W[c, r] = np.clip(C * interp(points), minval, None)

def compress(W, R, xyz, i0, dx, I):
    classes, rotations, pixels = W.shape
    
    I.fill(0)
    overlap = np.zeros(I.shape, dtype=int)
    
    points = np.empty((2, pixels))
    mi     = np.empty((pixels,), dtype=int)
    mj     = np.empty((pixels,), dtype=int)
    for c in tqdm(range(classes), desc = 'Compress'):
        for r in range(rotations):
            points[0, :] = R[r, 0, 0] * xyz[0] + R[r, 0, 1] * xyz[1]
            points[1, :] = R[r, 1, 0] * xyz[0] + R[r, 1, 1] * xyz[1]
            mi[:]   = np.round(i0 + points[0]/dx)
            mj[:]   = np.round(i0 + points[1]/dx)

            np.add.at(I[c], (mi, mj), W[c, r])
            np.add.at(overlap[c], (mi, mj), 1)

    overlap[overlap==0] = 1
    I /= overlap


def calculate_probability_matrix(w, W, b, B, K, logR, P, beta):
    frames = len(K)
        
    # probability matrix
    for d in tqdm(range(frames), desc = 'calculating probability matrix'):
        T = w[d] * W + b[d] * B
        logR[d] = beta * np.sum(K[d] * np.log(T) - T, axis=-1)

        m = np.max(logR[d])
        P[d]     = logR[d] - m
        P[d]     = np.exp(P[d])
        P[d]    /= np.sum(P[d])

    print('expectation value: {:.2e}'.format(np.sum(P * logR)))
    print('log likelihood   : {:.2e}'.format(np.sum(logR)))

def update_W(w, W, b, B, P, K, minval = 1e-15, iters = 4):
    classes, rotations, pixels = W.shape
    frames = len(K)
    
    # xmax[t, r, i] = sum_d P[d, t, r] K[d, i] / \sum_d w[d] P[d, t, r] 
    c    = np.tensordot(w, P, axes = ((0,), (0,)))
    c[c<minval] = minval
    
    xmax = np.tensordot(P, K, axes = ((0,), (0,))) / c[..., None]
    xmax[xmax<minval] = minval

    step_min = -xmax/2
    
    #- gW[t, r, i] = sum_d w[d] P[d, t, r] (K[d, i] / T[d, t, r, i] - 1)
    #    = sum_d P[d, t, r] K[d, i] w[d] / T[d, t, r, i]  - \sum_d w[d] P[d, t, r] 
    f  = np.empty((classes, rotations, pixels))
    g  = np.empty((classes, rotations, pixels))
    T  = np.empty((classes, rotations, pixels))
    PK = np.empty((classes, rotations, pixels))
    step = np.empty((classes, rotations, pixels))
    for i in range(iters) :
        f.fill(0)
        g.fill(0)
        for d in tqdm(range(frames), desc = f'updating classes, iteration {i}'):
            T[:]  = W + b[d] * B / w[d]
            PK[:] = P[d, :, :, None] * K[d]
            f += PK / T
            g -= PK / T**2
        
        print('T^2 min:', (T**2).min())
        print('T min:', T.min())
        print('g min max:', g.min(), g.max())
        print('f min max:', f.min(), f.max())
        step[:] = f / g * (1 - f / c[..., None])
        print('step min max:', step.min(), step.max())
        print('c min:', c.min())
        g[g > -minval] = -minval
        step[:] = f / g * (1 - f / c[..., None])
        np.clip(step, step_min, None, step)
        W   += step
        m = W > minval
        np.clip(W, minval, xmax, W)    
        
        print(i, np.mean((f - c[..., None])[m]**2)**0.5)

def update_w(w, W, b, B, P, K, minval = 1e-8, iters = 4):
    classes, rotations, pixels = W.shape
    frames = len(K)
    
    # xmax[d] = (sum_tr P[d, t, r]) (sum_i K[d, i]) / sum_tr (sum_i W[t, r, i]) P[d, t, r]
    #                               (sum_i K[d, i]) / sum_tr (sum_i W[t, r, i]) P[d, t, r]
    ksums = np.sum(K, axis=-1)
    Wsums = np.sum(W, axis=-1)
    c    = np.sum(P * Wsums, axis = (1,2))
    c[c<minval] = minval
    xmax = ksums / c
    xmax[xmax<minval] = minval
    
    #- gw[t, r, i] = sum_tri W[t, r, i] P[d, t, r] (K[d, i] / T[d, t, r, i] - 1)
    #    = sum_tri P[d, t, r] K[d, i] W[t, r, i] / T[d, t, r, i]  - \sum_tri W[t, r, i] P[d, t, r] 
    f  = np.empty((frames,))
    g  = np.empty((frames,))
    T  = np.empty((classes, rotations, pixels))
    PK = np.empty((classes, rotations, pixels))
    for i in range(iters) :
        f.fill(0)
        g.fill(0)
        
        for d in tqdm(range(frames), desc = f'updating fluence estimates, iteration {i}'):
            T[:]  = w[d] + b[d] * B / W
            PK[:] = P[d, :, :, None] * K[d]
            f[d]  = np.sum(PK / T)
            g[d] -= np.sum(PK / T**2)
        
        g[g > -minval] = -minval
        w[:] += f / g * (1 - f / c)
        m = w > minval
        w[:] = np.clip(w, minval, xmax)    
        
        print(i, np.mean((f - c)[m]**2)**0.5)


# only works for one background class
def update_b(w, W, b, B, P, K, minval = 1e-8, iters = 4):
    # gb[d] = sum_tri P[d, t, r] K[d, i] B[l, i] / T[d, t, r, i]  - \sum_i B[l, i] 
    classes, rotations, pixels = W.shape
    frames = len(K)
    
    # xmax[d, l] = sum_i K[d, i] / sum_i B[l, i]
    ksums = np.sum(K, axis=-1)
    Bsums = np.sum(B, axis=-1)
    c     = Bsums
    c     = max(minval, c)
    xmax = ksums / c
    xmax[xmax<minval] = minval
    
    f  = np.empty((frames,))
    g  = np.empty((frames,))
    T  = np.empty((classes, rotations, pixels))
    PK = np.empty((classes, rotations, pixels))
    for i in range(iters) :
        f.fill(0)
        g.fill(0)
        
        for d in tqdm(range(frames), desc = f'updating background weights, iteration {i}'):
            T[:]  = w[d] * W / B + b[d] 
            PK[:] = P[d, :, :, None] * K[d]
            f[d]  = np.sum(PK / T)
            g[d] -= np.sum(PK / T**2)
        
        g[g > -minval] = -minval
        b[:] += f / g * (1 - f / c)
        m = b > minval
        b[:] = np.clip(b, minval, xmax)    
        
        print(i, np.mean((f - c)[m]**2)**0.5)

def make_W_ims(W, mask, s, shape_2d):
    classes, rotations, pixels = W.shape
    ims = np.zeros((classes, rotations) + shape_2d)
    for c in range(classes):
        for r in range(rotations):
            ims[c, r][mask[s]] = W[c, r]
            #ims[c, r][mask[s]] = xmax[c, r]

    # show C
    #C_im = np.zeros(shape_2d)
    #C_im[mask[s]] = C
    return ims
