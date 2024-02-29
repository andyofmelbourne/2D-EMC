constant sampler_t trilinear = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR ;


// T[i, t, r]    = w[d] W[i, t, r] + b[d] B[i]
// logR[d, t, r] = beta sum_i K[i, d] log T[i, t, r] - T[i, t, r]

__kernel void calculate_LR (
image2d_array_t I, 
global float *LR,  
global unsigned char *K, 
global float *w,
global float *b, 
global float *B, 
global float *C, 
global float *R, 
global float *rx, 
global float *ry, 
const float beta, 
const float i0,
const float dx,
const int pixels)
{
int frame = get_global_id(0);
int class = get_global_id(1);
int rotation = get_global_id(2);

int frames = get_global_size(0);
int classes = get_global_size(1);
int rotations = get_global_size(2);

float R_l[4];
float T;
float logR = 0.;

int i;

for (i=0; i<4; i++) {
    R_l[i] = R[4*rotation + i];
}

float4 coord ;
float4 W;

coord.z = class ;

for (i=0; i<pixels; i++) {
    coord.y = i0 + (R_l[0] * rx[i] + R_l[1] * ry[i]) / dx + 0.5;
    coord.x = i0 + (R_l[2] * rx[i] + R_l[3] * ry[i]) / dx + 0.5;
    
    W = read_imagef(I, trilinear, coord);
    
    T = w[frame] * C[i] * W.x + b[frame] * B[i];
    logR += K[frames * i + frame] * log(T) - T ;
}

LR[rotations * classes * frame + rotations * class + rotation] = beta * logR;
}


__kernel void test (
image2d_array_t I, 
global float *out,  
global float *LR,  
global unsigned char *K, 
global float *w,
global float *b, 
global float *B, 
global float *R, 
global float *rx, 
global float *ry, 
const float beta, 
const float i0,
const float dx,
const int pixels)
{
int frame = get_global_id(0);
int class = get_global_id(1);
int rotation = get_global_id(2);

int frames = get_global_size(0);
int classes = get_global_size(1);
int rotations = get_global_size(2);

float R_l[4];
float T;
float logR = 0.;

int i;

for (i=0; i<4; i++) {
    R_l[i] = R[4*rotation + i];
}

float4 coord ;
float4 W;

coord.z = class ;


for (i=0; i<pixels; i++) {
    coord.x = i0 + (R_l[0] * rx[i] + R_l[1] * ry[i]) / dx + 0.5;
    coord.y = i0 + (R_l[2] * rx[i] + R_l[3] * ry[i]) / dx + 0.5;
    
    W = read_imagef(I, trilinear, coord);
    
    out[pixels * rotations * class + pixels * rotation + i] = W.x ;
    
}

}


//loop over (t, r) with pixel-chunking 
//    - sparsity: find frames with low P value
//    load P[:, t, r] # small
//    c       = sum_d w[d] P[d]
//    xmax[i] = sum_d P[d] K[d, i] / c
//    
//    loop iters:
//        calculate W[i] <-- I
//        loop d :
//            T[i]    = W[i] + b[d] B[i] / w[d]
//            PK      = P[d] K[d, i] 
//            f[i]   += PK / T[i]
//            g[i]   += PK / T[i]^2
//        
//        step[i] = f[i] / g[i] * (1 - f[i] / c)
//            
//        W[i] += step[i]
            

__kernel void update_W (
global float *Wout,  
global float *B,  
global float *w,
global float *b, 
global unsigned char *K, 
global float *P, 
const float c,
const int iters,
const int P_offset,
const int P_stride,
const int frames,
const int pixels)
{
int d, iter;
float xmax = 0.;
float T, f, g, step, PK;

int i = get_global_id(0);

for (d=0; d<frames; d++){
    xmax += P[d * P_stride + P_offset] * K[d * pixels + i];
}
xmax /= c;

float W = xmax / 2.;

for (iter=0; iter<iters; iter++){
    f = 0.;
    g = 0.;
    for (d=0; d<frames; d++){
        T  = W + b[d] * B[i] / w[d];
        PK = P[d * P_stride + P_offset] * K[d * pixels + i];
        f += PK / T ;
        g -= PK / (T*T) ;
    }

    step = f / g * (1 - f / c);
    
    W += step;
    W = clamp(W, (float)1e-8, xmax) ;
}

Wout[i] = W;
}












//    each worker has a d-index
//    w = w[d]
//    b = b[d]
//    loop iters:
//        loop t,r,i :
//            W[t, r, i] <-- I, C 
//            T     = w + b B[i] / W[t, r, i]
//            PK    = P[t, r, d] K[i, d]
//            f[d] += PK / T
//            g[d] -= PK / T^2

kernel void update_w(
image2d_array_t I, 
global float* B, 
global float* w, 
global float* b, 
global unsigned char* K, 
global float* P, 
global float* c, 
global float* xmax, 
global float* C, 
global float* R, 
global float* rx, 
global float* ry, 
const float i0, 
const float dx, 
const int   iters, 
const int   frames, 
const int   classes,
const int   rotations,
const int   pixels
)
{

int d = get_global_id(0);


int i, r, t, iter;
float x, c_l, b_l, xmax_l, f, g, T, step, PK;

//float R_l[4];
//for (i=0; i<4; i++) {
//    R_l[i] = R[4*rotation + i];
//}

float4 coord ;
float4 W;

// I wonder if these should be local arrays?
x      = w[d];
b_l    = b[d];
xmax_l = xmax[d];
c_l    = c[d];

float B_l, rx_l, ry_l;
unsigned char K_l;

for (iter=0; iter<iters; iter++){
    f = 0.;
    g = 0.;
    for (i=0; i<pixels; i++){
        rx_l = rx[i];
        ry_l = ry[i];
        B_l  = b_l * B[i] / C[i];
        K_l  = K[i * frames + d];
    
    for (r=0; r<rotations; r++){
        coord.y = i0 + (R[4*r + 0] * rx_l + R[4*r + 1] * ry_l) / dx + 0.5;
        coord.x = i0 + (R[4*r + 2] * rx_l + R[4*r + 3] * ry_l) / dx + 0.5;
        
    for (t=0; t<classes; t++){
        coord.z = t ;
        
        W = read_imagef(I, trilinear, coord);
        
        T = x + B_l / W.x;
        
        PK = P[t * rotations * frames + r * frames + d] * K_l;
        f += PK / T ;
        g -= PK / (T*T) ;
    }}}

    step = f / g * (1 - f / c_l);
    
    x += step;
    x = clamp(x, (float)1e-8, xmax_l) ;
}

w[d] = x;
}


kernel void update_b(
image2d_array_t I, 
global float* B, 
global float* w, 
global float* b, 
global unsigned char* K, 
global float* P, 
const  float  c, 
global float* xmax, 
global float* C, 
global float* R, 
global float* rx, 
global float* ry, 
const float i0, 
const float dx, 
const int   iters, 
const int   frames, 
const int   classes,
const int   rotations,
const int   pixels
)
{

int d = get_global_id(0);


int i, r, t, iter;
float x, w_l, xmax_l, f, g, T, step, PK;

float4 coord ;
float4 W;

// I wonder if these should be local arrays?
x      = b[d];
w_l    = w[d];
xmax_l = xmax[d];

float B_l, rx_l, ry_l;
unsigned char K_l;

for (iter=0; iter<iters; iter++){
    f = 0.;
    g = 0.;
    for (i=0; i<pixels; i++){
        rx_l = rx[i];
        ry_l = ry[i];
        B_l  = w_l * C[i] / B[i];
        K_l  = K[i * frames + d];
    
    for (r=0; r<rotations; r++){
        coord.y = i0 + (R[4*r + 0] * rx_l + R[4*r + 1] * ry_l) / dx + 0.5;
        coord.x = i0 + (R[4*r + 2] * rx_l + R[4*r + 3] * ry_l) / dx + 0.5;
        
    for (t=0; t<classes; t++){
        coord.z = t ;
        
        W = read_imagef(I, trilinear, coord);
        
        // T    = b + w C W[t, r, i] / B[i] 
        T = x + B_l * W.x;
        
        PK = P[t * rotations * frames + r * frames + d] * K_l;
        f += PK / T ;
        g -= PK / (T*T) ;
    }}}

    step = f / g * (1 - f / c);
    
    x += step;
    x = clamp(x, (float)1e-8, xmax_l) ;
}

b[d] = x;
}
