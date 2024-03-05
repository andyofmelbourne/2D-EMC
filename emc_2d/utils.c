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
const int pixels, 
const int frames)
{
int frame = get_global_id(0);
int class = get_global_id(1);
int rotation = get_global_id(2);


//int frames = get_global_size(0);
int classes = get_global_size(1);
int rotations = get_global_size(2);

float R_l[4];
float T;
float logR = 0.;

int i;

//int g0 = get_num_groups(0);
//int g1 = get_num_groups(1);
//int g2 = get_num_groups(2);
//
//int w0 = get_local_size(0);
//int w1 = get_local_size(1);
//int w2 = get_local_size(2);
//printf("%d %d %d %d %d %d\n", w0, w1, w2, g0, g1, g2);

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

//int g0 = get_num_groups(0);
//int g1 = get_num_groups(1);
//int g2 = get_num_groups(2);
//
//int w0 = get_local_size(0);
//int w1 = get_local_size(1);
//int w2 = get_local_size(2);
//printf("%d %d\n", w0, g0);

if (i < pixels){

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

// W[t, r, i] = C[i] I[t, R_r[i].mi, R_r[i].mj]
kernel void calculate_tomograms(
image2d_array_t I, 
global float* W, 
global float* C, 
global float* R, 
global float* rx, 
global float* ry, 
const float i0, 
const float dx, 
const int   frames, 
const int   classes,
const int   rotations,
const int   pixels
){
int i = get_global_id(0);

float4 coord ;
float4 temp;

float rx_l = rx[i];
float ry_l = ry[i];
float C_l  = C[i];
for (int r=0; r<rotations; r++){
    coord.y = i0 + (R[4*r + 0] * rx_l + R[4*r + 1] * ry_l) / dx + 0.5;
    coord.x = i0 + (R[4*r + 2] * rx_l + R[4*r + 3] * ry_l) / dx + 0.5;
    
    for (int t=0; t<classes; t++){
        coord.z = t ;
        
        temp = read_imagef(I, trilinear, coord);

        W[pixels * rotations * t + pixels * r + i] = C_l * temp.x;
}}
}

kernel void update_w_test3(
image2d_array_t I, 
global float* B, 
global float* w, 
global float* b, 
global unsigned char* K, 
global float* P, 
global float* c, 
global float* xmax, 
const int   iters, 
const int   frames, 
const int   classes,
const int   rotations,
const int   pixels,
const int   d,
global float* C, 
global float* R, 
global float* rx, 
global float* ry, 
const float i0,
const float dx
)
{

//int d    = get_group_id(0);
int wid  = get_local_id(0);
int size = get_local_size(0);


int i, t, r, iter;
float c_l, b_l, xmax_l, T, step, PK;
unsigned char K_l;

local float f[256];
local float g[256];

local float x ;


// I wonder if these should be local arrays?
x      = w[d];
b_l    = b[d];
xmax_l = xmax[d];
c_l    = c[d];

float B_l;
float fsum, gsum;

float4 coord ;
float4 W;
float rx_l, ry_l, C_l;

for (iter=0; iter<iters; iter++){
    f[wid] = 0.;
    g[wid] = 0.;
    for (i=wid; i<pixels; i+=size){
        B_l  = b_l * B[i] ;
        K_l  = K[i];
        rx_l = rx[i];
        ry_l = ry[i];
        C_l  = C[i];
    
    for (r=0; r<rotations; r++){
        coord.y = i0 + (R[4*r + 0] * rx_l + R[4*r + 1] * ry_l) / dx + 0.5;
        coord.x = i0 + (R[4*r + 2] * rx_l + R[4*r + 3] * ry_l) / dx + 0.5;
    
    for (t=0; t<classes; t++){
        coord.z = t ;
        
        W = read_imagef(I, trilinear, coord);
        
        T = x + B_l / (C_l * W.x);
        
        PK      = P[t * rotations * frames + r * frames + d] * K_l;
        f[wid] += PK / T ;
        g[wid] -= PK / (T*T) ;
        //printf("%d %d %d %d %.2e %.2e %.2e \n", iter, t, r, K_l, P[t * rotations * frames + r * frames + d],  f[wid], g[wid]);
    }}}
    
    // work group reduce
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wid == 0) {
        fsum = 0 ;
        gsum = 0 ;
        for (i=0; i<size; i++) {
            fsum += f[i];
            gsum += g[i];
        }
    step = fsum / gsum * (1 - fsum / c_l);
    x   += step;
    x    = clamp(x, (float)1e-8, xmax_l) ;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
}

if (wid == 0) w[d] = x;
}

kernel void update_w_test2(
image2d_array_t I, 
global float* W, 
global float* B, 
global float* w, 
global float* b, 
global unsigned char* K, 
global float* P, 
global float* c, 
global float* xmax, 
const int   iters, 
const int   frames, 
const int   classes,
const int   rotations,
const int   pixels,
const int   d
)
{

//int d    = get_group_id(0);
int wid  = get_local_id(0);
int size = get_local_size(0);


int i, t, r, iter;
float c_l, b_l, xmax_l, T, step, PK;
unsigned char K_l;

local float f[256];
local float g[256];

local float x ;


// I wonder if these should be local arrays?
x      = w[d];
b_l    = b[d];
xmax_l = xmax[d];
c_l    = c[d];

float B_l;
float fsum, gsum;

for (iter=0; iter<iters; iter++){
    f[wid] = 0.;
    g[wid] = 0.;
    for (i=wid; i<pixels; i+=size){
        B_l  = b_l * B[i] ;
        K_l  = K[i];
    
    for (t=0; t<classes; t++){
    for (r=0; r<rotations; r++){
        T = x + B_l / W[rotations * pixels * t + pixels * r + i];
        
        PK      = P[t * rotations * frames + r * frames + d] * K_l;
        f[wid] += PK / T ;
        g[wid] -= PK / (T*T) ;
        //printf("%d %d %d %d %.2e %.2e %.2e \n", iter, t, r, K_l, P[t * rotations * frames + r * frames + d],  f[wid], g[wid]);
    }}}
    
    // work group reduce
    barrier(CLK_LOCAL_MEM_FENCE);
    if (wid == 0) {
        fsum = 0 ;
        gsum = 0 ;
        for (i=0; i<size; i++) {
            fsum += f[i];
            gsum += g[i];
        }
    step = fsum / gsum * (1 - fsum / c_l);
    x   += step;
    x    = clamp(x, (float)1e-8, xmax_l) ;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
}

if (wid == 0) w[d] = x;
}


// reduce repeated calculation of W 
// by writing to local memory

kernel void update_w_test(
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
int ri = get_local_id(0);


int i, t, r, iter;
float x, c_l, b_l, xmax_l, f, g, T, step, PK;

//float R_l[4];
//for (i=0; i<4; i++) {
//    R_l[i] = R[4*rotation + i];
//}

local float W_l[256];

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
    
    for (t=0; t<classes; t++){
        coord.z = t ;
        coord.y = i0 + (R[4*ri + 0] * rx_l + R[4*ri + 1] * ry_l) / dx + 0.5;
        coord.x = i0 + (R[4*ri + 2] * rx_l + R[4*ri + 3] * ry_l) / dx + 0.5;
        
        // use local memory to share W accross workers
        W = read_imagef(I, trilinear, coord);
        W_l[ri] = W.x;

        // need a barrier to make sure W_l is not read before writing has finished
        barrier(CLK_LOCAL_MEM_FENCE);

    for (r=0; r<rotations; r++){
        T = x + B_l / W_l[r];
        
        PK = P[t * rotations * frames + r * frames + d] * K_l;
        //PK = PK_l[r];
        f += PK / T ;
        g -= PK / (T*T) ;
    }}}

    step = f / g * (1 - f / c_l);
    
    x += step;
    x = clamp(x, (float)1e-8, xmax_l) ;
}

w[d] = x;
}

//    each worker has a d,t-index
//    w = w[d]
//    b = b[d]
//    loop iters:
//        loop r,i :
//            W[t, r, i] <-- I, C 
//            T     = w + b B[i] / W[t, r, i]
//            PK    = P[r, t, d] K[i, d]
//            f[d, t] += PK / T
//            g[d, t] -= PK / T^2

kernel void calculate_fg_w(
image2d_array_t I, 
global float* B, 
global float* w, 
global float* b, 
global unsigned char* K, 
global float* P, 
global float* C, 
global float* R, 
global float* rx, 
global float* ry, 
global float* fg, 
global float* gg, 
const float i0, 
const float dx, 
const int   iters, 
const int   frames, 
const int   classes,
const int   rotations,
const int   pixels
)
{

int t = get_global_id(0);
int d = get_global_id(1);


int i, r;
float x, b_l, f, g, T, PK;

float4 coord ;
float4 W;

// I wonder if these should be local arrays?
x      = w[d];
b_l    = b[d];

float B_l, rx_l, ry_l;
unsigned char K_l;

//if ((t==0) && (d==0)){
//    int g0 = get_num_groups(0);
//    int g1 = get_num_groups(1);
//
//    int w0 = get_local_size(0);
//    int w1 = get_local_size(1);
//    printf("%d %d %d %d\n", w0, w1, g0, g1);
//}

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
        
        coord.z = t ;
        
        W = read_imagef(I, trilinear, coord);
        
        T = x + B_l / W.x;
        
        PK = P[r * classes * frames + t * frames + d] * K_l;
        f += PK / T ;
        g -= PK / (T*T) ;
}}

fg[d * classes + t] = f;
gg[d * classes + t] = g;
}

kernel void calculate_fg_w_test(
image2d_array_t I, 
global float* B, 
global float* w, 
global float* b, 
global unsigned char* K, 
global float* P, 
global float* C, 
global float* R, 
global float* rx, 
global float* ry, 
global float* fg, 
global float* gg, 
const float i0, 
const float dx, 
const int   iters, 
const int   frames, 
const int   classes,
const int   rotations,
const int   pixels
)
{

int r = get_global_id(0);
int t = get_global_id(1);
int d = get_global_id(2);


int i;
float T, PK, f, g, wl, bl, rxl, ryl;

float4 coord ;
float4 W;

float R_l[4];

coord.z = t ;

for (i=0; i<4; i++) {
    R_l[i] = R[4*r + i];
}
f = g = 0.;

wl = w[d];
bl = b[d];
for (i=0; i<pixels; i++){
    rxl = rx[i];
    ryl = ry[i];
    coord.y = i0 + (R_l[0] * rxl + R_l[1] * ryl) / dx + 0.5;
    coord.x = i0 + (R_l[2] * rxl + R_l[3] * ryl) / dx + 0.5;
    
    
    W = read_imagef(I, trilinear, coord);
    
    T = wl + bl * B[i] / W.x / C[i];
    
    PK = P[r * classes * frames + t * frames + d] * K[i * frames + d];
    f += PK / T ;
    g -= PK / (T*T) ;
}

fg[r * classes * frames + d * classes + t] = f;
gg[r * classes * frames + d * classes + t] = g;

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




kernel void test2(
global unsigned char* A, 
global unsigned char* out, 
const int I,
const int J,
const int K
)
{
int i = get_global_id(0);
//int j = get_global_id(1);
//int k = get_global_id(2);

//int g0 = get_num_groups(0);
//int g1 = get_num_groups(1);
//int g2 = get_num_groups(2);
//
//int w0 = get_local_size(0);
//int w1 = get_local_size(1);
//int w2 = get_local_size(2);
//printf("%d %d %d %d %d %d\n", w0, w1, w2, g0, g1, g2);

//if (i < (I*J*K)) {
out[i] = A[0] + 1;
//}
//A[i] += 1;
//A[J*K*i + K*j + k] += 1;
}
