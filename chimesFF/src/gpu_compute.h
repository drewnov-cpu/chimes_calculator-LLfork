#pragma once

#ifndef gpu_compute_h
#define gpu_compute_h

extern __constant__ double dr_gpu[CHDIM];
extern __device__ double gpu_energy; // I think I will need to get rid of this if I want to run multiple processes on the GPU.

// 3 body constants
extern __constant__ double fcut_3b[3];
extern __constant__ double fcutderiv_3b[3];
extern __constant__ double fcut2_3b[3];
extern __constant__ double dr_3b[3*CHDIM];
extern __constant__ int pair_idx_3b[3];
#ifdef USE_DISTANCE_TENSOR  
    extern __constant__ double dr2_3b[CHDIM*CHDIM*3*3]; // makes sense to me to always use this on the GPU since it saves
    // computation time and memory accesses but for a reason unknown currently it may be required to still be a compile
    // time option.
#endif

struct poly_pointers_3b { 
    // stores all of the device pointers for the Chebyshev polynomials and their derivatives for the 3b version.
    // Still a lot of book keeping to be done in the main function but this should cut down on the amount of stuff I
    // have to pass to the kernel, as there is a limit on how much you can pass.
    // See the beginning of compute3b in chimesFF.cu to see where these come from and what they are used for.

    double *Tn_ij;
    double *Tn_ik;
    double *Tn_jk;
    double *Tnd_ij;
    double *Tnd_ik;
    double *Tnd_jk;

};

__global__ void compute2B_helper(int ncoeffs, double fcut, double fcut_deriv, double dx_inv,
double *chimes_params, int *chimes_pows, double *Tn, double *Tnd, double *force, double *stress);

__global__ void compute3b_helper(int ncoeffs, double fcut_all,
double *chimes_params, int *chimes_pows, double *force, double *stress, poly_pointers_3b cheby);




#endif