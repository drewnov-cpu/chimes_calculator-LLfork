#pragma once

#ifndef gpu_compute_h
#define gpu_compute_h

extern __constant__ double dr_gpu[CHDIM];

__global__ void compute2B_helper(int ncoeffs, double fcut, double fcut_deriv, double dx_inv,
double *chimes_params, int *chimes_pows, double *Tn, double *Tnd, double *force, double *stress);


#endif