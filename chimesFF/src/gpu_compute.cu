
#include "chimesFF.h"
#include "gpu_compute.h"   

// 2 body
__constant__ double dr_gpu[CHDIM];

// 3 body
__constant__ double fcut_3b[3];
__constant__ double fcutderiv_3b[3];
__constant__ double fcut2_3b[3];
__constant__ double dr_3b[3*CHDIM];
__constant__ int pair_idx_3b[3];
#ifdef USE_DISTANCE_TENSOR
    __constant__ double dr2_3b[CHDIM*CHDIM*3*3];
#endif


__device__ double gpu_energy = 0;
// make sure to set this right before call to compute methods if that is allowed.
// use atomic adds to make sure data isnt lost.
// note that this may become a problem with MPI
// may need to think of another solution.

__global__ void compute2B_helper(int ncoeffs, double fcut, double fcut_deriv, double dx_inv,
double *chimes_params, int *chimes_pows, double *Tn, double *Tnd, double *force, double *stress) {
    // 
    // this is a lot of arguments - think about if theres a way to break this down into
    // something nicer.
    // some of these arguments should probably be const
    // ncoeffs and fcut can be passed in the calls
    // chimes_params should already be on the GPU.
    // force and stress are to be updated
    // what is the most efficient way to update the force
    // and stress vectors at the end
    // idk lets start with making it work and then we can
    // optimize after that.

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int coeffs = bx * blockDim.x + tx;

    if (coeffs < ncoeffs) {
        double coeff_val = chimes_params[coeffs];
        double energy_result = coeff_val * fcut * Tn[ chimes_pows[coeffs] + 1];
        atomicAdd(&gpu_energy, energy_result);


        double deriv = fcut * Tnd[ chimes_pows[coeffs] + 1 ] + fcut_deriv * Tn[ chimes_pows[coeffs] + 1 ];

        double force_scalar = coeff_val * deriv * dx_inv;


        // will likely want to switch to reductions
        // to speed this up at some point.
        // the atomicAdds still have to be done in sequence
        // which does reduce parallelism but is fine for smaller scales
        
        atomicAdd(&(force[0*CHDIM+0]), force_scalar * dr_gpu[0]);
        atomicAdd(&(force[0*CHDIM+1]), force_scalar * dr_gpu[1]);
        atomicAdd(&(force[0*CHDIM+2]), force_scalar * dr_gpu[2]);

        atomicAdd(&(force[1*CHDIM+0]), - force_scalar * dr_gpu[0]);
        atomicAdd(&(force[1*CHDIM+1]), - force_scalar * dr_gpu[1]);
        atomicAdd(&(force[1*CHDIM+2]), - force_scalar * dr_gpu[2]);

        atomicAdd(&(stress[0]), - force_scalar * dr_gpu[0] * dr_gpu[0]);
        atomicAdd(&(stress[1]), - force_scalar * dr_gpu[0] * dr_gpu[1]);
        atomicAdd(&(stress[2]), - force_scalar * dr_gpu[0] * dr_gpu[2]);
        atomicAdd(&(stress[3]), - force_scalar * dr_gpu[1] * dr_gpu[1]);
        atomicAdd(&(stress[4]), - force_scalar * dr_gpu[1] * dr_gpu[2]);
        atomicAdd(&(stress[5]), - force_scalar * dr_gpu[2] * dr_gpu[2]);

        // will likely need some larger 2-body only ones
        // in order to test this version cause Im not actually
        // sure how much faster it will be with large numbers of
        // coefficients.

    }
    


}


__global__ void compute3b_helper(int ncoeffs, double fcut_all,
double *chimes_params, int *chimes_pows, double *force, double *stress, poly_pointers_3b cheby) {
    #define get_dr2(i, j, k, l) dr2_3b[i*CHDIM*3*CHDIM + j*3*CHDIM + k*CHDIM + l]
    
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int coeffs = bx * blockDim.x + tx;

    if (coeffs < ncoeffs) {
        double coeff_val = chimes_params[coeffs];

        int powers[3];
        powers[0] = chimes_pows[coeffs * 3 + pair_idx_3b[0]];
        powers[1] = chimes_pows[coeffs * 3 + pair_idx_3b[1]];
        powers[2] = chimes_pows[coeffs * 3 + pair_idx_3b[2]];

        atomicAdd(&gpu_energy, coeff_val * fcut_all * cheby.Tn_ij[powers[0]] * cheby.Tn_ik[powers[1]] * cheby.Tn_jk[powers[2]]);

        double deriv[3];
        deriv[0] = fcut_3b[0] * cheby.Tnd_ij[powers[0]] + fcutderiv_3b[0] * cheby.Tn_ij[powers[0]];
        deriv[1] = fcut_3b[1] * cheby.Tnd_ik[powers[1]] + fcutderiv_3b[1] * cheby.Tn_ik[powers[1]];
        deriv[2] = fcut_3b[2] * cheby.Tnd_jk[powers[2]] + fcutderiv_3b[2] * cheby.Tn_jk[powers[2]];

        double force_scalar[3];
        force_scalar[0] = coeff_val * deriv[0] * fcut2_3b[0] * cheby.Tn_ik[powers[1]] * cheby.Tn_jk[powers[2]];
        force_scalar[1] = coeff_val * deriv[1] * fcut2_3b[1] * cheby.Tn_ij[powers[0]] * cheby.Tn_jk[powers[2]];
        force_scalar[2] = coeff_val * deriv[2] * fcut2_3b[2] * cheby.Tn_ij[powers[0]] * cheby.Tn_ik[powers[1]];

        // Accumulate forces/stresses on/from the ij pair

        atomicAdd(&(force[0*CHDIM+0]), force_scalar[0] * dr_3b[0*CHDIM+0]);
        atomicAdd(&(force[0*CHDIM+1]), force_scalar[0] * dr_3b[0*CHDIM+1]);
        atomicAdd(&(force[0*CHDIM+2]), force_scalar[0] * dr_3b[0*CHDIM+2]);
        
        atomicAdd(&(force[1*CHDIM+0]), - force_scalar[0] * dr_3b[0*CHDIM+0]);
        atomicAdd(&(force[1*CHDIM+1]), - force_scalar[0] * dr_3b[0*CHDIM+1]);
        atomicAdd(&(force[1*CHDIM+2]), - force_scalar[0] * dr_3b[0*CHDIM+2]);

        #ifdef USE_DISTANCE_TENSOR

        atomicAdd(&(stress[0]), - force_scalar[0] * get_dr2(0,0,0,0));
        atomicAdd(&(stress[1]), - force_scalar[0] * get_dr2(0,0,0,1));
        atomicAdd(&(stress[2]), - force_scalar[0] * get_dr2(0,0,0,2));
        atomicAdd(&(stress[3]), - force_scalar[0] * get_dr2(0,1,0,1));
        atomicAdd(&(stress[4]), - force_scalar[0] * get_dr2(0,1,0,2));
        atomicAdd(&(stress[5]), - force_scalar[0] * get_dr2(0,2,0,2));

        #endif

        // Accumulate forces/stresses on/from the ik pair

        atomicAdd(&(force[0*CHDIM+0]), force_scalar[1] * dr_3b[1*CHDIM+0]);
        atomicAdd(&(force[0*CHDIM+1]), force_scalar[1] * dr_3b[1*CHDIM+1]);
        atomicAdd(&(force[0*CHDIM+2]), force_scalar[1] * dr_3b[1*CHDIM+2]);
        
        atomicAdd(&(force[2*CHDIM+0]), - force_scalar[1] * dr_3b[1*CHDIM+0]);
        atomicAdd(&(force[2*CHDIM+1]), - force_scalar[1] * dr_3b[1*CHDIM+1]);
        atomicAdd(&(force[2*CHDIM+2]), - force_scalar[1] * dr_3b[1*CHDIM+2]);

        #ifdef USE_DISTANCE_TENSOR

        atomicAdd(&(stress[0]), - force_scalar[1] * get_dr2(1,0,1,0));
        atomicAdd(&(stress[1]), - force_scalar[1] * get_dr2(1,0,1,1));
        atomicAdd(&(stress[2]), - force_scalar[1] * get_dr2(1,0,1,2));
        atomicAdd(&(stress[3]), - force_scalar[1] * get_dr2(1,1,1,1));
        atomicAdd(&(stress[4]), - force_scalar[1] * get_dr2(1,1,1,2));
        atomicAdd(&(stress[5]), - force_scalar[1] * get_dr2(1,2,1,2));

        #endif

        // Accumulate forces/stresses on/from the jk pair

        atomicAdd(&(force[1*CHDIM+0]), force_scalar[2] * dr_3b[2*CHDIM+0]);
        atomicAdd(&(force[1*CHDIM+1]), force_scalar[2] * dr_3b[2*CHDIM+1]);
        atomicAdd(&(force[1*CHDIM+2]), force_scalar[2] * dr_3b[2*CHDIM+2]);
        
        atomicAdd(&(force[2*CHDIM+0]), - force_scalar[2] * dr_3b[2*CHDIM+0]);
        atomicAdd(&(force[2*CHDIM+1]), - force_scalar[2] * dr_3b[2*CHDIM+1]);
        atomicAdd(&(force[2*CHDIM+2]), - force_scalar[2] * dr_3b[2*CHDIM+2]);

        #ifdef USE_DISTANCE_TENSOR

        atomicAdd(&(stress[0]), - force_scalar[2] * get_dr2(2,0,2,0));
        atomicAdd(&(stress[1]), - force_scalar[2] * get_dr2(2,0,2,1));
        atomicAdd(&(stress[2]), - force_scalar[2] * get_dr2(2,0,2,2));
        atomicAdd(&(stress[3]), - force_scalar[2] * get_dr2(2,1,2,1));
        atomicAdd(&(stress[4]), - force_scalar[2] * get_dr2(2,1,2,2));
        atomicAdd(&(stress[5]), - force_scalar[2] * get_dr2(2,2,2,2));

        #endif

    }

    #undef get_dr2
}