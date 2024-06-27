
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

// 4 body
__constant__ double fcut_4b[6];
__constant__ double fcutderiv_4b[6];
#ifdef USE_DISTANCE_TENSOR  
    __constant__ double dr2_4b[CHDIM*CHDIM*6*6]; // npairs = 6
#endif
__constant__ double fcut5_4b[6];
//__constant__ double gpu_dx[6];
__constant__ double dr_4b[6 * CHDIM];
__constant__ int pair_idx_4b[6];



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

        double energy_result = coeff_val * fcut_all * cheby.Tn_ij[powers[0]] * cheby.Tn_ik[powers[1]] * cheby.Tn_jk[powers[2]];

        atomicAdd(&gpu_energy, energy_result);

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


__global__ void compute4b_helper(int ncoeffs, double fcut_all,
double *chimes_params, int *chimes_pows, double *force, double *stress, poly_pointers_4b cheby) {
    #define get_dr2(i, j, k, l) dr2_4b[i*CHDIM*6*CHDIM + j*6*CHDIM + k*CHDIM + l]

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int coeffs = bx * blockDim.x + tx;

    if (coeffs < ncoeffs) {

        double coeff = chimes_params[coeffs];

        int powers[6]; // npairs

        for (int i = 0; i < 6; i++)
            powers[i] = chimes_pows[coeffs * 6 + pair_idx_4b[i]];

        double Tn_ij_ik_il = cheby.Tn_ij[powers[0]] * cheby.Tn_ik[powers[1]] * cheby.Tn_il[powers[2]];
        double Tn_jk_jl = cheby.Tn_jk[powers[3]] * cheby.Tn_jl[powers[4]];
        double Tn_kl_5 = cheby.Tn_kl[powers[5]];

        double energy_result = coeff * fcut_all * Tn_ij_ik_il * Tn_jk_jl * Tn_kl_5;
        atomicAdd(&gpu_energy, energy_result);

        double deriv[6]; // npairs

        deriv[0] = fcut_4b[0] * cheby.Tnd_ij[powers[0]] + fcutderiv_4b[0] * cheby.Tn_ij[powers[0]];
        deriv[1] = fcut_4b[1] * cheby.Tnd_ik[powers[1]] + fcutderiv_4b[1] * cheby.Tn_ik[powers[1]];
        deriv[2] = fcut_4b[2] * cheby.Tnd_il[powers[2]] + fcutderiv_4b[2] * cheby.Tn_il[powers[2]];
        deriv[3] = fcut_4b[3] * cheby.Tnd_jk[powers[3]] + fcutderiv_4b[3] * cheby.Tn_jk[powers[3]];
        deriv[4] = fcut_4b[4] * cheby.Tnd_jl[powers[4]] + fcutderiv_4b[4] * cheby.Tn_jl[powers[4]];
        deriv[5] = fcut_4b[5] * cheby.Tnd_kl[powers[5]] + fcutderiv_4b[5] * cheby.Tn_kl[powers[5]];


        double force_scalar[6];

        force_scalar[0]  = coeff * deriv[0] * fcut5_4b[0] * cheby.Tn_ik[powers[1]] * cheby.Tn_il[powers[2]] * Tn_jk_jl * Tn_kl_5;
        force_scalar[1]  = coeff * deriv[1] * fcut5_4b[1] * cheby.Tn_ij[powers[0]] * cheby.Tn_il[powers[2]] * Tn_jk_jl * Tn_kl_5;
        force_scalar[2]  = coeff * deriv[2] * fcut5_4b[2] * cheby.Tn_ij[powers[0]] * cheby.Tn_ik[powers[1]] * Tn_jk_jl * Tn_kl_5;
        force_scalar[3]  = coeff * deriv[3] * fcut5_4b[3] * Tn_ij_ik_il * cheby.Tn_jl[powers[4]] * Tn_kl_5;
        force_scalar[4]  = coeff * deriv[4] * fcut5_4b[4] * Tn_ij_ik_il * cheby.Tn_jk[powers[3]] * Tn_kl_5;
        force_scalar[5]  = coeff * deriv[5] * fcut5_4b[5] * Tn_ij_ik_il * Tn_jk_jl;

        // Accumulate forces/stresses on/from the ij pair
        

        atomicAdd(&(force[0*CHDIM+0]), force_scalar[0] * dr_4b[0*CHDIM+0]);
        atomicAdd(&(force[0*CHDIM+1]), force_scalar[0] * dr_4b[0*CHDIM+1]);
        atomicAdd(&(force[0*CHDIM+2]), force_scalar[0] * dr_4b[0*CHDIM+2]);

        atomicAdd(&(force[1*CHDIM+0]), - force_scalar[0] * dr_4b[0*CHDIM+0]);
        atomicAdd(&(force[1*CHDIM+1]), - force_scalar[0] * dr_4b[0*CHDIM+1]);
        atomicAdd(&(force[1*CHDIM+2]), - force_scalar[0] * dr_4b[0*CHDIM+2]);

        #ifdef USE_DISTANCE_TENSOR
            atomicAdd(&(stress[0]), - force_scalar[0] * get_dr2(0,0,0,0));
            atomicAdd(&(stress[1]), - force_scalar[0] * get_dr2(0,0,0,1));
            atomicAdd(&(stress[2]), - force_scalar[0] * get_dr2(0,0,0,2));
            atomicAdd(&(stress[3]), - force_scalar[0] * get_dr2(0,1,0,1));
            atomicAdd(&(stress[4]), - force_scalar[0] * get_dr2(0,1,0,2));
            atomicAdd(&(stress[5]), - force_scalar[0] * get_dr2(0,2,0,2));
        #endif

        // Accumulate forces/stresses on/from the ik pair

        atomicAdd(&(force[0*CHDIM+0]), force_scalar[1] * dr_4b[1*CHDIM+0]);
        atomicAdd(&(force[0*CHDIM+1]), force_scalar[1] * dr_4b[1*CHDIM+1]);
        atomicAdd(&(force[0*CHDIM+2]), force_scalar[1] * dr_4b[1*CHDIM+2]);

        atomicAdd(&(force[2*CHDIM+0]), - force_scalar[1] * dr_4b[1*CHDIM+0]);
        atomicAdd(&(force[2*CHDIM+1]), - force_scalar[1] * dr_4b[1*CHDIM+1]);
        atomicAdd(&(force[2*CHDIM+2]), - force_scalar[1] * dr_4b[1*CHDIM+2]);

        #ifdef USE_DISTANCE_TENSOR
            atomicAdd(&(stress[0]), - force_scalar[1] * get_dr2(1,0,1,0));
            atomicAdd(&(stress[1]), - force_scalar[1] * get_dr2(1,0,1,1));
            atomicAdd(&(stress[2]), - force_scalar[1] * get_dr2(1,0,1,2));
            atomicAdd(&(stress[3]), - force_scalar[1] * get_dr2(1,1,1,1));
            atomicAdd(&(stress[4]), - force_scalar[1] * get_dr2(1,1,1,2));
            atomicAdd(&(stress[5]), - force_scalar[1] * get_dr2(1,2,1,2));
        #endif

        // Accumulate forces/stresses on/from the il pair

        atomicAdd(&(force[0*CHDIM+0]), force_scalar[2] * dr_4b[2*CHDIM+0]);
        atomicAdd(&(force[0*CHDIM+1]), force_scalar[2] * dr_4b[2*CHDIM+1]);
        atomicAdd(&(force[0*CHDIM+2]), force_scalar[2] * dr_4b[2*CHDIM+2]);

        atomicAdd(&(force[3*CHDIM+0]), - force_scalar[2] * dr_4b[2*CHDIM+0]);
        atomicAdd(&(force[3*CHDIM+1]), - force_scalar[2] * dr_4b[2*CHDIM+1]);
        atomicAdd(&(force[3*CHDIM+2]), - force_scalar[2] * dr_4b[2*CHDIM+2]);

        #ifdef USE_DISTANCE_TENSOR
            atomicAdd(&(stress[0]), - force_scalar[2]  * get_dr2(2,0,2,0));
            atomicAdd(&(stress[1]), - force_scalar[2]  * get_dr2(2,0,2,1));
            atomicAdd(&(stress[2]), - force_scalar[2]  * get_dr2(2,0,2,2));
            atomicAdd(&(stress[3]), - force_scalar[2]  * get_dr2(2,1,2,1));
            atomicAdd(&(stress[4]), - force_scalar[2]  * get_dr2(2,1,2,2));
            atomicAdd(&(stress[5]), - force_scalar[2]  * get_dr2(2,2,2,2));
        #endif

        // Accumulate forces/stresses on/from the jk pair

        atomicAdd(&(force[1*CHDIM+0]), force_scalar[3] * dr_4b[3*CHDIM+0]);
        atomicAdd(&(force[1*CHDIM+1]), force_scalar[3] * dr_4b[3*CHDIM+1]);
        atomicAdd(&(force[1*CHDIM+2]), force_scalar[3] * dr_4b[3*CHDIM+2]);

        atomicAdd(&(force[2*CHDIM+0]), - force_scalar[3] * dr_4b[3*CHDIM+0]);
        atomicAdd(&(force[2*CHDIM+1]), - force_scalar[3] * dr_4b[3*CHDIM+1]);
        atomicAdd(&(force[2*CHDIM+2]), - force_scalar[3] * dr_4b[3*CHDIM+2]);


        #ifdef USE_DISTANCE_TENSOR
            atomicAdd(&(stress[0]), - force_scalar[3]  * get_dr2(3,0,3,0));
            atomicAdd(&(stress[1]), - force_scalar[3]  * get_dr2(3,0,3,1));
            atomicAdd(&(stress[2]), - force_scalar[3]  * get_dr2(3,0,3,2));
            atomicAdd(&(stress[3]), - force_scalar[3]  * get_dr2(3,1,3,1));
            atomicAdd(&(stress[4]), - force_scalar[3]  * get_dr2(3,1,3,2));
            atomicAdd(&(stress[5]), - force_scalar[3]  * get_dr2(3,2,3,2));
        #endif
        // Accumulate forces/stresses on/from the jl pair

        atomicAdd(&(force[1*CHDIM+0]), force_scalar[4] * dr_4b[4*CHDIM+0]);
        atomicAdd(&(force[1*CHDIM+1]), force_scalar[4] * dr_4b[4*CHDIM+1]);
        atomicAdd(&(force[1*CHDIM+2]), force_scalar[4] * dr_4b[4*CHDIM+2]);

        atomicAdd(&(force[3*CHDIM+0]), - force_scalar[4] * dr_4b[4*CHDIM+0]);
        atomicAdd(&(force[3*CHDIM+1]), - force_scalar[4] * dr_4b[4*CHDIM+1]);
        atomicAdd(&(force[3*CHDIM+2]), - force_scalar[4] * dr_4b[4*CHDIM+2]);

        #ifdef USE_DISTANCE_TENSOR
            atomicAdd(&(stress[0]), - force_scalar[4]  * get_dr2(4,0,4,0));
            atomicAdd(&(stress[1]), - force_scalar[4]  * get_dr2(4,0,4,1));
            atomicAdd(&(stress[2]), - force_scalar[4]  * get_dr2(4,0,4,2));
            atomicAdd(&(stress[3]), - force_scalar[4]  * get_dr2(4,1,4,1));
            atomicAdd(&(stress[4]), - force_scalar[4]  * get_dr2(4,1,4,2));
            atomicAdd(&(stress[5]), - force_scalar[4]  * get_dr2(4,2,4,2));
        #endif
        // Accumulate forces/stresses on/from the kl pair

        atomicAdd(&(force[2*CHDIM+0]), force_scalar[5] * dr_4b[5*CHDIM+0]);
        atomicAdd(&(force[2*CHDIM+1]), force_scalar[5] * dr_4b[5*CHDIM+1]);
        atomicAdd(&(force[2*CHDIM+2]), force_scalar[5] * dr_4b[5*CHDIM+2]);

        atomicAdd(&(force[3*CHDIM+0]), - force_scalar[5] * dr_4b[5*CHDIM+0]);
        atomicAdd(&(force[3*CHDIM+1]), - force_scalar[5] * dr_4b[5*CHDIM+1]);
        atomicAdd(&(force[3*CHDIM+2]), - force_scalar[5] * dr_4b[5*CHDIM+2]);

        #ifdef USE_DISTANCE_TENSOR
            atomicAdd(&(stress[0]), - force_scalar[5]  * get_dr2(5,0,5,0));
            atomicAdd(&(stress[1]), - force_scalar[5]  * get_dr2(5,0,5,1));
            atomicAdd(&(stress[2]), - force_scalar[5]  * get_dr2(5,0,5,2));
            atomicAdd(&(stress[3]), - force_scalar[5]  * get_dr2(5,1,5,1));
            atomicAdd(&(stress[4]), - force_scalar[5]  * get_dr2(5,1,5,2));
            atomicAdd(&(stress[5]), - force_scalar[5]  * get_dr2(5,2,5,2));
        #endif

    }

    #undef get_dr2
}