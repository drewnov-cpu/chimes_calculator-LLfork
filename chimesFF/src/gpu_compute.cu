
#include "chimesFF.h"
#include "gpu_compute.h"   

__constant__ double dr_gpu[CHDIM];

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
        if (coeffs == 0) {
            printf("force = %f\n", force[0]);
            printf("stress = %f\n", stress[0]);
        }
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