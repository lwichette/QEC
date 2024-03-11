#include <cstdlib>
#include <fstream>
#include <iostream>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>
#include <cmath>
#include <math.h>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <filesystem>
#include <boost/program_options.hpp>

__global__ void init_randombond(
    signed char* interactions, const int seed, const long long nx, const long long ny, const float p
){

        const long long tid = static_cast<long long>(threadIdx.x + blockIdx.x * blockDim.x);

        if (tid >= 2*nx*ny) return;

        curandStatePhilox4_32_10_t st;
        curand_init(seed, tid, 0, &st);

        float bondrandval = curand_uniform(&st);

        signed char bondval = (bondrandval<p)? -1 : 1;
        
        interactions[tid] = bondval;
}

void write_bonds(signed char* interactions, std::string filename, long nx, long ny){
    printf("Writing bonds to %s ...\n", filename.c_str());

    std::vector<signed char> interactions_host(2*nx*ny);

    cudaMemcpy(interactions_host.data(),interactions, 2*nx*ny*sizeof(*interactions), cudaMemcpyDeviceToHost);

    std::ofstream f;
    f.open(filename + std::string(".txt"));
    if (f.is_open()) {
        for (int i = 0; i < 2*nx; i++) {
            for (int j = 0; j < ny; j++) {
                f << (int)interactions_host[i * ny + j] << " ";
            }
            f << std::endl;
        }
    }
    f.close();

}

void write_lattice_to_disc(signed char *lattice_b, signed char *lattice_w, std::string filename, long long nx, long long ny) {
    printf("Writing lattice to %s...\n", filename.c_str());

    std::vector<signed char> lattice_h(nx*ny);
    std::vector<signed char> lattice_w_h(nx*ny/2);
    std::vector<signed char> lattice_b_h(nx*ny/2);

    cudaMemcpy(lattice_b_h.data(), lattice_b, nx * ny/2 * sizeof(*lattice_b), cudaMemcpyDeviceToHost);
    cudaMemcpy(lattice_w_h.data(), lattice_w, nx * ny/2 * sizeof(*lattice_w), cudaMemcpyDeviceToHost);


    for (int i = 0; i < nx; i++){
        for (int j=0; j < ny/2; j++){
            if (i%2){
                lattice_h[i*ny+2*j+1] = lattice_w_h[i*ny/2+j];
                lattice_h[i*ny+2*j] = lattice_b_h[i*ny/2+j];
            }
            else{
                lattice_h[i*ny+2*j] = lattice_w_h[i*ny/2+j];
                lattice_h[i*ny+2*j+1] = lattice_b_h[i*ny/2+j];
            }
        }
    }

    std::ofstream f;
    f.open(filename + std::string(".txt"));

    if (f.is_open()) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                f << (int)lattice_h[i * ny + j] << " ";
            }
            f << std::endl;
        }
    }
    f.close();
}

__global__ void init_spins(
    signed char* lattice, const int seed, const int offset, const long long nx, const long long ny
){
        const long long  tid = static_cast<long long>(blockDim.x * blockIdx.x + threadIdx.x);
        
        if (tid >= nx * ny) return;

        // Random number generator
        curandStatePhilox4_32_10_t st;
        curand_init(seed, tid, static_cast<long long>(offset), &st);

        float randval = curand_uniform(&st);
        signed char val = (randval < 0.5f) ? -1 : 1;
        lattice[tid] = val;
}

template<bool is_black>
__global__ void update_lattice(
    signed char* lattice, signed char* __restrict__ op_lattice, signed char* interactions,
    int seed, int it, const long long nx, const long long ny, const float coupling_constant
) {

    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= nx*ny) return;

    int i = tid/ny;
    int j = tid%ny;

    // Set up periodic boundary conditions
    int ipp = (i + 1 < nx) ? i + 1 : 0;
    int inn = (i - 1 >= 0) ? i - 1: nx - 1;
    int jpp = (j + 1 < ny) ? j + 1 : 0;
    int jnn = (j - 1 >= 0) ? j - 1: ny - 1;

    int joff;
    int jcouplingoff;
    int icouplingpp;
    int icouplingnn;

    if (!is_black) {
        icouplingpp = 2*(nx-1)*ny + 2*(ny*(i+1) + j) + (i+1)%2;
        icouplingnn = 2*(nx-1)*ny + 2*(ny*(inn+1) + j) + (i+1)%2;

        joff = (i % 2) ? jnn : jpp;

        if (i % 2) {
            jcouplingoff = 2 * (i * ny + joff) + 1;
        } else {
            if (j + 1 >= ny) {
                jcouplingoff = 2 * (i * ny + j + 1) - 1;
            } else {
                jcouplingoff = 2 * (i * ny + joff) - 1;
            }
        }
    } else {
        icouplingpp = 2*(nx-1)*ny + 2*(ny*(i+1) + j) + i%2;
        icouplingnn = 2*(nx-1)*ny + 2*(ny*(inn+1) + j) + i%2;

        joff = (i % 2) ? jpp : jnn;

        if (i % 2) {
            if (j+1 >= ny) {
                jcouplingoff = 2 * (i * ny + j + 1) - 1;
            } else {
                jcouplingoff = 2 * (i * ny + joff) - 1;
            }
        } else {
            jcouplingoff = 2 * (i * ny + joff) + 1;
        }
    }

    // Compute sum of nearest neighbor spins times the coupling
    signed char nn_sum = op_lattice[inn*ny + j]*interactions[icouplingnn] + op_lattice[i*ny + j]*interactions[2*(i*ny + j)]
                        + op_lattice[ipp*ny + j]*interactions[icouplingpp] + op_lattice[i*ny + joff]*interactions[jcouplingoff];


    signed char lij = lattice[i*ny + j];

    // set device energy for each temp and each spin on lattice
    float diff = coupling_constant*nn_sum*lij;

    // Determine whether to flip spin
    float acceptance_ratio = exp(-2 * diff);

    curandStatePhilox4_32_10_t st;
	curand_init(seed, tid, static_cast<long long>(2*it+int(is_black)), &st);
    
    float randval = curand_uniform(&st);

    if (randval < acceptance_ratio) {
        lattice[i*ny + j] = -lij;
    }
}

int main(int argc, char **argv){
    float p = 0.06;
    float inv_temp = 1/1.7;
    int niters = 1000000;
    int L = 128;

    const int seed = 42;

    int THREADS = 128;

    int blocks = (L*L*2+ THREADS -1) / THREADS; 

    //Setup interaction lattice on device
    signed char *d_interactions;
    cudaMalloc(&d_interactions, L*L*2*sizeof(*d_interactions));
    
    init_randombond<<<blocks, THREADS>>>(d_interactions, seed + 1, L, L, p);
    
    // Setup black and white lattice arrays on device
    signed char *lattice_b, *lattice_w;
    cudaMalloc(&lattice_b, L * L/2 * sizeof(*lattice_b));
    cudaMalloc(&lattice_w, L * L/2 * sizeof(*lattice_w));
    
    init_spins<<<blocks, THREADS>>>(lattice_b, seed, 0, L, L/2);
    init_spins<<<blocks, THREADS>>>(lattice_w, seed, 1, L, L/2);

    write_bonds(d_interactions, "bonds", L, L);
    write_lattice_to_disc(lattice_b, lattice_w, "lattice_before", L, L);

    for (int j=0; j < niters; j++){
        
        update_lattice<true><<<blocks,THREADS>>>(lattice_b, lattice_w, d_interactions, seed, j+1, L, L/2, inv_temp);

        update_lattice<false><<<blocks,THREADS>>>(lattice_w, lattice_b, d_interactions, seed, j+1, L, L/2, inv_temp);
    
    }

    write_lattice_to_disc(lattice_b, lattice_w, "lattice", L, L);

    return 0;
}
