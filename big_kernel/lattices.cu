#include <fstream>
#include <iostream>
#include <time.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>
#include <math.h>
#include <vector>
#include <string>
#include <sys/stat.h>

#include "defines.h"

using namespace std;


__global__ void init_randombond(signed char* interactions, const float* __restrict__ interaction_randvals,
    const long long nx, const long long ny, const int num_lattices, const float p){
        
        const long long tid = static_cast<long long>(threadIdx.x + blockIdx.x * blockDim.x);
        
        if (tid >= 2*nx*ny*num_lattices) return;

        float bondrandval = interaction_randvals[tid];
        signed char bondval = (bondrandval<p)? -1 : 1;
        interactions[tid] = bondval;                                  
}

// Initialize lattice spins
__global__ void init_spins(signed char* lattice, const float* __restrict__ randvals,
    const long long nx, const long long ny, const int num_lattices) {
        const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
        if (tid >= nx * ny * num_lattices) return;
        
        float randval = randvals[tid];
        signed char val = (randval < 0.5f) ? -1 : 1;
        lattice[tid] = val;
}

void init_interactions_with_seed(signed char* interactions, const long long seed, const long long nx, const long long ny, const int num_lattices, const float p){
    int blocks = (nx*ny*2*num_lattices + THREADS -1)/THREADS;

    //Setup cuRAND generator for the random bond sign
    curandGenerator_t interaction_rng;
    curandCreateGenerator(&interaction_rng,CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(interaction_rng,seed);
    
    float *interaction_randvals;
    cudaMalloc(&interaction_randvals,num_lattices*nx*ny*2*sizeof(*interaction_randvals));

    curandGenerateUniform(interaction_rng,interaction_randvals, num_lattices*nx*ny*2);
    init_randombond<<<blocks, THREADS>>>(interactions, interaction_randvals, nx, ny, num_lattices, p);
    
    cudaFree(interaction_randvals); 
    curandDestroyGenerator(interaction_rng);
}

void init_spins_with_seed(signed char* lattice_b, signed char* lattice_w, const long long seed, const long long nx, const long long ny, const int num_lattices){
    
    int blocks = (nx*ny*2*num_lattices + THREADS -1)/THREADS;
    
    // Setup cuRAND generator
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(rng, seed);

    float *randvals;
    cudaMalloc(&randvals, num_lattices * nx * ny/2 * sizeof(*randvals));

    //Initialize the arrays for white and black lattice
    curandGenerateUniform(rng, randvals, nx*ny/2*num_lattices);
    init_spins<<<blocks, THREADS>>>(lattice_b, randvals, nx, ny/2, num_lattices);

    curandGenerateUniform(rng, randvals, nx*ny/2*num_lattices);
    init_spins<<<blocks, THREADS>>>(lattice_w, randvals, nx, ny/2, num_lattices);

    curandDestroyGenerator(rng);
    cudaFree(randvals); 
}

void write_lattice(signed char *lattice_b, signed char *lattice_w, std::string filename, long long nx, long long ny, const int num_lattices) {
    printf("Writing lattice to %s...\n", filename.c_str());
    
    std::vector<signed char> lattice_h(nx*ny);
    std::vector<signed char> lattice_w_h(nx*ny/2*num_lattices);
    std::vector<signed char> lattice_b_h(nx*ny/2*num_lattices);

    cudaMemcpy(lattice_b_h.data(), lattice_b, num_lattices * nx * ny/2 * sizeof(*lattice_b), cudaMemcpyDeviceToHost);
    cudaMemcpy(lattice_w_h.data(), lattice_w, num_lattices * nx * ny/2 * sizeof(*lattice_w), cudaMemcpyDeviceToHost);

    int offset; 

    for (int l = 0; l < num_lattices; l++){
        
        offset = l*nx*ny/2;

        for (int i = 0; i < nx; i++){
            for (int j=0; j < ny/2; j++){
                if (i%2){
                    lattice_h[i*ny+2*j+1] = lattice_w_h[offset+i*ny/2+j];
                    lattice_h[i*ny+2*j] = lattice_b_h[offset+i*ny/2+j];
                }
                else{
                    lattice_h[i*ny+2*j] = lattice_w_h[offset+i*ny/2+j];
                    lattice_h[i*ny+2*j+1] = lattice_b_h[offset+i*ny/2+j];
                }
            }
        }

        std::ofstream f;
        f.open(filename + std::to_string(l) + std::string(".txt"));
        
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
}

void write_bonds(signed char* interactions, std::string filename, long nx, long ny, const int num_lattices){
    printf("Writing bonds to %s ...\n", filename.c_str());
    
    std::vector<signed char> interactions_host(2*nx*ny*num_lattices);
    
    cudaMemcpy(interactions_host.data(),interactions, 2*num_lattices*nx*ny*sizeof(*interactions), cudaMemcpyDeviceToHost);
    
    int offset;

    for (int l=0; l<num_lattices; l++){
        
        offset = l*nx*ny*2;
        
        std::ofstream f;
        f.open(filename + std::to_string(l) + std::string(".txt"));
        if (f.is_open()) {
            for (int i = 0; i < 2*nx; i++) {
                for (int j = 0; j < ny; j++) {
                    f << (int)interactions_host[offset + i * ny + j] << " ";
                }
                f << std::endl;
            }
        }
        f.close();
    }
}

template<bool is_black>
__global__ void update_lattice(signed char* lattice, signed char* __restrict__ op_lattice, const float* __restrict__ randvals, signed char* interactions,
                               const float inv_temp,
                               const long long nx,
                               const long long ny,
                               const int num_lattices,
                               const float coupling_constant) {

    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= nx*ny*num_lattices) return;
    
    // Calculate in which lattice we are
    int l_id = tid/(nx*ny);
    
    // Project tid back to single lattice 
    int tid_sl = tid - l_id*nx*ny;

    int i = tid_sl/ny;
    int j = tid_sl%ny;

    // Set up periodic boundary conditions
    int ipp = (i + 1 < nx) ? i + 1 : 0;
    int inn = (i - 1 >= 0) ? i - 1: nx - 1;
    int jpp = (j + 1 < ny) ? j + 1 : 0;
    int jnn = (j - 1 >= 0) ? j - 1: ny - 1;

    int joff;
    int jcouplingoff;
    int icouplingpp;
    int icouplingnn;
    
    int offset = l_id * nx * ny;
    int offset_i = l_id * nx * ny * 4;
    
    if (is_black) {
        icouplingpp = offset_i + 2*(nx-1)*ny + 2*(ny*(i+1) + j) + (i+1)%2;
        icouplingnn = offset_i + 2*(nx-1)*ny + 2*(ny*(inn+1) + j) + (i+1)%2;
        
        joff = (i % 2) ? jnn : jpp;

        if (i % 2) {
            jcouplingoff = offset_i + 2 * (i * ny + joff) + 1;
        } else {
            if (j + 1 >= ny) {
                jcouplingoff = offset_i + 2 * (i * ny + j + 1) - 1;
            } else {
                jcouplingoff = offset_i + 2 * (i * ny + joff) - 1;
            }
        }
    } else {
        icouplingpp = offset_i + 2*(nx-1)*ny + 2*(ny*(i+1) + j) + i%2;
        icouplingnn = offset_i + 2*(nx-1)*ny + 2*(ny*(inn+1) + j) + i%2;
        
        joff = (i % 2) ? jpp : jnn;

        if (i % 2) {
            if (j+1 >= ny) {
                jcouplingoff = offset_i + 2 * (i * ny + j + 1) - 1;
            } else {
                jcouplingoff = offset_i + 2 * (i * ny + joff) - 1;
            }
        } else {
            jcouplingoff = offset_i + 2 * (i * ny + joff) + 1;
        }
    }

    // Compute sum of nearest neighbor spins times the coupling
    signed char nn_sum = op_lattice[offset + inn*ny + j]*interactions[icouplingnn] + op_lattice[offset + i*ny + j]*interactions[offset_i + 2*(i*ny + j)] 
                        + op_lattice[offset + ipp*ny + j]*interactions[icouplingpp] + op_lattice[offset + i*ny + joff]*interactions[jcouplingoff];

    // Determine whether to flip spin
    signed char lij = lattice[offset + i*ny + j];
    float acceptance_ratio = exp(-2 * coupling_constant * nn_sum * lij);
    if (randvals[offset + i*ny + j] < acceptance_ratio) {
        lattice[offset + i*ny + j] = -lij;
    }  
}

void update(signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, signed char* interactions, float inv_temp, long long nx, long long ny, const int num_lattices, float coupling_constant) {
 
    // Setup CUDA launch configuration
    int blocks = (nx * ny/2 * num_lattices + THREADS - 1) / THREADS;

    // Update black
    curandGenerateUniform(rng, randvals, num_lattices*nx*ny/2);
    update_lattice<true><<<blocks, THREADS>>>(lattice_b, lattice_w, randvals,interactions, inv_temp, nx, ny/2, num_lattices, coupling_constant);

    // Update white
    curandGenerateUniform(rng, randvals, num_lattices*nx*ny/2);
    update_lattice<false><<<blocks, THREADS>>>(lattice_w, lattice_b, randvals,interactions, inv_temp, nx, ny/2, num_lattices, coupling_constant);
}



int main(int argc, char **argv) {
    
    long nx = 1000;
    long ny = 1000;
    int niters = 10000;
    float alpha = 1.0f;
    int nwarmup = 100;
    float TCRIT = 8.0f;
    float inv_temp = 1.0f / (alpha*TCRIT);

    unsigned long long seed = 0ULL;
    const float p = 0.031091730001f;
    const float coupling_constant = 0.5*TCRIT*log((1-p)/p);

    int num_lattices = 3;

    // Setup black and white lattice arrays on device
    signed char *lattice_b, *lattice_w;
    cudaMalloc(&lattice_b, num_lattices *nx * ny/2 * sizeof(*lattice_b));
    cudaMalloc(&lattice_w, num_lattices * nx * ny/2 * sizeof(*lattice_w));

    init_spins_with_seed(lattice_b, lattice_w, seed, nx, ny, num_lattices);

    signed char *d_interactions;
    cudaMalloc(&d_interactions, num_lattices*nx*ny*2*sizeof(*d_interactions));

    init_interactions_with_seed(d_interactions, seed, nx, ny, num_lattices, p);

    // Setup cuRAND generator
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(rng, seed);
    
    float *randvals;
    cudaMalloc(&randvals, num_lattices * nx * ny/2 * sizeof(*randvals));

    cudaDeviceSynchronize();

    // Warmup iterations
    printf("Starting warmup...\n");
    for (int j = 0; j < nwarmup; j++) {
        update(lattice_b, lattice_w, randvals, rng, d_interactions, inv_temp, nx, ny, num_lattices, coupling_constant);
    }
    
    cudaDeviceSynchronize();

    for (int j = 0; j < niters; j++) {
        update(lattice_b, lattice_w, randvals, rng, d_interactions, inv_temp, nx, ny, num_lattices, coupling_constant);
        if (j % 1000 == 0) printf("Completed %d/%d iterations...\n", j+1, niters);
    }

    write_lattice(lattice_b, lattice_w, "lattice/final_lattice_", nx, ny, num_lattices);
    write_bonds(d_interactions, "bonds/final_bonds_", nx, ny, num_lattices);
}