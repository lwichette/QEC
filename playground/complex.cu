#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <curand.h>

using namespace std;

const int THREADS = 256;


// Initialize lattice spins
__global__ void init_spins(signed char* lattice, const float* __restrict__ randvals,
    const long long nx, const long long ny) {
        const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
        if (tid >= nx * ny) return;
        
        float randval = randvals[tid];
        signed char val = (randval < 0.5f) ? -1 : 1;
        lattice[tid] = val;
}


int main(void){
    // Lattice size, probability, factors,...
    int nx = 10;
    int ny = 10;

    int blocks =(nx * ny * 2 + THREADS - 1) / THREADS;
    
    int num_lattices = 10;

    
    // Setup cuRAND generator
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(rng, seed);
    float *randvals;
    cudaMalloc(&randvals, nx * ny * sizeof(*randvals));
    curandGenerateUniform(rng, randvals, nx*ny);
    
    // Initialize lattice
    signed char *h_lattice = (signed char *)malloc(nx*ny*sizeof(signed char));
    signed char *d_lattice;
    cudaMalloc(&d_lattice, nx * ny * sizeof(*d_lattice));
    init_spins<<<blocks, THREADS>>>(d_lattice, randvals, nx, ny);

    cudaMemcpy(h_lattice, d_lattice, nx*ny*sizeof(signed char), cudaMemcpyDeviceToHost);

    // Check if they are the same
    float *randvals1;
    cudaMalloc(&randvals1, nx * ny * sizeof(*randvals));
    curandGenerateUniform(rng, randvals1, nx*ny);
    
    // Initialize lattice
    signed char *h_lattice1 = (signed char *)malloc(nx*ny*sizeof(signed char));
    signed char *d_lattice1;
    cudaMalloc(&d_lattice1, nx * ny * sizeof(*d_lattice1));
    init_spins<<<blocks, THREADS>>>(d_lattice1, randvals1, nx, ny);

    cudaMemcpy(h_lattice1, d_lattice1, nx*ny*sizeof(signed char), cudaMemcpyDeviceToHost);

    int equal = 0;

    for (int i=0; i<nx*ny;i++){
        if (h_lattice[i] == h_lattice1[i]){
            equal += 1;
        }
    }

    printf("%d", equal);
}