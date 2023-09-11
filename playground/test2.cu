#include <cstdlib>
#include <iostream>
#include <time.h>
#include <fstream>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>

using namespace std;

#define THREADS 128



// Initialize lattice spins
__global__ void init_spins(signed char* lattice,
    const float* __restrict__ randvals,
    const long long nx,
    const long long ny) {
    
    const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (tid >= nx * ny) return;

    float randval = randvals[tid];
    signed char val = (randval < 0.5f) ? -1 : 1;
    lattice[tid] = val;
}

 // Write lattice configuration to file
 void write_lattice(signed char *lattice_b, signed char *lattice_w, std::string filename, long long nx, long long ny) {
    printf("Writing lattice to %s...\n", filename.c_str());
    signed char *lattice_h, *lattice_b_h, *lattice_w_h;
    lattice_h = (signed char*) malloc(nx * ny * sizeof(*lattice_h));
    lattice_b_h = (signed char*) malloc(nx * ny/2 * sizeof(*lattice_b_h));
    lattice_w_h = (signed char*) malloc(nx * ny/2 * sizeof(*lattice_w_h));
  
    cudaMemcpy(lattice_b_h, lattice_b, nx * ny/2 * sizeof(*lattice_b), cudaMemcpyDeviceToHost);
    cudaMemcpy(lattice_w_h, lattice_w, nx * ny/2 * sizeof(*lattice_w), cudaMemcpyDeviceToHost);
  
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny/2; j++) {
        if (i % 2) {
          lattice_h[i*ny + 2*j+1] = lattice_w_h[i*ny/2 + j];
          lattice_h[i*ny + 2*j] = lattice_b_h[i*ny/2 + j];
        } else {
          lattice_h[i*ny + 2*j] = lattice_w_h[i*ny/2 + j];
          lattice_h[i*ny + 2*j+1] = lattice_b_h[i*ny/2 + j];
        }
      }
    }
  
    std::ofstream f;
    f.open(filename);
    if (f.is_open()) {
      for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
           f << (int)lattice_h[i * ny + j] << " ";
        }
        f << std::endl;
      }
    }
    f.close();
  
    free(lattice_h);
    free(lattice_b_h);
    free(lattice_w_h);
}


int main(int argc, char **argv) {
    long nx = 6;
    long ny = 6;
    unsigned long long seed = 1234ULL;
    
    int blocks = (nx*ny*2 + THREADS -1)/THREADS;

    // Setup cuRAND generator
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(rng, seed);
    float *randvals;
    cudaMalloc(&randvals, nx * ny/2 * sizeof(*randvals));

    // Setup black and white lattice arrays on device
    signed char *lattice_b, *lattice_w;
    cudaMalloc(&lattice_b, nx * ny/2 * sizeof(*lattice_b));
    cudaMalloc(&lattice_w, nx * ny/2 * sizeof(*lattice_w));

    //Initialize the arrays for white and black lattice
    curandGenerateUniform(rng, randvals, nx*ny/2);
    init_spins<<<blocks, THREADS>>>(lattice_b, randvals, nx, ny/2);
    curandGenerateUniform(rng, randvals, nx*ny/2);
    init_spins<<<blocks, THREADS>>>(lattice_w, randvals, nx, ny/2);
    
    write_lattice(lattice_b, lattice_w, "check.txt", nx, ny);

    signed char *lattice_b_h, *lattice_w_h;

    lattice_b_h = (signed char*) malloc(nx * ny/2 * sizeof(*lattice_b_h));
    lattice_w_h = (signed char*) malloc(nx * ny/2 * sizeof(*lattice_w_h));
  
    cudaMemcpy(lattice_b_h, lattice_b, nx * ny/2 * sizeof(*lattice_b), cudaMemcpyDeviceToHost);
  
    for (int i=0; i<nx*ny/2;i++){
        cout << (int)lattice_b_h[i];
        cout << "\n";
    }
}