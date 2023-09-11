#include <cstdlib>
#include <fstream>
#include <iostream>
#include <time.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>
#include <cmath>

using namespace std;

#define THREADS 128


void write_bonds(signed char* interactions, std::string filename, long long nx, long long ny){
    printf("Writing bonds to %s ...\n", filename.c_str());
    signed char *interactions_host;
    interactions_host = (signed char*)malloc(2*nx*ny*sizeof(*interactions_host));
    cudaMemcpy(interactions_host,interactions, 2*nx*ny*sizeof(*interactions), cudaMemcpyDeviceToHost);
        
      std::ofstream f;
      f.open(filename);
      if (f.is_open()) {
        for (int i = 0; i < 2*nx; i++) {
          for (int j = 0; j < ny; j++) {
             f << (int)interactions_host[i * ny + j] << " ";
          }
          f << std::endl;
        }
      }
      f.close();
      cudaFree(interactions);
      free(interactions_host);
}

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
          lattice_h[i*ny + 2*j+1] = lattice_b_h[i*ny/2 + j];
          lattice_h[i*ny + 2*j] = lattice_w_h[i*ny/2 + j];
        } else {
          lattice_h[i*ny + 2*j] = lattice_b_h[i*ny/2 + j];
          lattice_h[i*ny + 2*j+1] = lattice_w_h[i*ny/2 + j];
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

__global__ void init_randombond(signed char* interactions, const float* __restrict__ interaction_randvals,
    const long long nx, const long long ny, const float p){
        
        const long long tid = static_cast<long long>(threadIdx.x + blockIdx.x * blockDim.x);
        
        if (tid >= 2*nx*ny) return;

        float bondrandval = interaction_randvals[tid];
        signed char bondval = (bondrandval<p)? -1 : 1;
        interactions[tid] = bondval;                                  
}

void init_interactions_with_seed(signed char* interactions, const long long seed, const long long nx, const long long ny, const float p){
    int blocks = (nx*ny*2 + THREADS -1)/THREADS;

    //Setup cuRAND generator for the random bond sign
    curandGenerator_t interaction_rng;
    curandCreateGenerator(&interaction_rng,CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(interaction_rng,seed);
    
    float *interaction_randvals;
    cudaMalloc(&interaction_randvals,nx*ny*2*sizeof(*interaction_randvals));

    curandGenerateUniform(interaction_rng,interaction_randvals,nx*ny*2);
    init_randombond<<<blocks, THREADS>>>(interactions, interaction_randvals,nx,ny,p);
    
    cudaFree(interaction_randvals); 
}

__global__ void init_spins(signed char* lattice, const float* __restrict__ randvals,
    const long long nx, const long long ny) {
        const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
        if (tid >= nx * ny) return;
        
        float randval = randvals[tid];
        signed char val = (randval < 0.5f) ? -1 : 1;
        lattice[tid] = val;
}

void init_spins_with_seed(signed char* lattice_b, signed char* lattice_w, const long long seed, const long long nx, const long long ny){
    
    int blocks = (nx*ny*2 + THREADS -1)/THREADS;
    
    // Setup cuRAND generator
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(rng, seed);

    float *randvals;
    cudaMalloc(&randvals, nx * ny/2 * sizeof(*randvals));

    //Initialize the arrays for white and black lattice
    curandGenerateUniform(rng, randvals, nx*ny/2);
    init_spins<<<blocks, THREADS>>>(lattice_b, randvals, nx, ny/2);
    curandGenerateUniform(rng, randvals, nx*ny/2);
    init_spins<<<blocks, THREADS>>>(lattice_w, randvals, nx, ny/2);

    cudaFree(randvals); 
}

template<bool is_black>
__global__ void update_lattice(signed char* lattice, signed char* __restrict__ op_lattice, const float* __restrict__ randvals, signed char* interactions,
                               const float inv_temp,
                               const long long nx,
                               const long long ny,
                               const float coupling_constant) {

    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;
    
    const int i = tid/ny;
    const int j = tid%ny;

    if (i>=nx || j >= ny) return;

    // Set up periodic boundary conditions
    int ipp = (i + 1 < nx) ? i + 1 : 0;
    int inn = (i - 1 >= 0) ? i - 1: nx - 1;
    int jpp = (j + 1 < ny) ? j + 1 : 0;
    int jnn = (j - 1 >= 0) ? j - 1: ny - 1;

    int joff;
    int jcouplingoff;
    int icouplingpp;
    int icouplingnn;

    if (is_black) {
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
    signed char nn_sum = op_lattice[inn * ny + j]*interactions[icouplingnn] + op_lattice[i * ny + j]*interactions[2*(i*ny + j)] 
                        + op_lattice[ipp * ny + j]*interactions[icouplingpp] + op_lattice[i * ny + joff]*interactions[jcouplingoff];

    // Compute sum of nearest neighbor spins
    //signed char nn_sum = op_lattice[inn * ny + j] + op_lattice[i * ny + j] + op_lattice[ipp * ny + j] + op_lattice[i * ny + joff];

    // Determine whether to flip spin
    signed char lij = lattice[i * ny + j];
    float acceptance_ratio = exp(-2 * coupling_constant * nn_sum * lij);
    if (randvals[i*ny + j] < acceptance_ratio) {
        lattice[i * ny + j] = -lij;
    }  
}

void update(signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, signed char* interactions, float inv_temp, long long nx, long long ny, float coupling_constant) {
 
    // Setup CUDA launch configuration
    int blocks = (nx * ny/2 + THREADS - 1) / THREADS;

    // Update black
    curandGenerateUniform(rng, randvals, nx*ny/2);
    update_lattice<true><<<blocks, THREADS>>>(lattice_b, lattice_w, randvals,interactions, inv_temp, nx, ny/2,coupling_constant);

    // Update white
    curandGenerateUniform(rng, randvals, nx*ny/2);
    update_lattice<false><<<blocks, THREADS>>>(lattice_w, lattice_b, randvals,interactions, inv_temp, nx, ny/2, coupling_constant);
}

int main(int argc, char **argv) {
    // Initialize all possible parameters
    int niters = 1000;
    int nwarmup = 100;
    long nx = 1000;
    long ny = 1000;  
    //float p = 0.15;
    float p = 0.031091730001f;
    float alpha = 1.0f;
    float TCRIT = 8.0f;
    float inv_temp = 1.0f / (alpha*TCRIT);
    const float coupling_constant = 0.5*TCRIT*log((1-p)/p);

    int num_iterations = 1;
    
    int blocks = (nx*ny*2 + THREADS -1)/THREADS;

    // Initialize seeds used for spin and interaction initialization
    unsigned long long seeds_spins = 1234ULL;
    unsigned long long seeds_interactions =  1234ULL;
    
    // Allocate the wave vector and copy it to GPU memory
    float wave_vector[2] = {0,0};

    float *d_wave_vector;
    cudaMalloc(&d_wave_vector, 2 * sizeof(*d_wave_vector));
    cudaMemcpy(d_wave_vector, wave_vector, 2*sizeof(float), cudaMemcpyHostToDevice);

    // Initialize arrays on the GPU to store results per spin system for energy and sum of B2
    thrust::complex<float> *d_store_sum;
    cudaMalloc(&d_store_sum, num_iterations*sizeof(*d_store_sum));

    float *d_store_energy;
    cudaMalloc(&d_store_energy, num_iterations*sizeof(*d_store_energy));

    //Setup interaction lattice on device
    signed char *d_interactions;
    cudaMalloc(&d_interactions, nx*ny*2*sizeof(*d_interactions));

    init_interactions_with_seed(d_interactions, seeds_interactions, nx, ny, p);

    //Synchronize devices
    cudaDeviceSynchronize();

    // Loop over number of iterations
    for (int i=0; i<num_iterations; i++){
        
        // Setup black and white lattice arrays on device
        signed char *lattice_b, *lattice_w;
        cudaMalloc(&lattice_b, nx * ny/2 * sizeof(*lattice_b));
        cudaMalloc(&lattice_w, nx * ny/2 * sizeof(*lattice_w));

        init_spins_with_seed(lattice_b, lattice_w, seeds_spins, nx, ny); 

        // Setup cuRAND generator
        curandGenerator_t rng;
        curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        curandSetPseudoRandomGeneratorSeed(rng, seeds_spins);
        float *randvals;
        cudaMalloc(&randvals, nx * ny/2 * sizeof(*randvals));
        
        //Synchronize devices
        cudaDeviceSynchronize();

        // Warmup iterations
        printf("Starting warmup...\n");

        for (int i = 0; i < nwarmup; i++) {
            update(lattice_b, lattice_w, randvals, rng, d_interactions, inv_temp, nx, ny, coupling_constant);
        }

        //Synchronize devices
        cudaDeviceSynchronize();
        
        for (int i = 0; i < niters; i++) {
            update(lattice_b, lattice_w, randvals, rng, d_interactions, inv_temp, nx, ny,coupling_constant);
            if (i % 1000 == 0) printf("Completed %d/%d iterations...\n", i+1, niters);
        }
    
        cudaDeviceSynchronize();

        write_lattice(lattice_b, lattice_w, "final.txt", nx, ny);
        write_bonds(d_interactions, "final_bonds.txt" ,nx, ny);
    }
}