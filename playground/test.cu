#include <cstdlib>
#include <iostream>
#include <time.h>
#include <fstream>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>

using namespace std;

#define THREADS 128


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

// Initialize lattice spins
__global__ void init_spins(signed char* lattice, const float* __restrict__ randvals,
    const long long nx, const long long ny) {
        const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
        if (tid >= nx * ny) return;
        
        float randval = randvals[tid];
        signed char val = (randval < 0.5f) ? -1 : 1;
        lattice[tid] = val;
}

template<bool is_black>
__global__ void calc_energy(float* sum, signed char* lattice, signed char* __restrict__ op_lattice, 
    signed char* interactions, const long long nx, const long long ny, const float coupling_constant){
    
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
    sum[tid] = -coupling_constant*lattice[i*ny+j]*(op_lattice[inn * ny + j]*interactions[icouplingnn] + op_lattice[i * ny + j]*interactions[2*(i*ny + j)] 
               + op_lattice[ipp * ny + j]*interactions[icouplingpp] + op_lattice[i * ny + joff]*interactions[jcouplingoff]);
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

void calculate_energy(signed char *lattice_b, signed char *lattice_w, signed char *d_interactions, float *d_store_energy, float coupling_constant, int i, long nx, long ny){
    // Calculate energy and reduce sum
    int blocks = (nx*ny*2 + THREADS -1)/THREADS;
    
    float *d_energy;
    cudaMalloc(&d_energy, nx*ny/2*sizeof(*d_energy));

    calc_energy<true><<<blocks,THREADS>>>(d_energy, lattice_b, lattice_w, d_interactions, nx, ny/2, coupling_constant);

    // Run sum-reduction
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_energy, &d_store_energy[i], nx*ny/2);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_energy, &d_store_energy[i], nx*ny/2);
    
    cudaFree(d_energy);
}

// Write interaction bonds to file
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

int main(void){
    
    // Initialize all possible parameters
    long nx = 1000;
    long ny = 1000;
    double p = 0.3f;
    float alpha = 1.0f;
    float TCRIT = 2.0f;
    float inv_temp = 1.0f / (alpha*TCRIT);
    const float coupling_constant = 0.5*TCRIT*log((1-p)/p);
    int seeds_interactions = 1234ULL;
    
    int blocks = (nx*ny/2 + THREADS -1)/THREADS;

    printf("Coupling constant %f \n", coupling_constant);
    
    //Setup interaction lattice on device
    signed char *d_interactions;
    cudaMalloc(&d_interactions, nx*ny*2*sizeof(*d_interactions));

    init_interactions_with_seed(d_interactions, seeds_interactions, nx, ny, p);

    // Setup cuRAND generator
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(rng, seeds_interactions);
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

    cudaDeviceSynchronize();

    float *d_energy;
    cudaMalloc(&d_energy, nx*ny/2*sizeof(float));

    calc_energy<true><<<blocks, THREADS>>>(d_energy, lattice_b, lattice_w, d_interactions, nx, ny/2, coupling_constant);

    float *h_energy = (float *)malloc(nx*ny/2*sizeof(float));

    cudaMemcpy(h_energy, d_energy, nx*ny/2*sizeof(float), cudaMemcpyDeviceToHost);

    float h_sum = 0;

    for (int i=0;i<nx*ny/2;i++){
        h_sum += h_energy[i];
    }

    printf("Total energy: %f \n", h_sum);

    write_lattice(lattice_b, lattice_w, "check.txt", nx, ny);
    write_bonds(d_interactions, "jamesbonds.txt", nx, ny);
    
    /*
    float h_sum = 0;

    for (int i=0; i<1;i++){
    h_sum += h_energy[i];
    }
    printf("Total energy: %f \n", h_sum);

    //

    signed char *h_lattice_w;
    h_lattice_w = (signed char*) malloc(nx * ny/2 * sizeof(*h_lattice_w));

    cudaMemcpy(h_lattice_w, lattice_w, nx*ny/2*sizeof(*h_lattice_w), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    for (int i=0; i<nx*ny/2;i++){
        printf("%u \n", h_lattice_w[i]);
    }
    */

}
