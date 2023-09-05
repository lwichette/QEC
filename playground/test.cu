#include <cstdlib>
#include <iostream>
#include <time.h>
#include <fstream>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>

using namespace std;

#define TCRIT 1
#define THREADS 128


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


int main(void){
    
    // Initialize all possible parameters
    long nx = 200;
    long ny = 200;  
    double p = 0.5;
    float alpha = 1.0f;
    float inv_temp = 1.0f / (alpha*TCRIT);
    const float coupling_constant = 0.5*TCRIT*log((1-p)/p);
    const float K = 0.5*TCRIT*log(1/((1-p)*p));

    int blocks = (nx*ny*2 + THREADS - 1)/THREADS;

    unsigned long long seeds_interactions =  123ULL;

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    //Setup interaction lattice on device
    signed char *h_interactions = (signed char *)malloc(nx*ny*2*sizeof(signed char));
    signed char *d_interactions;
    cudaMalloc(&d_interactions, nx*ny*2*sizeof(*d_interactions));

    init_interactions_with_seed(d_interactions, seeds_interactions, nx, ny, p);

    // Check how many errors occurred
    int *h_sum_of_interactions = (int *)malloc(sizeof(int));
    int *d_sum_of_interactions;
    cudaMalloc(&d_sum_of_interactions, sizeof(*d_sum_of_interactions));

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_interactions, d_sum_of_interactions, nx*ny*2);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_interactions, d_sum_of_interactions, nx*ny*2);
    
    cudaMemcpy(h_sum_of_interactions, d_sum_of_interactions, sizeof(int), cudaMemcpyDeviceToHost);
    
    cout << *h_sum_of_interactions;

    write_bonds(d_interactions, "final_bonds.txt" , nx, ny);
}
