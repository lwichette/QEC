#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <curand.h>

using namespace std;

const int THREADS = 256;

__global__ void HE(const signed char *lattice, float J, float *summand, int ny, int nx){
    
    // Calculate indices i, j
    const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    const int i = tid / ny;
    const int j = tid % ny;
  
    if (i >= nx || j >= ny) return;
    
    // Set stencil indices with periodicity
    int inn = (i - 1 >= 0) ? i - 1: nx - 1;
    int jnn = (j - 1 >= 0) ? j - 1: ny - 1;

    summand[tid] = J * lattice[tid] * lattice[inn*ny+j] + J * lattice[tid] * lattice[i*ny+jnn];
}

// Initialize random bond signs
__global__ void init_randombond(signed char* interactions, const float* __restrict__ interaction_randvals,
    const long long nx, const long long ny, const float p){
    const long long tid = static_cast<long long>(threadIdx.x + blockIdx.x * blockDim.x);
    if (tid >= 2*nx*ny) return;

    float bondrandval = interaction_randvals[tid];
    signed char bondval = (bondrandval<p)? -1 : 1;
    interactions[tid] = bondval;                                  
}

int main(void){
    // Initialize parameters like probability of error, inv_temp, J and K
    float p = 0.3;
    float beta = 0.5f;
    float J = 1/(2*beta)*log((1-p)/p);
    float K = 1/(2*beta)*log(1/(p*(1-p))); 

    // Initialize lattice size
    int ny = 10;
    int nx = 10;

    // Set seed
    unsigned long long seed = 1234ULL;

    // Launch the Vector Add CUDA Kernel
    int blocks =(nx * ny * 2 + THREADS - 1) / THREADS;

    //Setup cuRAND generator for the random bond sign
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(rng, seed);

    float *interaction_randvals;
    cudaMalloc(&interaction_randvals, nx*ny*2*sizeof(float));
    curandGenerateUniform(rng, interaction_randvals, nx*ny*2*sizeof(float));

    //Setup interaction lattice on device
    signed char *interactions;
    cudaMalloc(&interactions, nx*ny*2*sizeof(*interactions));
    
    init_randombond<<<blocks, THREADS>>>(interactions, interaction_randvals, nx, ny, p);

    //Synchronize devices
    cudaDeviceSynchronize();

    signed char *h_interactions = (signed char *)malloc(nx*ny*2*sizeof(signed char));
    cudaMemcpy(interactions, h_interactions, nx*ny*2*sizeof(signed char), cudaMemcpyDeviceToHost);

    for (int i=0; i<nx*ny;i++){
        printf("Interactions %d\n", h_interactions[i]);
    }
    /*
    // Initialize spin
    signed char *h_lattice = (signed char *)malloc(nx*ny*sizeof(signed char));

    // Initialize the host spins
    for (int i = 0; i < nx*ny; ++i){
        if (i<nx*ny/2){
            h_lattice[i] = 1;
        }
        else{
            h_lattice[i] = -1;
        }   
    }

    // Initialize device spins
    signed char *d_lattice = NULL;
    cudaMalloc(&d_lattice, nx*ny*sizeof(signed char));

    // Allocate the host output vector C
    float *h_summand = (float *)malloc(nx*ny*sizeof(float));

    // Allocate the device output vector C
    float *d_summand = NULL;
    cudaMalloc(&d_summand, nx*ny*sizeof(float));
    
    // Copy lattice from host do device
    cudaMemcpy(d_lattice, h_lattice, nx*ny*sizeof(signed char), cudaMemcpyHostToDevice);

    HE<<<blocksPerGrid, threadsPerBlock>>>(d_lattice, J, d_summand, ny, nx);

    float* d_out = NULL;
    cudaMalloc(&d_out, sizeof(float));

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_summand, d_out, nx*ny);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_summand, d_out, nx*ny);

    float* hostsum = (float *)malloc(sizeof(float));
    cudaMemcpy(hostsum, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    
    cout << *hostsum;
    */
}   