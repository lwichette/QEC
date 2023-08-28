#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <iostream>

using namespace std;

const int THREADS = 256;


__global__ void
B2(const signed char *A, const float *B, thrust::complex<float> *C, int ny, int nx)
{
    /*
    Calculates the inner sum of eq B2. Sum of blocks and absolute value, square needs to be done on the host.
    */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    thrust::complex<float> imag = thrust::complex<float>(0, 1.0f);

    if (tid < nx*ny){       
        int i = tid/ny;
        int j = tid%ny;
        
        float dot = B[0]*i + B[1]*j;
        C[tid] = A[tid]*exp(imag*dot);
    }
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


__global__ void init_randombond(signed char* interactions, const float* __restrict__ interaction_randvals,
    const long long nx, const long long ny, const float p){
        
        const long long tid = static_cast<long long>(threadIdx.x + blockIdx.x * blockDim.x);
        
        if (tid >= 2*nx*ny) return;

        float bondrandval = interaction_randvals[tid];
        signed char bondval = (bondrandval<p)? -1 : 1;
        interactions[tid] = bondval;                                  
}

__global__ void HE(const signed char *lattice, const signed char *interactions, float J, float *summand, int ny, int nx){
    
    // Calculate indices i, j
    const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    const int i = tid / ny;
    const int j = tid % ny;
  
    if (i >= nx || j >= ny) return;
    
    // Set stencil indices with periodicity
    int inn = (i - 1 >= 0) ? i - 1: nx - 1;
    int jnn = (j - 1 >= 0) ? j - 1: ny - 1;

    summand[tid] = J * lattice[tid] * lattice[inn*ny+j]*interactions[tid] + J * lattice[tid] * lattice[i*ny+jnn]*interactions[tid+nx*ny];
}

int main(void){

    int *seeds = (int *)malloc(1000*sizeof(int));

    srand(time(NULL));
    
    for (int i=0; i < 1000; i++){
        seeds[i] = rand();
    }
    
    for (int i = 0; i< 1000; i++){
        for (int j = 0; j<1000; j++){
            if (i==j){
                continue;
            }
            if(seeds[i]==seeds[j]){
                printf("SAME VALUES\n");
            }
        }
    } 
}

int main(void){
    // Lattice size, probability, factors,...
    int nx = 30;
    int ny = 30;
    float p = 0.3;
    float beta = 0.5f;
    float J = 1/(2*beta)*log((1-p)/p);
    float K = 1/(2*beta)*log(1/(p*(1-p))); 

    unsigned long long seed = 1234ULL;

    // Launch the Vector Add CUDA Kernel
    int blocks =(nx * ny * 2 + THREADS - 1) / THREADS;

    // Initialize spin
    signed char *h_lattice = (signed char *)malloc(nx*ny*sizeof(signed char));

    // Initialize the host spins
    for (int i = 0; i < nx*ny; ++i){
        h_lattice[i] = 1;
    }

    // Initialize device spins
    signed char *d_lattice = NULL;
    cudaMalloc(&d_lattice, nx*ny*sizeof(signed char));

    // Copy lattice from host do device
    cudaMemcpy(d_lattice, h_lattice, nx*ny*sizeof(signed char), cudaMemcpyHostToDevice);

    //Setup cuRAND generator for the random bond sign
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(rng, seed);
    
    // Generate uniform numbers onto interaction randvals
    float *interaction_randvals;
    cudaMalloc(&interaction_randvals, nx*ny*2*sizeof(*interaction_randvals));
    curandGenerateUniform(rng, interaction_randvals, nx*ny*2);

    //Setup interaction lattice on gpu
    signed char *d_interactions;
    cudaMalloc(&d_interactions, nx*ny*2*sizeof(*d_interactions));
    
    init_randombond<<<blocks, THREADS>>>(d_interactions, interaction_randvals, nx, ny, p);

    // Synchronize
    cudaDeviceSynchronize();

    // Allocate the device output vector C
    float *d_summand;
    cudaMalloc(&d_summand, nx*ny*sizeof(float));

    // Calculate energy
    HE<<<blocks, THREADS>>>(d_lattice, d_interactions, J, d_summand, ny, nx);

    // Reduce sum
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
    
    hostsum += nx*ny*(-2*K);
    hoststum = -1*hostsum;
    
    printf("Reduced sum %f", *hostsum);
}