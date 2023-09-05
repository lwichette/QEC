#include <cstdlib>
#include <fstream>
#include <iostream>
#include <time.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>
#include <cmath>

using namespace std;

#define TCRIT 2.23f
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


// Initialize lattice spins
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

__global__ void B2_lattices(signed char* lattice_b, signed char* lattice_w, const float *wave_vector, thrust::complex<float> *sum,  int nx, int ny){
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    int i = tid/ny;
    int j = tid%ny;

    if (i>=nx || j >= ny) return;

    int b_orig_j;
    int w_orig_j; 

    if (i%2==0){
        b_orig_j = 2*j +1;
        w_orig_j = 2*j;
    }
    else{
        b_orig_j = 2*j;
        w_orig_j = 2*j + 1;
    }

    thrust::complex<float> imag = thrust::complex<float>(0, 1.0f);

    float dot_b = wave_vector[0]*i + wave_vector[1]*b_orig_j;
    float dot_w = wave_vector[0]*i + wave_vector[1]*w_orig_j;
    sum[tid] = lattice_b[tid]*exp(imag*dot_b) + lattice_w[tid]*exp(imag*dot_w);
}


template<bool is_black>
__global__ void calc_energy(float* sum, signed char* lattice, signed char* __restrict__ op_lattice, 
    signed char* interactions, const long long nx, const long long ny, const float coupling_constant, const float K){
    
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
    sum[tid] = coupling_constant*lattice[i*ny+j]*(op_lattice[inn * ny + j]*interactions[icouplingnn] + op_lattice[i * ny + j]*interactions[2*(i*ny + j)] 
               + op_lattice[ipp * ny + j]*interactions[icouplingpp] + op_lattice[i * ny + joff]*interactions[jcouplingoff]);
    
    sum[tid] = sum[tid] - 4*K;
    sum[tid] = -1 * sum[tid];
}


__global__ void abs_square(thrust::complex<float> *d_store_sum, const int num_iterations){
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= num_iterations) return;

    d_store_sum[tid] = thrust::abs(d_store_sum[tid]) * thrust::abs(d_store_sum[tid]);
}


__global__ void exp_beta(float *d_store_energy, float inv_temp, const int num_iterations){
    
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= num_iterations) return;

    d_store_energy[tid] = exp(-inv_temp*d_store_energy[tid]);
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


int main(void){

    // Initialize all possible parameters
    long nx = 1000;
    long ny = 1000;  
    float p = 0.3;
    //float p = 0.031091730001f;
    float alpha = 1.0f;
    float inv_temp = 1.0f / (alpha*TCRIT);
    const float coupling_constant = 0.5*TCRIT*log((1-p)/p);
    const float K = 0.5*TCRIT*log(1/((1-p)*p));

    int num_iterations = 2;
    
    int blocks = (nx*ny*2 + THREADS -1)/THREADS;

    // Initialize seeds used for spin and interaction initialization
    unsigned long long seeds_spins;
    unsigned long long seeds_interactions =  1234ULL;
    
    // Allocate the wave vector and copy it to GPU memory
    float *wave_vector = (float *)malloc(2*sizeof(float));
    wave_vector[0] = 0;
    wave_vector[1] = 0;

    float *d_wave_vector;
    cudaMalloc(&d_wave_vector, 2 * sizeof(*d_wave_vector));
    cudaMemcpy(d_wave_vector, wave_vector, 2*sizeof(float), cudaMemcpyHostToDevice);

    // Initialize arrays on the GPU to store results per spin system for energy and sum of B2
    thrust::complex<float> *d_store_sum;
    cudaMalloc(&d_store_sum, num_iterations*sizeof(*d_store_sum));

    float *d_store_energy;
    cudaMalloc(&d_store_energy, num_iterations*sizeof(*d_store_energy));

    // Initialize arrays on the CPU to store results per spin system for energy and sum of B2
    thrust::complex<float> *h_sum = (thrust::complex<float> *)malloc(num_iterations*sizeof(thrust::complex<float>));;
    float *h_energy = (float *)malloc(num_iterations*sizeof(float));

    // Initialize parameters needed later for the sum reduction procedure
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    //Setup interaction lattice on device
    signed char *d_interactions;
    
    cudaMalloc(&d_interactions, nx*ny*2*sizeof(*d_interactions));

    init_interactions_with_seed(d_interactions, seeds_interactions, nx, ny, p);

    // Loop over number of iterations
    for (int i=0; i<num_iterations; i++){

        seeds_spins = i;

        // Setup black and white lattice arrays on device
        signed char *lattice_b, *lattice_w;
        cudaMalloc(&lattice_b, nx * ny/2 * sizeof(*lattice_b));
        cudaMalloc(&lattice_w, nx * ny/2 * sizeof(*lattice_w));

        init_spins_with_seed(lattice_b, lattice_w, seeds_spins, nx, ny);
        
        // Calculate B2 and reduce sum
        thrust::complex<float> *d_sum;
        cudaMalloc(&d_sum, nx*ny/2*sizeof(*d_sum));

        B2_lattices<<<blocks, THREADS>>>(lattice_b, lattice_w, d_wave_vector, d_sum, nx, ny/2);

        // Run sum-reduction
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, &d_store_sum[i], nx*ny/2);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, &d_store_sum[i], nx*ny/2);
        
        // Calculate energy
        float *d_energy;
        cudaMalloc(&d_energy, nx*ny/2*sizeof(*d_energy));

        calc_energy<true><<<blocks,THREADS>>>(d_energy, lattice_b, lattice_w, d_interactions, nx, ny/2, coupling_constant, K);

        // Run sum-reduction
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_energy, &d_store_energy[i], nx*ny/2);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_energy, &d_store_energy[i], nx*ny/2);
    }
    
    float *h_energy_before = (float *)malloc(num_iterations*sizeof(float));
    cudaMemcpy(h_energy_before, d_store_energy, num_iterations*sizeof(float), cudaMemcpyDeviceToHost);

    abs_square<<<blocks, THREADS>>>(d_store_sum, num_iterations); 
    exp_beta<<<blocks, THREADS>>>(d_store_energy, inv_temp, num_iterations);

    cudaMemcpy(h_sum, d_store_sum, num_iterations*sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_energy, d_store_energy, num_iterations*sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i=0; i<num_iterations;i++){
        printf("Energy before exponentiating %f \n", h_energy_before[i]);
        printf("Energy after exponentiating %f \n", h_energy[i]);
    }
    

    /*
    int *d_out;
    cudaMalloc(&d_out, sizeof(int));

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_interactions, d_out, nx*ny*2);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_interactions, d_out, nx*ny*2);

    int *h_out = (int *)malloc(sizeof(int));

    cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d \n", *h_out);

    write_bonds(d_interactions, "final_bonds.txt" , nx, ny);
    /*
    float *d_sum_energy;
    cudaMalloc(&d_sum_energy, sizeof(*d_sum_energy));

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_energy, d_sum_energy, nx*ny/2);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_energy, d_sum_energy, nx*ny/2);

    float *h_sum_energy = (float *)malloc(sizeof(float));

    cudaMemcpy(h_sum_energy, d_sum_energy, sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f \n", h_sum_energy);


   





    
    
    // Check how many errors occurred
    int *h_sum_of_interactions = (int *)malloc(sizeof(int));
    int *d_sum_of_interactions;
    cudaMalloc(&d_sum_of_interactions, sizeof(*d_sum_of_interactions));

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_interactions, d_sum_of_interactions, nx*ny*2);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_interactions, d_sum_of_interactions, nx*ny*2);
    
    cudaMemcpy(h_sum_of_interactions, d_sum_of_interactions, sizeof(int), cudaMemcpyDeviceToHost);
    
    int plus_one = (2*nx*ny + *h_sum_of_interactions)/2;
    int minus_one = (2*nx*ny - *h_sum_of_interactions)/2;

    // cout << *h_sum_of_interactions;
    float prob_error = pow(p, minus_one)*pow(1-p, plus_one);

    // printf("\n%f \n", prob_error);

    write_bonds(d_interactions, "final_bonds.txt" , nx, ny);





    abs_square<<<blocks, THREADS>>>(d_store_sum, num_iterations); 

    exp_beta<<<blocks, THREADS>>>(d_store_energy, inv_temp, num_iterations);

    cudaMemcpy(h_sum, d_store_sum, num_iterations*sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_energy, d_store_energy, num_iterations*sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i=0; i<num_iterations;i++){
        printf("%f \n", h_energy[i]);
    }
    
    float *d_partition_function;
    cudaMalloc(&d_partition_function, sizeof(float));
    
    // Run sum-reduction for partition function
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_store_energy, d_partition_function, num_iterations);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_store_energy, d_partition_function, num_iterations);
    */
}