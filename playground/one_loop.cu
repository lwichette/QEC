#include <cstdlib>
#include <fstream>
#include <iostream>
#include <time.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>
#include <cmath>
#include <math.h>
#include <vector>

using namespace std;

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
    curandDestroyGenerator(interaction_rng);
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

    curandDestroyGenerator(rng);
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
    sum[tid] = -1 * coupling_constant*lattice[i*ny+j]*(op_lattice[inn * ny + j]*interactions[icouplingnn] + op_lattice[i * ny + j]*interactions[2*(i*ny + j)] 
               + op_lattice[ipp * ny + j]*interactions[icouplingpp] + op_lattice[i * ny + joff]*interactions[jcouplingoff]);
}

__global__ void abs_square(thrust::complex<float> *d_store_sum, const int num_iterations){
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= num_iterations) return;

    d_store_sum[tid] = thrust::abs(d_store_sum[tid]) * thrust::abs(d_store_sum[tid]);
}

__global__ void exp_beta(float *d_store_energy, float inv_temp, const int num_iterations, const int L){
    
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= num_iterations) return;

    d_store_energy[tid] = exp(-inv_temp*d_store_energy[tid]/(L*L));
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

void calculate_B2(signed char *lattice_b, signed char *lattice_w, thrust::complex<float> *d_store_sum, float *d_wave_vector, int i, const long nx, const long ny){
    // Calculate B2 and reduce sum
    int blocks = (nx*ny*2 + THREADS -1)/THREADS;

    thrust::complex<float> *d_sum;
    cudaMalloc(&d_sum, nx*ny/2*sizeof(*d_sum));

    B2_lattices<<<blocks, THREADS>>>(lattice_b, lattice_w, d_wave_vector, d_sum, nx, ny/2);

    // Run sum-reduction
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, &d_store_sum[i], nx*ny/2);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, &d_store_sum[i], nx*ny/2);

    cudaFree(d_sum);
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

__global__ void weighted_energies(float *d_weighted_energies, float *d_store_energy, thrust::complex<float> *d_store_sum, float *d_partition_function, const int num_iterations){
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= num_iterations) return;

    d_weighted_energies[tid] = d_store_energy[tid]*d_store_sum[tid].real() / *d_partition_function;
}

void calculate_weighted_energies(float *d_error_weight, float *d_store_energy, thrust::complex<float> *d_store_sum, float *d_partition_function, const int num_iterations, const int blocks, const int j){
    // Calculate energy and reduce sum

    float *d_weighted_energies;
    cudaMalloc(&d_weighted_energies, num_iterations*sizeof(*d_weighted_energies));

    weighted_energies<<<blocks, THREADS>>>(d_weighted_energies, d_store_energy, d_store_sum, d_partition_function, num_iterations);

    // Run sum-reduction
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_weighted_energies, &d_error_weight[j], num_iterations);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_weighted_energies, &d_error_weight[j], num_iterations);
    
    cudaFree(d_weighted_energies);
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


int main(void){
    // Initialize all possible parameters
    float p = 0.06f;
    float alpha = 1.0f;
  
    int num_iterations_seeds = 10;
    int num_iterations_error = 10;

    int niters = 10;
    int nwarmup = 10;
    


    int L[6] = {12, 14, 18, 24, 28, 36};
    float temp[13] = {1.8f, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4};
    
    float inv_temp;
    float coupling_constant;

    unsigned long long seeds_spins = 0ULL;
    unsigned long long seeds_interactions = 0ULL;

    // Variables used for sum reduction
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
        
    // Allocate the wave vectors and copy it to GPU memory
    float wave_vector_0[2] = {0,0};
    float *d_wave_vector_0;
    cudaMalloc(&d_wave_vector_0, 2 * sizeof(*d_wave_vector_0));
    cudaMemcpy(d_wave_vector_0, wave_vector_0, 2*sizeof(float), cudaMemcpyHostToDevice);
    
    for (int l=0; l<6;l++){
        long nx = L[l];
        long ny = L[l];
    
        int blocks = (nx*ny*2 + THREADS -1)/THREADS;
        float wave_vector_k[2] = {2.0*M_PI/nx,0};
        float *d_wave_vector_k;
        cudaMalloc(&d_wave_vector_k, 2 * sizeof(*d_wave_vector_k));
        cudaMemcpy(d_wave_vector_k, wave_vector_k, 2*sizeof(float), cudaMemcpyHostToDevice);
    
        std::vector< float > psi_l;

        for (int t = 0; t < (int)sizeof(temp)/sizeof(temp[0]); t++){

            printf("Started temperature %f \n", temp[t]);

            inv_temp = 1/(alpha*temp[t]);
            coupling_constant = inv_temp;

            // Weighted error
            float *d_error_weight_0, *d_error_weight_k;
            cudaMalloc(&d_error_weight_0, num_iterations_error*sizeof(*d_error_weight_0));
            cudaMalloc(&d_error_weight_k, num_iterations_error*sizeof(*d_error_weight_k));

            // Loop over different errors
            for (int j=0; j<num_iterations_error; j++){

                // Initialize arrays on the GPU to store results per spin system for energy and sum of B2
                thrust::complex<float> *d_store_sum_0, *d_store_sum_k;
                cudaMalloc(&d_store_sum_0, num_iterations_seeds*sizeof(*d_store_sum_0));
                cudaMalloc(&d_store_sum_k, num_iterations_seeds*sizeof(*d_store_sum_k));

                float *d_store_energy;
                cudaMalloc(&d_store_energy, num_iterations_seeds*sizeof(*d_store_energy));

                //Setup interaction lattice on device
                signed char *d_interactions;
                cudaMalloc(&d_interactions, nx*ny*2*sizeof(*d_interactions));

                init_interactions_with_seed(d_interactions, seeds_interactions, nx, ny, p);
                
                //Synchronize devices
                cudaDeviceSynchronize();

                // Loop over number of iterations
                for (int i=0; i<num_iterations_seeds; i++){
                    
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
                    //printf("Starting warmup...\n");
                    for (int j = 0; j < nwarmup; j++) {
                        update(lattice_b, lattice_w, randvals, rng, d_interactions, inv_temp, nx, ny, coupling_constant);
                    }

                    //Synchronize devices
                    cudaDeviceSynchronize();

                    for (int j = 0; j < niters; j++) {
                        update(lattice_b, lattice_w, randvals, rng, d_interactions, inv_temp, nx, ny,coupling_constant);
                        //if (j % 1000 == 0) printf("Completed %d/%d iterations...\n", j+1, niters);
                    }
                    
                    cudaDeviceSynchronize();
                    
                    calculate_B2(lattice_b, lattice_w, d_store_sum_0, d_wave_vector_0, i, nx, ny);
                    calculate_B2(lattice_b, lattice_w, d_store_sum_k, d_wave_vector_k, i, nx, ny);
                    
                    cudaDeviceSynchronize();

                    calculate_energy(lattice_b, lattice_w, d_interactions, d_store_energy, coupling_constant, i, nx, ny);

                    cudaDeviceSynchronize();

                    seeds_spins += 1;

                    //write_lattice(lattice_b, lattice_w, "lattice/final_lattice_" + std::to_string(i) + ".txt", nx, ny);

                    cudaFree(lattice_b);
                    cudaFree(lattice_w);
                    cudaFree(randvals);
                    curandDestroyGenerator(rng);
                }
            
                // Take absolute square + exp
                abs_square<<<blocks, THREADS>>>(d_store_sum_0, num_iterations_seeds);
                abs_square<<<blocks, THREADS>>>(d_store_sum_k, num_iterations_seeds);
                
                exp_beta<<<blocks, THREADS>>>(d_store_energy, inv_temp, num_iterations_seeds, nx);
                
                // Calculate partition function
                float *d_partition_function;
                cudaMalloc(&d_partition_function, sizeof(float));
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_store_energy, d_partition_function, num_iterations_seeds);
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_store_energy, d_partition_function, num_iterations_seeds);

                calculate_weighted_energies(d_error_weight_0, d_store_energy, d_store_sum_0, d_partition_function, num_iterations_seeds, blocks, j);
                calculate_weighted_energies(d_error_weight_k, d_store_energy, d_store_sum_k, d_partition_function, num_iterations_seeds, blocks, j);

                cudaDeviceSynchronize();

                seeds_interactions += 1;

                //write_bonds(d_interactions, "lattice/final_bonds.txt", nx, ny);
                cudaFree(d_store_sum_0);
                cudaFree(d_store_sum_k);
                cudaFree(d_store_energy);
                cudaFree(d_interactions);
                cudaFree(d_partition_function);
            }

            // Magnetic susceptibility 
            float *d_magnetic_susceptibility_0, *d_magnetic_susceptibility_k;
            cudaMalloc(&d_magnetic_susceptibility_0, sizeof(*d_magnetic_susceptibility_0));
            cudaMalloc(&d_magnetic_susceptibility_k, sizeof(*d_magnetic_susceptibility_k));

            // Sum reduction for both
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_error_weight_0, d_magnetic_susceptibility_0, num_iterations_error);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_error_weight_0, d_magnetic_susceptibility_0, num_iterations_error);

            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_error_weight_k, d_magnetic_susceptibility_k, num_iterations_error);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_error_weight_k, d_magnetic_susceptibility_k, num_iterations_error);

            cudaDeviceSynchronize();

            float *h_magnetic_susceptibility_0 = (float *)malloc(sizeof(float));
            float *h_magnetic_susceptibility_k = (float *)malloc(sizeof(float));
            
            cudaMemcpy(h_magnetic_susceptibility_0, d_magnetic_susceptibility_0, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_magnetic_susceptibility_k, d_magnetic_susceptibility_k, sizeof(float), cudaMemcpyDeviceToHost);

            float psi = 1/(2*sin(M_PI/nx))*sqrt(*h_magnetic_susceptibility_0 / *h_magnetic_susceptibility_k - 1);

            psi_l.push_back(psi/nx);

            cudaFree(d_magnetic_susceptibility_0);
            cudaFree(d_magnetic_susceptibility_k);
            cudaFree(d_error_weight_0);
            cudaFree(d_error_weight_k);
        }

        std::ofstream f;
        f.open("lattice/psi_L_" + std::to_string(nx) + ".txt");
        if (f.is_open()) {
            for (int i = 0; i < (int)sizeof(temp)/sizeof(temp[0]); i++) {
                f << psi_l[i] << " " << temp[i] "\n";
            }
        }
        f.close();

        cudaFree(d_wave_vector_k);
    }


}