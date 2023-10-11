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
#include <string>
#include <sys/stat.h>

#include "../header/defines.h"
#include "../header/utils_big.cuh"

void *d_temp = NULL;
size_t temp_storage = 0;

__global__ void init_randombond(
    signed char* interactions, const float* __restrict__ interaction_randvals,
    const long long nx, const long long ny, const int num_lattices, const float p
){
        
        const long long tid = static_cast<long long>(threadIdx.x + blockIdx.x * blockDim.x);
        
        if (tid >= 2*nx*ny*num_lattices) return;

        float bondrandval = interaction_randvals[tid];
        signed char bondval = (bondrandval<p)? -1 : 1;
        interactions[tid] = bondval;                                  
}

// Initialize lattice spins
__global__ void init_spins(
    signed char* lattice, const float* __restrict__ randvals,
    const long long nx, const long long ny, const int num_lattices
){
        const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
        if (tid >= nx * ny * num_lattices) return;
        
        float randval = randvals[tid];
        signed char val = (randval < 0.5f) ? -1 : 1;
        lattice[tid] = val;
}

void init_interactions_with_seed(
    signed char* interactions, const long long seed, curandGenerator_t interaction_rng, float* interaction_randvals,
    const long long nx, const long long ny, const int num_lattices, const float p
){
    int blocks = (nx*ny*2*num_lattices + THREADS -1)/THREADS;

    // Set Seed 
    curandSetPseudoRandomGeneratorSeed(interaction_rng,seed);
    
    curandGenerateUniform(interaction_rng,interaction_randvals, num_lattices*nx*ny*2);
    init_randombond<<<blocks, THREADS>>>(interactions, interaction_randvals, nx, ny, num_lattices, p);
}

void init_spins_with_seed(
    signed char* lattice_b, signed char* lattice_w, const long long seed, curandGenerator_t lattice_rng, float* lattice_randvals,
    const long long nx, const long long ny, const int num_lattices
){
    
    int blocks = (nx*ny*2*num_lattices + THREADS -1)/THREADS;
    
    // Set seed
    curandSetPseudoRandomGeneratorSeed(lattice_rng, seed);

    //Initialize the arrays for white and black lattice
    curandGenerateUniform(lattice_rng, lattice_randvals, nx*ny/2*num_lattices);
    init_spins<<<blocks, THREADS>>>(lattice_b, lattice_randvals, nx, ny/2, num_lattices);

    curandGenerateUniform(lattice_rng, lattice_randvals, nx*ny/2*num_lattices);
    init_spins<<<blocks, THREADS>>>(lattice_w, lattice_randvals, nx, ny/2, num_lattices);
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
__global__ void update_lattice(
    signed char* lattice, signed char* __restrict__ op_lattice, const float* __restrict__ randvals, signed char* interactions,
    const float *inv_temp,const long long nx, const long long ny, const int num_lattices, const float *coupling_constant
) {

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
    float acceptance_ratio = exp(-2 * coupling_constant[l_id] * nn_sum * lij);
    if (randvals[offset + i*ny + j] < acceptance_ratio) {
        lattice[offset + i*ny + j] = -lij;
    }  
}

void update(
    signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, signed char* interactions, 
    float *inv_temp, long long nx, long long ny, const int num_lattices, float *coupling_constant
) {
 
    // Setup CUDA launch configuration
    int blocks = (nx * ny/2 * num_lattices + THREADS - 1) / THREADS;

    // Update black
    curandGenerateUniform(rng, randvals, num_lattices*nx*ny/2);
    update_lattice<true><<<blocks, THREADS>>>(lattice_b, lattice_w, randvals,interactions, inv_temp, nx, ny/2, num_lattices, coupling_constant);

    // Update white
    curandGenerateUniform(rng, randvals, num_lattices*nx*ny/2);
    update_lattice<false><<<blocks, THREADS>>>(lattice_w, lattice_b, randvals,interactions, inv_temp, nx, ny/2, num_lattices, coupling_constant);
}

__global__ void B2_lattices(signed char* lattice_b, signed char* lattice_w, const float *wave_vector, thrust::complex<float> *sum,  int nx, int ny, int num_lattices){
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid >= nx*ny*num_lattices);

    int lid = tid/(nx*ny);
    int offset = lid*nx*ny;
    int tid_sl = tid - offset;

    int i = tid_sl/ny;
    int j = tid_sl%ny;

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
__global__ void calc_energy(
    float* sum, signed char* lattice, signed char* __restrict__ op_lattice,
    signed char* interactions, const long long nx, const long long ny, const int num_lattices, const float *coupling_constant
){

    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid>num_lattices*nx*ny) return;
    
    const int lid = tid/(nx*ny);
    const int offset = lid*nx*ny;
    int offset_i = lid * nx * ny * 4;
    const int tid_sl = tid - offset;
    const int i = tid_sl/ny;
    const int j = tid_sl%ny;

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
    sum[tid] = -1 * coupling_constant[lid]*lattice[offset + i*ny+j]*(op_lattice[offset + inn*ny + j]*interactions[icouplingnn] + op_lattice[offset + i*ny + j]*interactions[offset_i + 2*(i*ny + j)] 
    + op_lattice[offset + ipp*ny + j]*interactions[icouplingpp] + op_lattice[offset + i*ny + joff]*interactions[jcouplingoff]);
}

void calculate_B2(
    thrust::complex<float> *d_sum, signed char *lattice_b, signed char *lattice_w, thrust::complex<float> *d_store_sum, float *d_wave_vector, 
    int loc, const long nx, const long ny, const int num_lattices, const int num_iterations_seeds
){
    // Calculate B2 and reduce sum
    int blocks = (nx*ny*2*num_lattices + THREADS -1)/THREADS;

    B2_lattices<<<blocks, THREADS>>>(lattice_b, lattice_w, d_wave_vector, d_sum, nx, ny/2, num_lattices);

    for (int i=0; i<num_lattices; i++){
        cub::DeviceReduce::Sum(d_temp, temp_storage, d_sum + i*nx*ny/2, &d_store_sum[loc + i*num_iterations_seeds], nx*ny/2);
        cudaMalloc(&d_temp, temp_storage);
        cub::DeviceReduce::Sum(d_temp, temp_storage, d_sum + i*nx*ny/2, &d_store_sum[loc + i*num_iterations_seeds], nx*ny/2);
    }
}

void calculate_energy(
    float* d_energy, signed char *lattice_b, signed char *lattice_w, signed char *d_interactions, float *d_store_energy, 
    float *coupling_constant, const int loc, const int nx, const int ny, const int num_lattices, const int num_iterations_seeds
){
    // Calculate energy and reduce sum
    int blocks = (nx*ny*2*num_lattices + THREADS -1)/THREADS;

    calc_energy<true><<<blocks,THREADS>>>(d_energy, lattice_b, lattice_w, d_interactions, nx, ny/2, num_lattices, coupling_constant);

    for (int i=0; i<num_lattices; i++){
        cub::DeviceReduce::Sum(d_temp, temp_storage, d_energy + i*nx*ny/2, &d_store_energy[loc + i*num_iterations_seeds], nx*ny/2);
        cudaMalloc(&d_temp, temp_storage);
        cub::DeviceReduce::Sum(d_temp, temp_storage, d_energy + i*nx*ny/2, &d_store_energy[loc + i*num_iterations_seeds], nx*ny/2);
    }
}

__global__ void abs_square(thrust::complex<float> *d_store_sum, const int num_lattices, const int num_iterations){
    
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= num_lattices * num_iterations) return;

    d_store_sum[tid] = thrust::abs(d_store_sum[tid]) * thrust::abs(d_store_sum[tid]);
}

__global__ void exp_beta(float *d_store_energy, float *inv_temp, const int num_lattices, const int num_iterations, const int L){
    
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= num_iterations*num_lattices) return;

    int lid = tid/num_iterations;

    d_store_energy[tid] = exp(-inv_temp[lid]*d_store_energy[tid]/(L*L));
}

__global__ void weighted_energies(float *d_weighted_energies, float *d_store_energy, thrust::complex<float> *d_store_sum, float *d_partition_function, const int num_lattices, const int num_iterations){
    
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= num_lattices*num_iterations) return;
    
    int lid = tid/num_iterations;
    
    d_weighted_energies[tid] = d_store_energy[tid]*d_store_sum[tid].real() / d_partition_function[lid];
}

void calculate_weighted_energies(
    float* d_weighted_energies, float *d_error_weight, float *d_store_energy, thrust::complex<float> *d_store_sum, float *d_partition_function, 
    const int num_lattices, const int num_iterations_seeds, const int num_iterations_error, const int blocks, const int e
){
    weighted_energies<<<blocks, THREADS>>>(d_weighted_energies, d_store_energy, d_store_sum, d_partition_function, num_lattices, num_iterations_seeds);

    for (int i=0; i<num_lattices; i++){
        cub::DeviceReduce::Sum(d_temp, temp_storage, d_weighted_energies + i*num_iterations_seeds, &d_error_weight[e + i*num_iterations_error], num_iterations_seeds);
        cudaMalloc(&d_temp, temp_storage);
        cub::DeviceReduce::Sum(d_temp, temp_storage, d_weighted_energies + i*num_iterations_seeds, &d_error_weight[e + i*num_iterations_error], num_iterations_seeds);
    }
}

int create_results_folder(char* results){
    struct stat sb;

    if (stat(results, &sb) == 0){
        std::cout << "Results already exist, check file name";
        return 0;
    }
    else{
        mkdir(results, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        return 1;
    }
}

template<bool is_black>
__global__ void update_lattice_ob(
    signed char* lattice, signed char* __restrict__ op_lattice, const float* __restrict__ randvals, signed char* interactions,
    const float *inv_temp, const long long nx, const long long ny, const int num_lattices, const float *coupling_constant
){

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

    int c_up = 1-inn/(nx-1);
    int c_down = 1-(i+1)/nx;
    int c_side;

    if (is_black) {
        icouplingpp = offset_i + 2*(nx-1)*ny + 2*(ny*(i+1) + j) + (i+1)%2;
        icouplingnn = offset_i + 2*(nx-1)*ny + 2*(ny*(inn+1) + j) + (i+1)%2;
        
        joff = (i % 2) ? jnn : jpp;

        if (i % 2) {
            jcouplingoff = offset_i + 2 * (i * ny + joff) + 1;
            
            c_side = 1 - jnn/(ny-1);
        } else {
            c_side = 1 - (j+1)/ny;

            if (j + 1 >= ny) {
                jcouplingoff = offset_i + 2 * (i * ny + j + 1) - 1;
            } else {
                jcouplingoff = offset_i + 2 * (i * ny + joff) - 1;
            }
        }
    } 
    else {
        icouplingpp = offset_i + 2*(nx-1)*ny + 2*(ny*(i+1) + j) + i%2;
        icouplingnn = offset_i + 2*(nx-1)*ny + 2*(ny*(inn+1) + j) + i%2;
        
        joff = (i % 2) ? jpp : jnn;

        if (i % 2) {
            c_side = 1-(j+1)/ny;

            if (j+1 >= ny) {
                jcouplingoff = offset_i + 2 * (i * ny + j + 1) - 1;
            } else {
                jcouplingoff = offset_i + 2 * (i * ny + joff) - 1;
            }
        } else {
            c_side = 1-jnn/(ny-1);
            jcouplingoff = offset_i + 2 * (i * ny + joff) + 1;
        }
    }

    signed char nn_sum = op_lattice[offset + inn*ny + j]*interactions[icouplingnn]*c_up + op_lattice[offset + i*ny + j]*interactions[offset_i + 2*(i*ny + j)] 
                        + op_lattice[offset + ipp*ny + j]*interactions[icouplingpp]*c_down + op_lattice[offset + i*ny + joff]*interactions[jcouplingoff]*c_side;

    // Determine whether to flip spin
    signed char lij = lattice[offset + i*ny + j];
    float acceptance_ratio = exp(-2 * coupling_constant[l_id] * nn_sum * lij);
    if (randvals[offset + i*ny + j] < acceptance_ratio) {
        lattice[offset + i*ny + j] = -lij;
    }  
}

void update_ob(
    signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, signed char* interactions, 
    float *inv_temp, long long nx, long long ny, const int num_lattices, float *coupling_constant
) {
 
    // Setup CUDA launch configuration
    int blocks = (nx * ny/2 * num_lattices + THREADS - 1) / THREADS;

    // Update black
    curandGenerateUniform(rng, randvals, num_lattices*nx*ny/2);
    update_lattice_ob<true><<<blocks, THREADS>>>(lattice_b, lattice_w, randvals,interactions, inv_temp, nx, ny/2, num_lattices, coupling_constant);

    // Update white
    curandGenerateUniform(rng, randvals, num_lattices*nx*ny/2);
    update_lattice_ob<false><<<blocks, THREADS>>>(lattice_w, lattice_b, randvals,interactions, inv_temp, nx, ny/2, num_lattices, coupling_constant);
}

template<bool is_black>
__global__ void calc_energy_ob(
    float* sum, signed char* lattice, signed char* __restrict__ op_lattice,
    signed char* interactions, const long long nx, const long long ny, const int num_lattices, const float *coupling_constant
){

    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid>num_lattices*nx*ny) return;
    
    const int lid = tid/(nx*ny);
    const int offset = lid*nx*ny;
    int offset_i = lid * nx * ny * 4;
    const int tid_sl = tid - offset;
    const int i = tid_sl/ny;
    const int j = tid_sl%ny;

    // Set up periodic boundary conditions
    int ipp = (i + 1 < nx) ? i + 1 : 0;
    int inn = (i - 1 >= 0) ? i - 1: nx - 1;
    int jpp = (j + 1 < ny) ? j + 1 : 0;
    int jnn = (j - 1 >= 0) ? j - 1: ny - 1;

    int joff;
    int jcouplingoff;
    int icouplingpp;
    int icouplingnn;

    int c_up = 1-inn/(nx-1);
    int c_down = 1-(i+1)/nx;
    int c_side;
    
    if (is_black) {
        icouplingpp = offset_i + 2*(nx-1)*ny + 2*(ny*(i+1) + j) + (i+1)%2;
        icouplingnn = offset_i + 2*(nx-1)*ny + 2*(ny*(inn+1) + j) + (i+1)%2;
        
        joff = (i % 2) ? jnn : jpp;

        if (i % 2) {
            jcouplingoff = offset_i + 2 * (i * ny + joff) + 1;
            
            c_side = 1-jnn/(ny-1);
        } else {
            c_side = 1-(j+1)/ny;
            
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
            c_side = 1-(j+1)/ny;

            if (j+1 >= ny) {
                jcouplingoff = offset_i + 2 * (i * ny + j + 1) - 1;
            } else {
                jcouplingoff = offset_i + 2 * (i * ny + joff) - 1;
            }
        } else {
            c_side = 1-jnn/(ny-1);

            jcouplingoff = offset_i + 2 * (i * ny + joff) + 1;
        }
    }

    // Compute sum of nearest neighbor spins times the coupling
    sum[tid] = -1 * coupling_constant[lid]*lattice[offset + i*ny+j]*(op_lattice[offset + inn*ny + j]*interactions[icouplingnn]*c_up + op_lattice[offset + i*ny + j]*interactions[offset_i + 2*(i*ny + j)] 
    + op_lattice[offset + ipp*ny + j]*interactions[icouplingpp]*c_down + op_lattice[offset + i*ny + joff]*interactions[jcouplingoff]*c_side);

}


void calculate_energy_ob(
    float* d_energy, signed char *lattice_b, signed char *lattice_w, signed char *d_interactions, float *d_store_energy, 
    float *coupling_constant, const int loc, const int nx, const int ny, const int num_lattices, const int num_iterations_seeds
){
    // Calculate energy and reduce sum
    int blocks = (nx*ny*2*num_lattices + THREADS -1)/THREADS;

    calc_energy_ob<true><<<blocks,THREADS>>>(d_energy, lattice_b, lattice_w, d_interactions, nx, ny/2, num_lattices, coupling_constant);

    for (int i=0; i<num_lattices; i++){
        cub::DeviceReduce::Sum(d_temp, temp_storage, d_energy + i*nx*ny/2, &d_store_energy[loc + i*num_iterations_seeds], nx*ny/2);
        cudaMalloc(&d_temp, temp_storage);
        cub::DeviceReduce::Sum(d_temp, temp_storage, d_energy + i*nx*ny/2, &d_store_energy[loc + i*num_iterations_seeds], nx*ny/2);
    }
}

