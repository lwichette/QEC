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

#include "utils_big.cuh"
#include "defines.h"

using namespace std;

// Run sum-reduction
void *d_temp = NULL;
size_t temp_storage = 0;

__global__ void init_randombond(signed char* interactions, const float* __restrict__ interaction_randvals,
    const long long nx, const long long ny, const int num_lattices, const float* p){
        
        const long long tid = static_cast<long long>(threadIdx.x + blockIdx.x * blockDim.x);
        
        if (tid >= 2*nx*ny*num_lattices) return;

        const int lid = tid/(2*nx*ny);

        float bondrandval = interaction_randvals[tid];
        signed char bondval = (bondrandval < p[lid])? -1 : 1;
        interactions[tid] = bondval;                                 
}

// Initialize lattice spins
__global__ void init_spins(signed char* lattice, const float* __restrict__ randvals,
    const long long nx, const long long ny, const int num_lattices) {
        const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
        if (tid >= nx * ny * num_lattices) return;
        
        float randval = randvals[tid];
        signed char val = (randval < 0.5f) ? -1 : 1;
        lattice[tid] = val;
}

void init_interactions_with_seed(
    signed char* interactions, const long long seed, curandGenerator_t interaction_rng, float* interaction_randvals,
    const long long nx, const long long ny, const int num_lattices, const float* p
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
__global__ void update_lattice(signed char* lattice, signed char* __restrict__ op_lattice, const float* __restrict__ randvals, signed char* interactions,
                               const float *inv_temp,
                               const long long nx,
                               const long long ny,
                               const int num_lattices,
                               const float *coupling_constant) {

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
    float *inv_temp, long long nx, long long ny, const int num_lattices, float *coupling_constant) {
 
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
__global__ void calc_energy(float* sum, signed char* lattice, signed char* __restrict__ op_lattice,
    signed char* interactions, const long long nx, const long long ny, const int num_lattices, const float *coupling_constant){

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

int main(int argc, char **argv){

    char *results = "results/test_Nishimori";
    int check = create_results_folder(results);
    if (check == 0) return 0;
    
    // Number iterations and how many lattices
    int num_iterations_seeds = 10;
    int num_iterations_error = 10;
    int niters = 10;
    int nwarmup = 10;
    
    // Temp
    float start_prob = 0.102f;
    float step = 0.001;
    int num_lattices = 11;

    float run_probs;
    std::vector<float> probs; 
    std::vector<float> inv_temp;
    std::vector<float> coupling_constant;

    for (int i=0; i < num_lattices; i++){
        run_probs = start_prob + i*step;
        probs.push_back(run_probs);
        inv_temp.push_back(1.0f/2.0f*log((1-run_probs)/run_probs));
        coupling_constant.push_back(1.0f/2.0f*log((1-run_probs)/run_probs));
    }

    float *d_probs;
    cudaMalloc(&d_probs, num_lattices*sizeof(float));
    cudaMemcpy(d_probs, probs.data(), num_lattices*sizeof(float), cudaMemcpyHostToDevice);

    float *d_inv_temp, *d_coupling_constant;
    cudaMalloc(&d_inv_temp, num_lattices*sizeof(float));
    cudaMalloc(&d_coupling_constant, num_lattices*sizeof(float));
    cudaMemcpy(d_inv_temp, inv_temp.data(), num_lattices*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coupling_constant, coupling_constant.data(), num_lattices*sizeof(float), cudaMemcpyHostToDevice);  

    // Lattice size
    std::array<int, 1> L_size = {12};
    
    for(int ls = 0; ls < L_size.size(); ls++){

        int L = L_size[ls];

        cout << "Started Simulation of Lattice " << L << endl;
        
        // SEEDs
        unsigned long long seeds_spins = 0ULL;
        unsigned long long seeds_interactions = 0ULL;
        
        int blocks = (num_lattices*L*L*2 + THREADS -1)/THREADS;

        auto t0 = std::chrono::high_resolution_clock::now();

        // Allocate the wave vectors and copy it to GPU memory
        std::array<float, 2> wave_vector_0 = {0,0};
        float wv = 2.0f*M_PI/L;
        std::array<float, 2> wave_vector_k = {wv,0};

        float *d_wave_vector_0, *d_wave_vector_k;
        cudaMalloc(&d_wave_vector_0, 2 * sizeof(*d_wave_vector_0));
        cudaMalloc(&d_wave_vector_k, 2 * sizeof(*d_wave_vector_k));
        cudaMemcpy(d_wave_vector_0, wave_vector_0.data(), 2*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wave_vector_k, wave_vector_k.data(), 2*sizeof(float), cudaMemcpyHostToDevice);
        
        //Setup interaction lattice on device
        signed char *d_interactions;
        cudaMalloc(&d_interactions, num_lattices*L*L*2*sizeof(*d_interactions));

        // Setup black and white lattice arrays on device
        signed char *lattice_b, *lattice_w;
        cudaMalloc(&lattice_b, num_lattices * L * L/2 * sizeof(*lattice_b));
        cudaMalloc(&lattice_w, num_lattices * L * L/2 * sizeof(*lattice_w));

        // Weighted error
        float *d_error_weight_0, *d_error_weight_k;
        cudaMalloc(&d_error_weight_0, num_lattices*num_iterations_error*sizeof(*d_error_weight_0));
        cudaMalloc(&d_error_weight_k, num_lattices*num_iterations_error*sizeof(*d_error_weight_k));

        // Initialize arrays on the GPU to store results per spin system for energy and sum of B2
        thrust::complex<float> *d_store_sum_0, *d_store_sum_k;
        float *d_store_energy;
        cudaMalloc(&d_store_sum_0, num_lattices*num_iterations_seeds*sizeof(*d_store_sum_0));
        cudaMalloc(&d_store_sum_k, num_lattices*num_iterations_seeds*sizeof(*d_store_sum_k));
        cudaMalloc(&d_store_energy, num_lattices*num_iterations_seeds*sizeof(*d_store_energy));

        // B2 Sum 
        thrust::complex<float> *d_sum;
        cudaMalloc(&d_sum, num_lattices*L*L/2*sizeof(*d_sum));

        // Weighted energies
        float *d_weighted_energies;
        cudaMalloc(&d_weighted_energies, num_lattices*num_iterations_seeds*sizeof(*d_weighted_energies));

        // energy
        float *d_energy;
        cudaMalloc(&d_energy, num_lattices*L*L/2*sizeof(*d_energy));
        
        // Setup cuRAND generator
        curandGenerator_t update_rng;
        curandCreateGenerator(&update_rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        float *randvals;
        cudaMalloc(&randvals, L * L/2 * sizeof(*randvals));

        // Setup cuRAND generator
        curandGenerator_t lattice_rng;
        curandCreateGenerator(&lattice_rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        float *lattice_randvals;
        cudaMalloc(&lattice_randvals, num_lattices * L * L/2 * sizeof(*lattice_randvals));

        // Setup cuRAND generator
        curandGenerator_t interaction_rng;
        curandCreateGenerator(&interaction_rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        float *interaction_randvals;
        cudaMalloc(&interaction_randvals,num_lattices*L*L*2*sizeof(*interaction_randvals));

        // Initialize array for partition function
        float *d_partition_function;
        cudaMalloc(&d_partition_function, num_lattices*sizeof(float));

        for (int e = 0; e < num_iterations_error; e++){
            
            cout << "Error " << e << " of " << num_iterations_error << endl;

            init_interactions_with_seed(d_interactions, seeds_interactions, interaction_rng, interaction_randvals, L, L, num_lattices, d_probs);

            for (int s = 0; s < num_iterations_seeds; s++){
                
                init_spins_with_seed(lattice_b, lattice_w, seeds_spins, lattice_rng, lattice_randvals, L, L, num_lattices);

                curandSetPseudoRandomGeneratorSeed(update_rng, seeds_spins);
                
                //write_lattice(lattice_b, lattice_w, "lattices/lattice_"+std::to_string(e) + std::string("_") + std::to_string(s) + std::string("_"), L, L, num_lattices);

                cudaDeviceSynchronize();

                // Warmup iterations
                //printf("Starting warmup...\n");
                for (int j = 0; j < nwarmup; j++) {
                    update(lattice_b, lattice_w, randvals, update_rng, d_interactions, d_inv_temp, L, L, num_lattices, d_coupling_constant);
                }
                
                cudaDeviceSynchronize();

                for (int j = 0; j < niters; j++) {
                    update(lattice_b, lattice_w, randvals, update_rng, d_interactions, d_inv_temp, L, L, num_lattices, d_coupling_constant);
                    //if (j % 1000 == 0) printf("Completed %d/%d iterations...\n", j+1, niters);
                }
                
                cudaDeviceSynchronize();

                calculate_B2(d_sum, lattice_b, lattice_w, d_store_sum_0, d_wave_vector_0, s, L, L, num_lattices, num_iterations_seeds);
                calculate_B2(d_sum, lattice_b, lattice_w, d_store_sum_k, d_wave_vector_k, s, L, L, num_lattices, num_iterations_seeds);

                calculate_energy(d_energy, lattice_b, lattice_w, d_interactions, d_store_energy, d_coupling_constant, s, L, L, num_lattices, num_iterations_seeds);

                seeds_spins += 1;
            }

            // Take absolute square + exp
            abs_square<<<blocks, THREADS>>>(d_store_sum_0, num_lattices, num_iterations_seeds);
            abs_square<<<blocks, THREADS>>>(d_store_sum_k, num_lattices, num_iterations_seeds);

            exp_beta<<<blocks, THREADS>>>(d_store_energy, d_inv_temp, num_lattices, num_iterations_seeds, L);
            
            for (int l=0; l<num_lattices; l++){
                cub::DeviceReduce::Sum(d_temp, temp_storage, d_store_energy + l*num_iterations_seeds, &d_partition_function[l], num_iterations_seeds);
                cudaMalloc(&d_temp, temp_storage);
                cub::DeviceReduce::Sum(d_temp, temp_storage, d_store_energy + l*num_iterations_seeds, &d_partition_function[l], num_iterations_seeds);
            }
            
            calculate_weighted_energies(d_weighted_energies, d_error_weight_0, d_store_energy, d_store_sum_0, d_partition_function, num_lattices, num_iterations_seeds, num_iterations_error, blocks, e);
            calculate_weighted_energies(d_weighted_energies, d_error_weight_k, d_store_energy, d_store_sum_k, d_partition_function, num_lattices, num_iterations_seeds, num_iterations_error, blocks, e);

            seeds_interactions += 1;
        }

        // Magnetic susceptibility 
        float *d_magnetic_susceptibility_0, *d_magnetic_susceptibility_k;
        cudaMalloc(&d_magnetic_susceptibility_0, num_lattices*sizeof(*d_magnetic_susceptibility_0));
        cudaMalloc(&d_magnetic_susceptibility_k, num_lattices*sizeof(*d_magnetic_susceptibility_k));

        for (int l=0; l < num_lattices; l++){
            // Sum reduction for both
            cub::DeviceReduce::Sum(d_temp, temp_storage, d_error_weight_0 + l*num_iterations_error, &d_magnetic_susceptibility_0[l], num_iterations_error);
            cudaMalloc(&d_temp, temp_storage);
            cub::DeviceReduce::Sum(d_temp, temp_storage, d_error_weight_0 + l*num_iterations_error, &d_magnetic_susceptibility_0[l], num_iterations_error);

            cub::DeviceReduce::Sum(d_temp, temp_storage, d_error_weight_k + l*num_iterations_error, &d_magnetic_susceptibility_k[l], num_iterations_error);
            cudaMalloc(&d_temp, temp_storage);
            cub::DeviceReduce::Sum(d_temp, temp_storage, d_error_weight_k + l*num_iterations_error, &d_magnetic_susceptibility_k[l], num_iterations_error);
        }

        cudaDeviceSynchronize();

        std::vector<float> h_magnetic_susceptibility_0(num_lattices);
        std::vector<float> h_magnetic_susceptibility_k(num_lattices);
        
        cudaMemcpy(h_magnetic_susceptibility_0.data(), d_magnetic_susceptibility_0, num_lattices*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_magnetic_susceptibility_k.data(), d_magnetic_susceptibility_k, num_lattices*sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> psi(num_lattices);
        
        for (int l=0; l < num_lattices; l++){
            psi[l] = (1/(2*sin(M_PI/L))*sqrt(h_magnetic_susceptibility_0[l] / h_magnetic_susceptibility_k[l] - 1))/L;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double duration = (double) std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count();

        printf("Elapsed time for temperature loop min %f \n", duration/60);

        // Write results
        std::ofstream f;
        f.open(results + std::string("/L_") + std::to_string(L) + std::string("_ns_") + std::to_string(num_iterations_seeds) + std::string("_ne_") + std::to_string(num_iterations_error) + std::string("_ni_") + std::to_string(niters) + std::string("_nw_") + std::to_string(nwarmup) + std::string(".txt"));
        if (f.is_open()) {
            for (int i = 0; i < num_lattices; i++) {
                f << psi[i] << " " << 1/inv_temp[i] << "\n";
            }
        }
        f.close();

        cudaFree(d_wave_vector_0);
        cudaFree(d_wave_vector_k);
        cudaFree(d_interactions);
        cudaFree(lattice_b);
        cudaFree(lattice_w);
        cudaFree(d_error_weight_0);
        cudaFree(d_error_weight_k);
        cudaFree(d_store_sum_0);
        cudaFree(d_store_sum_k);
        cudaFree(d_store_energy);
        cudaFree(d_sum);
        cudaFree(d_weighted_energies);
        cudaFree(d_energy);
        cudaFree(randvals);
        cudaFree(lattice_randvals);
        cudaFree(interaction_randvals);
        cudaFree(d_partition_function);
        cudaFree(d_magnetic_susceptibility_0);
        cudaFree(d_magnetic_susceptibility_k);

        curandDestroyGenerator(update_rng);
        curandDestroyGenerator(interaction_rng);
        curandDestroyGenerator(lattice_rng);
    }
}