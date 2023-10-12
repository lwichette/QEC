#include <fstream>
#include <iostream>
#include <time.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>
#include <math.h>
#include <vector>
#include <string>
#include <sys/stat.h>

#include "../header/utils.cuh"
#include "../header/defines.h"

using namespace std;

int main(void){

    // Initialize all possible parameters
    int num_iterations_seeds = 100;
    int num_iterations_error = 100;

    int niters = 1000;
    int nwarmup = 100;

    int nx = 14;
    int ny = 14;

    float prob = 0.085;
    float coupling_constant = 1.0f/2.0f*log((1-prob)/prob);
    float inv_temp = coupling_constant;

    unsigned long long seeds_spins = 0ULL;
    unsigned long long seeds_interactions = 0ULL;

    // Variables used for sum reduction
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // Allocate the wave vectors and copy it to GPU memory
    std::array<float, 2> wave_vector_0 = {0,0};

    float *d_wave_vector_0;
    cudaMalloc(&d_wave_vector_0, 2 * sizeof(*d_wave_vector_0));
    cudaMemcpy(d_wave_vector_0, wave_vector_0.data(), 2*sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (nx*ny*2 + THREADS -1)/THREADS;

    std::array<float, 2> wave_vector_k = {2.0*M_PI/nx,0};

    float *d_wave_vector_k;
    cudaMalloc(&d_wave_vector_k, 2 * sizeof(*d_wave_vector_k));
    cudaMemcpy(d_wave_vector_k, wave_vector_k.data(), 2*sizeof(float), cudaMemcpyHostToDevice);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Weighted error
    float *d_error_weight_0, *d_error_weight_k;
    cudaMalloc(&d_error_weight_0, num_iterations_error*sizeof(*d_error_weight_0));
    cudaMalloc(&d_error_weight_k, num_iterations_error*sizeof(*d_error_weight_k));

    // Loop over different errors
    for (int j=0; j < num_iterations_error; j++){

        // Initialize arrays on the GPU to store results per spin system for energy and sum of B2
        thrust::complex<float> *d_store_sum_0, *d_store_sum_k;
        cudaMalloc(&d_store_sum_0, num_iterations_seeds*sizeof(*d_store_sum_0));
        cudaMalloc(&d_store_sum_k, num_iterations_seeds*sizeof(*d_store_sum_k));

        float *d_store_energy;
        cudaMalloc(&d_store_energy, num_iterations_seeds*sizeof(*d_store_energy));

        //Setup interaction lattice on device
        signed char *d_interactions;
        cudaMalloc(&d_interactions, nx*ny*2*sizeof(*d_interactions));

        init_interactions_with_seed(d_interactions, seeds_interactions, nx, ny, prob);

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

            cudaDeviceSynchronize();

            // Warmup iterations
            //printf("Starting warmup...\n");
            for (int j = 0; j < nwarmup; j++) {
                update(lattice_b, lattice_w, randvals, rng, d_interactions, inv_temp, nx, ny, coupling_constant);
            }

            cudaDeviceSynchronize();

            for (int j = 0; j < niters; j++) {
                update(lattice_b, lattice_w, randvals, rng, d_interactions, inv_temp, nx, ny,coupling_constant);
                //if (j % 1000 == 0) printf("Completed %d/%d iterations...\n", j+1, niters);
            }

            cudaDeviceSynchronize();

            calculate_B2(lattice_b, lattice_w, d_store_sum_0, d_wave_vector_0, i, nx, ny);
            calculate_B2(lattice_b, lattice_w, d_store_sum_k, d_wave_vector_k, i, nx, ny);

            calculate_energy(lattice_b, lattice_w, d_interactions, d_store_energy, coupling_constant, i, nx, ny);

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

        seeds_interactions += 1;

        //write_bonds(d_interactions, "lattice/final_bonds.txt", nx, ny);
        cudaFree(d_store_sum_0);
        cudaFree(d_store_sum_k);
        cudaFree(d_store_energy);
        cudaFree(d_interactions);
        cudaFree(d_partition_function);
    }

    float psi = calc_psi(d_error_weight_0, d_error_weight_k, num_iterations_error, nx);

    printf("%f \n", psi/nx); 

    auto t1 = std::chrono::high_resolution_clock::now();
    double duration = (double) std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count();

    printf("Elapsed time for temperature loop sec %f \n", duration);

    cudaFree(d_error_weight_0);
    cudaFree(d_error_weight_k);
    cudaFree(d_wave_vector_k);
}

