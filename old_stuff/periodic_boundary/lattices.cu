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
#include <filesystem>
#include <boost/program_options.hpp>

#include "../header/defines.h"
#include "../header/utils_big.cuh"

using namespace std;

namespace po = boost::program_options;

int main(int argc, char **argv){

    float p, start_temp, step;
    int num_iterations_error, num_iterations_seeds, niters, nwarmup, num_lattices, num_reps_temp;
    vector<int> L_size;
    std::string folderName;
    bool up;

    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help options")
      ("p", po::value<float>(), "probability")
      ("temp", po::value<float>(), "start_temp")
      ("step", po::value<float>(), "step size temperature")
      ("up", po::value<bool>(), "spins up or random")
      ("nie", po::value<int>(), "num iterations error")
      ("nis", po::value<int>(), "num iterations seeds")
      ("nit", po::value<int>(), "niters updates")
      ("nw", po::value<int>(), "nwarmup updates")
      ("nl", po::value<int>(), "num lattices")
      ("nrt", po::value<int>(), "num reps temp")
      ("L", po::value<std::vector<int>>()->multitoken(), "Lattice")
      ("folder", po::value<std::string>(), "folder")
    ;
  
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    
  
    if (vm.count("p")) {
        p = vm["p"].as<float>();
    }
    if (vm.count("temp")) {
        start_temp = vm["temp"].as<float>();
    }
    if (vm.count("step")) {
        step = vm["step"].as<float>();
    }
    if (vm.count("up")) {
        up = vm["up"].as<bool>();
    }
    if (vm.count("nie")) {
        num_iterations_error = vm["nie"].as<int>();
    }
    if (vm.count("nis")) {
        num_iterations_seeds = vm["nis"].as<int>();
    }
    if (vm.count("nit")) {
        niters = vm["nit"].as<int>();
    }
    if (vm.count("nw")) {
        nwarmup = vm["nw"].as<int>();
    }
    if (vm.count("nl")) {
        num_lattices = vm["nl"].as<int>();
    }
    if (vm.count("nrt")) {
        num_reps_temp = vm["nrt"].as<int>();
    }
    if (vm.count("L")){
      L_size = vm["L"].as<vector<int>>();
    }
    if (vm.count("folder")) {
        folderName = vm["folder"].as<std::string>();
    }

    if (std::filesystem::create_directory("results/" + folderName)) {
        std::cout << "Folder created successfully: " << folderName << std::endl;
    } else {
        std::cerr << "Failed to create folder: " << folderName << std::endl;
        return 0;
    }

    std::vector<float> inv_temp;
    std::vector<float> coupling_constant;
    float run_temp;

    for (int j=0; j < num_reps_temp; j++){
        for (int i=0; i < num_lattices; i++){
            run_temp = start_temp+i*step;
            inv_temp.push_back(1/run_temp);
            coupling_constant.push_back(1/run_temp);
        }
    }

    num_lattices = num_lattices * num_reps_temp;

    float *d_inv_temp, *d_coupling_constant;
    cudaMalloc(&d_inv_temp, num_lattices*sizeof(float));
    cudaMalloc(&d_coupling_constant, num_lattices*sizeof(float));
    cudaMemcpy(d_inv_temp, inv_temp.data(), num_lattices*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coupling_constant, coupling_constant.data(), num_lattices*sizeof(float), cudaMemcpyHostToDevice);  

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
        cudaMalloc(&randvals, num_lattices * L * L/2 * sizeof(*randvals));

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
            
            init_interactions_with_seed(d_interactions, seeds_interactions, interaction_rng, interaction_randvals, L, L, num_lattices, p);

            for (int s = 0; s < num_iterations_seeds; s++){
                
                if (up){
                    // Initialize spins up
                    init_spins_up<<<blocks,THREADS>>>(lattice_b, L, L/2, num_lattices);
                    init_spins_up<<<blocks,THREADS>>>(lattice_w, L, L/2, num_lattices);
                }
                else{
                    init_spins_with_seed(lattice_b, lattice_w, seeds_spins, lattice_rng, lattice_randvals, L, L, num_lattices);
                }

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
                if(temp_storage_nis == 0){
                    cub::DeviceReduce::Sum(d_temp_nis, temp_storage_nis, d_store_energy + l*num_iterations_seeds, &d_partition_function[l], num_iterations_seeds);
                    cudaMalloc(&d_temp_nis, temp_storage_nis);
                }
                
                cub::DeviceReduce::Sum(d_temp_nis, temp_storage_nis, d_store_energy + l*num_iterations_seeds, &d_partition_function[l], num_iterations_seeds);
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
            if (temp_storage_nie==0){
                cub::DeviceReduce::Sum(d_temp_nie, temp_storage_nie, d_error_weight_0 + l*num_iterations_error, &d_magnetic_susceptibility_0[l], num_iterations_error);
                cudaMalloc(&d_temp_nie, temp_storage_nie);
            }

            cub::DeviceReduce::Sum(d_temp_nie, temp_storage_nie, d_error_weight_0 + l*num_iterations_error, &d_magnetic_susceptibility_0[l], num_iterations_error);
            cub::DeviceReduce::Sum(d_temp_nie, temp_storage_nie, d_error_weight_k + l*num_iterations_error, &d_magnetic_susceptibility_k[l], num_iterations_error);
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
        f.open("results/" + folderName + std::string("/L_") + std::to_string(L) + std::string("_p_") + std::to_string(p) + std::string("_ns_") + std::to_string(num_iterations_seeds) + std::string("_ne_") + std::to_string(num_iterations_error) + std::string("_ni_") + std::to_string(niters) + std::string("_nw_") + std::to_string(nwarmup) + std::string(".txt"));
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

        CHECK_CUDA(cudaFree(d_temp_nie));
        CHECK_CUDA(cudaFree(d_temp_nis));
        CHECK_CUDA(cudaFree(d_temp_nx));
        CHECK_CUDA(cudaFree(d_temp_nx_thrust));

        d_temp_nie = NULL;
        d_temp_nis = NULL;
        d_temp_nx = NULL;
        d_temp_nx_thrust = NULL;
        
        temp_storage_nie = 0;
        temp_storage_nis = 0;
        temp_storage_nx = 0;
        temp_storage_nx_thrust = 0;
        
        curandDestroyGenerator(update_rng);
        curandDestroyGenerator(interaction_rng);
        curandDestroyGenerator(lattice_rng);
    }   
}
