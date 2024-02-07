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

#include "./header/defines.h"
#include "./header/utils.cuh"
#include "./header/cudamacro.h"

// General problem with multiple header files - methods on higher hierarchy overload the lower once such that I couldnt use my dev methods while header exists on higher hierarchy

using namespace std;

namespace po = boost::program_options;
namespace fs = std::filesystem;

int main(int argc, char **argv){

    float p, start_temp, step;
    int num_iterations_error, niters, nwarmup, num_lattices, num_reps_temp, normalization_factor;
    std::vector<int> L_size;
    std::string folderName;
    bool up, write_lattice, read_lattice;

    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help options")
      ("p", po::value<float>(), "probability")
      ("temp", po::value<float>(), "start_temp")
      ("step", po::value<float>(), "step size temperature")
      ("up", po::value<bool>(), "step size temperature")
      ("nie", po::value<int>(), "num iterations error")
      ("nis", po::value<int>(), "num iterations seeds")
      ("nit", po::value<int>(), "niters updates")
      ("nw", po::value<int>(), "nwarmup updates")
      ("nl", po::value<int>(), "num lattices")
      ("nrt", po::value<int>(), "num reps temp")
      ("L", po::value<std::vector<int>>()->multitoken(), "Lattice")
      ("folder", po::value<std::string>(), "folder")
      ("write_lattice", po::value<bool>(), "signifier if updated lattices shall be written to file")
      ("read_lattice", po::value<bool>(), "signifier if lattice shall be initialized from file")
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
    if (vm.count("write_lattice")) {
        write_lattice = vm["write_lattice"].as<bool>();
    }
    if (vm.count("read_lattice")) {
        read_lattice = vm["read_lattice"].as<bool>();
    }

    std::string folderPath = "results/" + folderName;

    if (!fs::exists(folderPath)) {
        if (fs::create_directory(folderPath)) {
            std::cout << "Directory created successfully." << std::endl;
        } else {
            std::cout << "Failed to create directory." << std::endl;
        }
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

    // This knots my thoughts - num_lattices is amount of different temperatures and reps temp is the multiplicity of each temp.
    num_lattices = num_lattices * num_reps_temp;

    float *d_inv_temp, *d_coupling_constant;
    CHECK_CUDA(cudaMalloc(&d_inv_temp, num_lattices*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_coupling_constant, num_lattices*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_inv_temp, inv_temp.data(), num_lattices*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_coupling_constant, coupling_constant.data(), num_lattices*sizeof(float), cudaMemcpyHostToDevice));

    for(int ls = 0; ls < L_size.size(); ls++){

        int L = L_size[ls];

        normalization_factor = 0;

        std::string result_name = std::string("L_") + std::to_string(L) + std::string("_p_") + std::to_string(p) + std::string("_ne_") + std::to_string(num_iterations_error) + std::string("_ni_") + std::to_string(niters) + std::string("_nw_") + std::to_string(nwarmup) + std::string("_up_") + std::to_string(up) + std::string("_temp_") + std::to_string(start_temp) + std::string("_step_") + std::to_string(step) + std::string("_nl_") + std::to_string(num_lattices/num_reps_temp) + std::string("_nrt_") + std::to_string(num_reps_temp) + std::string("_read_lattice_") + std::to_string(read_lattice) + std::string("_write_lattice_") + std::to_string(write_lattice)  + std::string(".txt");

        // if (fs::exists(folderPath + "/" + result_name)){
        //     cout << "Results already exist" << result_name << std::endl;
        //     cout << "Continuing with next lattice size" << endl;
        //     continue;
        // }

        cout << "Started Simulation of Lattice " << L << endl;

        // SEEDs
        unsigned long long seeds_spins = 42ULL;

        int blocks_inter = (num_lattices*L*L*2 + THREADS - 1)/THREADS;
        int blocks_spins = (L*L/2*num_lattices + THREADS - 1)/THREADS;
        int blocks_temperature_parallel = (num_lattices + THREADS - 1)/THREADS;

        auto t0 = std::chrono::high_resolution_clock::now();

        // Allocate the wave vectors and copy it to GPU memory
        std::array<float, 2> wave_vector_0 = {0,0};
        float wv = 2.0f*M_PI/L;
        std::array<float, 2> wave_vector_k = {wv,0};

        float *d_wave_vector_0, *d_wave_vector_k;
        CHECK_CUDA(cudaMalloc(&d_wave_vector_0, 2 * sizeof(*d_wave_vector_0)));
        CHECK_CUDA(cudaMalloc(&d_wave_vector_k, 2 * sizeof(*d_wave_vector_k)));
        CHECK_CUDA(cudaMemcpy(d_wave_vector_0, wave_vector_0.data(), 2*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_wave_vector_k, wave_vector_k.data(), 2*sizeof(float), cudaMemcpyHostToDevice));

        //Setup interaction lattice on device
        signed char *d_interactions;
        CHECK_CUDA(cudaMalloc(&d_interactions, num_lattices*L*L*2*sizeof(*d_interactions)));

        // Setup black and white lattice arrays on device
        signed char *lattice_b, *lattice_w;
        CHECK_CUDA(cudaMalloc(&lattice_b, num_lattices * L * L/2 * sizeof(*lattice_b)));
        CHECK_CUDA(cudaMalloc(&lattice_w, num_lattices * L * L/2 * sizeof(*lattice_w)));

        // Initialize arrays on the GPU to store results per spin system for energy and sum of B2
        thrust::complex<float> *d_store_sum_0, *d_store_sum_k;
        float *d_store_energy;
        CHECK_CUDA(cudaMalloc(&d_store_sum_0, num_lattices*sizeof(*d_store_sum_0)));
        CHECK_CUDA(cudaMalloc(&d_store_sum_k, num_lattices*sizeof(*d_store_sum_k)));
        CHECK_CUDA(cudaMalloc(&d_store_energy, num_lattices*sizeof(*d_store_energy)));

        // Initialize array on the GPU to store incremental sums of the magnetization sums time boltzmann factors over update steps.
        float *d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector, *d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector;
        CHECK_CUDA(cudaMalloc(&d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector, num_lattices*sizeof(*d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector)));
        CHECK_CUDA(cudaMalloc(&d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector, num_lattices*sizeof(*d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector)));

        // B2 Sum
        thrust::complex<float> *d_sum;
        CHECK_CUDA(cudaMalloc(&d_sum, num_lattices*L*L/2*sizeof(thrust::complex<float>)));

        // energy
        float *d_energy;
        CHECK_CUDA(cudaMalloc(&d_energy, num_lattices*L*L/2*sizeof(*d_energy)));

        // Setup cuRAND generators
        curandGenerator_t rng;
        CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seeds_spins));

        curandGenerator_t rng_errors;
        CHECK_CURAND(curandCreateGenerator(&rng_errors, CURAND_RNG_PSEUDO_PHILOX4_32_10));
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng_errors, seeds_spins));

        float *randvals;
        CHECK_CUDA(cudaMalloc(&randvals, num_lattices * L * L/2 * sizeof(*randvals)));

        float *lattice_randvals;
        CHECK_CUDA(cudaMalloc(&lattice_randvals, num_lattices * L * L/2 * sizeof(*lattice_randvals)));

        float *interaction_randvals;
        CHECK_CUDA(cudaMalloc(&interaction_randvals,num_lattices*L*L*2*sizeof(*interaction_randvals)));

        for (int e = 0; e < num_iterations_error; e++){

            // the directory lattices inside the folderPath must already exist, there is no mkdir included here!
            std::string lattice_b_file_name = folderPath + "/lattices/lattice_b_e" + std::to_string(e) + std::string("_L") + std::to_string(L) + std::string("_p") + std::to_string(p) + std::string("_num_lattices") + std::to_string(num_lattices) + std::string("_start_temp") + std::to_string(start_temp) + std::string("_step") + std::to_string(step) + std::string(".txt");
            std::string lattice_w_file_name = folderPath + "/lattices/lattice_w_e" + std::to_string(e) + std::string("_L") + std::to_string(L) + std::string("_p") + std::to_string(p) + std::string("_num_lattices") + std::to_string(num_lattices) + std::string("_start_temp") + std::to_string(start_temp) + std::string("_step") + std::to_string(step) + std::string(".txt");

            cout << "Error " << e << " of " << num_iterations_error << endl;

            init_interactions_with_seed(d_interactions, rng_errors, interaction_randvals, L, L, num_lattices, p, blocks_inter);

            initialize_spins(lattice_b, lattice_w, rng, lattice_randvals, L, L, num_lattices, up, blocks_spins, read_lattice, lattice_b_file_name, lattice_w_file_name);

            for (int j = 0; j < nwarmup; j++) {
                update_ob(lattice_b, lattice_w, randvals, rng, d_interactions, d_inv_temp, L, L, num_lattices, d_coupling_constant, blocks_spins, d_energy);
            }

            CHECK_CUDA(cudaDeviceSynchronize());

            for(int j = 0; j < niters; j++){
                update_ob(lattice_b, lattice_w, randvals, rng, d_interactions, d_inv_temp, L, L, num_lattices, d_coupling_constant, blocks_spins, d_energy);

                // combine cross term hamiltonian values from d_energy array (dim: num_lattices*sublattice_dof) and store in d_store_energy array (dim: num_lattices) to whole lattice energy for each temperature.
                // reduce autocorrelation between snapshots with this if ?
                if(j%2){
                    combine_cross_subset_hamiltonians_to_whole_lattice_hamiltonian(d_energy, d_store_energy, L, L, num_lattices);

                    // Calculate suscetibilitites for each temperature (hence dimension of d_store_sum equals num_lattices)
                    calculate_B2(d_sum, lattice_b, lattice_w, d_store_sum_0, d_wave_vector_0, L, L, num_lattices, blocks_spins);
                    calculate_B2(d_sum, lattice_b, lattice_w, d_store_sum_k, d_wave_vector_k, L, L, num_lattices, blocks_spins);


                    // Take abs squares of previous B2 sums for each temperature and store again to d_store_sum array.
                    abs_square<<<blocks_temperature_parallel, THREADS>>>(d_store_sum_0, num_lattices);
                    abs_square<<<blocks_temperature_parallel, THREADS>>>(d_store_sum_k, num_lattices);

                    // Summation over errors and update steps is incrementally executed and stored in the d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_._wave_vector arrays for each temperature.
                    incremental_summation_of_product_of_magnetization_and_boltzmann_factor<<<blocks_temperature_parallel, THREADS>>>(d_store_energy, d_store_sum_0, d_store_sum_k, num_lattices, d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector, d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector);

                    normalization_factor += 1;
                }
            }

            CHECK_CUDA(cudaDeviceSynchronize());

            if(write_lattice){
                // Write last results from updates to txt format
                write_updated_lattices(lattice_b, lattice_w, L, L, num_lattices, lattice_b_file_name, lattice_w_file_name);
            }

        }

        // copying new result to host.
        std::vector<float> h_store_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector(num_lattices);
        std::vector<float> h_store_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector(num_lattices);
        CHECK_CUDA(cudaMemcpy(h_store_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector.data(), d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector, num_lattices*sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_store_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector.data(), d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector, num_lattices*sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> zeta(num_lattices);

        for (int l=0; l < num_lattices; l++){
            zeta[l] = 1/(2*sin(M_PI/L))*sqrt(h_store_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector[l]/h_store_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector[l] - 1);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double duration = (double) std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count();

        printf("Elapsed time for temperature loop min %f \n", duration/60);

        std::ofstream f;
        f.open(folderPath + "/" + result_name);
        if (f.is_open()) {
            for (int i = 0; i < num_lattices; i++) {
                f << zeta[i] << " " << 1/inv_temp[i] << "\n";
            }
        }
        f.close();

        cout << normalization_factor << endl;

        CHECK_CUDA(cudaFree(d_wave_vector_0));
        CHECK_CUDA(cudaFree(d_wave_vector_k));
        CHECK_CUDA(cudaFree(d_interactions));
        CHECK_CUDA(cudaFree(lattice_b));
        CHECK_CUDA(cudaFree(lattice_w));
        CHECK_CUDA(cudaFree(d_store_sum_0));
        CHECK_CUDA(cudaFree(d_store_sum_k));
        CHECK_CUDA(cudaFree(d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector));
        CHECK_CUDA(cudaFree(d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector));
        CHECK_CUDA(cudaFree(d_store_energy));
        CHECK_CUDA(cudaFree(d_sum));
        CHECK_CUDA(cudaFree(d_energy));
        CHECK_CUDA(cudaFree(randvals));
        CHECK_CUDA(cudaFree(lattice_randvals));
        CHECK_CUDA(cudaFree(interaction_randvals));
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

        CHECK_CURAND(curandDestroyGenerator(rng));
    }
}
