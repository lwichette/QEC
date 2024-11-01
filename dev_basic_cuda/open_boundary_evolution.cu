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
#include "../header/utils.cuh"
#include "../header/cudamacro.h"

// General problem with multiple header files - methods on higher hierarchy overload the lower once such that I couldnt use my dev methods while header exists on higher hierarchy

using namespace std;

namespace po = boost::program_options;
namespace fs = std::filesystem;

int main(int argc, char **argv){

    float p, start_temp, step;
    int num_iterations_error, num_iterations_seeds, niters, nwarmup, num_lattices, num_reps_temp, normalization_factor;
    std::vector<int> L_size;
    std::string folderName;
    bool up;

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

        std::string result_name = std::string("L_") + std::to_string(L) + std::string("_p_") + std::to_string(p) + std::string("_ns_") + std::to_string(num_iterations_seeds) + std::string("_ne_") + std::to_string(num_iterations_error) + std::string("_ni_") + std::to_string(niters) + std::string("_nw_") + std::to_string(nwarmup) + std::string("_up_") + std::to_string(up) + std::string(".txt");

        if (fs::exists(folderPath + "/" + result_name)){
            cout << "Results already exist" << result_name << std::endl;
            cout << "Continuing with next lattice size" << endl;
            continue;
        }

        cout << "Started Simulation of Lattice " << L << endl;

        // SEEDs
        unsigned long long seeds_spins = 0ULL;
        unsigned long long seeds_interactions = 0ULL;

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
        float *d_store_energy, *d_store_partition_function;
        CHECK_CUDA(cudaMalloc(&d_store_sum_0, num_lattices*sizeof(*d_store_sum_0)));
        CHECK_CUDA(cudaMalloc(&d_store_sum_k, num_lattices*sizeof(*d_store_sum_k)));
        CHECK_CUDA(cudaMalloc(&d_store_energy, num_lattices*sizeof(*d_store_energy)));
        CHECK_CUDA(cudaMalloc(&d_store_partition_function, num_lattices*sizeof(*d_store_partition_function)));

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

        // Setup cuRAND generator
        curandGenerator_t update_rng;
        CHECK_CURAND(curandCreateGenerator(&update_rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));

        float *randvals;
        CHECK_CUDA(cudaMalloc(&randvals, num_lattices * L * L/2 * sizeof(*randvals)));

        // Setup cuRAND generator
        curandGenerator_t lattice_rng;
        CHECK_CURAND(curandCreateGenerator(&lattice_rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));

        float *lattice_randvals;
        CHECK_CUDA(cudaMalloc(&lattice_randvals, num_lattices * L * L/2 * sizeof(*lattice_randvals)));

        // Setup cuRAND generator
        curandGenerator_t interaction_rng;
        CHECK_CURAND(curandCreateGenerator(&interaction_rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));

        float *interaction_randvals;
        CHECK_CUDA(cudaMalloc(&interaction_randvals,num_lattices*L*L*2*sizeof(*interaction_randvals)));

        // copying new result to host.
        std::vector<float> h_store_local_step_result(num_lattices);

        std::ofstream stream;
        stream.open(folderPath + "/" + "energy_evolution_" + std::to_string(L));

         normalization_factor = 0;

        // summation over errors can be parallized right?
        // Change this !!!
        for (int e = 0; e < num_iterations_error; e++){

            cout << "Error " << e << " of " << num_iterations_error << endl;

            init_interactions_with_seed(d_interactions, seeds_interactions, interaction_rng, interaction_randvals, L, L, num_lattices, p, blocks_inter);

            init_spins_with_seed(lattice_b, lattice_w, seeds_spins, lattice_rng, lattice_randvals, L, L, num_lattices, up, blocks_spins);

            // what exactly does this and why after using the spin?
            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(update_rng, seeds_spins));

            for (int j = 0; j < nwarmup; j++) {
                update_ob(lattice_b, lattice_w, randvals, update_rng, d_interactions, d_inv_temp, L, L, num_lattices, d_coupling_constant, blocks_spins, d_energy);

                // following only for print energy time evolution
                combine_cross_subset_hamiltonians_to_whole_lattice_hamiltonian(d_energy, d_store_energy, L, L, num_lattices);
                normalization_factor += 1;
                CHECK_CUDA(cudaMemcpy(h_store_local_step_result.data(), d_store_energy, num_lattices*sizeof(float), cudaMemcpyDeviceToHost));
                if (stream.is_open() && !(normalization_factor%1000)) {
                    for (int i = 0; i < num_lattices; i++) {
                        stream << h_store_local_step_result[i] << " " << normalization_factor << " " << 1/inv_temp[i] << "\n";
                    }
                }

            }

            CHECK_CUDA(cudaDeviceSynchronize());

            for (int j = 0; j < niters; j++){
                update_ob(lattice_b, lattice_w, randvals, update_rng, d_interactions, d_inv_temp, L, L, num_lattices, d_coupling_constant, blocks_spins, d_energy);

                // device sync needed here somewhere?
                // combine cross term hamiltonian values from d_energy array (dim: num_lattices*sublattice_dof) and store in d_store_energy array (dim: num_lattices) to whole lattice energy for each temperature.
                // reduce autocorrelation between snapshots with this if ?
                if(!j%2){
                    combine_cross_subset_hamiltonians_to_whole_lattice_hamiltonian(d_energy, d_store_energy, L, L, num_lattices);

                    // maybe include here storage of energy values to plot over time to see thermalization process for a specific error maybe not all?

                    // Calculate suscetibilitites for each temperature (hence dimension of d_store_sum equals num_lattices)
                    calculate_B2(d_sum, lattice_b, lattice_w, d_store_sum_0, d_wave_vector_0, L, L, num_lattices, blocks_spins);
                    calculate_B2(d_sum, lattice_b, lattice_w, d_store_sum_k, d_wave_vector_k, L, L, num_lattices, blocks_spins);

                    // Take abs squares of previous B2 sums for each temperature and store again to d_store_sum array.
                    abs_square<<<blocks_temperature_parallel, THREADS>>>(d_store_sum_0, num_lattices);
                    abs_square<<<blocks_temperature_parallel, THREADS>>>(d_store_sum_k, num_lattices);

                    // Calculate boltzman factor time lattice dim normalization factor for each temperature.
                    // exp_beta<<<blocks_temperature_parallel, THREADS>>>(d_store_energy, d_inv_temp, num_lattices, L);

                    // Summation over errors and update steps is incrementally executed and stored in the d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_._wave_vector arrays for each temperature.

                    // incremental_summation_of_product_of_magnetization_and_boltzmann_factor<<<blocks_temperature_parallel, THREADS>>>(d_store_energy, d_store_sum_0, d_store_sum_k, num_lattices, d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector, d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector);

                    normalization_factor += 1;

                    // write time evolution of magnetization
                    // CHECK_CUDA(cudaMemcpy(h_store_local_step_result.data(), d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector, num_lattices*sizeof(float), cudaMemcpyDeviceToHost));

                    // if (stream.is_open() && !j%1000) {
                    //     for (int i = 0; i < num_lattices; i++) {
                    //         stream << h_store_local_step_result[i]/(normalization_factor*L*L) << " " << normalization_factor << " " << 1/inv_temp[i] << "\n";
                    //     }
                    // }

                    // write time evolution of energy

                    CHECK_CUDA(cudaMemcpy(h_store_local_step_result.data(), d_store_energy, num_lattices*sizeof(float), cudaMemcpyDeviceToHost));

                    if (stream.is_open() && !(j%1000)) {
                        for (int i = 0; i < num_lattices; i++) {
                            stream << h_store_local_step_result[i] << " " << normalization_factor << " " << 1/inv_temp[i] << "\n";
                        }
                    }

                    // missing normalization factor of number errors times partitition function times number of update steps (this should be computed here incrementally)
                    // though this is not contributing to correlation function
                    // partition function computation. d_store_energy contains for current update config the e^-beta*H value which gets added up incrementally for each temperature in parallel.
                    // incremental_summation_of_partition_function<<<blocks_temperature_parallel, THREADS>>>(d_store_energy, num_lattices, d_store_partition_function);
                }
            }

            CHECK_CUDA(cudaDeviceSynchronize());

        }

        stream.close();

        CHECK_CUDA(cudaDeviceSynchronize());

        // copying incrmental sums to host.
        std::vector<float> h_store_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector(num_lattices);
        std::vector<float> h_store_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector(num_lattices);
        CHECK_CUDA(cudaMemcpy(h_store_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector.data(), d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector, num_lattices*sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_store_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector.data(), d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector, num_lattices*sizeof(float), cudaMemcpyDeviceToHost));


        // for (int i=0; i < num_lattices; i++){
        //     cout << h_store_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector[i] << endl;
        //     cout << h_store_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector[i] << endl;
        //     cout << "Frac" << h_store_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector[i]/h_store_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector[i] - 1 << endl;
        // }

        // std::vector<float> zeta(num_lattices);

        // CHECK_CUDA(cudaDeviceSynchronize());

        // for (int l=0; l < num_lattices; l++){
        //     zeta[l] = (1/(2*sin(M_PI/L))*sqrt(h_store_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector[l]/h_store_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector[l] - 1));
        // }

        // auto t1 = std::chrono::high_resolution_clock::now();
        // double duration = (double) std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count();

        // printf("Elapsed time for temperature loop min %f \n", duration/60);

        // std::ofstream f;
        // f.open(folderPath + "/" + result_name);
        // if (f.is_open()) {
        //     for (int i = 0; i < num_lattices; i++) {
        //         f << zeta[i] << " " << 1/inv_temp[i] << "\n";
        //     }
        // }
        // f.close();

        // cout << normalization_factor << endl;

        // std::ofstream f;
        // f.open(folderPath + "/" + result_name);
        // if (f.is_open()) {
        //     for (int i = 0; i < num_lattices; i++) {
        //         f << h_store_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector[i]/(normalization_factor*L*L) << " " << 1/inv_temp[i] << "\n";
        //     }
        // }
        // f.close();

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
        CHECK_CUDA(cudaFree(d_store_partition_function));
        CHECK_CUDA(cudaFree(d_sum));
        CHECK_CUDA(cudaFree(d_energy));
        CHECK_CUDA(cudaFree(randvals));
        CHECK_CUDA(cudaFree(lattice_randvals));
        CHECK_CUDA(cudaFree(interaction_randvals));
        CHECK_CUDA(cudaFree(d_partition_function));
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

        CHECK_CURAND(curandDestroyGenerator(update_rng));
        CHECK_CURAND(curandDestroyGenerator(interaction_rng));
        CHECK_CURAND(curandDestroyGenerator(lattice_rng));
    }
}
