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

__global__ void store_sum(thrust::complex<double> *d_store_sum, double *d_store_mag, const int loc, const int nsteps, const int num_lattices){

    const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (tid > 0) return;

    for (int i=0; i< num_lattices; i++){
        d_store_mag[i*num_lattices*nsteps+loc] = d_store_sum[i].real();
    }
}

int main(int argc, char **argv){

    double p, start_temp, step;
    int num_iterations_error, niters, nwarmup, num_lattices, num_reps_temp, leave_out;
    int seed_adder = 0;
    std::vector<int> L_size;
    std::string folderName;
    bool up, open;
    bool write_lattice = false;
    bool read_lattice = false;

    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help options")
      ("p", po::value<double>(), "probability")
      ("open", po::value<bool>(), "open boundary")
      ("temp", po::value<double>(), "start_temp")
      ("step", po::value<double>(), "step size temperature")
      ("up", po::value<bool>(), "up initialization")
      ("nie", po::value<int>(), "num iterations error")
      ("leave_out", po::value<int>(), "leave_out")
      ("nit", po::value<int>(), "niters updates")
      ("nw", po::value<int>(), "nwarmup updates")
      ("nl", po::value<int>(), "num lattices")
      ("seed_adder", po::value<int>(), "seed adder")
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
        p = vm["p"].as<double>();
    }
    if (vm.count("open")) {
        open = vm["open"].as<bool>();
    }
    if (vm.count("temp")) {
        start_temp = vm["temp"].as<double>();
    }
    if (vm.count("step")) {
        step = vm["step"].as<double>();
    }
    if (vm.count("up")) {
        up = vm["up"].as<bool>();
    }
    if (vm.count("nie")) {
        num_iterations_error = vm["nie"].as<int>();
    }
    if (vm.count("seed_adder")) {
        seed_adder = vm["seed_adder"].as<int>();
    }
    if (vm.count("nit")) {
        niters = vm["nit"].as<int>();
    }
    if (vm.count("leave_out")) {
        leave_out = vm["leave_out"].as<int>();
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
    
    if (open){
        folderPath = "results/open_boundary/" + folderName;
    }
    else{
        folderPath = "results/periodic_boundary/" + folderName; 
    }

    if (!fs::exists(folderPath)) {
        if (fs::create_directory(folderPath)) {
            fs::create_directory(folderPath + "/lattices");
            std::cout << "Directory created successfully." << std::endl;
        } else {
            std::cout << "Failed to create directory." << std::endl;
        }
    } else {
        if (!fs::exists(folderPath + "/lattices")){
            if (fs::create_directory(folderPath+"/lattices")){
                std::cout << "Lattice folder created successfully" << std::endl;
            }
	    }
        else {
            std::cout << "Failed to create lattice directory" << std::endl;
        }
    }

    // Function Maps
    std::map<bool, std::function<void(
        signed char*, signed char*, float*, curandGenerator_t, signed char*, double*,
        long long, long long, const int, const int, double*
    )>> updateMap;

    updateMap[false] = &update;
    updateMap[true] = &update_ob;
    
    std::vector<double> inv_temp;

    for (int j=0; j < num_reps_temp; j++){
        for (int i=0; i < num_lattices; i++){
            inv_temp.push_back(1/(start_temp+i*step));
        }
    }

    num_lattices = num_lattices * num_reps_temp;

    double *d_inv_temp;
    CHECK_CUDA(cudaMalloc(&d_inv_temp, num_lattices*sizeof(*d_inv_temp)));
    CHECK_CUDA(cudaMemcpy(d_inv_temp, inv_temp.data(), num_lattices*sizeof(*d_inv_temp), cudaMemcpyHostToDevice));

    for(int ls = 0; ls < L_size.size(); ls++){

        int L = L_size[ls];

        std::string result_name = std::string("L_") + std::to_string(L) + std::string("_p_") + std::to_string(p) + std::string("_lo_") + std::to_string(leave_out) + std::string("_ne_") + std::to_string(num_iterations_error) + std::string("_ni_") + std::to_string(niters) + std::string("_nw_") + std::to_string(nwarmup) + std::string("_up_") + std::to_string(up) + std::string("_temp_") + std::to_string(start_temp) + std::string("_step_") + std::to_string(step) + std::string("_nl_") + std::to_string(num_lattices/num_reps_temp) + std::string("_nrt_") + std::to_string(num_reps_temp) + std::string("_read_lattice_") + std::to_string(read_lattice) + std::string("_write_lattice_") + std::to_string(write_lattice) + std::string("_seed_adder_") + std::to_string(seed_adder) + std::string(".txt");

        // if (fs::exists(folderPath + "/" + result_name)){
        //     cout << "Results already exist" << result_name << std::endl;
        //     cout << "Continuing with next lattice size" << endl;
        //     continue;
        // }

        cout << "Started Simulation of Lattice " << L << endl;

        // SEEDs
        unsigned long long seed = 42ULL;

        int blocks_inter = (num_lattices*L*L*2 + THREADS - 1)/THREADS;
        int blocks_spins = (L*L/2*num_lattices + THREADS - 1)/THREADS;
        int blocks_temperature_parallel = (num_lattices + THREADS - 1)/THREADS;

        auto t0 = std::chrono::high_resolution_clock::now();

        // Allocate the wave vectors and copy it to GPU memory
        std::array<double, 2> wave_vector_0 = {0,0};
        double wv = 2.0f*M_PI/L;
        std::array<double, 2> wave_vector_k = {wv,0};

        double *d_wave_vector_0, *d_wave_vector_k;
        CHECK_CUDA(cudaMalloc(&d_wave_vector_0, 2 * sizeof(*d_wave_vector_0)));
        CHECK_CUDA(cudaMalloc(&d_wave_vector_k, 2 * sizeof(*d_wave_vector_k)));
        CHECK_CUDA(cudaMemcpy(d_wave_vector_0, wave_vector_0.data(), 2*sizeof(*d_wave_vector_0), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_wave_vector_k, wave_vector_k.data(), 2*sizeof(*d_wave_vector_k), cudaMemcpyHostToDevice));

        //Setup interaction lattice on device
        signed char *d_interactions;
        CHECK_CUDA(cudaMalloc(&d_interactions, num_lattices*L*L*2*sizeof(*d_interactions)));

        // Setup black and white lattice arrays on device
        signed char *lattice_b, *lattice_w;
        CHECK_CUDA(cudaMalloc(&lattice_b, num_lattices * L * L/2 * sizeof(*lattice_b)));
        CHECK_CUDA(cudaMalloc(&lattice_w, num_lattices * L * L/2 * sizeof(*lattice_w)));

        // Initialize arrays on the GPU to store results per spin system for energy and sum of B2
        thrust::complex<double> *d_store_sum_0, *d_store_sum_k;
        double *d_store_energy;
        CHECK_CUDA(cudaMalloc(&d_store_sum_0, num_lattices*sizeof(*d_store_sum_0)));
        CHECK_CUDA(cudaMalloc(&d_store_sum_k, num_lattices*sizeof(*d_store_sum_k)));
        CHECK_CUDA(cudaMalloc(&d_store_energy, num_lattices*sizeof(*d_store_energy)));

        double* d_store_mag;
        CHECK_CUDA(cudaMalloc(&d_store_mag, niters*sizeof(*d_store_mag)));

        // Initialize array on the GPU to store incremental sums of the magnetization sums time boltzmann factors over update steps.
        double *d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector, *d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector;
        CHECK_CUDA(cudaMalloc(&d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector, num_lattices*sizeof(*d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector)));
        CHECK_CUDA(cudaMalloc(&d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector, num_lattices*sizeof(*d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector)));

        // B2 Sum
        thrust::complex<double> *d_sum;
        CHECK_CUDA(cudaMalloc(&d_sum, num_lattices*L*L/2*sizeof(*d_sum)));

        // energy
        double *d_energy;
        CHECK_CUDA(cudaMalloc(&d_energy, num_lattices*L*L/2*sizeof(*d_energy)));

        // Setup cuRAND generators
        curandGenerator_t rng;
        CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed+1));

        // Setup cuRAND generator
        curandGenerator_t rng_errors;
        CHECK_CURAND(curandCreateGenerator(&rng_errors, CURAND_RNG_PSEUDO_PHILOX4_32_10));
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng_errors, seed));

        // Setup cuRAND generator
        curandGenerator_t update_rng;
        CHECK_CURAND(curandCreateGenerator(&update_rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(update_rng, seed + 2 + seed_adder));

        float *randvals;
        CHECK_CUDA(cudaMalloc(&randvals, num_lattices * L * L/2 * sizeof(*randvals)));

        float *lattice_randvals;
        CHECK_CUDA(cudaMalloc(&lattice_randvals, num_lattices * L * L/2 * sizeof(*lattice_randvals)));

        float *interaction_randvals;
        CHECK_CUDA(cudaMalloc(&interaction_randvals, num_lattices * L * L * 2 *sizeof(*interaction_randvals)));

        for (int e = 0; e < num_iterations_error; e++){

            // the directory lattices inside the folderPath must already exist, there is no mkdir included here!
            std::string lattice_b_file_name = folderPath + "/lattices/lattice_b_e" + std::to_string(e) + std::string("_L") + std::to_string(L) + std::string("_p") + std::to_string(p) + std::string("_num_lattices") + std::to_string(num_lattices) + std::string("_start_temp") + std::to_string(start_temp) + std::string("_step") + std::to_string(step) + std::string(".txt");
            std::string lattice_w_file_name = folderPath + "/lattices/lattice_w_e" + std::to_string(e) + std::string("_L") + std::to_string(L) + std::string("_p") + std::to_string(p) + std::string("_num_lattices") + std::to_string(num_lattices) + std::string("_start_temp") + std::to_string(start_temp) + std::string("_step") + std::to_string(step) + std::string(".txt");

            cout << "Error " << e << " of " << num_iterations_error << endl;

            auto start = std::chrono::high_resolution_clock::now();

            init_interactions_with_seed(d_interactions, rng_errors, interaction_randvals, L, L, num_lattices, p, blocks_inter);

            initialize_spins(lattice_b, lattice_w, rng, lattice_randvals, L, L, num_lattices, up, blocks_spins, read_lattice, lattice_b_file_name, lattice_w_file_name);
            
            for (int j = 0; j < nwarmup; j++) {
                updateMap[open](lattice_b, lattice_w, randvals, update_rng, d_interactions, d_inv_temp, L, L, num_lattices, blocks_spins, d_energy);
            }
            
            auto end = std::chrono::high_resolution_clock::now();

            // Calculate the duration of the operation
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            // Print the duration
            std::cout << "Operation took " << duration.count() << " milliseconds." << std::endl;

            CHECK_CUDA(cudaDeviceSynchronize());

            for(int j = 0; j < niters*leave_out; j++){
                
                updateMap[open](lattice_b, lattice_w, randvals, update_rng, d_interactions, d_inv_temp, L, L, num_lattices, blocks_spins, d_energy);

                if(j%leave_out == 0){
                    
                    //write_lattice_to_disc(lattice_b, lattice_w, "test_lattice_" + std::to_string(j/leave_out), L, L, num_lattices);

                    calculate_B2(d_sum, lattice_b, lattice_w, d_store_sum_0, d_wave_vector_0, L, L, num_lattices, blocks_spins);
                    calculate_B2(d_sum, lattice_b, lattice_w, d_store_sum_k, d_wave_vector_k, L, L, num_lattices, blocks_spins);

                    store_sum<<<1,1>>>(d_store_sum_0, d_store_mag, j/leave_out, niters, num_lattices);
                    
                    abs_square<<<blocks_temperature_parallel, THREADS>>>(d_store_sum_0, num_lattices);
                    abs_square<<<blocks_temperature_parallel, THREADS>>>(d_store_sum_k, num_lattices);

                    incrementalSumMagnetization<<<blocks_temperature_parallel, THREADS>>>(d_store_sum_0, d_store_sum_k, num_lattices, d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector, d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector);
                }
            }

            std::vector<double> h_store_mag(num_lattices*niters);
            CHECK_CUDA(cudaMemcpy(h_store_mag.data(), d_store_mag, num_lattices*niters*sizeof(double), cudaMemcpyDeviceToHost));

            std::ofstream m;
            m.open("mag_nit_"+std::to_string(niters)+"_nw_" + std::to_string(nwarmup) + "_lo_" + std::to_string(leave_out) + "_t_" + std::to_string(start_temp) + "_p_" +std::to_string(p) + "_l_" + std::to_string(L) +".txt");
            if (m.is_open()) {
                for (int i = 0; i < num_lattices*niters; i++) {
                    m << h_store_mag[i]  << "\n";
                }
            }
            m.close();

            CHECK_CUDA(cudaDeviceSynchronize());

            if(write_lattice){
                write_updated_lattices(lattice_b, lattice_w, L, L, num_lattices, lattice_b_file_name, lattice_w_file_name);
            }
        }

        // copying new result to host.
        std::vector<double> h_store_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector(num_lattices);
        std::vector<double> h_store_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector(num_lattices);
        CHECK_CUDA(cudaMemcpy(h_store_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector.data(), d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_0_wave_vector, num_lattices*sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_store_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector.data(), d_store_incremental_summation_of_product_of_magnetization_and_boltzmann_factor_k_wave_vector, num_lattices*sizeof(double), cudaMemcpyDeviceToHost));

        std::vector<double> zeta(num_lattices);

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
                f << 1/inv_temp[i] << " " << zeta[i]  << "\n";
            }
        }
        f.close();

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
