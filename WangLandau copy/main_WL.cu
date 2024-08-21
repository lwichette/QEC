#include "./header/cudamacro.h"
#include "./header/wlutils.cuh"
#include <thrust/extrema.h> //addition needed for my (Linnea's) version of thrust -- comment out if this causes issues

/*
To Do:
    - blockID to blockIdx.x to save storage?
    - d_indices array not needed, could in theory use shared memory in each fisher yates call
    - init flag for found new energy with seperate kernel and only update in wang landau inverted to current setting
    - maybe implement runtime balanced subdivision as in https://www.osti.gov/servlets/purl/1567362
    - Add sort to lattice read function
    - update histogram not working 
*/

__global__ void check_energy(int *d_energy, int *d_check_energy, int num_walker_total, int iterator){
    long long tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= num_walker_total) return;

    if (d_energy[tid] != d_check_energy[tid]){
        printf("Energy in blockId %d and walkerId %d is mismatching at iterator %d with energywl %d and energycalc %d\n", blockIdx.x, threadIdx.x, iterator, d_energy[tid], d_check_energy[tid]);
        assert(0);
    }

    return;
}

int main(int argc, char **argv){

    // Get the device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming device 0
    int max_threads_per_block = prop.maxThreadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now(); 

    Options options;
    parse_args(argc, argv, &options);

    const int num_walker_total = options.num_intervals * options.walker_per_interval;

    char *histogram_file = constructFilePath(options.prob_interactions, options.X, options.Y, options.seed_histogram, "histogram", options.logical_error_type, options.boundary_type);

    // Energy spectrum from pre_run
    std::vector<int> h_expected_energy_spectrum;
    if (read_histogram(histogram_file, h_expected_energy_spectrum, &options.E_min, &options.E_max) != 0){
        fprintf(stderr, "Error reading histogram file.\n");
        return 1;
    }
    
    const int len_energy_spectrum = h_expected_energy_spectrum.size();

    // Get interval information
    IntervalResult interval_result = generate_intervals(options.E_min, options.E_max, options.num_intervals, options.walker_per_interval, options.overlap_decimal);
    
    // Start end energies of the intervals 
    int *d_start, *d_end; 
    CHECK_CUDA(cudaMalloc(&d_start, options.num_intervals * sizeof(*d_start))); 
    CHECK_CUDA(cudaMalloc(&d_end, options.num_intervals * sizeof(*d_end)));
    CHECK_CUDA(cudaMemcpy(d_start, interval_result.h_start.data(), options.num_intervals * sizeof(*d_start), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_end, interval_result.h_end.data(), options.num_intervals * sizeof(*d_start), cudaMemcpyHostToDevice));

    // Histogramm and G array
    unsigned long long *d_H;
    CHECK_CUDA(cudaMalloc(&d_H, interval_result.len_histogram_over_all_walkers * sizeof(*d_H)));
    CHECK_CUDA(cudaMemset(d_H, 0, interval_result.len_histogram_over_all_walkers * sizeof(*d_H)));

    double *d_logG;
    CHECK_CUDA(cudaMalloc(&d_logG, interval_result.len_histogram_over_all_walkers * sizeof(*d_logG)));
    CHECK_CUDA(cudaMemset(d_logG, 0, interval_result.len_histogram_over_all_walkers * sizeof(*d_logG)));

    int size_shared_log_G = (options.num_intervals-1)*interval_result.len_interval + (interval_result.h_end[options.num_intervals-1] - interval_result.h_start[options.num_intervals-1] + 1);
    
    double *d_shared_logG;
    CHECK_CUDA(cudaMalloc(&d_shared_logG, size_shared_log_G*sizeof(*d_shared_logG)));
    CHECK_CUDA(cudaMemset(d_shared_logG, 0, size_shared_log_G*sizeof(*d_shared_logG)));
    
    long long *d_offset_shared_logG;
    CHECK_CUDA(cudaMalloc(&d_offset_shared_logG, options.num_intervals*sizeof(*d_offset_shared_logG)));
    CHECK_CUDA(cudaMemset(d_offset_shared_logG, 0, options.num_intervals*sizeof(*d_offset_shared_logG)));

    // Offset histograms, lattice, seed_iterator
    int *d_offset_histogramm, *d_offset_lattice;
    unsigned long long *d_offset_iter;
    CHECK_CUDA(cudaMalloc(&d_offset_histogramm, num_walker_total * sizeof(*d_offset_histogramm)));
    CHECK_CUDA(cudaMalloc(&d_offset_lattice, num_walker_total * sizeof(*d_offset_lattice)));
    CHECK_CUDA(cudaMalloc(&d_offset_iter, num_walker_total * sizeof(*d_offset_iter)));
    CHECK_CUDA(cudaMemset(d_offset_iter, 0, num_walker_total * sizeof(*d_offset_iter)));
    
    // f Factors for each walker
    std::vector<double> h_factor(num_walker_total, exp(1.0));

    double *d_factor;
    CHECK_CUDA(cudaMalloc(&d_factor, num_walker_total * sizeof(*d_factor)));
    CHECK_CUDA(cudaMemcpy(d_factor, h_factor.data(), num_walker_total * sizeof(*d_factor), cudaMemcpyHostToDevice));

    // Indices used for replica exchange later
    int *d_indices;
    CHECK_CUDA(cudaMalloc(&d_indices, num_walker_total * sizeof(*d_indices)));

    // lattice, interactions
    signed char *d_lattice, *d_interactions;
    CHECK_CUDA(cudaMalloc(&d_lattice, num_walker_total * options.X * options.Y * sizeof(*d_lattice)));
    CHECK_CUDA(cudaMalloc(&d_interactions, options.X * options.Y * 2 * sizeof(*d_interactions)));

    // Hamiltonian of lattices
    int *d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, num_walker_total * sizeof(*d_energy)));
    CHECK_CUDA(cudaMemset(d_energy, 0, num_walker_total * sizeof(*d_energy)));

    // Binary indicator of energies were found or not
    int *d_expected_energy_spectrum;
    CHECK_CUDA(cudaMalloc(&d_expected_energy_spectrum, h_expected_energy_spectrum.size() * sizeof(*d_expected_energy_spectrum)));
    CHECK_CUDA(cudaMemcpy(d_expected_energy_spectrum, h_expected_energy_spectrum.data(), h_expected_energy_spectrum.size() * sizeof(*d_expected_energy_spectrum), cudaMemcpyHostToDevice));

    // To catch energies which are outside of expected spectrum
    int *d_newEnergies, *d_foundNewEnergyFlag; 
    CHECK_CUDA(cudaMalloc(&d_newEnergies, num_walker_total * sizeof(*d_newEnergies)));
    CHECK_CUDA(cudaMalloc(&d_foundNewEnergyFlag, num_walker_total * sizeof(*d_foundNewEnergyFlag)));

    double* d_finished_walkers_ratio;
    CHECK_CUDA(cudaMalloc(&d_finished_walkers_ratio, 1 * sizeof(*d_finished_walkers_ratio)));

    signed char* d_cond;
    CHECK_CUDA(cudaMalloc(&d_cond, options.num_intervals*sizeof(*d_cond)));
    CHECK_CUDA(cudaMemset(d_cond, 0, options.num_intervals*sizeof(*d_cond)));

    /*
    ----------------------------------------------
    ------------ Actual WL Starts Now ------------
    ----------------------------------------------
    */

    // Initialization of lattices, interactions, offsets and indices
    init_offsets_lattice<<<options.num_intervals, options.walker_per_interval>>>(d_offset_lattice, options.X, options.Y, num_walker_total);
    init_offsets_histogramm<<<options.num_intervals, options.walker_per_interval>>>(d_offset_histogramm, d_start, d_end);
    init_indices<<<options.num_intervals, options.walker_per_interval>>>(d_indices);
    cudaDeviceSynchronize();

    char *interaction_file = constructFilePath(options.prob_interactions, options.X, options.Y, options.seed_histogram, "interactions", options.logical_error_type, options.boundary_type);
    std::vector<signed char> h_interactions;
    read(h_interactions, interaction_file);
    CHECK_CUDA(cudaMemcpy(d_interactions, h_interactions.data(), options.X * options.Y * 2 * sizeof(*d_interactions), cudaMemcpyHostToDevice));
    
    std::vector<signed char> h_lattice = get_lattice_with_pre_run_result(options.prob_interactions, options.seed_histogram, options.X, options.Y, interval_result.h_start, interval_result.h_end, options.num_intervals, num_walker_total, options.walker_per_interval, options.logical_error_type, options.boundary_type);
    CHECK_CUDA(cudaMemcpy(d_lattice, h_lattice.data(), num_walker_total * options.X * options.Y * sizeof(*d_lattice), cudaMemcpyHostToDevice));

    calc_energy(options.num_intervals, options.walker_per_interval, options.boundary_type, d_lattice, d_interactions, d_energy, d_offset_lattice, options.X, options.Y, num_walker_total);
    cudaDeviceSynchronize();
    
    check_energy_ranges<<<options.num_intervals, options.walker_per_interval>>>(d_energy, d_start, d_end);
    cudaDeviceSynchronize();

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time before Wang Landau has started: " << elapsed.count() << " seconds" << std::endl;

    double max_factor = exp(1.0);
    int max_newEnergyFlag = 0;

    int block_count = (interval_result.len_histogram_over_all_walkers + max_threads_per_block - 1) / max_threads_per_block;

    while (max_factor > exp(options.beta)){
        printf("Max Factor %8f \n", max_factor);

        wang_landau<<<options.num_intervals, options.walker_per_interval>>>(d_lattice, d_interactions, d_energy, d_start, d_end, d_H, d_logG, d_offset_histogramm, d_offset_lattice, options.num_iterations, options.X, options.Y, options.seed_run, d_factor, d_offset_iter, d_expected_energy_spectrum, d_newEnergies, d_foundNewEnergyFlag, num_walker_total, options.beta, d_cond, options.boundary_type);
        cudaDeviceSynchronize(); 

        // get max of found new energy flag array to condition break and update the histogramm file with value in new energy array
        thrust::device_ptr<int> d_newEnergyFlag_ptr(d_foundNewEnergyFlag);
        thrust::device_ptr<int> max_newEnergyFlag_ptr = thrust::max_element(d_newEnergyFlag_ptr, d_newEnergyFlag_ptr + num_walker_total);
        max_newEnergyFlag = *max_newEnergyFlag_ptr;

        // If flag shows new energies get the device arrays containing these to the host, update histogramm file and print error message.
        if (max_newEnergyFlag != 0){
            int h_newEnergies[num_walker_total];
            int h_newEnergyFlag[num_walker_total];
            CHECK_CUDA(cudaMemcpy(h_newEnergies, d_newEnergies, num_walker_total * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_newEnergyFlag, d_foundNewEnergyFlag, num_walker_total * sizeof(int), cudaMemcpyDeviceToHost));

            handleNewEnergyError(h_newEnergies, h_newEnergyFlag, histogram_file, num_walker_total); 
            return 1;
        }

        check_histogram<<<options.num_intervals, options.walker_per_interval>>>(d_H, d_logG, d_shared_logG, d_offset_histogramm, d_end, d_start, d_factor, options.X, options.Y, options.alpha, options.beta, d_expected_energy_spectrum, len_energy_spectrum, num_walker_total, d_cond);
        cudaDeviceSynchronize();
        
        calc_average_log_g<<<block_count, max_threads_per_block>>>(options.num_intervals, interval_result.len_histogram_over_all_walkers, options.walker_per_interval, d_logG, d_shared_logG, d_end, d_start, d_expected_energy_spectrum, d_cond);
        cudaDeviceSynchronize();
        
        redistribute_g_values<<<block_count, max_threads_per_block>>>(options.num_intervals, interval_result.len_histogram_over_all_walkers, options.walker_per_interval,  d_logG, d_shared_logG, d_end, d_start, d_factor, options.beta, d_expected_energy_spectrum, d_cond);
        cudaDeviceSynchronize();

        CHECK_CUDA(cudaMemset(d_shared_logG, 0, size_shared_log_G*sizeof(*d_shared_logG)));

        // get max factor over walkers for abort condition of while loop 
        thrust::device_ptr<double> d_factor_ptr(d_factor);
        thrust::device_ptr<double> max_factor_ptr = thrust::max_element(d_factor_ptr, d_factor_ptr + num_walker_total);
        max_factor = *max_factor_ptr;

        // std::vector<int> h_offset_lattice(num_walker_total);
        // std::vector<int> h_energy(num_walker_total);
        // CHECK_CUDA(cudaMemcpy(h_offset_lattice.data(), d_offset_lattice, num_walker_total*sizeof(*d_offset_lattice), cudaMemcpyDeviceToHost));
        // CHECK_CUDA(cudaMemcpy(h_energy.data(), d_energy, num_walker_total*sizeof(*d_energy), cudaMemcpyDeviceToHost));
        // for (int i=0; i<num_walker_total; i++){
        //     std::cout << h_offset_lattice[i] << " " << h_energy[i] << std::endl;
        // }

        replica_exchange<<<options.num_intervals, options.walker_per_interval>>>(d_offset_lattice, d_energy, d_start, d_end, d_indices, d_logG, d_offset_histogramm, true, options.seed_run, d_offset_iter);
        cudaDeviceSynchronize();

        replica_exchange<<<options.num_intervals, options.walker_per_interval>>>(d_offset_lattice, d_energy, d_start, d_end, d_indices, d_logG, d_offset_histogramm, false, options.seed_run, d_offset_iter);
        cudaDeviceSynchronize();

    }

    result_handling(options, interval_result, d_logG);

}