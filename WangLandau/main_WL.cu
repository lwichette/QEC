#include <thrust/extrema.h> //addition needed for my (Linnea's) version of thrust -- comment out if this causes issues
#include <cub/cub.cuh>
#include "./header/cudamacro.h"
#include "./header/wlutils.cuh"

/*
To Do:
    - blockID to blockIdx.x to save storage?
    - d_indices array not needed, could in theory use shared memory in each
fisher yates call
    - init flag for found new energy with seperate kernel and only update in
wang landau inverted to current setting
    - maybe implement runtime balanced subdivision as in
https://www.osti.gov/servlets/purl/1567362
    - Add sort to lattice read function
    - update histogram not working
*/

__global__ void check_sums(int *d_cond_interactions, int num_intervals, int num_interactions){
    
    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= num_interactions) return;

    if (d_cond_interactions[tid] == num_intervals){
        d_cond_interactions[tid] = 1;
    }
}

void check_interactions_finished(
    signed char *d_cond, int *d_cond_interactions, 
    int *d_offset_intervals, int num_intervals, int num_interactions,
    void *d_temp_storage, size_t& temp_storage_bytes
    ){
    
    // Determine the amount of temporary storage needed
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_cond, d_cond_interactions, num_interactions, d_offset_intervals, d_offset_intervals + 1);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Perform the segmented reduction
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_cond, d_cond_interactions, num_interactions, d_offset_intervals, d_offset_intervals + 1);
    cudaDeviceSynchronize();

    check_sums<<<num_interactions,1>>>(d_cond_interactions, num_intervals, num_interactions);
    cudaDeviceSynchronize();

    cudaFree(d_temp_storage);
}


int main(int argc, char **argv)
{

    // Temporary storage size
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // Get the device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming device 0
    int max_threads_per_block = prop.maxThreadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();

    Options options;
    parse_args(argc, argv, &options);

    const int total_walker = options.num_interactions * options.num_intervals * options.walker_per_interval;
    const int total_intervals = options.num_interactions * options.num_intervals;
    const int walker_per_interactions = options.num_intervals * options.walker_per_interval;

    // Ŕead histograms for all different seeds
    std::vector<signed char> h_expected_energy_spectrum;
    std::vector<int> h_len_energy_spectrum;
    std::vector<int> h_offset_energy_spectrum;
    int total_len_energy_spectrum = 0;

    for (int i = 0; i < options.num_interactions; i++)
    {
        std::string hist_path =
            constructFilePath(options.prob_interactions, options.X, options.Y,
                              options.seed_histogram + i, "histogram",
                              options.logical_error_type, options.boundary_type);
        std::vector<signed char> energy_spectrum = read_histogram(hist_path, options.E_min, options.E_max);

        h_expected_energy_spectrum.insert(h_expected_energy_spectrum.end(),
                                          energy_spectrum.begin(),
                                          energy_spectrum.end());
        h_offset_energy_spectrum.push_back(total_len_energy_spectrum);
        total_len_energy_spectrum += energy_spectrum.size();
        h_len_energy_spectrum.push_back(energy_spectrum.size());
    }

    int *d_offset_energy_spectrum, *d_len_energy_spectrum;
    CHECK_CUDA(cudaMalloc(&d_offset_energy_spectrum, options.num_interactions * sizeof(*d_offset_energy_spectrum)));
    CHECK_CUDA(cudaMemcpy(d_offset_energy_spectrum, h_offset_energy_spectrum.data(), options.num_interactions * sizeof(*d_offset_energy_spectrum), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_len_energy_spectrum,options.num_interactions * sizeof(*d_len_energy_spectrum)));
    CHECK_CUDA(cudaMemcpy(d_len_energy_spectrum, h_len_energy_spectrum.data(), options.num_interactions * sizeof(*d_len_energy_spectrum), cudaMemcpyHostToDevice));

    // Generate intervals for all different energy spectrums
    std::vector<int> h_end_int;
    std::vector<int> h_start_int;
    std::vector<int> len_histogram_int;
    std::vector<int> len_interval_int;

    long long total_len_histogram = 0;

    for (int i = 0; i < options.num_interactions; i++)
    {
        IntervalResult run_result = generate_intervals(
            options.E_min[i], options.E_max[i], options.num_intervals,
            options.walker_per_interval, options.overlap_decimal);

        h_end_int.insert(h_end_int.end(), run_result.h_end.begin(),
                         run_result.h_end.end());

        h_start_int.insert(h_start_int.end(), run_result.h_start.begin(),
                           run_result.h_start.end());

        len_histogram_int.push_back(run_result.len_histogram_over_all_walkers);
        len_interval_int.push_back(run_result.len_interval);
        total_len_histogram += run_result.len_histogram_over_all_walkers;
    }

    int *d_len_histograms;
    CHECK_CUDA(cudaMalloc(&d_len_histograms, options.num_interactions * sizeof(*d_len_histograms)))
    CHECK_CUDA(cudaMemcpy(d_len_histograms, len_histogram_int.data(), options.num_interactions * sizeof(*d_len_histograms), cudaMemcpyHostToDevice));

    // Start end energies of the intervals
    int *d_start, *d_end;
    CHECK_CUDA(cudaMalloc(&d_start, total_intervals * sizeof(*d_start)));
    CHECK_CUDA(cudaMalloc(&d_end, total_intervals * sizeof(*d_end)));
    CHECK_CUDA(cudaMemcpy(d_start, h_start_int.data(), total_intervals * sizeof(*d_start), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_end, h_end_int.data(), total_intervals * sizeof(*d_start), cudaMemcpyHostToDevice));

    // Histogramm and G array
    unsigned long long *d_H;
    CHECK_CUDA(cudaMalloc(&d_H, total_len_histogram * sizeof(*d_H)));
    CHECK_CUDA(cudaMemset(d_H, 0, total_len_histogram * sizeof(*d_H)));

    double *d_logG;
    CHECK_CUDA(cudaMalloc(&d_logG, total_len_histogram * sizeof(*d_logG)));
    CHECK_CUDA(cudaMemset(d_logG, 0, total_len_histogram * sizeof(*d_logG)));

    std::vector<long long> h_offset_shared_log_G;
    int size_shared_log_G = 0;
    for (int i = 0; i < options.num_interactions; i++){
        for (int j=0; j < options.num_intervals; j++){
            h_offset_shared_log_G.push_back(size_shared_log_G);
            size_shared_log_G += (h_end_int[i*options.num_intervals + j] - h_start_int[i*options.num_intervals + j] + 1);
        }
    }

    double *d_shared_logG;
    CHECK_CUDA(cudaMalloc(&d_shared_logG, size_shared_log_G * sizeof(*d_shared_logG)));
    CHECK_CUDA(cudaMemset(d_shared_logG, 0, size_shared_log_G * sizeof(*d_shared_logG)));

    long long *d_offset_shared_logG;
    CHECK_CUDA(cudaMalloc(&d_offset_shared_logG, total_intervals * sizeof(*d_offset_shared_logG)));
    CHECK_CUDA(cudaMemcpy(d_offset_shared_logG, h_offset_shared_log_G.data(), total_intervals*sizeof(*d_offset_shared_logG), cudaMemcpyHostToDevice));

    // Offset histograms, lattice, seed_iterator
    int *d_offset_histogram, *d_offset_lattice;
    unsigned long long *d_offset_iter;
    CHECK_CUDA(cudaMalloc(&d_offset_histogram, total_walker * sizeof(*d_offset_histogram)));
    CHECK_CUDA(cudaMalloc(&d_offset_lattice, total_walker * sizeof(*d_offset_lattice)));
    CHECK_CUDA(cudaMalloc(&d_offset_iter, total_walker * sizeof(*d_offset_iter)));
    CHECK_CUDA(cudaMemset(d_offset_iter, 0, total_walker * sizeof(*d_offset_iter)));

    // f Factors for each walker
    std::vector<double> h_factor(total_walker, exp(1.0));

    double *d_factor;
    CHECK_CUDA(cudaMalloc(&d_factor, total_walker * sizeof(*d_factor)));
    CHECK_CUDA(cudaMemcpy(d_factor, h_factor.data(), total_walker * sizeof(*d_factor), cudaMemcpyHostToDevice));

    // Indices used for replica exchange later
    int *d_indices;
    CHECK_CUDA(cudaMalloc(&d_indices, total_walker * sizeof(*d_indices)));

    // lattice, interactions
    signed char *d_lattice, *d_interactions;
    CHECK_CUDA(cudaMalloc(&d_lattice, total_walker * options.X * options.Y * sizeof(*d_lattice)));
    CHECK_CUDA(cudaMalloc(&d_interactions, options.num_interactions * options.X *options.Y * 2 *sizeof(*d_interactions)));

    // Hamiltonian of lattices
    int *d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, total_walker * sizeof(*d_energy)));
    CHECK_CUDA(cudaMemset(d_energy, 0, total_walker * sizeof(*d_energy)));

    // Binary indicator of energies were found or not
    signed char *d_expected_energy_spectrum;
    CHECK_CUDA(cudaMalloc(&d_expected_energy_spectrum,h_expected_energy_spectrum.size() * sizeof(*d_expected_energy_spectrum)));
    CHECK_CUDA(cudaMemcpy(d_expected_energy_spectrum, h_expected_energy_spectrum.data(),h_expected_energy_spectrum.size() * sizeof(*d_expected_energy_spectrum), cudaMemcpyHostToDevice));

    // To catch energies which are outside of expected spectrum
    int *d_newEnergies, *d_foundNewEnergyFlag;
    CHECK_CUDA(cudaMalloc(&d_newEnergies, total_walker * sizeof(*d_newEnergies)));
    CHECK_CUDA(cudaMalloc(&d_foundNewEnergyFlag, total_walker * sizeof(*d_foundNewEnergyFlag)));

    double *d_finished_walkers_ratio;
    CHECK_CUDA(cudaMalloc(&d_finished_walkers_ratio, 1 * sizeof(*d_finished_walkers_ratio)));

    signed char *d_cond;
    CHECK_CUDA(cudaMalloc(&d_cond, total_intervals * sizeof(*d_cond)));
    CHECK_CUDA(cudaMemset(d_cond, 0, total_intervals * sizeof(*d_cond)));
    
    int *d_cond_interactions;
    CHECK_CUDA(cudaMalloc(&d_cond_interactions, options.num_interactions * sizeof(*d_cond_interactions)));
    CHECK_CUDA(cudaMemset(d_cond_interactions, 0, options.num_interactions * sizeof(*d_cond_interactions)));

    std::vector<int> h_offset_intervals(options.num_interactions + 1);

    for (int i = 0; i < options.num_interactions; i++) {
        h_offset_intervals[i] = i * options.num_intervals;
    }
    
    h_offset_intervals[options.num_interactions] = total_intervals;

    int *d_offset_intervals;
    CHECK_CUDA(cudaMalloc(&d_offset_intervals, h_offset_intervals.size() * sizeof(*d_offset_intervals)));
    CHECK_CUDA(cudaMemcpy(d_offset_intervals, h_offset_intervals.data(), h_offset_intervals.size() * sizeof(*d_offset_intervals), cudaMemcpyHostToDevice));

    /*
    ----------------------------------------------
    ------------ Actual WL Starts Now ------------
    ----------------------------------------------
    */

    // Initialization of lattices, interactions, offsets and indices
    init_offsets_lattice<<<total_intervals, options.walker_per_interval>>>(
        d_offset_lattice, options.X, options.Y, total_walker);
    
    init_offsets_histogramm<<<total_intervals, options.walker_per_interval>>>(
        d_offset_histogram, d_start, d_end, d_len_histograms,
        options.num_intervals, total_walker);
    
    init_indices<<<total_intervals, options.walker_per_interval>>>(d_indices,
                                                                   total_walker);
    cudaDeviceSynchronize();

    std::vector<signed char> h_interactions;
    for (int i = 0; i < options.num_interactions; i++)
    {
        std::string run_int_path = constructFilePath(options.prob_interactions, options.X, options.Y, options.seed_histogram + i, "interactions", options.logical_error_type, options.boundary_type);
        std::vector<signed char> run_interactions;
        read(run_interactions, run_int_path);

        h_interactions.insert(h_interactions.end(), run_interactions.begin(), run_interactions.end());
    }

    CHECK_CUDA(cudaMemcpy(d_interactions, h_interactions.data(), options.num_interactions * options.X * options.Y * 2 * sizeof(*d_interactions), cudaMemcpyHostToDevice));

    std::vector<signed char> h_lattice;

    for (int i = 0; i < options.num_interactions; i++)
    {
        std::vector<int> run_start(h_start_int.begin() + i * options.num_intervals,
                                   h_start_int.begin() + i * options.num_intervals +
                                       options.num_intervals);
        std::vector<int> run_end(
            h_end_int.begin() + i * options.num_intervals,
            h_end_int.begin() + i * options.num_intervals + options.num_intervals);

        std::vector<signed char> run_lattice = get_lattice_with_pre_run_result(
            options.prob_interactions, options.seed_histogram + i, options.X,
            options.Y, run_start, run_end, options.num_intervals, total_walker,
            options.walker_per_interval, options.logical_error_type,
            options.boundary_type);

        h_lattice.insert(h_lattice.end(), run_lattice.begin(), run_lattice.end());
    }

    CHECK_CUDA(cudaMemcpy(d_lattice, h_lattice.data(),total_walker * options.X * options.Y * sizeof(*d_lattice), cudaMemcpyHostToDevice));

    calc_energy(total_intervals, options.walker_per_interval,
                options.boundary_type, d_lattice, d_interactions, d_energy,
                d_offset_lattice, options.X, options.Y, total_walker,
                walker_per_interactions);
    cudaDeviceSynchronize();

    check_energy_ranges<<<total_intervals, options.walker_per_interval>>>(d_energy, d_start, d_end, total_walker);
    cudaDeviceSynchronize();

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time before Wang Landau has started: "
              << elapsed.count() << " seconds" << std::endl;

    double max_factor = exp(1.0);
    int max_newEnergyFlag = 0;

    int block_count = (total_len_histogram + max_threads_per_block - 1) / max_threads_per_block;

    while (max_factor > exp(options.beta)){
        printf("Max Factor %8f \n", max_factor);

        wang_landau<<<total_intervals, options.walker_per_interval>>>(
            d_lattice, d_interactions, d_energy, d_start, d_end, d_H, d_logG,
            d_offset_histogram, d_offset_lattice, options.num_iterations,
            options.X, options.Y, options.seed_run, d_factor, d_offset_iter,
            d_expected_energy_spectrum, d_newEnergies, d_foundNewEnergyFlag,
            total_walker, options.beta, d_cond, options.boundary_type,
            walker_per_interactions, options.num_intervals,
            d_offset_energy_spectrum, d_cond_interactions);
        cudaDeviceSynchronize();

        // get max of found new energy flag array to condition break and update the
        // histogramm file with value in new energy array
        thrust::device_ptr<int> d_newEnergyFlag_ptr(d_foundNewEnergyFlag);
        thrust::device_ptr<int> max_newEnergyFlag_ptr = thrust::max_element(d_newEnergyFlag_ptr, d_newEnergyFlag_ptr + total_walker);
        max_newEnergyFlag = *max_newEnergyFlag_ptr;

        // If flag shows new energies get the device arrays containing these to the
        // host, update histogramm file and print error message.
        if (max_newEnergyFlag != 0){
            int h_newEnergies[total_walker];
            int h_newEnergyFlag[total_walker];
            CHECK_CUDA(cudaMemcpy(h_newEnergies, d_newEnergies, total_walker * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_newEnergyFlag, d_foundNewEnergyFlag, total_walker * sizeof(int), cudaMemcpyDeviceToHost));

            // TO DO: Adjust for several interactions
            // handleNewEnergyError(h_newEnergies, h_newEnergyFlag, histogram_file,
            // total_walker);
            std::cerr << "Error: Found new energy:" << std::endl;
            return 1;
        }
        
        check_histogram<<<total_intervals, options.walker_per_interval>>>(
            d_H, d_logG, d_shared_logG, d_offset_histogram, d_end, d_start,
            d_factor, options.X, options.Y, options.alpha, options.beta,
            d_expected_energy_spectrum, d_len_energy_spectrum, total_walker, d_cond,
            walker_per_interactions, options.num_intervals,
            d_offset_energy_spectrum, d_cond_interactions);
        cudaDeviceSynchronize();

        calc_average_log_g<<<block_count, max_threads_per_block>>>(options.num_intervals, 
            d_len_histograms, options.walker_per_interval,
            d_logG, d_shared_logG, d_end, d_start, d_expected_energy_spectrum, d_cond, 
            d_offset_histogram, d_offset_energy_spectrum, options.num_interactions,
            d_offset_shared_logG, d_cond_interactions);
        cudaDeviceSynchronize();

        redistribute_g_values<<<block_count, max_threads_per_block>>>(options.num_intervals,
            d_len_histograms, options.walker_per_interval, d_logG, d_shared_logG, 
            d_end, d_start, d_factor, options.beta, d_expected_energy_spectrum, d_cond,
            d_offset_histogram, options.num_interactions, d_offset_shared_logG, d_cond_interactions); 
        cudaDeviceSynchronize();
        
        CHECK_CUDA(cudaMemset(d_shared_logG, 0, size_shared_log_G*sizeof(*d_shared_logG)));
        
        check_interactions_finished(
            d_cond, d_cond_interactions, d_offset_intervals, 
            options.num_intervals, options.num_interactions, 
            d_temp_storage, temp_storage_bytes);
    

        // get max factor over walkers for abort condition of while loop
        thrust::device_ptr<double> d_factor_ptr(d_factor);
        thrust::device_ptr<double> max_factor_ptr = thrust::max_element(d_factor_ptr, d_factor_ptr + total_walker);
        max_factor = *max_factor_ptr;

        replica_exchange<<<total_intervals, options.walker_per_interval>>>(
            d_offset_lattice, d_energy, d_start, d_end, d_indices, d_logG,
            d_offset_histogram, true, options.seed_run, d_offset_iter,
            options.num_intervals, walker_per_interactions, d_cond_interactions);
        cudaDeviceSynchronize();

        replica_exchange<<<total_intervals, options.walker_per_interval>>>(
            d_offset_lattice, d_energy, d_start, d_end, d_indices, d_logG,
            d_offset_histogram, false, options.seed_run, d_offset_iter,
            options.num_intervals, walker_per_interactions, d_cond_interactions);
        cudaDeviceSynchronize();

        // To Do d_cond_interaction and check_finished
        // Adjust result_handling individually
    }

    std::vector<int> h_cond_interactions(options.num_interactions);
    CHECK_CUDA(cudaMemcpy(h_cond_interactions.data(), d_cond_interactions, options.num_interactions * sizeof(*d_cond_interactions), cudaMemcpyDeviceToHost));

    for (int i=0; i < options.num_interactions; i++){
        std::cout << h_cond_interactions[i] << std::endl;
    }

    std::vector<double> h_logG(total_len_histogram);
    CHECK_CUDA(cudaMemcpy(h_logG.data(), d_logG, total_len_histogram * sizeof(*d_logG), cudaMemcpyDeviceToHost));

    std::vector<int> h_offset_histogram(total_walker);
    CHECK_CUDA(cudaMemcpy(h_offset_histogram.data(), d_offset_histogram, total_walker * sizeof(*d_offset_histogram), cudaMemcpyDeviceToHost));

    for (int i=0; i < options.num_interactions; i++){
        std::vector<int> run_start(h_start_int.begin() + i * options.num_intervals,
                                   h_start_int.begin() + (i + 1)*options.num_intervals);
                                       
        std::vector<int> run_end(h_end_int.begin() + i * options.num_intervals,
                                 h_end_int.begin() + (i + 1)*options.num_intervals);
        
        std::vector<double> run_logG;
        auto start = h_logG.begin() + h_offset_histogram[i * (options.num_intervals * options.walker_per_interval)];
        auto end = (i < options.num_interactions - 1) ? h_logG.begin() + h_offset_histogram[(i + 1) * (options.num_intervals * options.walker_per_interval)] : h_logG.end();
        run_logG.assign(start, end);
        
        result_handling(options, run_logG, run_start, run_end, i);   
    }
    
    return 0;
}
