#include "./header/cudamacro.h"
#include "./header/wlutils.cuh"

/*
To Do:
    - Paths in construct histogram path, and interaction path
    - blockID to blockIdx.x to save storage?
    - d_indices array not needed, could in theory use shared memory in each fisher yates call
    - still the read and write of interactions such that we initialize each WL run with a specific histogram and interaction data
    - Store results and normalize
    - init flag for found new energy with seperate kernel and only update in wang landau inverted to current setting
    - New energies smarter way to update histogram
    - print metric finished walker count / total walker count -> may use for finish condition
    - maybe implement runtime balanced subdivision as in https://www.osti.gov/servlets/purl/1567362
    - Get averages over all walkers per interval of log g after all are simultaneously flat 
    - update spin configs for totally finished walkers still but do not update hist and g anymore -> circumvents replica exchange problem
    - Add sort to lattice read function
*/

int main(int argc, char **argv){

    // Get the device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming device 0
    int max_threads_per_block = prop.maxThreadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now(); 

    const int seed = 42;

    Options options;
    parse_args(argc, argv, &options);
    
    const int num_walker_total = options.num_intervals * options.walker_per_interval;

    char *histogram_file = constructFilePath(options.prob_interactions, options.X, options.Y, seed, "histogram");

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
    init_offsets_lattice<<<options.num_intervals, options.walker_per_interval>>>(d_offset_lattice, options.X, options.Y);
    init_offsets_histogramm<<<options.num_intervals, options.walker_per_interval>>>(d_offset_histogramm, d_start, d_end);
    init_indices<<<options.num_intervals, options.walker_per_interval>>>(d_indices);
    
    char *interaction_file = constructFilePath(options.prob_interactions, options.X, options.Y, seed, "interactions");
    std::vector<signed char> h_interactions;
    read(h_interactions, interaction_file);
    CHECK_CUDA(cudaMemcpy(d_interactions, h_interactions.data(), options.X * options.Y * 2 * sizeof(*d_interactions), cudaMemcpyHostToDevice));
    
    std::vector<signed char> h_lattice = get_lattice_with_pre_run_result(options.prob_interactions, seed, options.X, options.Y, interval_result.h_start, interval_result.h_end, options.num_intervals, num_walker_total, options.walker_per_interval);
    CHECK_CUDA(cudaMemcpy(d_lattice, h_lattice.data(), num_walker_total * options.X * options.Y * sizeof(*d_lattice), cudaMemcpyHostToDevice));
    
    // Calculate energy and find right configurations
    calc_energy<<<options.num_intervals, options.walker_per_interval>>>(d_lattice, d_interactions, d_energy, d_offset_lattice, options.X, options.Y, num_walker_total);    
    check_energy_ranges<<<options.num_intervals, options.walker_per_interval>>>(d_energy, d_start, d_end);
    cudaDeviceSynchronize();

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time before Wang Landau has started: " << elapsed.count() << " seconds" << std::endl;

    double max_factor = exp(1.0);
    int max_newEnergyFlag = 0;
    double finished_walkers_ratio = 0;

    int block_count = (interval_result.len_histogram_over_all_walkers + max_threads_per_block - 1) / max_threads_per_block;

    while (max_factor > exp(options.beta)){

        wang_landau<<<options.num_intervals, options.walker_per_interval>>>(d_lattice, d_interactions, d_energy, d_start, d_end, d_H, d_logG, d_offset_histogramm, d_offset_lattice, options.num_iterations, options.X, options.Y, seed + 3, d_factor, d_offset_iter, d_expected_energy_spectrum, d_newEnergies, d_foundNewEnergyFlag, num_walker_total, options.beta, d_cond);
        cudaDeviceSynchronize(); 

        // get max of found new energy flag array to condition break and update the histogramm file with value in new energy array
        thrust::device_ptr<int> d_newEnergyFlag_ptr(d_foundNewEnergyFlag);
        thrust::device_ptr<int> max_newEnergyFlag_ptr = thrust::max_element(d_newEnergyFlag_ptr, d_newEnergyFlag_ptr + num_walker_total);
        max_newEnergyFlag = *max_newEnergyFlag_ptr;

        // If flag shows new energies get the device arrays containing these to the host, update histogramm file and print error message.
        if (max_newEnergyFlag != 0)
        {
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
        redistribute_g_values<<<block_count, max_threads_per_block>>>(options.num_intervals, interval_result.len_histogram_over_all_walkers, options.walker_per_interval, d_logG, d_shared_logG, d_end, d_start, d_factor, options.beta, d_expected_energy_spectrum, d_cond);
        cudaDeviceSynchronize();

        CHECK_CUDA(cudaMemset(d_shared_logG, 0, size_shared_log_G*sizeof(*d_shared_logG)));

        // get max factor over walkers for abort condition of while loop 
        thrust::device_ptr<double> d_factor_ptr(d_factor);
        thrust::device_ptr<double> max_factor_ptr = thrust::max_element(d_factor_ptr, d_factor_ptr + num_walker_total);
        max_factor = *max_factor_ptr;
        
        //replica_exchange<<<options.num_intervals, options.walker_per_interval>>>(d_offset_lattice, d_energy, d_start, d_end, d_indices, d_logG, d_offset_histogramm, true, seed + 3, d_offset_iter);
        //replica_exchange<<<options.num_intervals, options.walker_per_interval>>>(d_offset_lattice, d_energy, d_start, d_end, d_indices, d_logG, d_offset_histogramm, false, seed + 3, d_offset_iter);

        print_finished_walker_ratio<<<1, num_walker_total>>>(d_factor, num_walker_total, exp(options.beta), d_finished_walkers_ratio);


        // // This block here is mainly for testing the non convergence
        // // get ratio of finished walkers to control dump of histogram
        // thrust::device_ptr<double> d_finished_walkers_ratio_ptr(d_finished_walkers_ratio);
        // finished_walkers_ratio = *d_finished_walkers_ratio_ptr;
        // printf("ratio of finished walkers: %f\n", finished_walkers_ratio);
        // if(finished_walkers_ratio >= 0.9){
        //     std::vector<unsigned long long> h_hist(interval_result.len_histogram_over_all_walkers);
        //     CHECK_CUDA(cudaMemcpy(h_hist.data(), d_H, interval_result.len_histogram_over_all_walkers * sizeof(*d_H), cudaMemcpyDeviceToHost));

        //     std::ofstream hist_file("histogram_time_evolution.txt", std::ios::app);
        //     int index_h_hist = 0;
        //     for (int i = 0; i < options.num_intervals; i++)
        //     {
        //         int start_energy = interval_result.h_start[i];
        //         int end_energy = interval_result.h_end[i];
        //         int len_int = interval_result.h_end[i] - interval_result.h_start[i] + 1;
        //         for (int j = 0; j < options.walker_per_interval; j++)
        //         {
        //             for (int k = 0; k < len_int; k++)
        //             {
        //                 hist_file << (int)interval_result.h_start[i] + k << " : " << h_hist[index_h_hist] << " ,";
        //                 index_h_hist += 1;
        //             }
        //             hist_file << std::endl;
        //         }
        //     }
        //     hist_file << std::endl;
        //     hist_file.close();
        // }
    }

    /*
    ---------------------------------------------
    --------------Post Processing ---------------
    ---------------------------------------------
    */

    std::vector<double> h_log_density_per_walker(interval_result.len_histogram_over_all_walkers);
    CHECK_CUDA(cudaMemcpy(h_log_density_per_walker.data(), d_logG, interval_result.len_histogram_over_all_walkers * sizeof(*d_logG), cudaMemcpyDeviceToHost));

    std::ofstream f_log_density;

    std::stringstream result_path;
    result_path << "results/prob_" << std::fixed << std::setprecision(6) << options.prob_interactions
       << "/X_" << options.X
       << "_Y_" << options.Y
       << "/seed_" << seed
       << "/intervals" << options.num_intervals
       << "_iterations" << options.num_iterations
       << "_overlap" << options.overlap_decimal
       << "_walkers" << options.walker_per_interval
       << "_alpha" << options.alpha
       << "_beta"  << std::fixed << std::setprecision(10) << options.beta
       << ".txt";

    std::cout << options.beta;

    f_log_density.open(result_path.str());

    int index_h_log_g = 0;
    if (f_log_density.is_open())
    {
        for (int i = 0; i < options.num_intervals; i++)
        {

            int start_energy = interval_result.h_start[i];
            int end_energy = interval_result.h_end[i];
            int len_int = interval_result.h_end[i] - interval_result.h_start[i] + 1;

            for (int j = 0; j < options.walker_per_interval; j++)
            {
                for (int k = 0; k < len_int; k++)
                {
                    f_log_density << (int)interval_result.h_start[i] + k << " : " << (float)h_log_density_per_walker[index_h_log_g] << " ,";
                    index_h_log_g += 1;
                }
                f_log_density << std::endl;
            }
        }

    }
    f_log_density.close();
}
