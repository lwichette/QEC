#include <thrust/extrema.h> //addition needed for my (Linnea's) version of thrust -- comment out if this causes issues
#include <cub/cub.cuh>
#include "./header/cudamacro.h"
#include "./header/wlutils.cuh"

std::vector<signed char> get_lattice_with_pre_run_result(Options options, int seed_offset, std::vector<int> h_start, std::vector<int> h_end)
{
    std::string boundary;

    switch (options.boundary_type)
    {
    case 0:
        boundary = "periodic";
        break;
    case 1:
        boundary = "open";
        break;
    case 2:
        boundary = "cylinder";
        break;
    default:
        boundary = "unknown";
        break; // Handle any unexpected boundary_type values
    }

    namespace fs = std::filesystem;
    std::ostringstream oss;
    oss << "init/task_id_" << options.task_id << "/" << boundary << "/error_mean_" << std::fixed << std::setprecision(6) << options.error_mean << "/error_variance_" << options.error_variance;
    oss << "/X_" << options.X << "_Y_" << options.Y;
    oss << "/error_class_" << options.logical_error_type;
    oss << "/seed_" << (options.seed_histogram + seed_offset);
    oss << "/lattice";

    std::string lattice_path = oss.str();
    std::vector<signed char> lattice_over_all_walkers;

    for (int interval_iterator = 0; interval_iterator < options.num_intervals; interval_iterator++)
    {
        // std::cout << interval_iterator << " ";
        try
        {
            bool found_energy_in_interval = false;

            for (const auto &entry : fs::directory_iterator(lattice_path))
            {
                // Check if the entry is a regular file and has a .txt extension
                if (entry.is_regular_file() && entry.path().extension() == ".txt")
                {
                    // Extract the number from the filename
                    std::string filename = entry.path().stem().string(); // Get the filename without extension
                    std::regex regex("lattice_energy_(-?\\d+(\\.\\d+)?)");
                    std::smatch match;
                    if (std::regex_search(filename, match, regex))
                    {
                        float number = std::stof(match[1]);
                        // Check if the number is between interval boundaries
                        if (static_cast<int>(std::round(number)) >= h_start[interval_iterator] && static_cast<int>(std::round(number)) <= h_end[interval_iterator])
                        {
                            found_energy_in_interval = true;

                            for (int walker_per_interval_iterator = 0; walker_per_interval_iterator < options.walker_per_interval; walker_per_interval_iterator++)
                            {
                                read(lattice_over_all_walkers, entry.path().string());
                                // std::cout << "Lattice with energy: " << number << " for interval [" << h_start[interval_iterator] << ", " << h_end[interval_iterator] << "]" << " walker: " << walker_per_interval_iterator << std::endl;
                            }
                            break;
                        }
                    }
                    else
                    {
                        std::cerr << "Unable to open file: " << entry.path() << std::endl;
                    }
                }
            }
            if (!found_energy_in_interval)
            {
                std::cerr << "No Lattice found for interval [" << h_start[interval_iterator] << ", " << h_end[interval_iterator] << "]" << " and interaction seed " << options.seed_histogram + seed_offset << std::endl;
                assert(0);
            }
        }
        catch (const fs::filesystem_error &e)
        {
            std::cerr << "Filesystem error: " << e.what() << std::endl;
        }
    }
    return lattice_over_all_walkers;
}

int main(int argc, char **argv)
{
    int max_threads_per_block = 128;

    auto start = std::chrono::high_resolution_clock::now();

    Options options;
    int opt;
    while (1)
    {
        int option_index = 0;
        static struct option long_options[] = {
            {"X", 1, 0, 'x'},
            {"Y", 1, 0, 'y'},
            {"num_iterations", 1, 0, 'n'},
            {"alpha", 1, 0, 'a'},
            {"beta", 1, 0, 'b'},
            {"num_intervals", 1, 0, 'i'},
            {"walker_per_interval", 1, 0, 'w'},
            {"overlap_decimal", 1, 0, 'o'},
            {"seed_histogram", 1, 0, 'h'},
            {"seed_run", 1, 0, 's'},
            {"logical_error", 1, 0, 'e'},
            {"boundary_type", 1, 0, 't'},
            {"repetitions_interactions", 1, 0, 'r'},
            {"replica_exchange_offsets", 1, 0, 'c'},
            {"task_id", 1, 0, 'd'},
            {"error_mean", 1, 0, 'm'},
            {"error_variance", 1, 0, 'v'},
            {0, 0, 0, 0}};

        opt = getopt_long(argc, argv, "x:y:n:a:b:i:w:o:h:s:e:t:r:c:d:m:v:", long_options, &option_index);

        if (opt == -1)
            break;
        switch (opt)
        {
        case 'x':
            options.X = std::atoi(optarg);
            break;
        case 'y':
            options.Y = std::atoi(optarg);
            break;
        case 'n':
            options.num_iterations = std::atoi(optarg);
            break;
        case 'a':
            options.alpha = std::atof(optarg);
            break;
        case 'b':
            options.beta = std::atof(optarg);
            break;
        case 'i':
            options.num_intervals = std::atoi(optarg);
            break;
        case 'w':
            options.walker_per_interval = std::atoi(optarg);
            break;
        case 'o':
            options.overlap_decimal = std::atof(optarg);
            break;
        case 'h':
            options.seed_histogram = std::atoi(optarg);
            break;
        case 's':
            options.seed_run = std::atoi(optarg);
            break;
        case 'e':
            options.logical_error_type = *optarg;
            break;
        case 't':
            options.boundary_type = std::atoi(optarg);
            break;
        case 'r':
            options.num_interactions = std::atoi(optarg);
            break;
        case 'c':
            options.replica_exchange_offset = std::atoi(optarg);
            break;
        case 'd':
            options.task_id = std::atoi(optarg);
            break;
        case 'm':
            options.error_mean = std::atof(optarg);
            break;
        case 'v':
            options.error_variance = std::atof(optarg);
            break;
        default:
            fprintf(stderr, "Usage: %s [-i num_intervals] [-m E_min] [-M E_max] [-w walker_per_interval] [-o overlap_decimal] [-r num_iterations]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

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
            constructFilePath(options, i, "histogram", true);

        std::vector<signed char>
            energy_spectrum = read_histogram(hist_path, options.E_min, options.E_max);

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
    CHECK_CUDA(cudaMalloc(&d_len_energy_spectrum, options.num_interactions * sizeof(*d_len_energy_spectrum)));
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

        std::cout << "last interval: " << h_start_int.back() << " - " << h_end_int.back() << " other intervals len: " << len_interval_int[i] << " num intervals: " << options.num_intervals << std::endl;
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
    for (int i = 0; i < options.num_interactions; i++)
    {
        for (int j = 0; j < options.num_intervals; j++)
        {
            h_offset_shared_log_G.push_back(size_shared_log_G);
            size_shared_log_G += (h_end_int[i * options.num_intervals + j] - h_start_int[i * options.num_intervals + j] + 1);
        }
    }

    double *d_shared_logG;
    CHECK_CUDA(cudaMalloc(&d_shared_logG, size_shared_log_G * sizeof(*d_shared_logG)));
    CHECK_CUDA(cudaMemset(d_shared_logG, 0, size_shared_log_G * sizeof(*d_shared_logG)));

    long long *d_offset_shared_logG;
    CHECK_CUDA(cudaMalloc(&d_offset_shared_logG, total_intervals * sizeof(*d_offset_shared_logG)));
    CHECK_CUDA(cudaMemcpy(d_offset_shared_logG, h_offset_shared_log_G.data(), total_intervals * sizeof(*d_offset_shared_logG), cudaMemcpyHostToDevice));

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
    signed char *d_lattice;
    CHECK_CUDA(cudaMalloc(&d_lattice, total_walker * options.X * options.Y * sizeof(*d_lattice)));

    double *d_interactions;
    CHECK_CUDA(cudaMalloc(&d_interactions, options.num_interactions * options.X * options.Y * 2 * sizeof(*d_interactions)));

    // Hamiltonian of lattices
    double *d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, total_walker * sizeof(*d_energy)));
    CHECK_CUDA(cudaMemset(d_energy, 0, total_walker * sizeof(*d_energy)));

    // Binary indicator of energies were found or not
    signed char *d_expected_energy_spectrum;
    CHECK_CUDA(cudaMalloc(&d_expected_energy_spectrum, h_expected_energy_spectrum.size() * sizeof(*d_expected_energy_spectrum)));
    CHECK_CUDA(cudaMemcpy(d_expected_energy_spectrum, h_expected_energy_spectrum.data(), h_expected_energy_spectrum.size() * sizeof(*d_expected_energy_spectrum), cudaMemcpyHostToDevice));

    // To catch energies which are outside of expected spectrum
    double *d_newEnergies;
    CHECK_CUDA(cudaMalloc(&d_newEnergies, total_walker * sizeof(*d_newEnergies)));

    int *d_foundNewEnergyFlag;
    CHECK_CUDA(cudaMalloc(&d_foundNewEnergyFlag, total_walker * sizeof(*d_foundNewEnergyFlag)));

    signed char *d_cond;
    CHECK_CUDA(cudaMalloc(&d_cond, total_intervals * sizeof(*d_cond)));
    CHECK_CUDA(cudaMemset(d_cond, 0, total_intervals * sizeof(*d_cond)));

    int *d_cond_interactions;
    CHECK_CUDA(cudaMalloc(&d_cond_interactions, options.num_interactions * sizeof(*d_cond_interactions)));
    CHECK_CUDA(cudaMemset(d_cond_interactions, 0, options.num_interactions * sizeof(*d_cond_interactions)));

    // host storage for the is finished flag per interaction stored in d_cond_interaction
    int *h_cond_interactions;
    h_cond_interactions = (int *)malloc(options.num_interactions * sizeof(*h_cond_interactions));

    bool *h_result_is_dumped;
    h_result_is_dumped = (bool *)calloc(options.num_interactions, sizeof(*h_result_is_dumped));

    std::vector<int> h_offset_intervals(options.num_interactions + 1);

    for (int i = 0; i < options.num_interactions; i++)
    {
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

    init_indices<<<total_intervals, options.walker_per_interval>>>(d_indices, total_walker);
    cudaDeviceSynchronize();

    std::vector<double> h_interactions;
    for (int i = 0; i < options.num_interactions; i++)
    {
        std::string run_int_path = constructFilePath(options, i, "interactions", true);
        std::vector<double> run_interactions;
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

        std::vector<signed char> run_lattice = get_lattice_with_pre_run_result(options, i, run_start, run_end);

        h_lattice.insert(h_lattice.end(), run_lattice.begin(), run_lattice.end());
    }

    CHECK_CUDA(cudaMemcpy(d_lattice, h_lattice.data(), total_walker * options.X * options.Y * sizeof(*d_lattice), cudaMemcpyHostToDevice));

    calc_energy(total_intervals, options.walker_per_interval, options.boundary_type, d_lattice, d_interactions, d_energy, d_offset_lattice, options.X, options.Y, total_walker, walker_per_interactions);
    cudaDeviceSynchronize();

    check_energy_ranges<double><<<total_intervals, options.walker_per_interval>>>(d_energy, d_start, d_end, total_walker);
    cudaDeviceSynchronize();

    // // TEST BLOCK
    // // ---------------------
    // double *d_energy_test;
    // CHECK_CUDA(cudaMalloc(&d_energy_test, total_walker * sizeof(*d_energy_test)));
    // // ---------------------

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time before Wang Landau has started: "
              << elapsed.count() << " seconds" << std::endl;

    double max_factor = exp(1.0);
    int min_cond_interactions = 0;

    int max_newEnergyFlag = 0;

    int block_count = (total_len_histogram + max_threads_per_block - 1) / max_threads_per_block;
    long long wang_landau_counter = 1;

    while (max_factor - exp(options.beta) > 1e-10) // hardcoded precision for abort condition
    {
        wang_landau<<<total_intervals, options.walker_per_interval>>>(
            d_lattice, d_interactions, d_energy, d_start, d_end, d_H, d_logG,
            d_offset_histogram, d_offset_lattice, options.num_iterations,
            options.X, options.Y, options.seed_run, d_factor, d_offset_iter,
            d_expected_energy_spectrum, d_newEnergies, d_foundNewEnergyFlag,
            total_walker, options.beta, d_cond, options.boundary_type,
            walker_per_interactions, options.num_intervals,
            d_offset_energy_spectrum, d_cond_interactions);
        cudaDeviceSynchronize();

        // // TEST BLOCK
        // //-----------
        // std::vector<double> test_energies(total_walker);
        // std::vector<double> test_energies_wl(total_walker);
        // CHECK_CUDA(cudaMemcpy(test_energies_wl.data(), d_energy, total_walker * sizeof(*d_energy), cudaMemcpyDeviceToHost)); // get energies from wl step with energy diff calc
        // calc_energy_periodic_boundary<<<total_intervals, options.walker_per_interval>>>(d_lattice, d_interactions, d_energy_test, d_offset_lattice, options.X, options.Y, total_walker, walker_per_interactions);
        // cudaDeviceSynchronize();
        // CHECK_CUDA(cudaMemcpy(test_energies.data(), d_energy_test, total_walker * sizeof(*d_energy_test), cudaMemcpyDeviceToHost)); // get energies from calc energy function
        // for (int idx = 0; idx < total_walker; idx++)
        // {
        //     if (std::abs(test_energies_wl[idx] - test_energies[idx]) > 1e-10)
        //     {
        //         std::cerr << " walker idx: " << idx << " calc energy: " << test_energies[idx] << " wl calc energy: " << test_energies_wl[idx] << " Diff: " << std::abs(test_energies_wl[idx] - test_energies[idx]) << std::endl;
        //     }
        // }
        // //-----------

        // get max of found new energy flag array to condition break and update the
        // histogramm file with value in new energy array
        thrust::device_ptr<int> d_newEnergyFlag_ptr(d_foundNewEnergyFlag);
        thrust::device_ptr<int> max_newEnergyFlag_ptr = thrust::max_element(d_newEnergyFlag_ptr, d_newEnergyFlag_ptr + total_walker);
        max_newEnergyFlag = *max_newEnergyFlag_ptr;

        // If flag shows new energies get the device arrays containing these to the
        // host, update histogramm file and print error message.
        if (max_newEnergyFlag != 0)
        {
            double h_newEnergies[total_walker];
            int h_newEnergyFlag[total_walker];
            CHECK_CUDA(cudaMemcpy(h_newEnergies, d_newEnergies, total_walker * sizeof(*d_newEnergies), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_newEnergyFlag, d_foundNewEnergyFlag, total_walker * sizeof(*d_foundNewEnergyFlag), cudaMemcpyDeviceToHost));

            // TO DO: Adjust for several interactions
            // handleNewEnergyError(h_newEnergies, h_newEnergyFlag, histogram_file, total_walker);
            std::cerr << "Error: Found new energy:" << std::endl;
            return -1;
        }

        check_histogram<<<total_intervals, options.walker_per_interval>>>(
            d_H, d_logG, d_shared_logG, d_offset_histogram, d_end, d_start,
            d_factor, options.X, options.Y, options.alpha, options.beta,
            d_expected_energy_spectrum, d_len_energy_spectrum, total_walker, d_cond,
            walker_per_interactions, options.num_intervals,
            d_offset_energy_spectrum, d_cond_interactions);
        cudaDeviceSynchronize();

        calc_average_log_g<<<block_count, max_threads_per_block>>>(
            options.num_intervals, d_len_histograms, options.walker_per_interval, d_logG, d_shared_logG,
            d_end, d_start, d_expected_energy_spectrum, d_cond, d_offset_histogram,
            d_offset_energy_spectrum, options.num_interactions, d_offset_shared_logG, d_cond_interactions, total_len_histogram);
        cudaDeviceSynchronize();

        redistribute_g_values<<<block_count, max_threads_per_block>>>(options.num_intervals,
                                                                      d_len_histograms, options.walker_per_interval, d_logG, d_shared_logG,
                                                                      d_end, d_start, d_factor, options.beta, d_expected_energy_spectrum, d_cond,
                                                                      d_offset_histogram, options.num_interactions, d_offset_shared_logG,
                                                                      d_cond_interactions, total_len_histogram);
        cudaDeviceSynchronize();

        CHECK_CUDA(cudaMemset(d_shared_logG, 0, size_shared_log_G * sizeof(*d_shared_logG)));

        reset_d_cond<<<options.num_interactions, options.num_intervals>>>(d_cond, d_factor, total_intervals, options.beta, options.walker_per_interval);
        cudaDeviceSynchronize();

        check_interactions_finished(
            d_cond, d_cond_interactions, d_offset_intervals,
            options.num_intervals, options.num_interactions);
        cudaDeviceSynchronize();

        // get max factor over walkers for abort condition of while loop
        thrust::device_ptr<double> d_factor_ptr(d_factor);
        thrust::device_ptr<double> max_factor_ptr = thrust::max_element(d_factor_ptr, d_factor_ptr + total_walker);
        max_factor = *max_factor_ptr;

        // get flag if any interaction is completely done and thus already ready for dump out
        thrust::device_ptr<int> d_cond_interactions_ptr(d_cond_interactions);
        thrust::device_ptr<int> min_cond_interactions_ptr = thrust::min_element(d_cond_interactions_ptr, d_cond_interactions_ptr + options.num_interactions);
        min_cond_interactions = *min_cond_interactions_ptr;

        if (wang_landau_counter % options.replica_exchange_offset == 0)
        {
            replica_exchange<double><<<total_intervals, options.walker_per_interval>>>(
                d_offset_lattice, d_energy, d_start, d_end, d_indices, d_logG,
                d_offset_histogram, true, options.seed_run, d_offset_iter,
                options.num_intervals, walker_per_interactions, d_cond_interactions);
            cudaDeviceSynchronize();

            replica_exchange<double><<<total_intervals, options.walker_per_interval>>>(
                d_offset_lattice, d_energy, d_start, d_end, d_indices, d_logG,
                d_offset_histogram, false, options.seed_run, d_offset_iter,
                options.num_intervals, walker_per_interactions, d_cond_interactions);
            cudaDeviceSynchronize();
        }

        // results dump out: if a single interaction already finished
        if (min_cond_interactions == -1)
        {
            CHECK_CUDA(cudaMemcpy(h_cond_interactions, d_cond_interactions, options.num_interactions * sizeof(*d_cond_interactions), cudaMemcpyDeviceToHost));

            for (int i = 0; i < options.num_interactions; i++)
            {
                if (h_cond_interactions[i] == -1 && !h_result_is_dumped[i])
                {

                    int offset_of_interaction_histogram;
                    CHECK_CUDA(cudaMemcpy(&offset_of_interaction_histogram, d_offset_histogram + i * (options.num_intervals * options.walker_per_interval), sizeof(*d_offset_histogram), cudaMemcpyDeviceToHost));

                    int len_of_interaction_histogram = len_histogram_int[i];

                    if (offset_of_interaction_histogram + len_of_interaction_histogram > total_len_histogram)
                    {
                        std::cerr << "Error: Copy range exceeds histogram bounds" << std::endl;
                        return -1;
                    }

                    std::vector<double> h_logG(len_of_interaction_histogram);
                    CHECK_CUDA(cudaMemcpy(h_logG.data(), d_logG + offset_of_interaction_histogram, len_of_interaction_histogram * sizeof(*d_logG), cudaMemcpyDeviceToHost));

                    h_result_is_dumped[i] = true; // setting flag such that result for this interaction wont be dumped again

                    std::vector<int> run_start(h_start_int.begin() + i * options.num_intervals, h_start_int.begin() + (i + 1) * options.num_intervals); // stores start energies of intervals of currently handled interaction

                    std::vector<int> run_end(h_end_int.begin() + i * options.num_intervals, h_end_int.begin() + (i + 1) * options.num_intervals); // stores end energies of intervals of currently handled interaction

                    result_handling_stitched_histogram(options, h_logG, run_start, run_end, i, true); // reduced result dump with X, Y needed for rescaling
                    // result_handling(options, h_logG, run_start, run_end, i); // extended result dump
                }
            }
        }
    }

    // Free allocated device memory
    CHECK_CUDA(cudaFree(d_H));
    CHECK_CUDA(cudaFree(d_logG));
    CHECK_CUDA(cudaFree(d_shared_logG));
    CHECK_CUDA(cudaFree(d_offset_shared_logG));
    CHECK_CUDA(cudaFree(d_factor));
    CHECK_CUDA(cudaFree(d_indices));
    CHECK_CUDA(cudaFree(d_lattice));
    CHECK_CUDA(cudaFree(d_interactions));
    CHECK_CUDA(cudaFree(d_energy));
    CHECK_CUDA(cudaFree(d_expected_energy_spectrum));
    CHECK_CUDA(cudaFree(d_newEnergies));
    CHECK_CUDA(cudaFree(d_foundNewEnergyFlag));
    CHECK_CUDA(cudaFree(d_start));
    CHECK_CUDA(cudaFree(d_end));
    CHECK_CUDA(cudaFree(d_offset_energy_spectrum));
    CHECK_CUDA(cudaFree(d_len_energy_spectrum));
    CHECK_CUDA(cudaFree(d_len_histograms));
    CHECK_CUDA(cudaFree(d_offset_histogram));
    CHECK_CUDA(cudaFree(d_offset_lattice));
    CHECK_CUDA(cudaFree(d_offset_iter));

    return 0;
}