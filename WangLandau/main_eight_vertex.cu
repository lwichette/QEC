#include <thrust/extrema.h>
#include <cub/cub.cuh>
#include "./header/cudamacro.h"
#include "./header/wlutils.cuh"

int main(int argc, char **argv)
{

    // Temporary storage size
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    int threads_per_block = 128;

    auto start = std::chrono::high_resolution_clock::now();

    int X, Y;
    int boundary_type = 0;
    int num_interactions = 0;
    int num_iterations = 0;
    int num_intervals = 0;
    int walker_per_interval = 0;
    int histogram_scale = 1;
    int replica_exchange_offset = 0;
    int seed_hist = 0;
    int seed_run = 0;

    float prob_i_err = 0;
    float prob_x_err = 0;
    float prob_y_err = 0;
    float prob_z_err = 0;
    float overlap_decimal = 0;
    float alpha = 0;
    float beta = 0;
    float error_mean = 0;
    float error_variance = 0;

    bool x_horizontal_error = false;
    bool x_vertical_error = false;
    bool z_horizontal_error = false;
    bool z_vertical_error = false;
    bool is_qubit_specific_noise = false;

    int och;

    while (1)
    {
        int option_index = 0;
        static struct option long_options[] = {
            {"alpha", 1, 0, 'a'},
            {"beta", 1, 0, 'b'},
            {"x_horizontal_error", required_argument, 0, 'c'},
            {"x_vertical_error", required_argument, 0, 'd'},
            {"z_horizontal_error", required_argument, 0, 'e'},
            {"z_vertical_error", required_argument, 0, 'f'},
            {"prob_x", required_argument, 0, 'g'},
            {"prob_y", required_argument, 0, 'h'},
            {"prob_z", required_argument, 0, 'i'},
            {"qubit_specific_noise", no_argument, 0, 'k'},
            {"replica_exchange_offsets", 1, 0, 'l'},
            {"num_intervals", required_argument, 0, 'm'},
            {"num_iterations", required_argument, 0, 'n'},
            {"overlap_decimal", 1, 0, 'o'},
            {"seed_histogram", 1, 0, 'p'},
            {"seed_run", 1, 0, 'q'},
            {"hist_scale", required_argument, 0, 'r'},
            {"num_interactions", required_argument, 0, 's'},
            {"error_mean", required_argument, 0, 't'},
            {"error_variance", required_argument, 0, 'u'},
            {"walker_per_interval", required_argument, 0, 'w'},
            {"X", required_argument, 0, 'x'},
            {"Y", required_argument, 0, 'y'},
            {0, 0, 0, 0}};

        och = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:kl:m:n:o:p:q:r:s:t:u:w:x:y:", long_options, &option_index);

        if (och == -1)
            break;
        switch (och)
        {
        case 0: // handles long opts with non-NULL flag field
            break;
        case 'a':
            alpha = atof(optarg);
            break;
        case 'b':
            alpha = atof(optarg);
            break;
        case 'c':
            x_horizontal_error = atoi(optarg) != 0;
            break;
        case 'd':
            x_vertical_error = atoi(optarg) != 0;
            break;
        case 'e':
            z_horizontal_error = atoi(optarg) != 0;
            break;
        case 'f':
            z_vertical_error = atoi(optarg) != 0;
            break;
        case 'g':
            prob_x_err = atof(optarg);
            break;
        case 'h':
            prob_y_err = atof(optarg);
            break;
        case 'i':
            prob_z_err = atof(optarg);
            break;
        case 'k': // qubit_specific_noise flag
            is_qubit_specific_noise = true;
            std::cout << "Qubit-specific noise flag is set." << std::endl;
            break;
        case 'l':
            replica_exchange_offset = atoi(optarg);
            break;
        case 'm':
            num_intervals = atoi(optarg);
            break;
        case 'n':
            num_iterations = atoi(optarg);
            break;
        case 'o':
            overlap_decimal = atof(optarg);
            break;
        case 'p':
            seed_hist = atoi(optarg);
            break;
        case 'q':
            seed_run = atoi(optarg);
            break;
        case 'r':
            histogram_scale = atoi(optarg);
            break;
        case 's':
            num_interactions = atoi(optarg);
            break;
        case 't':
            error_mean = atof(optarg);
            break;
        case 'u':
            error_variance = atof(optarg);
            break;
        case 'w':
            walker_per_interval = atoi(optarg);
            break;
        case 'x':
            X = atoi(optarg);
            break;
        case 'y':
            Y = atoi(optarg);
            break;
        case '?':
            exit(EXIT_FAILURE);
        default:
            fprintf(stderr, "unknown option: %c\n", och);
            exit(EXIT_FAILURE);
        }
    }

    const int total_walker = num_interactions * num_intervals * walker_per_interval;
    const int total_intervals = num_interactions * num_intervals;
    const int walker_per_interactions = num_intervals * walker_per_interval;

    // Ŕead histograms for all different seeds
    std::vector<signed char> h_expected_energy_spectrum;
    std::vector<int> h_len_energy_spectrum;
    std::vector<int> h_offset_energy_spectrum;
    std::vector<int> E_min;
    std::vector<int> E_max;
    int total_len_energy_spectrum = 0;

    for (int i = 0; i < num_interactions; i++)
    {
        std::string hist_path = eight_vertex_histogram_path(is_qubit_specific_noise, error_mean, error_variance, X, Y, seed_hist + i, x_horizontal_error, x_vertical_error, z_horizontal_error, z_vertical_error, prob_x_err, prob_y_err, prob_z_err);

        std::vector<signed char> energy_spectrum = read_histogram(hist_path, E_min, E_max);

        h_expected_energy_spectrum.insert(h_expected_energy_spectrum.end(),
                                          energy_spectrum.begin(),
                                          energy_spectrum.end());
        h_offset_energy_spectrum.push_back(total_len_energy_spectrum);
        total_len_energy_spectrum += energy_spectrum.size();
        h_len_energy_spectrum.push_back(energy_spectrum.size());
    }
    int *d_offset_energy_spectrum, *d_len_energy_spectrum;
    CHECK_CUDA(cudaMalloc(&d_offset_energy_spectrum, num_interactions * sizeof(*d_offset_energy_spectrum)));
    CHECK_CUDA(cudaMemcpy(d_offset_energy_spectrum, h_offset_energy_spectrum.data(), num_interactions * sizeof(*d_offset_energy_spectrum), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_len_energy_spectrum, num_interactions * sizeof(*d_len_energy_spectrum)));
    CHECK_CUDA(cudaMemcpy(d_len_energy_spectrum, h_len_energy_spectrum.data(), num_interactions * sizeof(*d_len_energy_spectrum), cudaMemcpyHostToDevice));

    // Generate intervals for all different energy spectrums
    std::vector<int> h_end_int;
    std::vector<int> h_start_int;
    std::vector<int> len_histogram_int;
    std::vector<int> len_interval_int;

    long long total_len_histogram = 0;

    for (int i = 0; i < num_interactions; i++)
    {
        IntervalResult run_result = generate_intervals(
            E_min[i], E_max[i], num_intervals,
            walker_per_interval, overlap_decimal);

        h_end_int.insert(h_end_int.end(), run_result.h_end.begin(),
                         run_result.h_end.end());

        h_start_int.insert(h_start_int.end(), run_result.h_start.begin(),
                           run_result.h_start.end());

        len_histogram_int.push_back(run_result.len_histogram_over_all_walkers);
        len_interval_int.push_back(run_result.len_interval);
        total_len_histogram += run_result.len_histogram_over_all_walkers;
    }

    int *d_len_histograms;
    CHECK_CUDA(cudaMalloc(&d_len_histograms, num_interactions * sizeof(*d_len_histograms)))
    CHECK_CUDA(cudaMemcpy(d_len_histograms, len_histogram_int.data(), num_interactions * sizeof(*d_len_histograms), cudaMemcpyHostToDevice));

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
    for (int i = 0; i < num_interactions; i++)
    {
        for (int j = 0; j < num_intervals; j++)
        {
            h_offset_shared_log_G.push_back(size_shared_log_G);
            size_shared_log_G += (h_end_int[i * num_intervals + j] - h_start_int[i * num_intervals + j] + 1);
        }
    }

    double *d_shared_logG;
    CHECK_CUDA(cudaMalloc(&d_shared_logG, size_shared_log_G * sizeof(*d_shared_logG)));
    CHECK_CUDA(cudaMemset(d_shared_logG, 0, size_shared_log_G * sizeof(*d_shared_logG)));

    long long *d_offset_shared_logG;
    CHECK_CUDA(cudaMalloc(&d_offset_shared_logG, total_intervals * sizeof(*d_offset_shared_logG)));
    CHECK_CUDA(cudaMemcpy(d_offset_shared_logG, h_offset_shared_log_G.data(), total_intervals * sizeof(*d_offset_shared_logG), cudaMemcpyHostToDevice));

    std::vector<double> h_factor(total_walker, exp(1.0));

    double *d_factor;
    CHECK_CUDA(cudaMalloc(&d_factor, total_walker * sizeof(*d_factor)));
    CHECK_CUDA(cudaMemcpy(d_factor, h_factor.data(), total_walker * sizeof(*d_factor), cudaMemcpyHostToDevice));

    // Indices used for replica exchange later
    int *d_indices;
    CHECK_CUDA(cudaMalloc(&d_indices, total_walker * sizeof(*d_indices)));

    // interactions
    double *d_interactions_r, *d_interactions_b, *d_interactions_down_four_body, *d_interactions_right_four_body; // single set of interaction arrays for all walkers to share
    CHECK_CUDA(cudaMalloc(&d_interactions_r, num_interactions * 2 * X * Y * sizeof(*d_interactions_r)));
    CHECK_CUDA(cudaMalloc(&d_interactions_b, num_interactions * 2 * X * Y * sizeof(*d_interactions_b)));
    CHECK_CUDA(cudaMalloc(&d_interactions_down_four_body, num_interactions * X * Y * sizeof(*d_interactions_down_four_body)));
    CHECK_CUDA(cudaMalloc(&d_interactions_right_four_body, num_interactions * X * Y * sizeof(*d_interactions_right_four_body)));

    // declare b and r lattice
    signed char *d_lattice_r, *d_lattice_b;
    CHECK_CUDA(cudaMalloc(&d_lattice_b, total_walker * X * Y * sizeof(*d_lattice_b)));
    CHECK_CUDA(cudaMalloc(&d_lattice_r, total_walker * X * Y * sizeof(*d_lattice_r)));

    // Hamiltonian of lattices
    double *d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, total_walker * sizeof(*d_energy)));

    // Binary indicator of energies were found or not
    signed char *d_expected_energy_spectrum;
    CHECK_CUDA(cudaMalloc(&d_expected_energy_spectrum, h_expected_energy_spectrum.size() * sizeof(*d_expected_energy_spectrum)));
    CHECK_CUDA(cudaMemcpy(d_expected_energy_spectrum, h_expected_energy_spectrum.data(), h_expected_energy_spectrum.size() * sizeof(*d_expected_energy_spectrum), cudaMemcpyHostToDevice));

    // To catch energies which are outside of expected spectrum
    double *d_newEnergies;
    int *d_foundNewEnergyFlag;
    CHECK_CUDA(cudaMalloc(&d_newEnergies, total_walker * sizeof(*d_newEnergies)));
    CHECK_CUDA(cudaMalloc(&d_foundNewEnergyFlag, total_walker * sizeof(*d_foundNewEnergyFlag)));

    signed char *d_cond;
    CHECK_CUDA(cudaMalloc(&d_cond, total_intervals * sizeof(*d_cond)));
    CHECK_CUDA(cudaMemset(d_cond, 0, total_intervals * sizeof(*d_cond)));

    int *d_cond_interactions;
    CHECK_CUDA(cudaMalloc(&d_cond_interactions, num_interactions * sizeof(*d_cond_interactions)));
    CHECK_CUDA(cudaMemset(d_cond_interactions, 0, num_interactions * sizeof(*d_cond_interactions)));

    // host storage for the is finished flag per interaction stored in d_cond_interaction
    int *h_cond_interactions;
    h_cond_interactions = (int *)malloc(num_interactions * sizeof(*h_cond_interactions));

    bool *h_result_is_dumped;
    h_result_is_dumped = (bool *)calloc(num_interactions, sizeof(*h_result_is_dumped));

    // Offsets
    // lets write some comments about what these offsets are
    int *d_offset_histogram_per_walker, *d_offset_lattice_per_walker;
    unsigned long long *d_offset_iterator_per_walker;
    CHECK_CUDA(cudaMalloc(&d_offset_histogram_per_walker, total_walker * sizeof(*d_offset_histogram_per_walker)));
    CHECK_CUDA(cudaMalloc(&d_offset_lattice_per_walker, total_walker * sizeof(*d_offset_lattice_per_walker)));
    CHECK_CUDA(cudaMalloc(&d_offset_iterator_per_walker, total_walker * sizeof(*d_offset_iterator_per_walker)));
    CHECK_CUDA(cudaMemset(d_offset_iterator_per_walker, 0, total_walker * sizeof(*d_offset_iterator_per_walker)));

    std::vector<int> h_offset_intervals(num_interactions + 1);

    for (int i = 0; i < num_interactions; i++)
    {
        h_offset_intervals[i] = i * num_intervals;
    }

    h_offset_intervals[num_interactions] = total_intervals;

    int *d_offset_intervals;
    CHECK_CUDA(cudaMalloc(&d_offset_intervals, h_offset_intervals.size() * sizeof(*d_offset_intervals)));
    CHECK_CUDA(cudaMemcpy(d_offset_intervals, h_offset_intervals.data(), h_offset_intervals.size() * sizeof(*d_offset_intervals), cudaMemcpyHostToDevice));

    init_offsets_lattice<<<total_intervals, walker_per_interval>>>(d_offset_lattice_per_walker, X, Y, total_walker);
    init_offsets_histogramm<<<total_intervals, walker_per_interval>>>(d_offset_histogram_per_walker, d_start, d_end, d_len_histograms, num_intervals, total_walker);
    init_indices<<<total_intervals, walker_per_interval>>>(d_indices, total_walker);
    cudaDeviceSynchronize();

    // Load interactions from init
    std::vector<double> h_interactions_r;
    std::vector<double> h_interactions_b;
    std::vector<double> h_interactions_four_body_down;
    std::vector<double> h_interactions_four_body_right;

    for (int i = 0; i < num_interactions; i++)
    {
        std::string run_int_path_b = eight_vertex_interaction_path(is_qubit_specific_noise, error_mean, error_variance, X, Y, seed_hist + i, x_horizontal_error, x_vertical_error, z_horizontal_error, z_vertical_error, "b", prob_x_err, prob_y_err, prob_z_err);
        std::string run_int_path_r = eight_vertex_interaction_path(is_qubit_specific_noise, error_mean, error_variance, X, Y, seed_hist + i, x_horizontal_error, x_vertical_error, z_horizontal_error, z_vertical_error, "r", prob_x_err, prob_y_err, prob_z_err);
        std::string run_int_path_four_body_down = eight_vertex_interaction_path(is_qubit_specific_noise, error_mean, error_variance, X, Y, seed_hist + i, x_horizontal_error, x_vertical_error, z_horizontal_error, z_vertical_error, "four_body_down", prob_x_err, prob_y_err, prob_z_err);
        std::string run_int_path_four_body_right = eight_vertex_interaction_path(is_qubit_specific_noise, error_mean, error_variance, X, Y, seed_hist + i, x_horizontal_error, x_vertical_error, z_horizontal_error, z_vertical_error, "four_body_right", prob_x_err, prob_y_err, prob_z_err);

        std::vector<double> run_interactions_r;
        std::vector<double> run_interactions_b;
        std::vector<double> run_interactions_four_body_down;
        std::vector<double> run_interactions_four_body_right;

        read(run_interactions_r, run_int_path_r);
        read(run_interactions_b, run_int_path_b);
        read(run_interactions_four_body_down, run_int_path_four_body_down);
        read(run_interactions_four_body_right, run_int_path_four_body_right);

        h_interactions_r.insert(h_interactions_r.end(), run_interactions_r.begin(), run_interactions_r.end());
        h_interactions_b.insert(h_interactions_b.end(), run_interactions_b.begin(), run_interactions_b.end());
        h_interactions_four_body_down.insert(h_interactions_four_body_down.end(), run_interactions_four_body_down.begin(), run_interactions_four_body_down.end());
        h_interactions_four_body_right.insert(h_interactions_four_body_right.end(), run_interactions_four_body_right.begin(), run_interactions_four_body_right.end());
    }

    // Load lattices from init
    std::vector<signed char> h_lattice_r;
    std::vector<signed char> h_lattice_b;

    for (int i = 0; i < num_interactions; i++)
    {
        std::vector<int> run_start(h_start_int.begin() + i * num_intervals,
                                   h_start_int.begin() + i * num_intervals + num_intervals);

        std::vector<int> run_end(h_end_int.begin() + i * num_intervals,
                                 h_end_int.begin() + i * num_intervals + num_intervals);

        std::map<std::string, std::vector<signed char>> run_lattices = get_lattice_with_pre_run_result_eight_vertex(
            is_qubit_specific_noise, error_mean, error_variance, x_horizontal_error, x_vertical_error,
            z_horizontal_error, z_vertical_error, X, Y, run_start, run_end, num_intervals, walker_per_interval,
            seed_hist, prob_x_err, prob_y_err, prob_z_err);

        // Access the "r" and "b" lattices from the map
        std::vector<signed char> run_lattice_r = run_lattices["r"];
        std::vector<signed char> run_lattice_b = run_lattices["b"];

        h_lattice_r.insert(h_lattice_r.end(), run_lattice_r.begin(), run_lattice_r.end());
        h_lattice_b.insert(h_lattice_b.end(), run_lattice_b.begin(), run_lattice_b.end());
    }

    // Init device arrays with host lattices and interactions
    CHECK_CUDA(cudaMemcpy(d_interactions_r, h_interactions_r.data(), num_interactions * X * Y * 2 * sizeof(*d_interactions_r), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_interactions_b, h_interactions_b.data(), num_interactions * X * Y * 2 * sizeof(*d_interactions_b), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_interactions_down_four_body, h_interactions_four_body_down.data(), num_interactions * X * Y * sizeof(*d_interactions_down_four_body), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_interactions_right_four_body, h_interactions_four_body_right.data(), num_interactions * X * Y * sizeof(*d_interactions_right_four_body), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lattice_r, h_lattice_r.data(), total_walker * X * Y * sizeof(*d_lattice_r), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lattice_b, h_lattice_b.data(), total_walker * X * Y * sizeof(*d_lattice_b), cudaMemcpyHostToDevice));

    // block counts
    int blocks_qubit_x_thread = (num_interactions * 2 * X * Y + threads_per_block - 1) / threads_per_block;
    int blocks_spins_single_color_x_thread = (total_walker * X * Y + threads_per_block - 1) / threads_per_block;
    int blocks_total_walker_x_thread = (total_walker + threads_per_block - 1) / threads_per_block;
    int blocks_total_intervals_x_thread = (total_intervals + threads_per_block - 1) / threads_per_block;
    int blocks_toal_len_histogram_x_thread = (total_len_histogram + threads_per_block - 1) / threads_per_block;

    // init the energy array
    calc_energy_eight_vertex<<<blocks_total_walker_x_thread, threads_per_block>>>(d_energy, d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, 2 * X * Y, X, 2 * Y, total_walker, walker_per_interactions);
    cudaDeviceSynchronize();

    // // TEST BLOC
    // // ------------
    // std::vector<double> h_energy(total_walker);
    // std::vector<signed char> h_expected_energy_spectrum_copy(h_expected_energy_spectrum.size());

    // // Copy the device data to the host
    // CHECK_CUDA(cudaMemcpy(h_energy.data(), d_energy, total_walker * sizeof(double), cudaMemcpyDeviceToHost));
    // CHECK_CUDA(cudaMemcpy(h_expected_energy_spectrum_copy.data(), d_expected_energy_spectrum, h_expected_energy_spectrum.size() * sizeof(signed char), cudaMemcpyDeviceToHost));

    // // Print the energy array
    // std::cout << "d_energy:" << std::endl;
    // for (size_t i = 0; i < total_walker; ++i)
    // {
    //     int interaction_id = i / walker_per_interactions;
    //     int interval_id = (i % walker_per_interactions) / walker_per_interval;
    //     std::cout << h_energy[i] << " for interval_start: " << h_start_int[interaction_id * num_intervals + interval_id] << " interval_end: " << h_end_int[interaction_id * num_intervals + interval_id] << std::endl;
    // }
    // std::cout << std::endl;

    // // Print the expected energy spectrum array
    // std::cout << "d_expected_energy_spectrum:" << std::endl;
    // for (size_t i = 0; i < h_expected_energy_spectrum_copy.size(); ++i)
    // {
    //     std::cout << static_cast<int>(h_expected_energy_spectrum_copy[i]) << " ";
    // }
    // std::cout << std::endl;
    // // ------------

    // check if read of lattices matches expected energy range of intervals
    check_energy_ranges<double><<<total_intervals, walker_per_interval>>>(d_energy, d_start, d_end, total_walker);
    cudaDeviceSynchronize();

    return 0;

    // control flow variables
    double max_factor = exp(1.0);
    int min_cond_interactions = 0;
    int max_newEnergyFlag = 0;
    long long wang_landau_counter = 1;

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time before Wang Landau has started: "
              << elapsed.count() << " seconds" << std::endl;

    // // TEST BLOCK
    // for (int w = 0; w < total_walker; w++)
    // {
    //     int interval_id = w / walker_per_interval;
    //     std::cout << "E_start: " << h_start_int[interval_id] << " E_end: " << h_end_int[interval_id] << ": ";
    //     for (int s = 0; s < X * Y; s++)
    //     {
    //         std::cout << static_cast<int>(h_lattice_r[w * X * Y + s]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // for (int i = 0; i < num_interactions; i++)
    // {
    //     int offset_interactions = i * 2 * X * Y;       // for interactions closed on a single colored sublattice
    //     int offset_four_body_interactions = i * X * Y; // for interactions closed on a single colored sublattice
    //     int offset_lattice = i * num_intervals * X * Y;
    //     int offset_energies = i * num_intervals;

    //     create_directory("TEST_READ/interactions");

    //     write(h_interactions_b.data() + offset_interactions, "TEST_READ/interactions/interactions_b", 2 * Y, X, 1, false);
    //     write(h_interactions_r.data() + offset_interactions, "TEST_READ/interactions/interactions_r", 2 * Y, X, 1, false);
    //     write(h_interactions_four_body_right.data() + offset_four_body_interactions, "TEST_READ/interactions/interactions_four_body_right", Y, X, 1, false);
    //     write(h_interactions_four_body_down.data() + offset_four_body_interactions, "TEST_READ/interactions/interactions_four_body_down", Y, X, 1, false);
    // }
    // //

    /*
    ----------------------------------------------
    ------------ Actual WL Starts Now ------------
    ----------------------------------------------
    */
    // while (max_factor > exp(options.beta))
    // {
    wang_landau_eight_vertex<<<blocks_total_walker_x_thread, threads_per_block>>>(
        d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, d_energy, d_start, d_end, d_H,
        d_logG, d_offset_histogram_per_walker, d_offset_lattice_per_walker, num_iterations, X, Y,
        seed_run, d_factor, d_offset_iterator_per_walker, d_expected_energy_spectrum, d_newEnergies, d_foundNewEnergyFlag,
        total_walker, beta, d_cond, walker_per_interactions, num_intervals,
        d_offset_energy_spectrum, d_cond_interactions);

    cudaDeviceSynchronize();
    // }

    return 0;
}