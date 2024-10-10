#include <thrust/extrema.h>
#include <cub/cub.cuh>
#include "./header/cudamacro.h"
#include "./header/wlutils.cuh"

int main(int argc, char **argv)
{

    int threads_per_block = 128;

    auto start = std::chrono::high_resolution_clock::now();

    Options options;

    float prob_x_err = 0;
    float prob_y_err = 0;
    float prob_z_err = 0;
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
            {"num_interactions", required_argument, 0, 's'},
            {"error_mean", required_argument, 0, 't'},
            {"error_variance", required_argument, 0, 'u'},
            {"task_id", required_argument, 0, 'v'},
            {"walker_per_interval", required_argument, 0, 'w'},
            {"X", required_argument, 0, 'x'},
            {"Y", required_argument, 0, 'y'},
            {0, 0, 0, 0}};

        och = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:kl:m:n:o:p:q:s:t:u:v:w:x:y:", long_options, &option_index);

        if (och == -1)
            break;
        switch (och)
        {
        case 0: // handles long opts with non-NULL flag field
            break;
        case 'a':
            options.alpha = atof(optarg);
            break;
        case 'b':
            options.beta = atof(optarg);
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
            options.replica_exchange_offset = atoi(optarg);
            break;
        case 'm':
            options.num_intervals = atoi(optarg);
            break;
        case 'n':
            options.num_iterations = atoi(optarg);
            break;
        case 'o':
            options.overlap_decimal = atof(optarg);
            break;
        case 'p':
            options.seed_histogram = atoi(optarg);
            break;
        case 'q':
            options.seed_run = atoi(optarg);
            break;
        case 's':
            options.num_interactions = atoi(optarg);
            break;
        case 't':
            error_mean = atof(optarg);
            break;
        case 'u':
            error_variance = atof(optarg);
            break;
        case 'v':
            options.task_id = atoi(optarg);
            break;
        case 'w':
            options.walker_per_interval = atoi(optarg);
            break;
        case 'x':
            options.X = atoi(optarg);
            break;
        case 'y':
            options.Y = atoi(optarg);
            break;
        case '?':
            exit(EXIT_FAILURE);
        default:
            fprintf(stderr, "unknown option: %c\n", och);
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
    std::vector<int> E_min;
    std::vector<int> E_max;
    int total_len_energy_spectrum = 0;

    for (int i = 0; i < options.num_interactions; i++)
    {
        std::string hist_path = eight_vertex_histogram_path(is_qubit_specific_noise, error_mean, error_variance, options.X, options.Y, options.seed_histogram + i, x_horizontal_error, x_vertical_error, z_horizontal_error, z_vertical_error, prob_x_err, prob_y_err, prob_z_err, options.task_id);

        std::vector<signed char> energy_spectrum = read_histogram(hist_path, E_min, E_max);

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
            E_min[i], E_max[i], options.num_intervals,
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

    std::vector<double> h_factor(total_walker, exp(1.0));

    double *d_factor;
    CHECK_CUDA(cudaMalloc(&d_factor, total_walker * sizeof(*d_factor)));
    CHECK_CUDA(cudaMemcpy(d_factor, h_factor.data(), total_walker * sizeof(*d_factor), cudaMemcpyHostToDevice));

    // Indices used for replica exchange later
    int *d_indices;
    CHECK_CUDA(cudaMalloc(&d_indices, total_walker * sizeof(*d_indices)));

    // interactions
    double *d_interactions_r, *d_interactions_b, *d_interactions_down_four_body, *d_interactions_right_four_body; // single set of interaction arrays for all walkers to share
    CHECK_CUDA(cudaMalloc(&d_interactions_r, options.num_interactions * 2 * options.X * options.Y * sizeof(*d_interactions_r)));
    CHECK_CUDA(cudaMalloc(&d_interactions_b, options.num_interactions * 2 * options.X * options.Y * sizeof(*d_interactions_b)));
    CHECK_CUDA(cudaMalloc(&d_interactions_down_four_body, options.num_interactions * options.X * options.Y * sizeof(*d_interactions_down_four_body)));
    CHECK_CUDA(cudaMalloc(&d_interactions_right_four_body, options.num_interactions * options.X * options.Y * sizeof(*d_interactions_right_four_body)));

    // declare b and r lattice
    signed char *d_lattice_r, *d_lattice_b;
    CHECK_CUDA(cudaMalloc(&d_lattice_b, total_walker * options.X * options.Y * sizeof(*d_lattice_b)));
    CHECK_CUDA(cudaMalloc(&d_lattice_r, total_walker * options.X * options.Y * sizeof(*d_lattice_r)));

    // Hamiltonian of lattices
    double *d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, total_walker * sizeof(*d_energy)));

    // // TEST BLOCK
    // // ---------------------
    double *d_energy_test;
    CHECK_CUDA(cudaMalloc(&d_energy_test, total_walker * sizeof(*d_energy_test)));
    // // ---------------------

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
    CHECK_CUDA(cudaMalloc(&d_cond_interactions, options.num_interactions * sizeof(*d_cond_interactions)));
    CHECK_CUDA(cudaMemset(d_cond_interactions, 0, options.num_interactions * sizeof(*d_cond_interactions)));

    // host storage for the is finished flag per interaction stored in d_cond_interaction
    int *h_cond_interactions;
    h_cond_interactions = (int *)malloc(options.num_interactions * sizeof(*h_cond_interactions));

    bool *h_result_is_dumped;
    h_result_is_dumped = (bool *)calloc(options.num_interactions, sizeof(*h_result_is_dumped));

    // Offsets
    // lets write some comments about what these offsets are
    int *d_offset_histogram_per_walker, *d_offset_lattice_per_walker;
    unsigned long long *d_offset_iterator_per_walker;
    CHECK_CUDA(cudaMalloc(&d_offset_histogram_per_walker, total_walker * sizeof(*d_offset_histogram_per_walker)));
    CHECK_CUDA(cudaMalloc(&d_offset_lattice_per_walker, total_walker * sizeof(*d_offset_lattice_per_walker)));
    CHECK_CUDA(cudaMalloc(&d_offset_iterator_per_walker, total_walker * sizeof(*d_offset_iterator_per_walker)));
    CHECK_CUDA(cudaMemset(d_offset_iterator_per_walker, 0, total_walker * sizeof(*d_offset_iterator_per_walker)));

    std::vector<int> h_offset_intervals(options.num_interactions + 1);

    for (int i = 0; i < options.num_interactions; i++)
    {
        h_offset_intervals[i] = i * options.num_intervals;
    }

    h_offset_intervals[options.num_interactions] = total_intervals;

    int *d_offset_intervals;
    CHECK_CUDA(cudaMalloc(&d_offset_intervals, h_offset_intervals.size() * sizeof(*d_offset_intervals)));
    CHECK_CUDA(cudaMemcpy(d_offset_intervals, h_offset_intervals.data(), h_offset_intervals.size() * sizeof(*d_offset_intervals), cudaMemcpyHostToDevice));

    init_offsets_lattice<<<total_intervals, options.walker_per_interval>>>(d_offset_lattice_per_walker, options.X, options.Y, total_walker);
    init_offsets_histogramm<<<total_intervals, options.walker_per_interval>>>(d_offset_histogram_per_walker, d_start, d_end, d_len_histograms, options.num_intervals, total_walker);
    init_indices<<<total_intervals, options.walker_per_interval>>>(d_indices, total_walker);
    cudaDeviceSynchronize();

    // Load interactions from init
    std::vector<double> h_interactions_r;
    std::vector<double> h_interactions_b;
    std::vector<double> h_interactions_four_body_down;
    std::vector<double> h_interactions_four_body_right;

    for (int i = 0; i < options.num_interactions; i++)
    {
        std::string run_int_path_b = eight_vertex_interaction_path(is_qubit_specific_noise, error_mean, error_variance, options.X, options.Y, options.seed_histogram + i, x_horizontal_error, x_vertical_error, z_horizontal_error, z_vertical_error, "b", prob_x_err, prob_y_err, prob_z_err, options.task_id);
        std::string run_int_path_r = eight_vertex_interaction_path(is_qubit_specific_noise, error_mean, error_variance, options.X, options.Y, options.seed_histogram + i, x_horizontal_error, x_vertical_error, z_horizontal_error, z_vertical_error, "r", prob_x_err, prob_y_err, prob_z_err, options.task_id);
        std::string run_int_path_four_body_down = eight_vertex_interaction_path(is_qubit_specific_noise, error_mean, error_variance, options.X, options.Y, options.seed_histogram + i, x_horizontal_error, x_vertical_error, z_horizontal_error, z_vertical_error, "four_body_down", prob_x_err, prob_y_err, prob_z_err, options.task_id);
        std::string run_int_path_four_body_right = eight_vertex_interaction_path(is_qubit_specific_noise, error_mean, error_variance, options.X, options.Y, options.seed_histogram + i, x_horizontal_error, x_vertical_error, z_horizontal_error, z_vertical_error, "four_body_right", prob_x_err, prob_y_err, prob_z_err, options.task_id);

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

        // // TEST BLOCK
        // //-----------
        // std::string error_string = std::to_string(x_horizontal_error) + std::to_string(x_vertical_error) + std::to_string(z_horizontal_error) + std::to_string(z_vertical_error);
        // std::string path = "test/interactions/" + std::to_string(options.seed_histogram) + "_" + std::to_string(i) + "_" + error_string;
        // create_directory(path);
        // write(run_interactions_b.data(), path + "/interactions_b", 2 * options.Y, options.X, 1, false);
        // write(run_interactions_r.data(), path + "/interactions_r", 2 * options.Y, options.X, 1, false);
        // write(run_interactions_four_body_right.data(), path + "/interactions_four_body_right", options.Y, options.X, 1, false);
        // write(run_interactions_four_body_down.data(), path + "/interactions_four_body_down", options.Y, options.X, 1, false);
        // //-----------
    }

    // Load lattices from init
    std::vector<signed char> h_lattice_r;
    std::vector<signed char> h_lattice_b;

    for (int i = 0; i < options.num_interactions; i++)
    {
        std::vector<int> run_start(h_start_int.begin() + i * options.num_intervals,
                                   h_start_int.begin() + i * options.num_intervals + options.num_intervals);

        std::vector<int> run_end(h_end_int.begin() + i * options.num_intervals,
                                 h_end_int.begin() + i * options.num_intervals + options.num_intervals);

        std::map<std::string, std::vector<signed char>> run_lattices = get_lattice_with_pre_run_result_eight_vertex(
            is_qubit_specific_noise, error_mean, error_variance, x_horizontal_error, x_vertical_error,
            z_horizontal_error, z_vertical_error, options.X, options.Y, run_start, run_end, options.num_intervals, options.walker_per_interval,
            options.seed_histogram + i, prob_x_err, prob_y_err, prob_z_err, options.task_id);

        // Access the "r" and "b" lattices from the map
        std::vector<signed char> run_lattice_r = run_lattices["r"];
        std::vector<signed char> run_lattice_b = run_lattices["b"];

        h_lattice_r.insert(h_lattice_r.end(), run_lattice_r.begin(), run_lattice_r.end());
        h_lattice_b.insert(h_lattice_b.end(), run_lattice_b.begin(), run_lattice_b.end());

        // // TEST BLOCK
        // //-----------
        // if (i == 1)
        // {
        //     std::cout << "r lattice 6th walker second interaction:" << std::endl;
        //     for (int x = 0; x < options.X; x++)
        //     {
        //         for (int y = 0; y < options.Y; y++)
        //         {
        //             std::cout << static_cast<int>(run_lattice_r[5 * options.X * options.Y + x * options.Y + y]);
        //         }
        //         std::cout << std::endl;
        //     }
        //     std::cout << "b lattice 6th walker second interaction:" << std::endl;
        //     for (int x = 0; x < options.X; x++)
        //     {
        //         for (int y = 0; y < options.Y; y++)
        //         {
        //             std::cout << static_cast<int>(run_lattice_b[5 * options.X * options.Y + x * options.Y + y]);
        //         }
        //         std::cout << std::endl;
        //     }
        // }

        // std::string error_string = std::to_string(x_horizontal_error) + std::to_string(x_vertical_error) + std::to_string(z_horizontal_error) + std::to_string(z_vertical_error);
        // create_directory("test/lattice/interaction_" + std::to_string(options.seed_histogram) + "_" + std::to_string(i) + "_" + error_string);
        // write(run_lattice_b.data(), "test/lattice/interaction_" + std::to_string(options.seed_histogram) + "_" + std::to_string(i) + "_" + error_string + "/lattice_b", options.Y, options.X, walker_per_interactions, true);
        // write(run_lattice_r.data(), "test/lattice/interaction_" + std::to_string(options.seed_histogram) + "_" + std::to_string(i) + "_" + error_string + "/lattice_r", options.Y, options.X, walker_per_interactions, true);
        // //-----------
    }

    // Init device arrays with host lattices and interactions
    CHECK_CUDA(cudaMemcpy(d_interactions_r, h_interactions_r.data(), options.num_interactions * options.X * options.Y * 2 * sizeof(*d_interactions_r), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_interactions_b, h_interactions_b.data(), options.num_interactions * options.X * options.Y * 2 * sizeof(*d_interactions_b), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_interactions_down_four_body, h_interactions_four_body_down.data(), options.num_interactions * options.X * options.Y * sizeof(*d_interactions_down_four_body), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_interactions_right_four_body, h_interactions_four_body_right.data(), options.num_interactions * options.X * options.Y * sizeof(*d_interactions_right_four_body), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lattice_r, h_lattice_r.data(), total_walker * options.X * options.Y * sizeof(*d_lattice_r), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lattice_b, h_lattice_b.data(), total_walker * options.X * options.Y * sizeof(*d_lattice_b), cudaMemcpyHostToDevice));

    // block counts
    int blocks_total_walker_x_thread = (total_walker + threads_per_block - 1) / threads_per_block;

    // init the energy array
    calc_energy_eight_vertex<<<blocks_total_walker_x_thread, threads_per_block>>>(d_energy, d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, 2 * options.X * options.Y, options.X, 2 * options.Y, total_walker, walker_per_interactions, d_offset_lattice_per_walker);
    cudaDeviceSynchronize();

    // check if read of lattices matches expected energy range of intervals
    check_energy_ranges<double><<<total_intervals, options.walker_per_interval>>>(d_energy, d_start, d_end, total_walker);
    cudaDeviceSynchronize();

    // control flow variables
    double max_factor = exp(1.0);
    int min_cond_interactions = 0;
    int max_newEnergyFlag = 0;
    long long wang_landau_counter = 1;

    int block_count = (total_len_histogram + threads_per_block - 1) / threads_per_block;

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time before Wang Landau with "
              << "X=" << options.X << " Y=" << options.Y << " probX=" << prob_x_err << " probY=" << prob_y_err << " probZ=" << prob_z_err << " xh=" << x_horizontal_error << " xv=" << x_vertical_error << " zh=" << z_horizontal_error << " zv=" << z_vertical_error
              << " has started: "
              << elapsed.count() << " seconds" << std::endl;

    // /*
    // ----------------------------------------------
    // ------------ Actual WL Starts Now ------------
    // ----------------------------------------------
    // */

    // while (max_factor - exp(options.beta) > 1e-10) // set precision for abort condition
    for (int iterator = 0; iterator < 1000; iterator++)
    {
        std::cout << iterator << " " << max_factor << " : " << std::setprecision(15) << exp(options.beta) << std::endl;
        // std::cout << max_factor << std::setprecision(7) << std::endl;
        wang_landau_eight_vertex<<<blocks_total_walker_x_thread, threads_per_block>>>(
            d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, d_energy, d_start, d_end, d_H,
            d_logG, d_offset_histogram_per_walker, d_offset_lattice_per_walker, options.num_iterations, options.X, 2 * options.Y,
            options.seed_run, d_factor, d_offset_iterator_per_walker, d_expected_energy_spectrum, d_newEnergies, d_foundNewEnergyFlag,
            total_walker, options.beta, d_cond, walker_per_interactions, options.num_intervals,
            d_offset_energy_spectrum, d_cond_interactions, options.walker_per_interval);
        cudaDeviceSynchronize();

        // // TEST BLOCK
        // //-----------
        // std::vector<double> test_energies(total_walker);
        // std::vector<double> test_energies_wl(total_walker);
        // CHECK_CUDA(cudaMemcpy(test_energies_wl.data(), d_energy, total_walker * sizeof(*d_energy), cudaMemcpyDeviceToHost)); // get energies from wl step with energy diff calc
        // calc_energy_eight_vertex<<<blocks_total_walker_x_thread, threads_per_block>>>(d_energy_test, d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, 2 * options.X * options.Y, options.X, 2 * options.Y, total_walker, walker_per_interactions, d_offset_lattice_per_walker);
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
            int h_newEnergies[total_walker];
            int h_newEnergyFlag[total_walker];
            CHECK_CUDA(cudaMemcpy(h_newEnergies, d_newEnergies, total_walker * sizeof(*d_newEnergies), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_newEnergyFlag, d_foundNewEnergyFlag, total_walker * sizeof(*d_foundNewEnergyFlag), cudaMemcpyDeviceToHost));

            // TO DO: Adjust for several interactions
            // handleNewEnergyError(h_newEnergies, h_newEnergyFlag, histogram_file, total_walker);
            std::cerr << "Error: Found new energy:" << std::endl;
            return -1;
        }

        check_histogram<<<total_intervals, options.walker_per_interval>>>(
            d_H, d_logG, d_shared_logG, d_offset_histogram_per_walker, d_end, d_start,
            d_factor, options.X, options.Y, options.alpha, options.beta, d_expected_energy_spectrum,
            d_len_energy_spectrum, total_walker, d_cond, walker_per_interactions,
            options.num_intervals, d_offset_energy_spectrum, d_cond_interactions);
        cudaDeviceSynchronize();

        calc_average_log_g<<<block_count, threads_per_block>>>(
            options.num_intervals, d_len_histograms, options.walker_per_interval, d_logG, d_shared_logG,
            d_end, d_start, d_expected_energy_spectrum, d_cond, d_offset_histogram_per_walker,
            d_offset_energy_spectrum, options.num_interactions, d_offset_shared_logG, d_cond_interactions, total_len_histogram);
        cudaDeviceSynchronize();

        redistribute_g_values<<<block_count, threads_per_block>>>(
            options.num_intervals, d_len_histograms, options.walker_per_interval, d_logG, d_shared_logG,
            d_end, d_start, d_factor, options.beta, d_expected_energy_spectrum, d_cond,
            d_offset_histogram_per_walker, options.num_interactions, d_offset_shared_logG, d_cond_interactions, total_len_histogram);
        cudaDeviceSynchronize();

        if (iterator == 148)
        {
            return;
        }

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

        // // TEST BLOCK
        // //-----------
        // CHECK_CUDA(cudaMemcpy(test_energies_wl.data(), d_energy, total_walker * sizeof(*d_energy), cudaMemcpyDeviceToHost)); // get energies from wl step with energy diff calc
        // calc_energy_eight_vertex<<<blocks_total_walker_x_thread, threads_per_block>>>(d_energy_test, d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, 2 * options.X * options.Y, options.X, 2 * options.Y, total_walker, walker_per_interactions, d_offset_lattice_per_walker);
        // cudaDeviceSynchronize();
        // CHECK_CUDA(cudaMemcpy(test_energies.data(), d_energy_test, total_walker * sizeof(*d_energy_test), cudaMemcpyDeviceToHost)); // get energies from calc energy function
        // if (iterator == 0)
        // {
        //     std::cerr << "Before Replica Exchange - iterator: " << iterator << " walker idx: " << "0" << " calc energy: " << test_energies[0] << " wl calc energy: " << test_energies_wl[0] << " Diff: " << std::abs(test_energies_wl[0] - test_energies[0]) << std::endl;
        // }
        // //-----------
        if (wang_landau_counter % options.replica_exchange_offset == 0)
        {
            replica_exchange<double><<<total_intervals, options.walker_per_interval>>>(
                d_offset_lattice_per_walker, d_energy, d_start, d_end, d_indices, d_logG,
                d_offset_histogram_per_walker, true, options.seed_run, d_offset_iterator_per_walker,
                options.num_intervals, walker_per_interactions, d_cond_interactions);

            replica_exchange<double><<<total_intervals, options.walker_per_interval>>>(
                d_offset_lattice_per_walker, d_energy, d_start, d_end, d_indices, d_logG,
                d_offset_histogram_per_walker, false, options.seed_run, d_offset_iterator_per_walker,
                options.num_intervals, walker_per_interactions, d_cond_interactions);
        }
        // // TEST BLOCK
        // //-----------
        // CHECK_CUDA(cudaMemcpy(test_energies_wl.data(), d_energy, total_walker * sizeof(*d_energy), cudaMemcpyDeviceToHost)); // get energies from wl step with energy diff calc
        // calc_energy_eight_vertex<<<blocks_total_walker_x_thread, threads_per_block>>>(d_energy_test, d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, 2 * options.X * options.Y, options.X, 2 * options.Y, total_walker, walker_per_interactions, d_offset_lattice_per_walker);
        // cudaDeviceSynchronize();
        // CHECK_CUDA(cudaMemcpy(test_energies.data(), d_energy_test, total_walker * sizeof(*d_energy_test), cudaMemcpyDeviceToHost)); // get energies from calc energy function
        // if (iterator == 0)
        // {
        //     std::cerr << "After Replica Exchange - iterator: " << iterator << " walker idx: " << "0" << " calc energy: " << test_energies[0] << " wl calc energy: " << test_energies_wl[0] << " Diff: " << std::abs(test_energies_wl[0] - test_energies[0]) << std::endl;
        // }
        // //-----------

        // results dump out: if a single interaction already finished
        if (min_cond_interactions == -1)
        {
            CHECK_CUDA(cudaMemcpy(h_cond_interactions, d_cond_interactions, options.num_interactions * sizeof(*d_cond_interactions), cudaMemcpyDeviceToHost));

            for (int i = 0; i < options.num_interactions; i++)
            {
                if (h_cond_interactions[i] == -1 && !h_result_is_dumped[i])
                {

                    int offset_of_interaction_histogram;
                    CHECK_CUDA(cudaMemcpy(&offset_of_interaction_histogram, d_offset_histogram_per_walker + i * (options.num_intervals * options.walker_per_interval), sizeof(*d_offset_histogram_per_walker), cudaMemcpyDeviceToHost));

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

                    eight_vertex_result_handling_stitched_histogram(
                        options, h_logG, error_mean, error_variance, prob_x_err, prob_y_err, prob_z_err, run_start, run_end, i,
                        is_qubit_specific_noise, x_horizontal_error, x_vertical_error, z_horizontal_error, z_vertical_error);
                }
            }
        }
    }

    return 0;
}