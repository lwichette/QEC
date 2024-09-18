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
        std::string hist_path = eight_vertex_path(is_qubit_specific_noise, error_mean, error_variance, X, Y, seed_hist + i, "histogram", x_horizontal_error, x_vertical_error, z_horizontal_error, z_vertical_error);
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
}