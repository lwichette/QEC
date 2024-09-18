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
    float overlap = 0;
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
            overlap = atof(optarg);
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

    // for (int i = 0; i < num_interactions; i++)
    // {
    //     std::cout << E_min[i] << " " << E_max[i] << std::endl;
    //     for (int j = 0; j < h_len_energy_spectrum[i]; j++)
    //     {
    //         std::cout << static_cast<int>(h_expected_energy_spectrum[j]) << std::endl;
    //     }
    // }

    // f Factors for each walker
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

    /*
    ----------------------------------------------
    ------------ Actual WL Starts Now ------------
    ----------------------------------------------
    */
    return 0;
}