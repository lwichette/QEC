#include "./header/cudamacro.h"
#include "./header/wlutils.cuh"
#include <thrust/extrema.h> //addition needed for my (Linnea's) version of thrust -- comment out if this causes issues

int THREADS = 128;

using namespace std;

int main(int argc, char **argv)
{

    int X, Y;

    float prob_interactions;

    int num_wl_loops, num_iterations, walker_per_interactions;

    int seed;

    int num_intervals;

    char logical_error_type = 'I';

    int boundary_type = 0;

    int och;

    int num_interactions;

    while (1)
    {
        int option_index = 0;
        static struct option long_options[] = {
            {"X", required_argument, 0, 'x'},
            {"Y", required_argument, 0, 'y'},
            {"prob", required_argument, 0, 'p'},
            {"nit", required_argument, 0, 'n'},
            {"nl", required_argument, 0, 'l'},
            {"nw", required_argument, 0, 'w'},
            {"seed", required_argument, 0, 's'},
            {"num_intervals", required_argument, 0, 'i'},
            {"logical_error", required_argument, 0, 'e'},
            {"boundary", required_argument, 0, 'b'},
            {"replicas", required_argument, 0, 'r'},
            {0, 0, 0, 0}};

        och = getopt_long(argc, argv, "x:y:p:n:l:w:s:i:e:b:r:", long_options, &option_index);

        if (och == -1)
            break;
        switch (och)
        {
        case 0: // handles long opts with non-NULL flag field
            break;
        case 'x':
            X = atoi(optarg);
            break;
        case 'y':
            Y = atoi(optarg);
            break;
        case 'p':
            prob_interactions = atof(optarg);
            break;
        case 'n':
            num_iterations = atoi(optarg);
            break;
        case 'l':
            num_wl_loops = atoi(optarg);
            break;
        case 'w':
            walker_per_interactions = atoi(optarg);
            break;
        case 's':
            seed = atoi(optarg);
            break;
        case 'i':
            num_intervals = atoi(optarg);
            break;
        case 'e':
            logical_error_type = *optarg;
            break;
        case 'b':
            boundary_type = atoi(optarg);
            break;
        case 'r':
            num_interactions = atoi(optarg);
            break;
        case '?':
            exit(EXIT_FAILURE);

        default:
            fprintf(stderr, "unknown option: %c\n", och);
            exit(EXIT_FAILURE);
        }
    }

    double factor = std::exp(1);

    const int E_min = -2 * X * Y;
    const int E_max = -E_min;

    int total_walker = num_interactions * walker_per_interactions;
    int total_intervals = num_interactions * num_intervals;

    IntervalResult interval_result = generate_intervals(E_min, E_max, num_intervals, 1, 1.0f);

    std::cout << "Intervals for the run" << std::endl;

    for (int i = 0; i < num_intervals; i++)
    {
        std::cout << interval_result.h_start[i] << " " << interval_result.h_end[i] << std::endl;
    }

    long long len_histogram = E_max - E_min + 1;
    long long total_histogram = num_interactions * len_histogram;

    unsigned long long *d_H;
    CHECK_CUDA(cudaMalloc(&d_H, total_histogram * sizeof(*d_H)));
    CHECK_CUDA(cudaMemset(d_H, 0, total_histogram * sizeof(*d_H)));

    unsigned long long *d_iter;
    CHECK_CUDA(cudaMalloc(&d_iter, total_walker * sizeof(*d_iter)));
    CHECK_CUDA(cudaMemset(d_iter, 0, total_walker * sizeof(*d_iter)));

    // lattice & interactions and energies
    signed char *d_lattice;
    CHECK_CUDA(cudaMalloc(&d_lattice, total_walker * X * Y * sizeof(*d_lattice)));

    float *d_probs;
    CHECK_CUDA(cudaMalloc(&d_probs, total_walker * sizeof(*d_probs)));
    CHECK_CUDA(cudaMemset(d_probs, 0, total_walker * sizeof(*d_probs)));

    signed char *d_interactions;
    CHECK_CUDA(cudaMalloc(&d_interactions, num_interactions * X * Y * 2 * sizeof(*d_interactions)));

    int *d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, total_walker * sizeof(*d_energy)));

    signed char *d_store_lattice;
    CHECK_CUDA(cudaMalloc(&d_store_lattice, total_intervals * X * Y * sizeof(*d_store_lattice)));

    int *d_found_interval;
    CHECK_CUDA(cudaMalloc(&d_found_interval, total_intervals * sizeof(*d_found_interval)));
    CHECK_CUDA(cudaMemset(d_found_interval, 0, total_intervals * sizeof(*d_found_interval)));

    int *d_interval_energies;
    CHECK_CUDA(cudaMalloc(&d_interval_energies, total_intervals * sizeof(*d_interval_energies)));
    CHECK_CUDA(cudaMemset(d_interval_energies, 0, total_intervals * sizeof(*d_interval_energies)));

    int *d_offset_lattice_per_walker, *d_offset_lattice_per_interval;
    CHECK_CUDA(cudaMalloc(&d_offset_lattice_per_walker, total_walker * sizeof(*d_offset_lattice_per_walker)));
    CHECK_CUDA(cudaMalloc(&d_offset_lattice_per_interval, total_intervals * sizeof(*d_offset_lattice_per_interval)));

    int BLOCKS_INIT = (total_walker * X * Y * 2 + THREADS - 1) / THREADS;
    int BLOCKS_ENERGY = (total_walker + THREADS - 1) / THREADS;
    int BLOCKS_INTERVAL = (total_intervals + THREADS - 1) / THREADS;

    init_interactions<<<BLOCKS_INIT, THREADS>>>(d_interactions, X, Y, num_interactions, seed, prob_interactions, logical_error_type);
    cudaDeviceSynchronize();

    init_lattice<<<BLOCKS_INIT, THREADS>>>(d_lattice, d_probs, X, Y, total_walker, seed - 1);
    init_offsets_lattice<<<BLOCKS_ENERGY, THREADS>>>(d_offset_lattice_per_walker, X, Y, total_walker);
    init_offsets_lattice<<<BLOCKS_INTERVAL, THREADS>>>(d_offset_lattice_per_interval, X, Y, total_intervals);
    cudaDeviceSynchronize();

    calc_energy(BLOCKS_ENERGY, THREADS, boundary_type, d_lattice, d_interactions, d_energy, d_offset_lattice_per_walker, X, Y, total_walker, walker_per_interactions);
    cudaDeviceSynchronize();

    int found_interval = 0;

    for (int i = 0; i < num_wl_loops; i++)
    {

        wang_landau_pre_run<<<BLOCKS_ENERGY, THREADS>>>(d_lattice, d_interactions, d_energy, d_H, d_iter, d_offset_lattice_per_walker, d_found_interval, d_store_lattice, E_min, E_max, num_iterations, X, Y, seed - 2, interval_result.len_interval, found_interval, total_walker, num_intervals, boundary_type, walker_per_interactions);
        cudaDeviceSynchronize();

        if (found_interval == 0)
        {
            thrust::device_ptr<int> d_found_interval_ptr(d_found_interval);
            thrust::device_ptr<int> min_found_interval_ptr = thrust::min_element(d_found_interval_ptr, d_found_interval_ptr + total_intervals);
            found_interval = *min_found_interval_ptr;
        }
    }

    calc_energy(BLOCKS_INTERVAL, THREADS, boundary_type, d_store_lattice, d_interactions, d_interval_energies, d_offset_lattice_per_interval, X, Y, total_intervals, num_intervals);
    cudaDeviceSynchronize();

    std::string boundary;

    if (boundary_type == 0)
    {
        boundary = "periodic";
    }
    else if (boundary_type == 1)
    {
        boundary = "open";
    }
    else if (boundary_type == 2)
    {
        boundary = "cylinder";
    }
    else
    {
        boundary = "unknown"; // Handle any unexpected boundary_type values
    }

    std::vector<int> h_interval_energies(total_intervals);
    std::vector<signed char> h_interactions(X * Y * 2 * num_interactions);
    std::vector<signed char> h_store_lattice(X * Y * total_intervals);
    std::vector<unsigned long long> h_H(total_histogram);

    CHECK_CUDA(cudaMemcpy(h_interval_energies.data(), d_interval_energies, total_intervals * sizeof(*d_energy), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_interactions.data(), d_interactions, X * Y * 2 * num_interactions * sizeof(*d_interactions), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_store_lattice.data(), d_store_lattice, X * Y * total_intervals * sizeof(*d_store_lattice), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_H.data(), d_H, total_histogram * sizeof(*d_H), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_interactions; i++)
    {
        std::string path = "init/" + boundary + "/prob_" + std::to_string(prob_interactions) + "/X_" + std::to_string(X) + "_Y_" + std::to_string(Y) + "/seed_" + std::to_string(seed + i) + "/error_class_" + logical_error_type;

        int offset_interactions = i * X * Y * 2;
        int offset_lattice = i * num_intervals * X * Y;
        int offset_energies = i * num_intervals;

        create_directory(path + "/interactions");
        create_directory(path + "/lattice");
        create_directory(path + "/histogram");

        write(h_interactions.data() + offset_interactions, path + "/interactions/interactions", X, Y, 1, false);
        write(h_store_lattice.data() + offset_lattice, path + "/lattice/lattice", X, Y, num_intervals, true, h_interval_energies.data() + offset_energies);
        write_histograms(h_H.data() + i * len_histogram, path + "/histogram/", (E_max - E_min + 1), seed, E_min);
    }

    printf("Finished prerun for Lattice %d x %d, boundary condition %s, probability %f, error type %c and %d interactions \n", X, Y, boundary.c_str(), prob_interactions, logical_error_type, num_interactions);
    return 0;
}
