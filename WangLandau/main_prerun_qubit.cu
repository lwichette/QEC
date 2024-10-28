#include "./header/cudamacro.h"
#include "./header/wlutils.cuh"
#include <thrust/extrema.h> //addition needed for my (Linnea's) version of thrust -- comment out if this causes issues

int THREADS = 128;

using namespace std;

// Define a functor to compute the absolute value
struct absolute_value
{
    __host__ __device__ double operator()(const double &x) const
    {
        return fabsf(x); // Compute the absolute value of each element
    }
};

__global__ void init_error_rates(float *d_prob_interactions, int num_interactions, int X, int Y, double error_mean, double error_variance, unsigned long long seed)
{
    const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= num_interactions * X * Y * 2)
        return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    // Generate a Gaussian distributed value with given mean and variance
    // double randomValue = curand_normal_double(&state) * sqrt(error_variance) + error_mean;
    double randomValue = error_mean;

    // // Bound error probabilities between 1e-25 and 0.5
    if (randomValue <= 0.0)
        randomValue = 1e-25; // error prob 0 is not allowed
    if (randomValue > 0.5)
        randomValue = 0.5;

    d_prob_interactions[tid] = randomValue;

    return;
}

__global__ void init_J_interactions(double *d_J, float *d_probs_interactions, int num_interactions, int X, int Y, unsigned long long seed, char logical_error_type)
{
    const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= num_interactions * X * Y * 2)
        return;

    double J = log((1 - d_probs_interactions[tid]) / d_probs_interactions[tid]);

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    if (curand_uniform(&state) < d_probs_interactions[tid])
    {
        d_J[tid] = -J;
    }
    else
    {
        d_J[tid] = J;
    }

    int lin_interaction_idx = tid % (X * Y * 2); // only needed for non trivial num lattices
    int i = lin_interaction_idx / Y;             // row index
    int j = lin_interaction_idx % Y;             // column index

    if (logical_error_type == 'I' && tid == 0)
    {
        printf("Id error class.\n");
    }
    else if (logical_error_type == 'X')
    {
        if (tid == 0)
        {
            printf("Row error class.\n");
        }
        if (i == X)
        { // flip all left interactions stored in first row --> changed to row nx, which should hopefully flip all up interactions instead
            d_J[tid] *= -1;
        }
    }
    else if (logical_error_type == 'Z')
    {
        if (tid == 0)
        {
            printf("Column error class.\n");
        }
        if (j == 0 && i < X)
        { // flip all up interactions stored in first column from row nx*ny onwards --> changed to row <nx, which should hopefully flip all left interactions instead
            d_J[tid] *= -1;
        }
    }
    else if (logical_error_type == 'Y')
    {
        if (tid == 0)
        {
            printf("Combined error class.\n");
        }
        if (i == X)
        { // flip all left interactions stored in first row --> changed to row nx
            d_J[tid] *= -1;
        }
        if (j == 0 && i < X)
        { // flip all up interactions stored in first column from row nx onwards in interaction matrix --> changed to row <nx
            d_J[tid] *= -1;
        }
    }

    return;
}

__global__ void rescale_interactions(double *d_interactions, double max_abs_val, int histogram_scale, int num_interactions, int X, int Y)
{
    const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= num_interactions * X * Y * 2)
        return;

    d_interactions[tid] = d_interactions[tid] * histogram_scale / max_abs_val;

    return;
}

int main(int argc, char **argv)
{
    int X, Y;

    int num_wl_loops, num_iterations, walker_per_interactions;

    int seed;

    int task_id;

    int num_intervals;

    char logical_error_type = 'I';

    int boundary_type = 0;

    int och;

    int num_interactions;

    float error_mean;

    float error_variance;

    int histogram_scale = 1;

    while (1)
    {
        int option_index = 0;
        static struct option long_options[] = {
            {"X", required_argument, 0, 'x'},
            {"Y", required_argument, 0, 'y'},
            {"nit", required_argument, 0, 'n'},
            {"nl", required_argument, 0, 'l'},
            {"nw", required_argument, 0, 'w'},
            {"seed", required_argument, 0, 's'},
            {"num_intervals", required_argument, 0, 'i'},
            {"logical_error", required_argument, 0, 'e'},
            {"boundary", required_argument, 0, 'b'},
            {"replicas", required_argument, 0, 'r'},
            {"task_id", required_argument, 0, 'd'},
            {"error_mean", required_argument, 0, 'm'},
            {"error_variance", required_argument, 0, 'v'},
            {"histogram_scale", required_argument, 0, 'h'},
            {0, 0, 0, 0}};

        och = getopt_long(argc, argv, "x:y:n:l:w:s:i:e:b:r:d:m:v:h:", long_options, &option_index);

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
        case 'd':
            task_id = atoi(optarg);
            break;
        case 'm':
            error_mean = atof(optarg);
            break;
        case 'v':
            error_variance = atof(optarg);
            break;
        case 'h':
            histogram_scale = atoi(optarg);
            break;
        case '?':
            exit(EXIT_FAILURE);

        default:
            fprintf(stderr, "unknown option: %c\n", och);
            exit(EXIT_FAILURE);
        }
    }

    double factor = std::exp(1);

    int total_walker = num_interactions * walker_per_interactions;
    int total_intervals = num_interactions * num_intervals;

    const int E_min = -2 * X * Y * histogram_scale;
    const int E_max = -E_min;

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

    float *d_probs_lattices;
    CHECK_CUDA(cudaMalloc(&d_probs_lattices, total_walker * sizeof(*d_probs_lattices)));
    CHECK_CUDA(cudaMemset(d_probs_lattices, 0, total_walker * sizeof(*d_probs_lattices)));

    float *d_probs_interactions;
    CHECK_CUDA(cudaMalloc(&d_probs_interactions, num_interactions * X * Y * 2 * sizeof(*d_probs_interactions)));
    CHECK_CUDA(cudaMemset(d_probs_interactions, 0, num_interactions * X * Y * 2 * sizeof(*d_probs_interactions)));

    double *d_interactions;
    CHECK_CUDA(cudaMalloc(&d_interactions, num_interactions * X * Y * 2 * sizeof(*d_interactions)));

    double *d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, total_walker * sizeof(*d_energy)));

    signed char *d_store_lattice;
    CHECK_CUDA(cudaMalloc(&d_store_lattice, total_intervals * X * Y * sizeof(*d_store_lattice)));

    int *d_found_interval;
    CHECK_CUDA(cudaMalloc(&d_found_interval, total_intervals * sizeof(*d_found_interval)));
    CHECK_CUDA(cudaMemset(d_found_interval, 0, total_intervals * sizeof(*d_found_interval)));

    double *d_interval_energies;
    CHECK_CUDA(cudaMalloc(&d_interval_energies, total_intervals * sizeof(*d_interval_energies)));
    CHECK_CUDA(cudaMemset(d_interval_energies, 0, total_intervals * sizeof(*d_interval_energies)));

    int *d_offset_lattice_per_walker, *d_offset_lattice_per_interval;
    CHECK_CUDA(cudaMalloc(&d_offset_lattice_per_walker, total_walker * sizeof(*d_offset_lattice_per_walker)));
    CHECK_CUDA(cudaMalloc(&d_offset_lattice_per_interval, total_intervals * sizeof(*d_offset_lattice_per_interval)));

    int BLOCKS_INIT = (total_walker * X * Y * 2 + THREADS - 1) / THREADS;
    int BLOCKS_ENERGY = (total_walker + THREADS - 1) / THREADS;
    int BLOCKS_INTERVAL = (total_intervals + THREADS - 1) / THREADS;

    init_lattice<<<BLOCKS_INIT, THREADS>>>(d_lattice, d_probs_lattices, X, Y, total_walker, seed + 1);
    init_offsets_lattice<<<BLOCKS_ENERGY, THREADS>>>(d_offset_lattice_per_walker, X, Y, total_walker);
    init_offsets_lattice<<<BLOCKS_INTERVAL, THREADS>>>(d_offset_lattice_per_interval, X, Y, total_intervals);
    cudaDeviceSynchronize();

    init_error_rates<<<BLOCKS_INIT, THREADS>>>(d_probs_interactions, num_interactions, X, Y, error_mean, error_variance, seed + 2);
    cudaDeviceSynchronize();

    init_J_interactions<<<BLOCKS_INIT, THREADS>>>(d_interactions, d_probs_interactions, num_interactions, X, Y, seed + 3, logical_error_type);
    cudaDeviceSynchronize();

    double *d_absolute_values;
    CHECK_CUDA(cudaMalloc(&d_absolute_values, num_interactions * X * Y * 2 * sizeof(*d_absolute_values)));

    thrust::device_ptr<double> dev_ptr_input(d_interactions);
    thrust::device_ptr<double> dev_ptr_abs(d_absolute_values);

    thrust::transform(dev_ptr_input, dev_ptr_input + num_interactions * X * Y * 2, dev_ptr_abs, absolute_value());

    double max_abs_val = thrust::reduce(dev_ptr_abs, dev_ptr_abs + num_interactions * X * Y * 2, 0.0, thrust::maximum<double>());

    rescale_interactions<<<BLOCKS_INIT, THREADS>>>(d_interactions, max_abs_val, histogram_scale, num_interactions, X, Y);
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

    std::vector<double> h_interval_energies(total_intervals);
    std::vector<double> h_interactions(X * Y * 2 * num_interactions);
    std::vector<signed char> h_store_lattice(X * Y * total_intervals);
    std::vector<unsigned long long> h_H(total_histogram);

    CHECK_CUDA(cudaMemcpy(h_interval_energies.data(), d_interval_energies, total_intervals * sizeof(*d_energy), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_interactions.data(), d_interactions, X * Y * 2 * num_interactions * sizeof(*d_interactions), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_store_lattice.data(), d_store_lattice, X * Y * total_intervals * sizeof(*d_store_lattice), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_H.data(), d_H, total_histogram * sizeof(*d_H), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_interactions; i++)
    {
        std::string path = "init/task_id_" + std::to_string(task_id) + "/" + boundary + "/error_mean_" + std::to_string(error_mean) + "/error_variance_" + std::to_string(error_variance) + "/X_" + std::to_string(X) + "_Y_" + std::to_string(Y) + "/error_class_" + logical_error_type + "/seed_" + std::to_string(seed + i);

        int offset_interactions = i * X * Y * 2;
        int offset_lattice = i * num_intervals * X * Y;
        int offset_energies = i * num_intervals;

        create_directory(path + "/interactions");
        create_directory(path + "/lattice");
        create_directory(path + "/histogram");

        write(h_interactions.data() + offset_interactions, path + "/interactions/interactions", 2 * X, Y, 1, false);
        write(h_store_lattice.data() + offset_lattice, path + "/lattice/lattice", X, Y, num_intervals, true, h_interval_energies.data() + offset_energies);
        write_histograms(h_H.data() + i * len_histogram, path + "/histogram/", (E_max - E_min + 1), seed, E_min);
    }

    printf("Finished prerun for Lattice %d x %d, boundary condition %s, error mean %f, error_variance %f, error type %c and %d interactions \n", X, Y, boundary.c_str(), error_mean, error_variance, logical_error_type, num_interactions);
    return 0;
}
