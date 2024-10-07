#include "wlutils.cuh"
#include "cudamacro.h"

// order is important 0: periodic, 1: open, 2: cylinder
__device__ RBIM (*rbim_func_map[])(signed char *, signed char *, int *, int *, unsigned long long *, curandStatePhilox4_32_10_t *, const long long, const int, const int, const int) = {periodic_boundary_random_bond_ising, open_boundary_random_bond_ising, cylinder_random_bond_ising};

void parse_args(int argc, char *argv[], Options *options)
{
    // overlap decimal is more like the reciprocal non overlap parameter here, i.e. 0 as overlap_decimal is full overlap of intervals.

    int opt;

    options->logical_error_type = 'I';
    options->replica_exchange_offset = 1;

    while (1)
    {
        int option_index = 0;
        static struct option long_options[] = {
            {"X", 1, 0, 'x'},
            {"Y", 1, 0, 'y'},
            {"num_iterations", 1, 0, 'n'},
            {"prob_interactions", 1, 0, 'p'},
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
            {0, 0, 0, 0}};

        opt = getopt_long(argc, argv, "x:y:n:p:a:b:i:w:o:h:s:e:t:r:c:", long_options, &option_index);

        if (opt == -1)
            break;
        switch (opt)
        {
        case 'x':
            options->X = std::atoi(optarg);
            break;
        case 'y':
            options->Y = std::atoi(optarg);
            break;
        case 'n':
            options->num_iterations = std::atoi(optarg);
            break;
        case 'p':
            options->prob_interactions = std::atof(optarg);
            break;
        case 'a':
            options->alpha = std::atof(optarg);
            break;
        case 'b':
            options->beta = std::atof(optarg);
            break;
        case 'i':
            options->num_intervals = std::atoi(optarg);
            break;
        case 'w':
            options->walker_per_interval = std::atoi(optarg);
            break;
        case 'o':
            options->overlap_decimal = std::atof(optarg);
            break;
        case 'h':
            options->seed_histogram = std::atoi(optarg);
            break;
        case 's':
            options->seed_run = std::atoi(optarg);
            break;
        case 'e':
            options->logical_error_type = *optarg;
            break;
        case 't':
            options->boundary_type = std::atoi(optarg);
            break;
        case 'r':
            options->num_interactions = std::atoi(optarg);
            break;
        case 'c':
            options->replica_exchange_offset = std::atoi(optarg);
            break;
        default:
            fprintf(stderr, "Usage: %s [-i num_intervals] [-m E_min] [-M E_max] [-w walker_per_interval] [-o overlap_decimal] [-r num_iterations]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return;
}

IntervalResult generate_intervals(const int E_min, const int E_max, int num_intervals, int num_walker, float overlap_decimal)
{
    IntervalResult interval_result;

    std::vector<int> h_start(num_intervals);
    std::vector<int> h_end(num_intervals);

    const int E_range = E_max - E_min + 1;
    const int len_interval = E_range / (1.0f + overlap_decimal * (num_intervals - 1)); // Len_interval computation stems from condition: len_interval + overlap * len_interval * (num_intervals - 1) = E_range
    const int step_size = overlap_decimal * len_interval;

    int start_interval = E_min;

    long long len_histogram_over_all_walkers = 0;

    for (int i = 0; i < num_intervals; i++)
    {

        h_start[i] = start_interval;

        if (i < num_intervals - 1)
        {
            h_end[i] = start_interval + len_interval - 1;
            len_histogram_over_all_walkers += num_walker * len_interval;
        }
        else
        {
            h_end[i] = E_max;
            len_histogram_over_all_walkers += num_walker * (E_max - h_start[i] + 1);
        }

        start_interval += step_size;
    }
    interval_result.h_start = h_start;
    interval_result.h_end = h_end;
    interval_result.len_histogram_over_all_walkers = len_histogram_over_all_walkers;
    interval_result.len_interval = len_interval;

    return interval_result;
}

void write(
    signed char *array_host, const std::string &filename, long nx, long ny,
    int num_lattices, bool lattice, const int *energies)
{
    // std::cout << "Writing to " << filename << " ..." << std::endl;

    int nx_w = (lattice) ? nx : 2 * nx;

    if (num_lattices == 1)
    {
        writeToFile(filename + ".txt", array_host, nx_w, ny);
    }
    else
    {
        for (int l = 0; l < num_lattices; l++)
        {
            int offset = l * nx_w * ny;

            if (energies)
            {
                if (energies[l] == 0 && array_host[offset] == 0)
                {
                    continue;
                }
            }

            std::string file_suffix = (!energies) ? std::to_string(l) : std::to_string(energies[l]);
            writeToFile(filename + "_" + file_suffix + ".txt", array_host + offset, nx_w, ny);
        }
    }

    return;
}

void create_directory(std::string path)
{
    if (!std::filesystem::exists(path))
    {
        // Create directory
        if (!std::filesystem::create_directories(path))
        {
            // std::cerr << "Failed to create directory: " << path << std::endl;
        }
    }

    return;
}

void write_histograms(unsigned long long *h_histogram, std::string path_histograms, int len_histogram, int seed, int E_min)
{

    // printf("Writing to %s ...\n", path_histograms.c_str());

    std::ofstream f;
    f.open(std::string(path_histograms + "/histogram.txt"));

    if (f.is_open())
    {
        for (int i = 0; i < len_histogram; i++)
        {
            int energy = E_min + i;
            if (h_histogram[i] > 0)
            {
                f << energy << " " << 1 << std::endl;
            }
            else
            {
                f << energy << " " << 0 << std::endl;
            }
        }
    }

    return;
}

std::vector<signed char> read_histogram(std::string filename, std::vector<int> &E_min, std::vector<int> &E_max)
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Failed to open the file." << std::endl;
        return std::vector<signed char>();
    }

    int min = INT_MAX;
    int max = INT_MIN;

    int value, count;

    while (file >> value >> count)
    {
        if (count > 0)
        {
            if (value < min)
                min = value;
            if (value > max)
                max = value;
        }
    }

    file.clear();
    file.seekg(0);

    std::vector<signed char> energy_spectrum;

    while (file >> value >> count)
    {
        if (value >= min && value <= max)
        {
            energy_spectrum.push_back(static_cast<signed char>(count));
        }
    }

    file.close();

    E_min.push_back(min);
    E_max.push_back(max);

    return energy_spectrum;
}

// It is important that the format of the histogramm file is like energy: count such that we can append a row with new_energy: 1
void handleNewEnergyError(int *new_energies, int *new_energies_flag, char *histogram_file, int num_walkers_total)
{
    std::cerr << "Error: Found new energy:" << std::endl;
    std::ofstream outfile;
    outfile.open(histogram_file, std::ios_base::app);
    if (!outfile.is_open())
    {
        std::cerr << "Error: Could not open file " << histogram_file << std::endl;
        return;
    }

    for (int i = 0; i < num_walkers_total; i++)
    {
        if (new_energies_flag[i] != 0)
        {
            int walker_idx = new_energies_flag[i] - 1;
            int new_energy = new_energies[walker_idx];

            // Write the new energy to the histogram file
            outfile << new_energy << " 1" << std::endl;
        }
    }

    // Close the file
    outfile.close();

    return;
}

std::string constructFilePath(float prob_interactions, int X, int Y, int seed, std::string type, char error_class, int boundary_type)
{
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

    std::stringstream strstr;
    strstr << "init/" << boundary << "/prob_" << std::fixed << std::setprecision(6) << prob_interactions;
    strstr << "/X_" << X << "_Y_" << Y;
    strstr << "/error_class_" << error_class;
    strstr << "/seed_" << seed;
    strstr << "/" << type << "/" << type << ".txt";

    return strstr.str();
}

std::vector<signed char> get_lattice_with_pre_run_result(float prob, int seed, int x, int y, std::vector<int> h_start, std::vector<int> h_end, int num_intervals, int num_walkers_total, int num_walkers_per_interval, char error_class, int boundary_type)
{
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

    namespace fs = std::filesystem;
    std::ostringstream oss;
    oss << "init/" << boundary << "/prob_" << std::fixed << std::setprecision(6) << prob;
    oss << "/X_" << x << "_Y_" << y;
    oss << "/error_class_" << error_class;
    oss << "/seed_" << seed;
    oss << "/lattice";

    std::string lattice_path = oss.str();
    std::vector<signed char> lattice_over_all_walkers;
    for (int interval_iterator = 0; interval_iterator < num_intervals; interval_iterator++)
    {
        // std::cout << interval_iterator << " ";
        try
        {
            for (const auto &entry : fs::directory_iterator(lattice_path))
            {
                // Check if the entry is a regular file and has a .txt extension
                if (entry.is_regular_file() && entry.path().extension() == ".txt")
                {
                    // Extract the number from the filename
                    std::string filename = entry.path().stem().string(); // Get the filename without extension
                    std::regex regex("lattice_(-?\\d+)");
                    std::smatch match;
                    if (std::regex_search(filename, match, regex))
                    {
                        int number = std::stoi(match[1]);
                        // Check if the number is between interval boundaries
                        if (number >= h_start[interval_iterator] && number <= h_end[interval_iterator])
                        {
                            // std::cout << "Lattice with energy: " << number << " for interval [" << h_start[interval_iterator] << ", " << h_end[interval_iterator] << "]" << std::endl;
                            for (int walker_per_interval_iterator = 0; walker_per_interval_iterator < num_walkers_per_interval; walker_per_interval_iterator++)
                            {
                                read(lattice_over_all_walkers, entry.path().string());
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
        }
        catch (const fs::filesystem_error &e)
        {
            std::cerr << "Filesystem error: " << e.what() << std::endl;
        }
    }
    return lattice_over_all_walkers;
}

std::map<std::string, std::vector<signed char>> get_lattice_with_pre_run_result_eight_vertex(
    bool is_qubit_specific_noise, float error_mean, float error_variance, bool x_horizontal_error, bool x_vertical_error, bool z_horizontal_error, bool z_vertical_error,
    int X, int Y, std::vector<int> h_start, std::vector<int> h_end, int num_intervals, int num_walkers_per_interval, int seed_hist, float prob_x_err, float prob_y_err, float prob_z_err)
{
    std::string error_string = std::to_string(x_horizontal_error) + std::to_string(x_vertical_error) + std::to_string(z_horizontal_error) + std::to_string(z_vertical_error);

    namespace fs = std::filesystem;
    std::ostringstream oss;
    oss << "init/eight_vertex/periodic/qubit_specific_noise_" << std::to_string(is_qubit_specific_noise) << "/" << std::fixed << std::setprecision(6);

    if (is_qubit_specific_noise)
    {
        oss << "error_mean_" << error_mean << "_error_variance_" << error_variance;
    }
    else
    {
        oss << "prob_X_" + std::to_string(prob_x_err) + "_prob_Y_" + std::to_string(prob_y_err) + "_prob_Z_" + std::to_string(prob_z_err);
    }

    oss << "/X_" << X << "_Y_" << Y;
    oss << "/error_class_" << error_string;
    oss << "/seed_" << seed_hist;
    oss << "/lattice";

    std::string lattice_path = oss.str();
    std::map<std::string, std::vector<signed char>> lattices;
    lattices["b"] = {};
    lattices["r"] = {};

    for (int interval_iterator = 0; interval_iterator < num_intervals; interval_iterator++)
    {
        try
        {
            for (const auto &entry : std::filesystem::directory_iterator(lattice_path))
            {
                // Check if the entry is a regular file and has a .txt extension
                if (entry.is_regular_file() && entry.path().extension() == ".txt")
                {
                    std::string filename = entry.path().stem().string(); // Get the filename without extension

                    std::string filename_b = filename;
                    // Check for "r" lattice
                    std::regex regex_r("lattice_r_energy_(-?\\d+(\\.\\d{6})?)");
                    std::smatch match_r;
                    if (std::regex_search(filename, match_r, regex_r))
                    {
                        float energy_r = std::stof(match_r[1]);
                        if (energy_r >= h_start[interval_iterator] && energy_r <= h_end[interval_iterator])
                        {

                            // Find the position of the substring "_r_" to replace
                            std::size_t pos = filename.find("_r_");

                            filename_b.replace(pos, 3, "_b_");

                            // Matching energy and within bounds, process both
                            for (int walker_per_interval_iterator = 0; walker_per_interval_iterator < num_walkers_per_interval; walker_per_interval_iterator++)
                            {
                                read(lattices["r"], lattice_path + "/" + filename + ".txt");
                                read(lattices["b"], lattice_path + "/" + filename_b + ".txt");
                                std::string file_b = lattice_path + "/" + filename_b + ".txt";
                                std::string file_r = lattice_path + "/" + filename + ".txt";
                                // printf("interval %d start %d stop %d energy %.2f path %s \n", interval_iterator, h_start[interval_iterator], h_end[interval_iterator], energy_r, filename.c_str());
                            }
                            break;
                        }

                        continue;
                    }
                }
            }
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            std::cerr << "Filesystem error: " << e.what() << std::endl;
        }
    }

    return lattices;
}

__device__ float atomicCAS_f32(float *p, float cmp, float val)
{
    return __int_as_float(atomicCAS((int *)p, __float_as_int(cmp), __float_as_int(val)));
}

__global__ void init_lattice(signed char *lattice, float *d_probs, const int nx, const int ny, const int num_lattices, const int seed)
{

    const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= nx * ny * num_lattices)
        return;

    long long lattice_id = tid / (nx * ny);

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);

    atomicCAS_f32(&d_probs[lattice_id], 0.0f, curand_uniform(&st));

    __syncthreads();

    double randval = curand_uniform(&st);
    signed char val = (randval < d_probs[lattice_id]) ? -1 : 1;

    lattice[tid] = val;

    return;
}

__global__ void init_interactionsOld(signed char *interactions, const int nx, const int ny, const int num_lattices, const int seed, const double prob, const char logical_error_type)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= nx * ny * 2 * num_lattices)
        return;

    int lattice_id = tid / (nx * ny * 2);

    curandStatePhilox4_32_10_t st;
    curand_init(seed + lattice_id, tid, 0, &st);

    double randval = curand_uniform(&st);
    signed char val = (randval < prob) ? -1 : 1;

    interactions[tid] = val;

    int lin_interaction_idx = tid % (nx * ny * 2); // only needed for non trivial num lattices
    int i = lin_interaction_idx / ny;              // row index
    int j = lin_interaction_idx % ny;              // column index

    if (logical_error_type == 'I' && tid == 0)
    {
        printf("Id error class.\n");
    }
    else if (logical_error_type == 'X')
    {
        if (tid == 0)
        {
            printf("X error class.\n");
        }
        if (i == 0)
        { // flip all left interactions stored in first row
            interactions[tid] *= -1;
        }
    }
    else if (logical_error_type == 'Z')
    {
        if (tid == 0)
        {
            printf("Z error class.\n");
        }
        if (j == 0 && i >= nx)
        { // flip all up interactions stored in first column from row nx*ny onwards
            interactions[tid] *= -1;
        }
    }
    else if (logical_error_type == 'Y')
    {
        if (tid == 0)
        {
            printf("Y error class.\n");
        }
        if (i == 0)
        { // flip all left interactions stored in first row
            interactions[tid] *= -1;
        }
        if (j == 0 && i >= nx)
        { // flip all up interactions stored in first column from row nx onwards in interaction matrix
            interactions[tid] *= -1;
        }
    }

    return;
}

// Test of changing init_interactions to swap up and left bonds. We want a column of left bonds and a row of up bonds

// When considering different error classes, we want to add a row of flipped vertical bonds, or a column of flipped horizontal bonds, or both
__global__ void init_interactions(signed char *interactions, const int nx, const int ny, const int num_lattices, const int seed, const double prob, const char logical_error_type)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= nx * ny * 2 * num_lattices)
        return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);

    double randval = curand_uniform(&st);
    signed char val = (randval < prob) ? -1 : 1;

    interactions[tid] = val;

    int lin_interaction_idx = tid % (nx * ny * 2); // only needed for non trivial num lattices
    int i = lin_interaction_idx / ny;              // row index
    int j = lin_interaction_idx % ny;              // column index

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
        if (i == nx)
        { // flip all left interactions stored in first row --> changed to row nx, which should hopefully flip all up interactions instead
            interactions[tid] *= -1;
        }
    }
    else if (logical_error_type == 'Z')
    {
        if (tid == 0)
        {
            printf("Column error class.\n");
        }
        if (j == 0 && i < nx)
        { // flip all up interactions stored in first column from row nx*ny onwards --> changed to row <nx, which should hopefully flip all left interactions instead
            interactions[tid] *= -1;
        }
    }
    else if (logical_error_type == 'Y')
    {
        if (tid == 0)
        {
            printf("Combined error class.\n");
        }
        if (i == nx)
        { // flip all left interactions stored in first row --> changed to row nx
            interactions[tid] *= -1;
        }
        if (j == 0 && i < nx)
        { // flip all up interactions stored in first column from row nx onwards in interaction matrix --> changed to row <nx
            interactions[tid] *= -1;
        }
    }

    return;
}

__global__ void calc_energy_periodic_boundary(signed char *lattice, signed char *interactions, int *d_energy, int *d_offset_lattice, const int nx, const int ny, const int num_lattices, const int walker_per_interactions)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= num_lattices)
        return;

    int int_id = tid / walker_per_interactions;
    int energy = 0;

    for (int l = 0; l < nx * ny; l++)
    {
        int i = l / ny;
        int j = l % ny;

        int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
        int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;

        energy += lattice[d_offset_lattice[tid] + i * ny + j] * (lattice[d_offset_lattice[tid] + inn * ny + j] * interactions[int_id * 2 * nx * ny + nx * ny + inn * ny + j] + lattice[d_offset_lattice[tid] + i * ny + jnn] * interactions[int_id * nx * ny * 2 + i * ny + jnn]);
    }

    d_energy[tid] = -energy;

    return;
}

__global__ void calc_energy_open_boundary(signed char *lattice, signed char *interactions, int *d_energy, int *d_offset_lattice, const int nx, const int ny, const int num_lattices, const int walker_per_interactions)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= num_lattices)
        return;

    int int_id = tid / walker_per_interactions;

    int energy = 0;
    int offset = d_offset_lattice[tid];

    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            signed char s_ij = lattice[offset + i * ny + j];
            signed char s_up = (i > 0) ? lattice[offset + (i - 1) * ny + j] : 0;
            signed char s_left = (j > 0) ? lattice[offset + i * ny + (j - 1)] : 0;

            // to avoid accessing interactions out of range for boundary terms the indices are arbitrarily set to 0
            int inn = (i > 0) ? nx * ny + (i - 1) * ny + j : 0;
            int jnn = (j > 0) ? i * ny + (j - 1) : 0;

            energy += s_ij * (s_up * interactions[int_id * nx * ny * 2 + inn] + s_left * interactions[int_id * nx * ny * 2 + jnn]);
        }
    }

    d_energy[tid] = -energy;

    return;
}

__global__ void calc_energy_cylinder(signed char *lattice, signed char *interactions, int *d_energy, int *d_offset_lattice, const int nx, const int ny, const int num_lattices, const int walker_per_interactions)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= num_lattices)
        return;

    int int_id = tid / walker_per_interactions;

    int energy = 0;
    int offset_lattice = d_offset_lattice[tid];
    int offset_interactions = int_id * nx * ny * 2;

    for (int i = 0; i < nx; i++) // row index up to nx
    {
        for (int j = 0; j < ny; j++) // column index up to ny
        {
            signed char s_ij = lattice[offset_lattice + i * ny + j];
            // up neighbor spin set to zero at boundary to set open condition in vertical lattice direction
            signed char s_up = (i > 0) ? lattice[offset_lattice + (i - 1) * ny + j] : 0;
            // left neighbor spin closed periodically to close cylinder in horizontal lattice direction - last horizontal index is ny-1
            signed char s_left = (j > 0) ? lattice[offset_lattice + i * ny + (j - 1)] : lattice[offset_lattice + i * ny + (ny - 1)];

            // down interaction linearised index set arbitrarily to zero
            int inn = (i > 0) ? nx * ny + (i - 1) * ny + j : 0;
            // right interacton linearised index closed periodically
            int jnn = (j > 0) ? i * ny + (j - 1) : i * ny + (ny - 1);

            // formula follows root spin times [up neighbor times down interaction rooted at up neighbor spin]
            energy += s_ij * (s_up * interactions[offset_interactions + inn] + s_left * interactions[offset_interactions + jnn]);
        }
    }

    d_energy[tid] = -energy;

    return;
}

__global__ void wang_landau_pre_run(
    signed char *d_lattice, signed char *d_interactions, int *d_energy, unsigned long long *d_H, unsigned long long *d_iter,
    int *d_offset_lattice, int *d_found_interval, signed char *d_store_lattice, const int E_min, const int E_max,
    const int num_iterations, const int nx, const int ny, const int seed, const int len_interval, const int found_interval,
    const int num_walker, const int num_interval, const int boundary_type, const int walker_per_interactions)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= num_walker)
        return;

    const int offset_lattice = tid * nx * ny;
    const int int_id = tid / walker_per_interactions;
    const int interaction_offset = int_id * 2 * nx * ny;

    const int len_hist = E_max - E_min + 1;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_iter[tid], &st);

    for (int it = 0; it < num_iterations; it++)
    {

        RBIM result = rbim_func_map[boundary_type](d_lattice, d_interactions, d_energy, d_offset_lattice, d_iter, &st, tid, nx, ny, interaction_offset);

        int d_new_energy = result.new_energy;

        if (d_new_energy > E_max || d_new_energy < E_min)
        {
            printf("Iterator %d \n", it);
            printf("Thread Id %lld \n", tid);
            printf("Energy out of range %d \n", d_new_energy);
            printf("Old energy %d \n", d_energy[tid]);
            assert(0);
            return;
        }
        else
        {
            int index_old = d_energy[tid] - E_min + int_id * len_hist;
            int index_new = d_new_energy - E_min + int_id * len_hist;

            double prob = exp(static_cast<double>(d_H[index_old]) - static_cast<double>(d_H[index_new]));

            if (curand_uniform(&st) < prob)
            {

                d_lattice[offset_lattice + result.i * ny + result.j] *= -1;
                d_energy[tid] = d_new_energy;

                atomicAdd(&d_H[index_new], 1);

                if (found_interval == 0)
                {
                    store_lattice(d_lattice, d_energy, d_found_interval, d_store_lattice, E_min, nx, ny, tid, len_interval, num_interval, int_id);
                }
            }
            else
            {
                atomicAdd(&d_H[index_old], 1);
            }
            d_iter[tid] += 1;
        }
    }

    return;
}

__global__ void wang_landau_pre_run_eight_vertex(
    signed char *d_lattice_b, signed char *d_lattice_r, double *d_interactions_b, double *d_interactions_r, double *d_interactions_right_four_body, double *d_interactions_down_four_body, double *d_energy, unsigned long long *d_H, unsigned long long *d_iter,
    int *d_found_interval, signed char *d_store_lattice_b, signed char *d_store_lattice_r, const int E_min, const int E_max,
    const int num_iterations, const int num_qubits, const int X, const int Y, const int seed, const int len_interval, const int found_interval,
    const int num_walker, const int num_interval, const int walker_per_interaction)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= num_walker)
        return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_iter[tid], &st);

    for (int it = 0; it < num_iterations; it++)
    {

        // may want to hand the offsets to this function and not compzte them inside
        RBIM_eight_vertex result = eight_vertex_periodic_wl_step(
            d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, d_energy, d_iter,
            &st, tid, num_qubits, X, Y, num_walker, walker_per_interaction);

        double d_new_energy = result.new_energy;

        if (d_new_energy > E_max || d_new_energy < E_min)
        {
            printf("Iterator %d \n", it);
            printf("Thread Id %lld \n", tid);
            printf("Energy out of range %f \n", d_new_energy);
            printf("Old energy %f \n", d_energy[tid]);
            assert(0);
            return;
        }
        else
        {
            const int len_hist = E_max - E_min + 1;
            const int int_id = tid / walker_per_interaction;

            // if (static_cast<int>(d_new_energy) % 2 == 1 && tid == 0)
            // {
            //     printf("old E before: %.10f ", d_energy[tid]);
            // }

            const int old_energy_int = static_cast<int>(round(d_energy[tid]));
            const int new_energy_int = static_cast<int>(round(d_new_energy));

            // if (static_cast<int>(d_new_energy) % 2 == 1 && tid == 0)
            // {
            //     printf("old E after: %d \n", old_energy_int);
            // }

            int index_old = old_energy_int - E_min + int_id * len_hist; // energy diff is double valued but binning is invoked by integer cast
            int index_new = new_energy_int - E_min + int_id * len_hist;

            // // // TEST BLOCK
            // // // ------
            // if (static_cast<int>(d_new_energy) % 2 == 1)
            // {
            //     printf("new index: %d double new index: %.10f newE: %.10f Emin: %d \n", index_new, (d_new_energy - E_min) + int_id * len_hist, d_new_energy, E_min);
            // }
            // // // ------

            double prob = exp(static_cast<double>(d_H[index_old]) - static_cast<double>(d_H[index_new]));

            if (curand_uniform(&st) < prob)
            {
                const int offset_lattice = tid * X * Y / 2;
                if (result.color)
                { // red lattice spin flip
                    d_lattice_r[offset_lattice + result.i * X + result.j] *= -1;
                }
                else // blue lattice spin flip
                {
                    d_lattice_b[offset_lattice + result.i * X + result.j] *= -1;
                }
                d_energy[tid] = d_new_energy;
                atomicAdd(&d_H[index_new], 1);
                if (found_interval == 0)
                {
                    // IMPORTANT: order of calling the store functions is necessary in order of the color parameter : 1st color false -> 2nd color true
                    // color parameter here does not have to coincide with the actual color it only remembers that a second color must still be processed.
                    store_lattice(d_lattice_r, d_lattice_b, d_energy, d_found_interval, d_store_lattice_r, d_store_lattice_b, E_min, X, Y / 2, tid, len_interval, num_interval, int_id);
                }
            }
            else
            {
                atomicAdd(&d_H[index_old], 1);
            }
            d_iter[tid] += 1;
        }
    }

    return;
}

__global__ void wang_landau_eight_vertex(
    signed char *d_lattice_b, signed char *d_lattice_r, double *d_interactions_b, double *d_interactions_r, double *d_interactions_right_four_body, double *d_interactions_down_four_body, double *d_energy, int *d_start, int *d_end, unsigned long long *d_H,
    double *d_logG, int *d_offset_histogramm, int *d_offset_lattice, const int num_iterations, const int nx, const int ny,
    const int seed, double *factor, unsigned long long *d_offset_iter, signed char *d_expected_energy_spectrum, double *d_newEnergies, int *foundFlag,
    const int num_lattices, const double beta, signed char *d_cond, const int walker_per_interactions, const int num_intervals,
    int *d_offset_energy_spectrum, int *d_cond_interaction, const int walker_per_interval)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= num_lattices)
        return;

    const int interval_id = (tid % walker_per_interactions) / walker_per_interval;
    const int interaction_id = tid / walker_per_interactions;

    if (d_cond_interaction[interaction_id] == -1)
        return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_offset_iter[tid], &st);

    if (d_cond[interval_id] == 0)
    {

        for (int it = 0; it < num_iterations; it++)
        {
            RBIM_eight_vertex result = eight_vertex_periodic_wl_step(
                d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, d_energy, d_offset_iter,
                &st, tid, nx * ny, nx, ny, num_lattices, walker_per_interactions);

            const int old_energy_int = static_cast<int>(round(d_energy[tid])); // Int cast for indexing via energy
            const int new_energy_int = static_cast<int>(round(result.new_energy));

            // If no new energy is found, set it to 0, else to tid + 1
            foundFlag[tid] = (d_expected_energy_spectrum[d_offset_energy_spectrum[interaction_id] + new_energy_int - d_start[interaction_id * num_intervals]] == 1) ? 0 : tid + 1;

            if (foundFlag[tid] != 0)
            {
                printf("new_energy %d index in spectrum %d \n", new_energy_int, new_energy_int - d_start[interaction_id * num_intervals]);
                d_newEnergies[tid] = result.new_energy;
                if (result.color)
                { // red lattice spin flip
                    d_lattice_r[d_offset_lattice[tid] + result.i * nx + result.j] *= -1;
                }
                else // blue lattice spin flip
                {
                    d_lattice_b[d_offset_lattice[tid] + result.i * nx + result.j] *= -1;
                }
                return;
            }

            int index_old = d_offset_histogramm[tid] + old_energy_int - d_start[interval_id];

            if (new_energy_int > d_end[interval_id] || new_energy_int < d_start[interval_id])
            {
                d_H[index_old] += 1;
                d_logG[index_old] += log(factor[tid]);
            }
            else
            {
                int index_new = d_offset_histogramm[tid] + new_energy_int - d_start[interval_id];
                double prob = exp(d_logG[index_old] - d_logG[index_new]);
                double randval = curand_uniform(&st);

                if (randval < prob)
                {

                    if (result.color)
                    { // red lattice spin flip
                        d_lattice_r[d_offset_lattice[tid] + result.i * nx + result.j] *= -1;
                    }
                    else // blue lattice spin flip
                    {
                        d_lattice_b[d_offset_lattice[tid] + result.i * nx + result.j] *= -1;
                    }

                    d_H[index_new] += 1;
                    d_logG[index_new] += log(factor[tid]);
                    d_energy[tid] = result.new_energy;
                }

                else
                {
                    d_H[index_old] += 1;
                    d_logG[index_old] += log(factor[tid]);
                }

                d_offset_iter[tid] += 1;
            }
        }
    }
    else
    {
        for (int it = 0; it < num_iterations; it++)
        {
            RBIM_eight_vertex result = eight_vertex_periodic_wl_step(
                d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, d_energy, d_offset_iter,
                &st, tid, nx * ny, nx, ny, num_lattices, walker_per_interactions);

            const int old_energy_int = static_cast<int>(round(d_energy[tid])); // Int cast for indexing via energy
            const int new_energy_int = static_cast<int>(round(result.new_energy));

            // If no new energy is found, set it to 0, else to tid + 1
            foundFlag[tid] = (d_expected_energy_spectrum[d_offset_energy_spectrum[interaction_id] + new_energy_int - d_start[interaction_id * num_intervals]] == 1) ? 0 : tid + 1;

            if (foundFlag[tid] != 0)
            {
                printf("new_energy %d index in spectrum %d \n", new_energy_int, new_energy_int - d_start[interaction_id * num_intervals]);
                d_newEnergies[tid] = result.new_energy;
                return;
            }

            if (result.new_energy <= d_end[interval_id] || result.new_energy >= d_start[interval_id])
            {
                int index_old = d_offset_histogramm[tid] + old_energy_int - d_start[interval_id];
                int index_new = d_offset_histogramm[tid] + new_energy_int - d_start[interval_id];

                double prob = min(1.0, exp(d_logG[index_old] - d_logG[index_new]));

                if (curand_uniform(&st) < prob)
                {
                    if (result.color)
                    { // red lattice spin flip
                        d_lattice_r[d_offset_lattice[tid] + result.i * nx + result.j] *= -1;
                    }
                    else // blue lattice spin flip
                    {
                        d_lattice_b[d_offset_lattice[tid] + result.i * nx + result.j] *= -1;
                    }
                    d_energy[tid] = result.new_energy;
                }
                d_offset_iter[tid] += 1;
            }
        }
    }

    return;
}

__global__ void find_spin_config_in_energy_range(signed char *d_lattice, signed char *d_interactions, const int nx, const int ny, const int num_lattices, const int seed, int *d_start, int *d_end, int *d_energy, int *d_offset_lattice)
{
    const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    int blockId = blockIdx.x;

    if (tid >= num_lattices)
        return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);

    int accept_spin_config = 0;

    while (accept_spin_config == 0)
    {
        if (d_energy[tid] <= d_end[blockId] && d_energy[tid] >= d_start[blockId])
        {
            // TO DO d_H and d_G update
            accept_spin_config = 1;
        }
        else
        {
            double randval = curand_uniform(&st);
            randval *= (nx * ny - 1 + 0.999999);
            int random_index = (int)trunc(randval);

            int i = random_index / ny;
            int j = random_index % ny;

            int ipp = (i + 1 < nx) ? i + 1 : 0;
            int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
            int jpp = (j + 1 < ny) ? j + 1 : 0;
            int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;

            signed char energy_diff = -2 * d_lattice[d_offset_lattice[tid] + i * ny + j] * (d_lattice[d_offset_lattice[tid] + inn * ny + j] * d_interactions[nx * ny + inn * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jnn] * d_interactions[i * ny + jnn] + d_lattice[d_offset_lattice[tid] + ipp * ny + j] * d_interactions[nx * ny + i * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jpp] * d_interactions[i * ny + j]);

            d_energy[tid] -= energy_diff;
            d_lattice[d_offset_lattice[tid] + i * ny + j] *= -1;
        }
    }

    return;
}

__device__ void store_lattice(
    signed char *d_lattice, int *d_energy, int *d_found_interval, signed char *d_store_lattice,
    const int E_min, const int nx, const int ny, const long long tid, const int len_interval,
    const int num_interval, const int int_id)
{

    int interval_index = ((d_energy[tid] - E_min) / len_interval < num_interval) ? (d_energy[tid] - E_min) / len_interval : num_interval - 1;

    if (atomicCAS(&d_found_interval[int_id * num_interval + interval_index], 0, 1) != 0)
        return;

    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            d_store_lattice[int_id * num_interval * nx * ny + interval_index * nx * ny + i * ny + j] = d_lattice[tid * nx * ny + i * ny + j];
        }
    }

    return;
}

__device__ void store_lattice(
    signed char *d_lattice_r, signed char *d_lattice_b, double *d_energy, int *d_found_interval, signed char *d_store_lattice_r, signed char *d_store_lattice_b,
    const int E_min, const int nx, const int ny, const long long tid, const int len_interval,
    const int num_interval, const int int_id)
{

    int interval_index = ((static_cast<int>(round(d_energy[tid])) - E_min) / len_interval < num_interval) ? (static_cast<int>(round(d_energy[tid])) - E_min) / len_interval : num_interval - 1;

    // Perform atomicCAS if color is true
    if (atomicCAS(&d_found_interval[int_id * num_interval + interval_index], 0, 1) != 0)
        return; // If CAS failed, interval was already claimed

    // printf("found in global %d local %d with E=%.2f \n", int_id * num_interval + interval_index, interval_index, d_energy[tid]);
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            d_store_lattice_r[int_id * num_interval * nx * ny + interval_index * nx * ny + i * ny + j] = d_lattice_r[tid * nx * ny + i * ny + j];
            d_store_lattice_b[int_id * num_interval * nx * ny + interval_index * nx * ny + i * ny + j] = d_lattice_b[tid * nx * ny + i * ny + j];
        }
    }

    return;
}

__global__ void init_indices(int *d_indices, int total_walker)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= total_walker)
        return;

    d_indices[tid] = threadIdx.x;

    return;
}

__global__ void init_offsets_histogramm(int *d_offset_histogramm, int *d_start, int *d_end, int *d_len_histograms, int num_intervals, int total_walker)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= total_walker)
        return;

    int interaction_id = blockIdx.x / num_intervals;
    long long tid_int = tid % (num_intervals * blockDim.x);

    int offset_hist = 0;
    for (int i = 0; i < interaction_id; i++)
    {
        offset_hist += d_len_histograms[i];
    }

    if (blockIdx.x % num_intervals == (num_intervals - 1))
    {
        d_offset_histogramm[tid] = offset_hist + (num_intervals - 1) * blockDim.x * (d_end[interaction_id * num_intervals] - d_start[interaction_id * num_intervals] + 1) +
                                   threadIdx.x * (d_end[(interaction_id + 1) * num_intervals - 1] - d_start[(interaction_id + 1) * num_intervals - 1] + 1);
    }
    else
    {
        d_offset_histogramm[tid] = offset_hist + tid_int * (d_end[interaction_id * num_intervals] - d_start[interaction_id * num_intervals] + 1);
    }

    return;
}

__global__ void init_offsets_lattice(int *d_offset_lattice, int nx, int ny, int num_lattices)
{
    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= num_lattices)
        return;

    d_offset_lattice[tid] = tid * nx * ny;

    return;
}

__global__ void check_histogram(
    unsigned long long *d_H, double *d_log_G, double *d_shared_logG, int *d_offset_histogramm,
    int *d_end, int *d_start, double *d_factor, int nx, int ny, double alpha, double beta,
    signed char *d_expected_energy_spectrum, int *d_len_energy_spectrum, int num_walker_total, signed char *d_cond,
    const int walker_per_interactions, const int num_intervals, int *d_offset_energy_spectrum,
    int *d_cond_interaction)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    int int_id = tid / walker_per_interactions;
    int blockId = blockIdx.x;

    if (tid >= num_walker_total)
        return;
    if (d_cond[blockId] == 1)
        return;

    if (d_cond_interaction[int_id] == -1)
        return;

    __shared__ int walkers_finished;

    if (threadIdx.x == 0)
    {
        walkers_finished = 0;
    }

    __syncthreads();

    unsigned long long min = INT_MAX;
    double average = 0;
    int len_reduced_energy_spectrum = 0;

    // Here is average and min calculation over all bins in histogram which correspond to values in expected energy spectrum
    for (int i = 0; i < (d_end[blockId] - d_start[blockId] + 1); i++)
    {
        if (d_expected_energy_spectrum[d_offset_energy_spectrum[int_id] + d_start[blockId] + i - d_start[int_id * num_intervals]] == 1)
        {
            if (d_H[d_offset_histogramm[tid] + i] < min)
            {
                min = d_H[d_offset_histogramm[tid] + i];
            }

            average += d_H[d_offset_histogramm[tid] + i];
            len_reduced_energy_spectrum += 1;
        }
    }

    __syncthreads();

    if (len_reduced_energy_spectrum > 0)
    {

        average = average / len_reduced_energy_spectrum;

        // printf("Walker %d in interval %d with min %lld average %.6f alpha %.6f alpha*average %.2f and factor %.10f and d_cond %d and end %d and start %d\n", threadIdx.x, blockIdx.x, min, average, alpha, alpha * average, d_factor[tid], d_cond[blockId], d_end[blockId], d_start[blockId]);

        if (min >= alpha * average)
        {
            atomicAdd(&walkers_finished, 1);
        }
    }
    else
    {
        printf("Error histogram has no sufficient length to check for flatness on walker %lld. \n", tid);
    }

    __syncthreads();

    if (walkers_finished == blockDim.x)
    {
        d_cond[blockId] = 1;

        for (int i = 0; i < (d_end[blockId] - d_start[blockId] + 1); i++)
        {
            d_H[d_offset_histogramm[tid] + i] = 0;
        }

        d_factor[tid] = sqrt(d_factor[tid]);
    }

    return;
}

__global__ void calc_average_log_g(
    int num_intervals_per_interaction, int *d_len_histograms,
    int num_walker_per_interval, double *d_log_G,
    double *d_shared_logG, int *d_end, int *d_start,
    signed char *d_expected_energy_spectrum, signed char *d_cond,
    int *d_offset_histogram, int *d_offset_energy_spectrum,
    int num_interactions, long long *d_offset_shared_logG,
    int *d_cond_interaction, int total_len_histogram)
{

    // block and threads as many as len_histogram_over_all_walkers
    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= total_len_histogram)
        return;

    // Calc interaction id
    int interaction_id = 0;
    for (int i = 0; i < num_interactions; i++)
    {
        if (i == num_interactions - 1)
        {
            interaction_id = i;
        }
        if (tid < d_offset_histogram[(i + 1) * num_intervals_per_interaction * num_walker_per_interval])
        {
            interaction_id = i;
            break;
        }
    }

    if (d_cond_interaction[interaction_id] == -1)
        return;

    // Index inside histogram of the interaction_id interaction
    int tid_int = tid - d_offset_histogram[interaction_id * num_intervals_per_interaction * num_walker_per_interval];
    int len_first_interval = (d_end[interaction_id * num_intervals_per_interaction] - d_start[interaction_id * num_intervals_per_interaction] + 1);
    int intervalId = (tid_int / (len_first_interval * num_walker_per_interval) < num_intervals_per_interaction) ? tid_int / (len_first_interval * num_walker_per_interval) : num_intervals_per_interaction - 1;

    int interval_over_interaction = interaction_id * num_intervals_per_interaction + intervalId;
    if (d_cond[interval_over_interaction] == 1 && tid_int < d_len_histograms[interaction_id])
    {

        int len_interval = d_end[interval_over_interaction] - d_start[interval_over_interaction] + 1;
        int energyId;

        if (intervalId != 0)
        {
            energyId = (tid_int % (len_first_interval * num_walker_per_interval * intervalId)) % len_interval;
        }
        else
        {
            energyId = tid_int % len_interval;
        }

        if (d_expected_energy_spectrum[d_offset_energy_spectrum[interaction_id] + d_start[interval_over_interaction] + energyId - d_start[interaction_id * num_intervals_per_interaction]] == 1)
        {
            atomicAdd(&d_shared_logG[d_offset_shared_logG[interval_over_interaction] + energyId], d_log_G[tid] / num_walker_per_interval);
        }
    }

    return;
}

__global__ void redistribute_g_values(
    int num_intervals_per_interaction, int *d_len_histograms, int num_walker_per_interval,
    double *d_log_G, double *d_shared_logG, int *d_end, int *d_start, double *d_factor,
    double beta, signed char *d_expected_energy_spectrum, signed char *d_cond,
    int *d_offset_histogram, int num_interactions, long long *d_offset_shared_logG,
    int *d_cond_interaction, int total_len_histogram)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= total_len_histogram)
        return;

    // Calc interaction_id
    int int_id = 0;
    for (int i = 0; i < num_interactions; i++)
    {
        if (i == num_interactions - 1)
        {
            int_id = i;
        }
        if (tid < d_offset_histogram[(i + 1) * num_intervals_per_interaction * num_walker_per_interval])
        {
            int_id = i;
            break;
        }
    }

    // Check if interaction is already finished and return if true
    if (d_cond_interaction[int_id] == -1)
        return;

    // thread id in interaction
    int tid_int = tid - d_offset_histogram[int_id * num_intervals_per_interaction * num_walker_per_interval];
    int len_first_interval = (d_end[int_id * num_intervals_per_interaction] - d_start[int_id * num_intervals_per_interaction] + 1);
    int intervalId = (tid_int / (len_first_interval * num_walker_per_interval) < num_intervals_per_interaction) ? tid_int / (len_first_interval * num_walker_per_interval) : num_intervals_per_interaction - 1;
    int interval_over_interaction = int_id * num_intervals_per_interaction + intervalId;

    // Check if in right range
    if (tid_int < d_len_histograms[int_id] && d_cond[interval_over_interaction] == 1)
    {
        int len_interval = d_end[interval_over_interaction] - d_start[interval_over_interaction] + 1;
        int energyId;

        if (intervalId != 0)
        {
            energyId = (tid_int % (len_first_interval * num_walker_per_interval * intervalId)) % len_interval;
        }
        else
        {
            energyId = tid_int % len_interval;
        }

        d_log_G[tid] = d_shared_logG[d_offset_shared_logG[interval_over_interaction] + energyId];
    }

    return;
}

__device__ RBIM periodic_boundary_random_bond_ising(
    signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_offset_lattice, unsigned long long *d_offset_iter,
    curandStatePhilox4_32_10_t *st, const long long tid, const int nx, const int ny, const int interaction_offset)
{
    double randval = curand_uniform(st);
    randval *= (nx * ny - 1 + 0.999999);
    int random_index = (int)trunc(randval);

    d_offset_iter[tid] += 1;

    int i = random_index / ny;
    int j = random_index % ny;

    int ipp = (i + 1 < nx) ? i + 1 : 0;
    int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
    int jpp = (j + 1 < ny) ? j + 1 : 0;
    int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;

    signed char energy_diff = -2 * d_lattice[d_offset_lattice[tid] + i * ny + j] * (d_lattice[d_offset_lattice[tid] + inn * ny + j] * d_interactions[interaction_offset + nx * ny + inn * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jnn] * d_interactions[interaction_offset + i * ny + jnn] + d_lattice[d_offset_lattice[tid] + ipp * ny + j] * d_interactions[interaction_offset + nx * ny + i * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jpp] * d_interactions[interaction_offset + i * ny + j]);

    int d_new_energy = d_energy[tid] - energy_diff;

    RBIM rbim;
    rbim.new_energy = d_new_energy;
    rbim.i = i;
    rbim.j = j;

    return rbim;
}

__device__ RBIM open_boundary_random_bond_ising(
    signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_offset_lattice, unsigned long long *d_offset_iter,
    curandStatePhilox4_32_10_t *st, const long long tid, const int nx, const int ny, const int interaction_offset)
{
    double randval = curand_uniform(st);
    randval *= (nx * ny - 1 + 0.999999);
    int random_index = (int)trunc(randval);

    d_offset_iter[tid] += 1;

    int i = random_index / ny;
    int j = random_index % ny;

    int ipp = (i + 1 < nx) ? i + 1 : 0;
    int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
    int jpp = (j + 1 < ny) ? j + 1 : 0;
    int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;

    int c_up = 1 - inn / (nx - 1);
    int c_down = 1 - (i + 1) / nx;
    int c_right = (j == (ny - 1)) ? 0 : 1;
    int c_left = (j == 0) ? 0 : 1;

    signed char energy_diff = -2 * d_lattice[d_offset_lattice[tid] + i * ny + j] * (c_up * d_lattice[d_offset_lattice[tid] + inn * ny + j] * d_interactions[interaction_offset + nx * ny + inn * ny + j] + c_left * d_lattice[d_offset_lattice[tid] + i * ny + jnn] * d_interactions[interaction_offset + i * ny + jnn] + c_down * d_lattice[d_offset_lattice[tid] + ipp * ny + j] * d_interactions[interaction_offset + nx * ny + i * ny + j] + c_right * d_lattice[d_offset_lattice[tid] + i * ny + jpp] * d_interactions[interaction_offset + i * ny + j]);

    int d_new_energy = d_energy[tid] - energy_diff;

    RBIM rbim;
    rbim.new_energy = d_new_energy;
    rbim.i = i;
    rbim.j = j;

    return rbim;
}

// gets called with a thread per walker
__device__ RBIM cylinder_random_bond_ising(
    signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_offset_lattice, unsigned long long *d_offset_iter,
    curandStatePhilox4_32_10_t *st, const long long tid, const int nx, const int ny, const int interaction_offset)
{
    double randval = curand_uniform(st);
    randval *= (nx * ny - 1 + 0.999999);
    int random_index = (int)trunc(randval);

    d_offset_iter[tid] += 1;

    int i = random_index / ny;
    int j = random_index % ny;

    int ipp = (i + 1) % nx;
    int inn = (i - 1 + nx) % nx;
    int jpp = (j + 1 < ny) ? j + 1 : 0;
    int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;

    int c_up = (i > 0);
    int c_down = (i < nx - 1);

    int index = d_offset_lattice[tid] + i * ny + j;
    signed char current_spin = d_lattice[index];

    // // for test
    // printf("nx: %d ny: %d i: %d j: %d ipp: %d inn: %d jpp: %d jnn: %d c_up: %d c_down: %d \n", nx, ny, i, j, ipp, inn, jpp, jnn, c_up, c_down);

    // "-" simulates spin flip and "2" stems from energy diff.
    signed char energy_diff = -2 * current_spin * (c_up * d_lattice[d_offset_lattice[tid] + inn * ny + j] * d_interactions[interaction_offset + nx * ny + inn * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jnn] * d_interactions[interaction_offset + i * ny + jnn] + c_down * d_lattice[d_offset_lattice[tid] + ipp * ny + j] * d_interactions[interaction_offset + nx * ny + i * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jpp] * d_interactions[interaction_offset + i * ny + j]);

    int d_new_energy = d_energy[tid] - energy_diff;

    RBIM rbim;
    rbim.new_energy = d_new_energy;
    rbim.i = i;
    rbim.j = j;

    return rbim;
}

__global__ void wang_landau(
    signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_start, int *d_end, unsigned long long *d_H,
    double *d_logG, int *d_offset_histogramm, int *d_offset_lattice, const int num_iterations, const int nx, const int ny,
    const int seed, double *factor, unsigned long long *d_offset_iter, signed char *d_expected_energy_spectrum, int *d_newEnergies, int *foundFlag,
    const int num_lattices, const double beta, signed char *d_cond, int boundary_type, const int walker_per_interactions, const int num_intervals,
    int *d_offset_energy_spectrum, int *d_cond_interaction)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= num_lattices)
        return;

    const int blockId = blockIdx.x;
    const int int_id = tid / walker_per_interactions;
    const int interaction_offset = int_id * 2 * nx * ny;

    if (d_cond_interaction[int_id] == -1)
        return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_offset_iter[tid], &st);

    if (d_cond[blockId] == 0)
    {

        for (int it = 0; it < num_iterations; it++)
        {

            RBIM result = rbim_func_map[boundary_type](d_lattice, d_interactions, d_energy, d_offset_lattice, d_offset_iter, &st, tid, nx, ny, interaction_offset);

            // If no new energy is found, set it to 0, else to tid + 1
            foundFlag[tid] = (d_expected_energy_spectrum[d_offset_energy_spectrum[int_id] + result.new_energy - d_start[int_id * num_intervals]] == 1) ? 0 : tid + 1;

            if (foundFlag[tid] != 0)
            {
                printf("new_energy %d index in spectrum %d \n", result.new_energy, result.new_energy - d_start[int_id * num_intervals]);
                d_newEnergies[tid] = result.new_energy;
                return;
            }

            int index_old = d_offset_histogramm[tid] + d_energy[tid] - d_start[blockId];

            if (result.new_energy > d_end[blockId] || result.new_energy < d_start[blockId])
            {

                d_H[index_old] += 1;
                d_logG[index_old] += log(factor[tid]);
            }
            else
            {
                int index_new = d_offset_histogramm[tid] + result.new_energy - d_start[blockId];
                double prob = min(1.0, exp(d_logG[index_old] - d_logG[index_new]));
                double randval = curand_uniform(&st);

                if (randval < prob)
                {
                    d_lattice[d_offset_lattice[tid] + result.i * ny + result.j] *= -1;
                    d_H[index_new] += 1;
                    d_logG[index_new] += log(factor[tid]);
                    d_energy[tid] = result.new_energy;
                }

                else
                {
                    d_H[index_old] += 1;
                    d_logG[index_old] += log(factor[tid]);
                }

                d_offset_iter[tid] += 1;
            }
        }
    }
    else
    {
        for (int it = 0; it < num_iterations; it++)
        {

            RBIM result = rbim_func_map[boundary_type](d_lattice, d_interactions, d_energy, d_offset_lattice, d_offset_iter, &st, tid, nx, ny, interaction_offset);

            // If no new energy is found, set it to 0, else to tid + 1
            foundFlag[tid] = (d_expected_energy_spectrum[d_offset_energy_spectrum[int_id] + result.new_energy - d_start[int_id * num_intervals]] == 1) ? 0 : tid + 1;

            if (foundFlag[tid] != 0)
            {
                printf("new_energy %d index in spectrum %d \n", result.new_energy, result.new_energy - d_start[int_id * num_intervals]);
                d_newEnergies[tid] = result.new_energy;
                return;
            }

            if (result.new_energy <= d_end[blockId] || result.new_energy >= d_start[blockId])
            {
                int index_old = d_offset_histogramm[tid] + d_energy[tid] - d_start[blockId];
                int index_new = d_offset_histogramm[tid] + result.new_energy - d_start[blockId];

                double prob = min(1.0, exp(d_logG[index_old] - d_logG[index_new]));

                if (curand_uniform(&st) < prob)
                {
                    d_lattice[d_offset_lattice[tid] + result.i * ny + result.j] *= -1;
                    d_energy[tid] = result.new_energy;
                }
                d_offset_iter[tid] += 1;
            }
        }
    }

    return;
}

__global__ void print_finished_walker_ratio(double *d_factor, int num_walker_total, const double exp_beta, double *d_finished_walkers_ratio)
{

    extern __shared__ int shared_count[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int threadId = threadIdx.x;

    if (threadId == 0)
    {
        shared_count[0] = 0;
    }

    __syncthreads();

    if (tid < num_walker_total)
    {
        if (d_factor[tid] <= exp_beta)
        {
            atomicAdd(&shared_count[0], 1);
        }
    }

    __syncthreads();

    if (threadId == 0)
    {
        double ratio_of_finished_walkers = (double)shared_count[0] / num_walker_total;
        printf("ratio of finished walkers: %f\n", ratio_of_finished_walkers);
        d_finished_walkers_ratio[0] = ratio_of_finished_walkers;
    }
    return;
}

double logSumExp(const std::vector<std::map<int, double>> &data)
{
    double maxVal = -std::numeric_limits<double>::infinity();

    // Get the maximum value to rescale for numerical reason
    for (const auto &data_map : data)
    {
        for (const auto &data_pair : data_map)
        {
            if (data_pair.second > maxVal)
            {
                maxVal = data_pair.second;
            }
        }
    }

    // Calculate sum of exp(values - maxVal)
    double sumExp = 0.0;
    for (const auto &data_map : data)
    {
        for (const auto &data_pair : data_map)
        {
            sumExp += std::exp(data_pair.second - maxVal);
        }
    }

    // rescale by maxVal to retrieve original log sum exp without overflow issues
    return maxVal + std::log(sumExp);
}

void rescaleMapValues(std::vector<std::map<int, double>> &data, double X, double Y)
{
    double offset = logSumExp(data);
    double log2XY = std::log(2) * X * Y; // scale to match high temperature limit of partition function - correct like this for random bond??

    for (auto &data_map : data)
    {
        for (auto &data_pair : data_map)
        {
            data_pair.second = data_pair.second + log2XY - offset;
        }
    }
}

void calc_energy(
    int blocks, int threads, const int boundary_type, signed char *lattice,
    signed char *interactions, int *d_energy, int *d_offset_lattice,
    const int nx, const int ny, const int total_walker, const int walker_per_interactions)
{

    switch (boundary_type)
    {
    case 2: // Cylinder closed horizontally
        calc_energy_cylinder<<<blocks, threads>>>(lattice, interactions, d_energy, d_offset_lattice, nx, ny, total_walker, walker_per_interactions);
        break;

    case 1: // Open boundary
        calc_energy_open_boundary<<<blocks, threads>>>(lattice, interactions, d_energy, d_offset_lattice, nx, ny, total_walker, walker_per_interactions);
        break;

    case 0: // Periodic boundary
        calc_energy_periodic_boundary<<<blocks, threads>>>(lattice, interactions, d_energy, d_offset_lattice, nx, ny, total_walker, walker_per_interactions);
        break;

    default:
        printf("Invalid boundary type!\n");
        break;
    }

    return;
}

void result_handling(
    Options options, std::vector<double> h_logG,
    std::vector<int> h_start, std::vector<int> h_end, int int_id)
{
    std::ofstream f_log_density;

    std::string boundary;

    if (options.boundary_type == 0)
    {
        boundary = "periodic";
    }
    else if (options.boundary_type == 1)
    {
        boundary = "open";
    }
    else if (options.boundary_type == 2)
    {
        boundary = "cylinder";
    }
    else
    {
        boundary = "unknown"; // Handle any unexpected boundary_type values
    }

    std::stringstream result_directory;
    result_directory << "results/" << boundary << "/prob_" << std::fixed << std::setprecision(6) << options.prob_interactions
                     << "/X_" << options.X
                     << "_Y_" << options.Y
                     << "/seed_" << options.seed_histogram + int_id
                     << "/error_class_" << options.logical_error_type;

    create_directory(result_directory.str());

    result_directory << "/intervals_" << options.num_intervals
                     << "_iterations_" << options.num_iterations
                     << "_overlap_" << options.overlap_decimal
                     << "_walkers_" << options.walker_per_interval
                     << "_seed_run_" << options.seed_run
                     << "_alpha_" << options.alpha
                     << "_beta_" << std::fixed << std::setprecision(10) << options.beta
                     << "exchange_offset" << options.replica_exchange_offset
                     << ".txt";

    f_log_density.open(result_directory.str());

    f_log_density << std::fixed << std::setprecision(10);

    int index_h_log_g = 0;
    if (f_log_density.is_open())
    {
        for (int i = 0; i < options.num_intervals; i++)
        {

            int start_energy = h_start[i];
            int end_energy = h_end[i];
            int len_int = h_end[i] - h_start[i] + 1;

            for (int j = 0; j < options.walker_per_interval; j++)
            {
                for (int k = 0; k < len_int; k++)
                {
                    f_log_density << (int)h_start[i] + k << " : " << (double)h_logG[index_h_log_g] << " ,";
                    index_h_log_g += 1;
                }
                f_log_density << std::endl;
            }
        }
    }
    f_log_density.close();
    return;
}

std::tuple<int, double> find_stitching_keys(const std::map<int, double> &current_interval, const std::map<int, double> &next_interval)
{
    int min_key = std::numeric_limits<int>::max();
    double min_diff = std::numeric_limits<double>::max();

    auto it1 = current_interval.begin();
    auto it2 = next_interval.begin();

    while (it1 != current_interval.end() && it2 != next_interval.end())
    {
        if (it1->first == it2->first)
        {
            if (std::next(it1) != current_interval.end() && std::next(it2) != next_interval.end())
            {
                double diff1 = (std::next(it1)->second - it1->second) / (std::next(it1)->first - it1->first);
                double diff2 = (std::next(it2)->second - it2->second) / (std::next(it2)->first - it2->first);

                // std::cout << "E = " << it1->first << " derivative = " << diff1 << std::endl; // sanity check

                double diff_between_intervals = std::abs(diff1 - diff2);
                if (diff_between_intervals < min_diff)
                {
                    min_diff = diff_between_intervals;
                    min_key = it1->first;
                }
            }
            ++it1;
            ++it2;
        }
        else if (it1->first < it2->first)
        {
            ++it1;
        }
        else
        {
            ++it2;
        }
    }

    return std::make_tuple(min_key, min_diff);
}

// Function to rescale the intervals for continuous concatenation
void rescale_intervals_for_concatenation(std::vector<std::map<int, double>> &interval_data, const std::vector<std::tuple<int, int>> &stitching_keys)
{
    for (size_t i = 0; i < stitching_keys.size(); ++i)
    {
        int e_concat = std::get<1>(stitching_keys[i]);
        int next_interval = std::get<0>(stitching_keys[i]);

        auto idx_in_preceding_interval = interval_data[0].find(e_concat);

        if (i != 0)
        {
            idx_in_preceding_interval = interval_data[std::get<0>(stitching_keys[i - 1])].find(e_concat);
        }

        auto idx_in_following_interval = interval_data[next_interval].find(e_concat);

        if (idx_in_preceding_interval == interval_data[i].end() || idx_in_following_interval == interval_data[next_interval].end())
        {
            throw std::runtime_error("stitching energy " + std::to_string(e_concat) + " not found in one of the intervals which may be caused by non overlapping intervals which can not be normalized properly.");
        }

        double shift_val = idx_in_preceding_interval->second - idx_in_following_interval->second; // difference by which the following interval results get affinely shifted

        // Apply the shift to all values in the following interval
        for (auto &[key, value] : interval_data[next_interval])
        {
            value += shift_val;
        }
    }
}

// Helper function to filter a map based on key conditions
std::map<int, double> filter_map_by_key(std::map<int, double> &data, int threshold, bool less_than)
{
    std::map<int, double> filtered_map;
    for (const auto &[key, value] : data)
    {
        if ((less_than && key < threshold) || (!less_than && key >= threshold))
        {
            filtered_map[key] = value;
        }
    }
    return filtered_map;
}

// Function to cut overlapping parts in the interval data based on stitching keys
std::vector<std::map<int, double>> cut_overlapping_histogram_parts(
    std::vector<std::map<int, double>> &interval_data,
    std::vector<std::tuple<int, int>> &stitching_keys)
{

    std::vector<std::map<int, double>> filtered_data;

    // Filter the first interval
    if (!stitching_keys.empty())
    {
        int first_energy = std::get<1>(stitching_keys[0]);
        auto filtered_map = filter_map_by_key(interval_data[0], first_energy, true);
        if (!filtered_map.empty())
        {
            filtered_data.push_back(filtered_map);
        }
    }

    // Filter the intermediate intervals
    for (size_t i = 1; i < stitching_keys.size(); ++i)
    {
        int energy = std::get<1>(stitching_keys[i]);
        int energy_prev = std::get<1>(stitching_keys[i - 1]);
        int previous_interval_index = std::get<0>(stitching_keys[i - 1]);
        auto filtered_map = filter_map_by_key(interval_data[previous_interval_index], energy, true);
        filtered_map = filter_map_by_key(filtered_map, energy_prev, false);

        if (!filtered_map.empty())
        {
            filtered_data.push_back(filtered_map);
        }
    }

    // Filter the last interval
    int last_energy = std::get<1>(stitching_keys.back());
    int last_interval_index = std::get<0>(stitching_keys.back());
    auto final_filtered_map = filter_map_by_key(interval_data[last_interval_index], last_energy, false);
    if (!final_filtered_map.empty())
    {
        filtered_data.push_back(final_filtered_map);
    }

    return filtered_data;
}

std::vector<std::map<int, double>> get_logG_data(std::vector<double> h_logG, std::vector<int> h_start, std::vector<int> h_end, Options options)
{
    int index_h_log_g = 0;

    // Store the results of the first walker for each interval as they are averaged already
    std::vector<std::map<int, double>> interval_data(options.num_intervals);

    for (int i = 0; i < options.num_intervals; i++)
    {
        int len_int = h_end[i] - h_start[i] + 1;

        for (int j = 0; j < options.walker_per_interval; j++)
        {
            if (j == 0)
            {
                for (int k = 0; k < len_int; k++)
                {

                    int key = h_start[i] + k;
                    double value = h_logG[index_h_log_g];

                    if (value != 0)
                    {
                        interval_data[i][key] = value; // Store the non-zero value with its key at correct map object according to interval
                    }

                    index_h_log_g += 1;
                }
            }
            else
            {
                index_h_log_g += len_int;
            }
        }
    }

    return interval_data;
}

std::vector<std::map<int, double>> rescaleByMinimum(std::vector<std::map<int, double>> &interval_data, const Options options)
{
    // finding minimum per interval
    std::vector<double> min_values(options.num_intervals, std::numeric_limits<double>::max());
    for (int i = 0; i < options.num_intervals; i++)
    {
        for (const auto &key_value_pair : interval_data[i])
        {
            if (key_value_pair.second < min_values[i])
            {
                min_values[i] = key_value_pair.second;
            }
        }

        // If no non-zero value was found, reset to 0 (or any other default)
        if (min_values[i] == std::numeric_limits<double>::max())
        {
            min_values[i] = 0;
        }
    }

    // rescaling by minimum
    for (int i = 0; i < options.num_intervals; i++)
    {
        for (auto &key_value_pair : interval_data[i])
        {
            key_value_pair.second -= min_values[i]; // each interval has a zero value now
        }
    }

    return interval_data;
}

std::vector<std::tuple<int, int>> calculate_stitching_points(std::vector<std::map<int, double>> interval_data, Options options)
{

    std::vector<std::tuple<int, int>> stitching_keys;

    for (int i = 0; i < options.num_intervals - 1; i++)
    {
        int absolute_min_key = std::numeric_limits<int>::max();
        double min_derivative = std::numeric_limits<double>::max();
        int interval_index = std::numeric_limits<int>::max();

        const auto &current_interval = interval_data[i];

        for (int j = i + 1; j < options.num_intervals; j++)
        {
            const auto &next_interval = interval_data[j];

            std::tuple<int, double> diff_derivatives = find_stitching_keys(current_interval, next_interval);

            int min_key = std::get<0>(diff_derivatives);
            double deriv = std::get<1>(diff_derivatives);

            if (min_key < std::numeric_limits<int>::max())
            {
                if (deriv < min_derivative)
                {
                    absolute_min_key = min_key;
                    min_derivative = deriv;
                    interval_index = j;
                }
            }
            else
            {
                break;
            }
        }

        stitching_keys.push_back(std::make_tuple(interval_index, absolute_min_key));
    }

    return stitching_keys;
}

void write_results(std::vector<std::map<int, double>> rescaled_data, Options options, int int_id)
{
    // From here on only write to csv
    std::stringstream result_directory;

    std::string boundary;

    if (options.boundary_type == 0)
    {
        boundary = "periodic";
    }
    else if (options.boundary_type == 1)
    {
        boundary = "open";
    }
    else if (options.boundary_type == 2)
    {
        boundary = "cylinder";
    }
    else
    {
        boundary = "unknown"; // Handle any unexpected boundary_type values
    }

    result_directory << "results/" << boundary << "/prob_" << std::fixed << std::setprecision(6) << options.prob_interactions
                     << "/X_" << options.X
                     << "_Y_" << options.Y
                     << "/error_class_" << options.logical_error_type;

    create_directory(result_directory.str());

    result_directory << "/StitchedHistogram"
                     << "_intervals_" << options.num_intervals
                     << "_iterations_" << options.num_iterations
                     << "_overlap_" << options.overlap_decimal
                     << "_walkers_" << options.walker_per_interval
                     << "_alpha_" << options.alpha
                     << "_beta_" << std::fixed << std::setprecision(10) << options.beta
                     << "_exchange_offset" << options.replica_exchange_offset
                     << ".txt";

    std::ofstream file(result_directory.str(), std::ios::app); // append mode to store multiple interaction results in same file

    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << result_directory.str() << std::endl;
        return;
    }

    file << "{\n";
    file << "  \"histogram_seed\": \"" << (options.seed_histogram + int_id) << "\",\n";
    file << "  \"run_seed\": \"" << options.seed_run << "\",\n";
    file << "  \"results\": {\n";
    file << std::fixed << std::setprecision(10);
    for (size_t i = 0; i < rescaled_data.size(); ++i)
    {
        const auto &interval_map = rescaled_data[i];
        for (auto iterator = interval_map.begin(); iterator != interval_map.end(); ++iterator)
        {
            int key = iterator->first;
            double value = iterator->second;

            // Formatting key-value pairs
            file << "      \"" << key << "\": " << value;

            // Add a comma unless it's the last element
            if (std::next(iterator) != interval_map.end() || i < rescaled_data.size() - 1)
            {
                file << ",";
            }
            file << "\n";
        }
    }
    file << "}\n";
    file << "},\n";
    file.close();
}

void result_handling_stitched_histogram(
    Options options, std::vector<double> h_logG,
    std::vector<int> h_start, std::vector<int> h_end, int int_id)
{

    std::vector<std::map<int, double>> interval_data = get_logG_data(h_logG, h_start, h_end, options);

    std::vector<std::map<int, double>> rescaled_data = rescaleByMinimum(interval_data, options);

    std::vector<std::tuple<int, int>> stitching_keys = calculate_stitching_points(rescaled_data, options);

    std::vector<std::tuple<int, int>> real_stitching_keys;

    for (int i = 0; i < stitching_keys.size(); i++)
    {
        int energy_key = std::get<1>(stitching_keys[i]);

        bool check_intersection = true;

        for (int j = i + 1; j < stitching_keys.size(); j++)
        {
            if (std::get<1>(stitching_keys[j]) <= energy_key)
            {
                check_intersection = false;
            }
        }
        if (check_intersection)
        {
            real_stitching_keys.push_back(stitching_keys[i]);
        }
    }

    // Vector to store the smallest values for each key
    std::vector<std::tuple<int, int>> smallest_values;

    // Initialize the first key-value tuple
    int current_key = std::get<0>(real_stitching_keys[0]);
    int current_min_value = std::get<1>(real_stitching_keys[0]);

    // Iterate over the key-value tuples (assuming sorted by key)
    for (const auto &tuple : real_stitching_keys)
    {
        int key = std::get<0>(tuple);   // Get the key from the tuple
        int value = std::get<1>(tuple); // Get the value from the tuple

        // If the key changes, store the smallest value for the previous key
        if (key != current_key)
        {
            smallest_values.push_back(std::make_tuple(current_key, current_min_value)); // Save the smallest value
            current_key = key;
            current_min_value = value; // reset for the new key
        }
        else
        {
            // If the key is the same, update the minimum value
            if (value < current_min_value)
            {
                current_min_value = value;
            }
        }
    }

    // Don't forget to add the last key-value pair after the loop ends
    smallest_values.push_back(std::make_tuple(current_key, current_min_value));

    rescale_intervals_for_concatenation(rescaled_data, smallest_values);

    std::vector<std::map<int, double>> cut_data = cut_overlapping_histogram_parts(rescaled_data, smallest_values);

    rescaleMapValues(cut_data, options.X, options.Y); // rescaling for high temperature interpretation of partition function

    write_results(cut_data, options, int_id);
}

void eight_vertex_result_handling_stitched_histogram(
    Options options, std::vector<double> h_logG, float error_mean, float error_variance,
    float prob_x, float prob_y, float prob_z, std::vector<int> h_start, std::vector<int> h_end, int int_id,
    bool isQubitSpecificNoise, bool x_horizontal_error, bool x_vertical_error, bool z_horizontal_error,
    bool z_vertical_error)
{

    std::vector<std::map<int, double>> interval_data = get_logG_data(h_logG, h_start, h_end, options);

    std::vector<std::map<int, double>> rescaled_data = rescaleByMinimum(interval_data, options);

    std::vector<std::tuple<int, int>> stitching_keys = calculate_stitching_points(rescaled_data, options);

    // Filter stitching keys based on energy keys
    std::vector<std::tuple<int, int>> real_stitching_keys;

    for (int i = 0; i < stitching_keys.size(); i++)
    {
        int energy_key = std::get<1>(stitching_keys[i]);

        bool check_intersection = true;

        for (int j = i + 1; j < stitching_keys.size(); j++)
        {
            if (std::get<1>(stitching_keys[j]) <= energy_key)
            {
                check_intersection = false;
            }
        }
        if (check_intersection)
        {
            real_stitching_keys.push_back(stitching_keys[i]);
        }
    }

    // Filter based on intervals
    std::vector<std::tuple<int, int>> smallest_values;

    // Initialize the first key-value tuple
    int current_key = std::get<0>(real_stitching_keys[0]);
    int current_min_value = std::get<1>(real_stitching_keys[0]);

    // Iterate over the key-value tuples (assuming sorted by key)
    for (const auto &tuple : real_stitching_keys)
    {
        int key = std::get<0>(tuple);   // Get the key from the tuple
        int value = std::get<1>(tuple); // Get the value from the tuple

        // If the key changes, store the smallest value for the previous key
        if (key != current_key)
        {
            smallest_values.push_back(std::make_tuple(current_key, current_min_value)); // Save the smallest value
            current_key = key;
            current_min_value = value; // reset for the new key
        }
        else
        {
            // If the key is the same, update the minimum value
            if (value < current_min_value)
            {
                current_min_value = value;
            }
        }
    }

    // Don't forget to add the last key-value pair after the loop ends
    smallest_values.push_back(std::make_tuple(current_key, current_min_value));

    rescale_intervals_for_concatenation(rescaled_data, smallest_values);

    std::vector<std::map<int, double>> cut_data = cut_overlapping_histogram_parts(rescaled_data, smallest_values);

    rescaleMapValues(interval_data, options.X, options.Y); // rescaling for high temperature interpretation of partition function

    // From here on only write to csv
    std::stringstream result_directory;

    std::string error_string = std::to_string(x_horizontal_error) + std::to_string(x_vertical_error) + std::to_string(z_horizontal_error) + std::to_string(z_vertical_error);
    std::string boundary = "periodic";

    result_directory << std::fixed << std::setprecision(6);
    if (isQubitSpecificNoise)
    {
        result_directory << "results/eight_vertex/"
                         << boundary
                         << "/qubit_specific_noise_1"
                         << "/error_mean_" << error_mean
                         << "/error_variance_" << std::fixed << std::setprecision(6) << error_variance
                         << "/X_" << options.X
                         << "_Y_" << options.Y
                         << "/error_class_" << error_string;
    }
    else
    {
        result_directory << "results/eight_vertex/"
                         << boundary
                         << "/qubit_specific_noise_0"
                         << "/prob_x_" << prob_x
                         << "/prob_y_" << prob_y
                         << "/prob_z_" << prob_z
                         << "/X_" << options.X
                         << "_Y_" << options.Y
                         << "/error_class_" << error_string;
    }

    create_directory(result_directory.str());

    result_directory << "/StitchedHistogram_"
                     << "_intervals_" << options.num_intervals
                     << "_iterations_" << options.num_iterations
                     << "_overlap_" << options.overlap_decimal
                     << "_walkers_" << options.walker_per_interval
                     << "_alpha_" << options.alpha
                     << "_beta_" << std::fixed << std::setprecision(10) << options.beta
                     << "_exchange_offset_" << options.replica_exchange_offset
                     << ".txt";

    std::ofstream file(result_directory.str(), std::ios::app); // append mode to store multiple interaction results in same file

    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << result_directory.str() << std::endl;
        return;
    }

    file << "{\n";
    file << "  \"histogram_seed\": \"" << (options.seed_histogram + int_id) << "\",\n";
    file << "  \"run_seed\": \"" << options.seed_run << "\",\n";
    file << "  \"results\": [\n";
    file << std::fixed << std::setprecision(10);
    for (size_t i = 0; i < cut_data.size(); ++i)
    {
        const auto &interval_map = cut_data[i];
        for (auto iterator = interval_map.begin(); iterator != interval_map.end(); ++iterator)
        {
            int key = iterator->first;
            double value = iterator->second;

            // Formatting key-value pairs
            file << "      \"" << key << "\": " << value;

            // Add a comma unless it's the last element
            if (std::next(iterator) != interval_map.end() || i < cut_data.size() - 1)
            {
                file << ",";
            }
            file << "\n";
        }
    }
    file << "]\n";
    file << "}\n";
    file.close();
}

__global__ void check_sums(int *d_cond_interactions, int num_intervals, int num_interactions)
{

    const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= num_interactions)
        return;

    if (d_cond_interactions[tid] == num_intervals)
    {
        d_cond_interactions[tid] = -1;
    }
}

void check_interactions_finished(
    signed char *d_cond, int *d_cond_interactions,
    int *d_offset_intervals, int num_intervals, int num_interactions)
{

    // Temporary storage size
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // Determine the amount of temporary storage needed
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_cond, d_cond_interactions, num_interactions, d_offset_intervals, d_offset_intervals + 1);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Perform the segmented reduction
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_cond, d_cond_interactions, num_interactions, d_offset_intervals, d_offset_intervals + 1);
    cudaDeviceSynchronize();

    check_sums<<<num_interactions, 1>>>(d_cond_interactions, num_intervals, num_interactions);
    cudaDeviceSynchronize();

    cudaFree(d_temp_storage);
}

__global__ void generate_pauli_errors(int *pauli_errors, const int num_qubits, const int X, const int num_interactions, const unsigned long seed, const double *p_I, const double *p_X, const double *p_Y, const double *p_Z, const bool x_horizontal_error, const bool x_vertical_error, const bool z_horizontal_error, const bool z_vertical_error)
{
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_qubits * num_interactions)
    {
        int interaction_id = idx / num_qubits;
        int qubit_id = idx % num_qubits;

        curandState state;
        curand_init(seed + interaction_id, idx, 0, &state);
        double rand_val = curand_uniform(&state); // between 0 and 1 here
        if (rand_val < p_I[qubit_id])
        {
            pauli_errors[idx] = 0; // I
        }
        else if (rand_val < p_I[qubit_id] + p_X[qubit_id])
        {
            pauli_errors[idx] = 1; // X
        }
        else if (rand_val < p_I[qubit_id] + p_X[qubit_id] + p_Y[qubit_id])
        {
            pauli_errors[idx] = 2; // Y
        }
        else
        {
            pauli_errors[idx] = 3; // Z
        }

        int i = qubit_id / X; // row index of qubit
        int j = qubit_id % X; // column index of qubit

        // // TEST BLOCK
        // // ------
        // int before_commutator_pauli = 0;
        // if (pauli_errors[idx] != 0)
        // {
        //     before_commutator_pauli = pauli_errors[idx];
        // }
        // // ------

        // here goes error chain application
        if (i == 0)
        {
            if (x_horizontal_error)
            {
                pauli_errors[idx] = commutator(1, pauli_errors[idx]); // x commutator with Pauli on qubit
            }
            if (z_horizontal_error)
            {
                pauli_errors[idx] = commutator(3, pauli_errors[idx]); // z commutator with Pauli on qubit
            }
        }
        if (j == 0 && i % 2 == 0) // IMPORTANT the row condition here is included as error chain shall go down a single column of qubits and not do a zigzag - this must be edited when changing from Y-Y/2 in input
        {
            if (x_vertical_error)
            {
                pauli_errors[idx] = commutator(1, pauli_errors[idx]); // x commutator with Pauli on qubit
            }
            if (z_vertical_error)
            {
                pauli_errors[idx] = commutator(3, pauli_errors[idx]); // z commutator with Pauli on qubit
            }
        }
        // // TEST BLOCK
        // // ------
        // if (before_commutator_pauli != pauli_errors[idx])
        // {
        //     printf("interaction %d walker %lld i %d j %d pauli before %d pauli after % d\n", interaction_id, idx % num_qubits, i, j, before_commutator_pauli, pauli_errors[idx]);
        // }
        // // ------
    }
}

__global__ void generate_pauli_errors(int *pauli_errors, const int num_qubits, const int X, const int num_interactions, const unsigned long seed, const double p_I, const double p_X, const double p_Y, const double p_Z, const bool x_horizontal_error, const bool x_vertical_error, const bool z_horizontal_error, const bool z_vertical_error)
{
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_qubits * num_interactions)
    {
        int interaction_id = idx / num_qubits;
        curandState state;
        curand_init(seed + interaction_id, idx, 0, &state);
        double rand_val = curand_uniform(&state); // between 0 and 1 here
        if (rand_val < p_I)
        {
            pauli_errors[idx] = 0; // I
        }
        else if (rand_val < p_I + p_X)
        {
            pauli_errors[idx] = 1; // X
        }
        else if (rand_val < p_I + p_X + p_Y)
        {
            pauli_errors[idx] = 2; // Y
        }
        else
        {
            pauli_errors[idx] = 3; // Z
        }

        int i = idx % num_qubits / X; // row index of qubit
        int j = idx % num_qubits % X; // column index of qubit

        // // TEST BLOCK
        // // ------
        // int before_commutator_pauli = 0;
        // if (pauli_errors[idx] != 0)
        // {
        //     before_commutator_pauli = pauli_errors[idx];
        // }
        // // ------

        // here goes error chain application
        if (i == 0)
        {
            if (x_horizontal_error)
            {
                pauli_errors[idx] = commutator(1, pauli_errors[idx]); // x commutator with Pauli on qubit
            }
            if (z_horizontal_error)
            {
                pauli_errors[idx] = commutator(3, pauli_errors[idx]); // z commutator with Pauli on qubit
            }
        }
        if (j == 0 && i % 2 == 0) // IMPORTANT the row condition here is included as error chain shall go down a single column of qubits and not do a zigzag - this must be edited when changing from Y-Y/2 in input
        {
            if (x_vertical_error)
            {
                pauli_errors[idx] = commutator(1, pauli_errors[idx]); // x commutator with Pauli on qubit
            }
            if (z_vertical_error)
            {
                pauli_errors[idx] = commutator(3, pauli_errors[idx]); // z commutator with Pauli on qubit
            }
        }
        // // TEST BLOCK
        // // ------
        // if (before_commutator_pauli != pauli_errors[idx])
        // {
        //     printf("interaction %d walker %lld i %d j %d pauli before %d pauli after % d\n", interaction_id, idx % num_qubits, i, j, before_commutator_pauli, pauli_errors[idx]);
        // }
        // // ------
    }
}

__device__ int commutator(int pauli1, int pauli2)
{
    // images of pauli2 under commutator with I: 0
    int mapping0[] = {0, 1, 2, 3}; // 0: I, 1: X, 2: Y, 3: Z

    // images of pauli2 under commutator with x: 1
    int mapping1[] = {1, 0, 3, 2}; // 0: I, 1: X, 2: Y, 3: Z

    // images of pauli2 under commutator with y: 2
    int mapping2[] = {2, 3, 0, 1}; // 0: I, 1: X, 2: Y, 3: Z

    // images of pauli2 under commutator with z: 3
    int mapping3[] = {3, 2, 1, 0}; // 0: I, 1: X, 2: Y, 3: Z

    int *mapping[] = {mapping0, mapping1, mapping2, mapping3};

    // Return the mapped value
    if (pauli1 >= 0 && pauli1 < 4 && pauli2 >= 0 && pauli2 < 4)
    {
        return mapping[pauli1][pauli2];
    }
    else
    {
        return -1;
    }
}

// this is not really a commutator as it is symmetric right? It gives 1 iff commuting and -1 iff not.
__device__ int scalar_commutator(int pauli1, int pauli2)
{
    // The action here should be compatible with the definition in https://arxiv.org/pdf/1809.10704
    //  Pauli operators are stored as: I = 0, X = 1, Y = 2, Z = 3 which yields:
    if ((pauli1 == 0 || pauli2 == 0) || pauli1 == pauli2)
        return 1;
    else
        return -1;
}

__global__ void get_interaction_from_commutator(int *pauli_errors, double *int_X, double *int_Y, double *int_Z, const int num_qubits, const int num_interactions, double J_X, double J_Y, double J_Z)
{
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_interactions * num_qubits)
    {

        int pauli = pauli_errors[idx];

        double comm_result_X = scalar_commutator(pauli, 1); // * J_X;
        double comm_result_Y = scalar_commutator(pauli, 2); // * J_Y;
        double comm_result_Z = scalar_commutator(pauli, 3); //  * J_Z;

        int_X[idx] = -comm_result_X * J_X;
        int_Y[idx] = -comm_result_Y * J_Y;
        int_Z[idx] = -comm_result_Z * J_Z;

        // printf("idx %lld int_X %f int_Y %f int_Z %f \n", idx, int_X[idx], int_Y[idx], int_Z[idx]);
    }
}

__global__ void get_interaction_from_commutator(int *pauli_errors, double *int_X, double *int_Y, double *int_Z, const int num_qubits, const int num_interactions, double *J_X, double *J_Y, double *J_Z)
{
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_interactions * num_qubits)
    {
        int qubit_id = idx % num_qubits;
        int pauli = pauli_errors[idx];

        double comm_result_X = scalar_commutator(pauli, 1);
        double comm_result_Y = scalar_commutator(pauli, 2);
        double comm_result_Z = scalar_commutator(pauli, 3);

        int_X[idx] = -comm_result_X * J_X[qubit_id];
        int_Y[idx] = -comm_result_Y * J_Y[qubit_id];
        int_Z[idx] = -comm_result_Z * J_Z[qubit_id];

        // printf("idx %lld int_X %f int_Y %f int_Z %f \n", idx, int_X[idx], int_Y[idx], int_Z[idx]);
    }
}

__global__ void init_interactions_eight_vertex(double *int_X, double *int_Y, double *int_Z, const int num_qubits, const int num_interactions, int X, int Y, double *int_r, double *int_b, double *d_interactions_down_four_body, double *d_interactions_right_four_body)
{

    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_interactions * num_qubits)
    {

        const int idx = tid % num_qubits;    // idx on the threads' interaction
        const int int_id = tid / num_qubits; // identifier of the threads' interaction

        const int offset_interactions_closed_on_sublattice = int_id * num_qubits; // offset on interaction arrays acting closed on sublattices
        const int offset_interactions_four_body = int_id * num_qubits / 2;        // offset on four body interaction arrays

        int i = idx / X; // row index
        int j = idx % X; // columns index

        double interaction_x = int_X[tid];
        double interaction_z = int_Z[tid];
        double interaction_y = int_Y[tid];

        if (i % 2 == 0)
        {
            // in even rows is x right interaction and z down. The 1/2 comes from ordering scheme as in pure bit flip implementation with first rows side neighbors and following once lateral.
            // in even rows is four body term a right version.
            if (j == 0)
            {
                int_b[offset_interactions_closed_on_sublattice + i / 2 * X + X - 1] = interaction_x;
                // printf("idx %lld int_b side at %d\n",idx, i/2*X+X-1);

                d_interactions_right_four_body[offset_interactions_four_body + i / 2 * X + X - 1] = interaction_y;
                // printf("idx %lld right four body at %d\n",idx, i/2*X+X-1);
            }
            else
            {
                int_b[offset_interactions_closed_on_sublattice + i / 2 * X + j - 1] = interaction_x;
                // printf("idx %lld int_b side at %d\n",idx, i/2*X+j-1);

                d_interactions_right_four_body[offset_interactions_four_body + i / 2 * X + j - 1] = interaction_y;
                // printf("idx %lld right four body at %d\n",idx, i/2*X+j-1);
            }
            if (i == 0)
            {
                int_r[offset_interactions_closed_on_sublattice + (Y - 1) * X + j] = interaction_z;
                // printf("idx %lld int_r down at %d\n",idx, (Y-1)*X+j);
            }
            else
            {
                int_r[offset_interactions_closed_on_sublattice + (Y / 2) * X + ((i / 2) - 1) * X + j] = interaction_z;
                // printf("idx %lld int_r down at %d\n",idx, (Y/2)*X+((i/2)-1)*X+j);
            }
        }

        else
        {
            // in odd rows is x down interaction and z right.
            // in odd rows is four body term a downward version.
            int_r[offset_interactions_closed_on_sublattice + (i / 2) * X + j] = interaction_z;
            // printf("idx %lld int_r side at %d\n",idx, (i/2)*X+j);

            int_b[offset_interactions_closed_on_sublattice + (Y / 2) * X + (i / 2) * X + j] = interaction_x;
            // printf("idx %lld int_b down at %d\n",idx, (Y/2)*X+(i/2)*X+j);

            d_interactions_down_four_body[offset_interactions_four_body + i / 2 * X + j] = interaction_y;
            // printf("idx %lld down four body at %d\n",idx, i/2*X+j);
        }
    }
}

__device__ double calc_energy_periodic_eight_vertex(signed char *lattice_b, signed char *lattice_r, double *interactions_b, double *interactions_r, double *interactions_four_body_right, double *interactions_four_body_down, const int num_qubits, const int X, const int Y, const int num_lattices_x_interaction)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    // const int lattice_in_interaction = tid % num_lattices_x_interaction;
    const int int_id = tid / num_lattices_x_interaction;

    const int offset_lattice = tid * num_qubits / 2;                          // offset on b r lattice arrays
    const int offset_interactions_closed_on_sublattice = int_id * num_qubits; // offset on interaction arrays acting closed on sublattices
    const int offset_interactions_four_body = int_id * num_qubits / 2;        // offset on four body interaction arrays

    double energy = 0;

    for (int l = 0; l < X * Y / 2; l++)
    {                  // dim of both Ising lattices is (X, Y/2)
        int i = l / X; // row index
        int j = l % X; // column index

        // these neighbor indices are used for interactions closed under an Ising lattice
        int i_dn = (i + 1 < Y / 2) ? i + 1 : 0;          // down neighbor row index
        int i_un = (i - 1 >= 0) ? (i - 1) : (Y / 2 - 1); // up neighbor row index
        int j_rn = (j + 1 < X) ? j + 1 : 0;              // right neighbor column index

        // these indices are used for right four body interaction
        int right_four_body_side_b = i * X + j_rn;
        int right_four_body_up_r = i_un * X + j_rn;
        int right_four_body_down_r = i * X + j_rn;
        // printf("i %d j %d : right_left_b %d right_side_b %d right_up_r %d right_down_r %d \n", i, j, i * X + j, right_four_body_side_b, right_four_body_up_r, right_four_body_down_r);

        // these indices are used for down four body interaction
        int down_four_body_left_r = i * X + j;
        int down_four_body_right_r = i * X + j_rn;
        int down_four_body_down_b = i_dn * X + j;
        // printf("i %d j %d : down_up_b %d down_down_b %d down_left_r %d down_right_r %d \n", i, j, i * X + j, i_dn * X + j, i * X + j, i * X + j_rn);

        energy += lattice_b[offset_lattice + i * X + j] * (lattice_b[offset_lattice + i_dn * X + j] * interactions_b[offset_interactions_closed_on_sublattice + num_qubits / 2 + i * X + j] + lattice_b[offset_lattice + i * X + j_rn] * interactions_b[offset_interactions_closed_on_sublattice + i * X + j]) + lattice_r[offset_lattice + i * X + j] * (lattice_r[offset_lattice + i_dn * X + j] * interactions_r[offset_interactions_closed_on_sublattice + num_qubits / 2 + i * X + j] + lattice_r[offset_lattice + i * X + j_rn] * interactions_r[offset_interactions_closed_on_sublattice + i * X + j]) + interactions_four_body_right[offset_interactions_four_body + i * X + j] * (lattice_b[offset_lattice + i * X + j] * lattice_b[offset_lattice + right_four_body_side_b] * lattice_r[offset_lattice + right_four_body_up_r] * lattice_r[offset_lattice + right_four_body_down_r]) + interactions_four_body_down[offset_interactions_four_body + i * X + j] * (lattice_b[offset_lattice + i * X + j] * lattice_b[offset_lattice + down_four_body_down_b] * lattice_r[offset_lattice + down_four_body_left_r] * lattice_r[offset_lattice + down_four_body_right_r]);
    }

    // if (tid == 0)
    // {
    //     printf("%.2f energies in calc_energy \n", energy);
    // }
    return energy;
}

__global__ void calc_energy_eight_vertex(double *energy_out, signed char *lattice_b, signed char *lattice_r, double *interactions_b, double *interactions_r, double *interactions_four_body_right, double *interactions_four_body_down, const int num_qubits, const int X, const int Y, const int num_lattices, const int num_lattices_x_interaction)
{
    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid < num_lattices)
    {
        energy_out[tid] = calc_energy_periodic_eight_vertex(lattice_b, lattice_r, interactions_b, interactions_r, interactions_four_body_right, interactions_four_body_down, num_qubits, X, Y, num_lattices_x_interaction);
    }
}

// gets called with a thread per walker
__device__ RBIM_eight_vertex eight_vertex_periodic_wl_step(
    signed char *d_lattice_b, signed char *d_lattice_r, double *d_interactions_b, double *d_interactions_r, double *d_interactions_four_body_right, double *d_interactions_four_body_down, double *d_energy, unsigned long long *d_offset_iter,
    curandStatePhilox4_32_10_t *st, const long long tid, const int num_qubits, const int X, const int Y, const int num_lattices, const int num_lattices_x_interaction)
{
    double randval = curand_uniform(st);
    randval *= (num_qubits - 1 + 0.999999); // num qubits is spin count over both sublattices
    int random_index = (int)trunc(randval);

    d_offset_iter[tid] += 1;

    const int int_id = tid / num_lattices_x_interaction;

    const int offset_lattice = tid * num_qubits / 2;                          // offset on b r lattice arrays
    const int offset_interactions_closed_on_sublattice = int_id * num_qubits; // offset on interaction arrays acting closed on sublattices
    const int offset_interactions_four_body = int_id * num_qubits / 2;        // offset on four body interaction arrays

    bool color = (random_index / (num_qubits / 2)) % 2; // which sublattice the spin gets flipped on. 0: b, 1: r
    int i = (random_index % (num_qubits / 2)) / X;      // row index on sublattice
    int j = (random_index % (num_qubits / 2)) % X;      // columns index on sublattice

    // these neighbor indices are used for interactions closed under an Ising lattice
    int i_dn = (i + 1 < Y / 2) ? (i + 1) : 0;    // down neighbor row index
    int i_un = (i - 1 >= 0) ? i - 1 : Y / 2 - 1; // up neighbor row index
    int j_rn = (j + 1 < X) ? (j + 1) : 0;        // right neighbor column index
    int j_ln = (j - 1 >= 0) ? j - 1 : X - 1;     // left neighbor column index

    double energy_diff = 0;

    if (!color) // color is blue
    {
        // these indices are used for right four body interaction (the one with blue spins on horizontal line and the right refers to storage of coupling strength labeled by blue spin at left end of the cross term) with root spin at left position
        int right_four_body_term_right_version_side_b = i * X + j_rn;
        int right_four_body_term_right_version_up_r = i_un * X + j_rn;
        int right_four_body_term_right_version_down_r = i * X + j_rn;

        // these indices are used for right four body interaction (the one with blue spins on horizontal line) with root spin at right position
        int right_four_body_term_left_version_side_b = i * X + j_ln;
        int right_four_body_term_left_version_up_r = i_un * X + j;
        int right_four_body_term_left_version_down_r = i * X + j;

        // these indices are used for down four body interaction (the one with blue spin on vertical line and the down refers to storage of coupling strength labeled by blue spin at up most end of the cross term) with root spin at up position
        int down_four_body_term_down_version_left_r = i * X + j;
        int down_four_body_term_down_version_right_r = i * X + j_rn;
        int down_four_body_term_down_version_down_b = i_dn * X + j;

        // these indices are used for down four body interaction (the one with blue spin on vertical line) with root spin at down position
        int down_four_body_term_up_version_left_r = i_un * X + j;
        int down_four_body_term_up_version_right_r = i_un * X + j_rn;
        int down_four_body_term_up_version_up_b = i_un * X + j;

        energy_diff = -2 * d_lattice_b[offset_lattice + i * X + j] * (d_lattice_b[offset_lattice + i_un * X + j] * d_interactions_b[offset_interactions_closed_on_sublattice + num_qubits / 2 + i_un * X + j] + d_lattice_b[offset_lattice + i_dn * X + j] * d_interactions_b[offset_interactions_closed_on_sublattice + num_qubits / 2 + i * X + j] + d_lattice_b[offset_lattice + i * X + j_rn] * d_interactions_b[offset_interactions_closed_on_sublattice + i * X + j] + d_lattice_b[offset_lattice + i * X + j_ln] * d_interactions_b[offset_interactions_closed_on_sublattice + i * X + j_ln] + d_interactions_four_body_right[offset_interactions_four_body + i * X + j] * (d_lattice_b[offset_lattice + right_four_body_term_right_version_side_b] * d_lattice_r[offset_lattice + right_four_body_term_right_version_up_r] * d_lattice_r[offset_lattice + right_four_body_term_right_version_down_r]) + d_interactions_four_body_down[offset_interactions_four_body + i * X + j] * (d_lattice_b[offset_lattice + down_four_body_term_down_version_down_b] * d_lattice_r[offset_lattice + down_four_body_term_down_version_left_r] * d_lattice_r[offset_lattice + down_four_body_term_down_version_right_r]) + d_interactions_four_body_right[offset_interactions_four_body + right_four_body_term_left_version_side_b] * (d_lattice_b[offset_lattice + right_four_body_term_left_version_side_b] * d_lattice_r[offset_lattice + right_four_body_term_left_version_up_r] * d_lattice_r[offset_lattice + right_four_body_term_left_version_down_r]) + d_interactions_four_body_down[offset_interactions_four_body + down_four_body_term_up_version_up_b] * (d_lattice_b[offset_lattice + down_four_body_term_up_version_up_b] * d_lattice_r[offset_lattice + down_four_body_term_up_version_left_r] * d_lattice_r[offset_lattice + down_four_body_term_up_version_right_r]));
    }
    else // color is red
    {
        // these indices are used for right four body interaction (the one with blue spins on horizontal line and the right refers to storage of coupling strength labeled by blue spin at left end of the cross term) with red root spin at up position
        int right_four_body_term_up_version_left_b = i_dn * X + j_ln; // roots the coupling strength
        int right_four_body_term_up_version_right_b = i_dn * X + j;
        int right_four_body_term_up_version_down_r = i_dn * X + j;

        // these indices are used for right four body interaction (the one with blue spins on horizontal line) with red root spin at down position
        int right_four_body_term_down_version_left_b = i * X + j_ln; // roots the coupling strength
        int right_four_body_term_down_version_right_b = i * X + j;
        int right_four_body_term_down_version_up_r = i_un * X + j;

        // these indices are used for down four body interaction (the one with blue spin on vertical line and the down refers to storage of coupling strength labeled by blue spin at up most end of the cross term) with red root spin at left position
        int down_four_body_term_left_version_right_r = i * X + j_rn;
        int down_four_body_term_left_version_up_b = i * X + j; // roots the coupling strength
        int down_four_body_term_left_version_down_b = i_dn * X + j;

        // these indices are used for down four body interaction (the one with blue spin on vertical line) with red root spin at right position
        int down_four_body_term_right_version_left_r = i * X + j_ln;
        int down_four_body_term_right_version_up_b = i * X + j_ln; // roots the coupling strength
        int down_four_body_term_right_version_down_b = i_dn * X + j_ln;

        double E_up = d_lattice_r[offset_lattice + i * X + j] * (d_lattice_r[offset_lattice + i_un * X + j] * d_interactions_r[offset_interactions_closed_on_sublattice + num_qubits / 2 + i_un * X + j]);
        double E_down = d_lattice_r[offset_lattice + i * X + j] * (d_lattice_r[offset_lattice + i_dn * X + j] * d_interactions_r[offset_interactions_closed_on_sublattice + num_qubits / 2 + i * X + j]);
        double E_right = d_lattice_r[offset_lattice + i * X + j] * (d_lattice_r[offset_lattice + i * X + j_rn] * d_interactions_r[offset_interactions_closed_on_sublattice + i * X + j]);
        double E_left = d_lattice_r[offset_lattice + i * X + j] * (d_lattice_r[offset_lattice + i * X + j_ln] * d_interactions_r[offset_interactions_closed_on_sublattice + i * X + j_ln]);
        double E_right_four_body_up_version = d_lattice_r[offset_lattice + i * X + j] * (d_interactions_four_body_right[offset_interactions_four_body + right_four_body_term_up_version_left_b] * (d_lattice_b[offset_lattice + right_four_body_term_up_version_left_b] * d_lattice_b[offset_lattice + right_four_body_term_up_version_right_b] * d_lattice_r[offset_lattice + right_four_body_term_up_version_down_r]));
        double E_right_four_body_down_version = d_lattice_r[offset_lattice + i * X + j] * (d_interactions_four_body_right[offset_interactions_four_body + right_four_body_term_down_version_left_b] * (d_lattice_b[offset_lattice + right_four_body_term_down_version_left_b] * d_lattice_b[offset_lattice + right_four_body_term_down_version_right_b] * d_lattice_r[offset_lattice + right_four_body_term_down_version_up_r]));
        double E_down_four_body_left_version = d_lattice_r[offset_lattice + i * X + j] * (d_interactions_four_body_down[offset_interactions_four_body + down_four_body_term_right_version_up_b] * (d_lattice_b[offset_lattice + down_four_body_term_right_version_up_b] * d_lattice_b[offset_lattice + down_four_body_term_right_version_down_b] * d_lattice_r[offset_lattice + down_four_body_term_right_version_left_r]));
        double E_down_four_body_right_version = d_lattice_r[offset_lattice + i * X + j] * (d_interactions_four_body_down[offset_interactions_four_body + down_four_body_term_left_version_up_b] * (d_lattice_b[offset_lattice + down_four_body_term_left_version_up_b] * d_lattice_b[offset_lattice + down_four_body_term_left_version_down_b] * d_lattice_r[offset_lattice + down_four_body_term_left_version_right_r]));

        energy_diff = -2 * (E_up + E_down + E_right + E_left + E_right_four_body_down_version + E_right_four_body_up_version + E_down_four_body_left_version + E_down_four_body_right_version);

        // if (tid == 0)
        // {
        // printf("E_up=%.6f E_down=%.6f E_right=%.6f E_left=%.6f E_right_four_body_up_version=%.6f E_right_four_body_down_version=%.6f E_down_four_body_left_version=%.6f E_down_four_body_right_version=%.6f \n", E_up, E_down, E_right, E_left, E_right_four_body_up_version, E_right_four_body_down_version, E_down_four_body_left_version, E_down_four_body_right_version);
        // printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n", d_lattice_r[offset_lattice + i * X + j], d_lattice_r[offset_lattice + i_un * X + j], d_lattice_r[offset_lattice + i_dn * X + j], d_lattice_r[offset_lattice + i * X + j_rn], d_lattice_r[offset_lattice + i * X + j_ln], d_lattice_b[offset_lattice + right_four_body_term_up_version_left_b], d_lattice_b[offset_lattice + right_four_body_term_up_version_right_b], d_lattice_r[offset_lattice + right_four_body_term_up_version_down_r], d_lattice_b[offset_lattice + right_four_body_term_down_version_left_b], d_lattice_b[offset_lattice + right_four_body_term_down_version_right_b], d_lattice_r[offset_lattice + right_four_body_term_down_version_up_r], d_lattice_b[offset_lattice + down_four_body_term_left_version_up_b], d_lattice_b[offset_lattice + down_four_body_term_left_version_down_b], d_lattice_r[offset_lattice + down_four_body_term_left_version_right_r], d_lattice_b[offset_lattice + down_four_body_term_right_version_up_b], d_lattice_b[offset_lattice + down_four_body_term_right_version_down_b], d_lattice_r[offset_lattice + down_four_body_term_right_version_left_r]);
        // // print indices for spin and interactions
        // printf("spin neighbor indices:\n");
        // printf("color=%d i=%d j=%d i_dn=%d i_un=%d j_rn=%d j_ln=%d \n", color, i, j, i_dn, i_un, j_rn, j_ln);
        // printf("right_four_body_term_up_version_left_b=%d right_four_body_term_up_version_right_b=%d right_four_body_term_up_version_down_r=%d\n", right_four_body_term_up_version_left_b, right_four_body_term_up_version_right_b, right_four_body_term_up_version_down_r);
        // printf("right_four_body_term_down_version_left_b=%d right_four_body_term_down_version_right_b=%d right_four_body_term_down_version_up_r=%d \n", right_four_body_term_down_version_left_b, right_four_body_term_down_version_right_b, right_four_body_term_down_version_up_r);
        // printf("down_four_body_term_left_version_right_r=%d down_four_body_term_left_version_up_b=%d down_four_body_term_left_version_down_b=%d \n", down_four_body_term_left_version_right_r, down_four_body_term_left_version_up_b, down_four_body_term_left_version_down_b);
        // printf("down_four_body_term_right_version_left_r=%d down_four_body_term_right_version_up_b=%d down_four_body_term_right_version_down_b=%d \n", down_four_body_term_right_version_left_r, down_four_body_term_right_version_up_b, down_four_body_term_right_version_down_b);
        // printf("interaction indices:\n");
        // printf("interactions_r: u_inter=%d d_inter=%d l_inter=%d right_inter=%d\n", offset_interactions_closed_on_sublattice + num_qubits / 2 + i_un * X + j, offset_interactions_closed_on_sublattice + num_qubits / 2 + i * X + j, offset_interactions_closed_on_sublattice + i * X + j_ln, offset_interactions_closed_on_sublattice + i * X + j);
        // printf("blue spins on horizontal line four body interactions: root_spin_up_inter=%d root_spin_down_inter=%d \n", offset_interactions_four_body + right_four_body_term_up_version_left_b, offset_interactions_four_body + right_four_body_term_down_version_left_b);
        // printf("blue spins on vertical line four body interactions: root_spin_left_inter=%d root_spin_right_inter=%d \n", offset_interactions_four_body + down_four_body_term_left_version_up_b, offset_interactions_four_body + down_four_body_term_right_version_up_b);
        // }
    }

    double d_new_energy = d_energy[tid] + energy_diff;

    // if (tid == 0 || tid == 1)
    // {
    //     printf("walker idx = %lld old energy = %.6f new energy = %.6f energy diff = %.6f\n", tid, d_energy[tid], d_new_energy, energy_diff);
    // }

    RBIM_eight_vertex rbim;
    rbim.new_energy = d_new_energy;
    rbim.i = i;
    rbim.j = j;
    rbim.color = color;
    return rbim;
}

__global__ void initialize_Gaussian_error_rates(double *d_prob_i, double *d_prob_x, double *d_prob_y, double *d_prob_z, int num_qubits, int num_interactions, double error_rate_mean, double error_rate_variance, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_qubits * num_interactions)
        return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    // Generate a Gaussian distributed value with given mean and variance
    double randomValue = curand_normal_double(&state) * sqrt(error_rate_variance) + error_rate_mean;

    // Bound error probabilities between 1e-25 and 0.5
    if (randomValue <= 0.0)
        randomValue = 1e-25; // error prob 0 is not allowed
    if (randomValue > 0.5)
        randomValue = 0.5;

    double unbiased_error_probability = randomValue / 3;

    d_prob_i[idx] = 1 - randomValue;
    d_prob_x[idx] = unbiased_error_probability;
    d_prob_y[idx] = unbiased_error_probability;
    d_prob_z[idx] = unbiased_error_probability;

    // printf("idx: %d prob_i: %.10f prob_x: %.10f prob_y: %.10f prob_z: %.10f \n", idx, d_prob_i[idx], d_prob_x[idx], d_prob_y[idx], d_prob_z[idx]);
}

__global__ void initialize_coupling_factors(double *prob_i_err, double *prob_x_err, double *prob_y_err, double *prob_z_err, int num_qubits, int num_interactions, int histogram_scale, double *d_J_i, double *d_J_x, double *d_J_y, double *d_J_z)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_qubits * num_interactions)
        return;

    // Coupling strength from Nishimori condition in https://arxiv.org/pdf/1809.10704 eq 15 with beta = 1.
    double J_i = log(prob_i_err[idx] * prob_x_err[idx] * prob_y_err[idx] * prob_z_err[idx]) / 4;
    double J_x = log((prob_i_err[idx] * prob_x_err[idx]) / (prob_y_err[idx] * prob_z_err[idx])) / 4;
    double J_y = log((prob_i_err[idx] * prob_y_err[idx]) / (prob_x_err[idx] * prob_z_err[idx])) / 4;
    double J_z = log((prob_i_err[idx] * prob_z_err[idx]) / (prob_x_err[idx] * prob_y_err[idx])) / 4;

    double abs_J_X = fabs(J_x);
    double abs_J_Y = fabs(J_y);
    double abs_J_Z = fabs(J_z);

    // Find the maximum of the absolute values using fmax
    double max_J = fmax(fmax(abs_J_X, abs_J_Y), abs_J_Z);

    // Ensure max_J is non-zero to avoid division by zero
    if (max_J > 0)
    {
        // Rescale the J values
        d_J_i[idx] = J_i * (histogram_scale / max_J);
        d_J_x[idx] = J_x * (histogram_scale / max_J);
        d_J_y[idx] = J_y * (histogram_scale / max_J);
        d_J_z[idx] = J_z * (histogram_scale / max_J);
    }

    // // // TEST BLOCK
    // // // -----------
    // // d_J_x[idx] = 1;
    // // d_J_y[idx] = 1;
    // // d_J_z[idx] = 1;
    // // // -----------

    // printf("J params are rescaled by histogram_scale / max J_i. \n");
    // printf("idx: %d j_i: %.10f j_x: %.10f j_y: %.10f j_z: %.10f \n", idx, d_J_i[idx], d_J_x[idx], d_J_y[idx], d_J_z[idx]);
}

std::string eight_vertex_histogram_path(
    bool is_qubit_specific_noise, float error_mean, float error_variance,
    int X, int Y, int seed_hist, bool x_horizontal_error, bool x_vertical_error,
    bool z_horizontal_error, bool z_vertical_error, float prob_x_err, float prob_y_err, float prob_z_err)
{
    std::string error_string = std::to_string(x_horizontal_error) + std::to_string(x_vertical_error) + std::to_string(z_horizontal_error) + std::to_string(z_vertical_error);
    std::stringstream strstr;
    strstr << "init/eight_vertex/periodic/qubit_specific_noise_" << std::to_string(is_qubit_specific_noise) << "/";
    if (is_qubit_specific_noise)
    {
        strstr << "error_mean_" << std::fixed << std::setprecision(6) << error_mean << "_error_variance_" << error_variance;
    }
    else
    {
        strstr << "prob_X_" + std::to_string(prob_x_err) + "_prob_Y_" + std::to_string(prob_y_err) + "_prob_Z_" + std::to_string(prob_z_err);
    }
    strstr << "/X_" << X << "_Y_" << Y;
    strstr << "/error_class_" << error_string;
    strstr << "/seed_" << seed_hist;
    strstr << "/histogram/histogram.txt";

    return strstr.str();
}

std::string eight_vertex_interaction_path(
    bool is_qubit_specific_noise, float error_mean, float error_variance,
    int X, int Y, int seed_hist, bool x_horizontal_error, bool x_vertical_error,
    bool z_horizontal_error, bool z_vertical_error, std::string interaction_type, float prob_x_err, float prob_y_err, float prob_z_err)
{
    std::string error_string = std::to_string(x_horizontal_error) + std::to_string(x_vertical_error) + std::to_string(z_horizontal_error) + std::to_string(z_vertical_error);
    std::stringstream strstr;
    strstr << "init/eight_vertex/periodic/qubit_specific_noise_" << std::to_string(is_qubit_specific_noise) << "/" << std::fixed << std::setprecision(6);
    if (is_qubit_specific_noise)
    {
        strstr << "error_mean_" << error_mean << "_error_variance_" << error_variance;
    }
    else
    {
        strstr << "prob_X_" + std::to_string(prob_x_err) + "_prob_Y_" + std::to_string(prob_y_err) + "_prob_Z_" + std::to_string(prob_z_err);
    }
    strstr << "/X_" << X << "_Y_" << Y;
    strstr << "/error_class_" << error_string;
    strstr << "/seed_" << seed_hist;
    strstr << "/interactions/interactions_" << interaction_type << ".txt";

    return strstr.str();
}

void read(std::vector<signed char> &lattice, std::string filename)
{

    std::ifstream inputFile(filename);

    if (!inputFile)
    {
        std::cerr << "Unable to open file " << filename << std::endl;
        return;
    }

    int spin = 0;

    while (inputFile >> spin)
    {
        lattice.push_back(static_cast<signed char>(spin));
    }

    return;
}

void read(std::vector<double> &lattice, std::string filename)
{

    std::ifstream inputFile(filename);

    if (!inputFile)
    {
        std::cerr << "Unable to open file " << filename << std::endl;
        return;
    }

    double spin = 0;

    while (inputFile >> spin)
    {
        lattice.push_back(spin);
    }

    return;
}

__global__ void reset_d_cond(signed char *d_cond, double *d_factor, int total_intervals, double beta, int walker_per_interval)
{
    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= total_intervals)
        return;

    if (d_cond[tid] == 1)
    {
        if (d_factor[tid * walker_per_interval] > exp(beta))
        {
            d_cond[tid] = 0;
        }
    }
}
