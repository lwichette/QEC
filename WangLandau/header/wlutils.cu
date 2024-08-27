#include "wlutils.cuh"
#include "cudamacro.h"

__device__ RBIM (*rbim_func_map[])(signed char *, signed char *, int *, int *, unsigned long long *,
                                   curandStatePhilox4_32_10_t *, const long long, const int, const int, const int) = {periodic_boundary_random_bond_ising, open_boundary_random_bond_ising};

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

void writeToFile(const std::string &filename, const signed char *data, int nx_w, int ny)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        for (int i = 0; i < nx_w; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                file << static_cast<int>(data[i * ny + j]) << " ";
            }
            file << std::endl;
        }
    }
    else
    {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
    file.close();

    return;
}

void write(
    signed char *array_host, const std::string &filename, long nx, long ny,
    int num_lattices, bool lattice, const int *energies)
{
    std::cout << "Writing to " << filename << " ..." << std::endl;

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
        if (std::filesystem::create_directories(path))
        {
            std::cout << "Successfully created directory: " << path << std::endl;
        }
        else
        {
            std::cerr << "Failed to create directory: " << path << std::endl;
        }
    }
    else
    {
        std::cout << "Directory already exists: " << path << std::endl;
    }

    return;
}

void write_histograms(unsigned long long *h_histogram, std::string path_histograms, int len_histogram, int seed, int E_min)
{

    printf("Writing to %s ...\n", path_histograms.c_str());

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
    std::string boundary = (boundary_type == 0) ? "periodic" : "open";

    std::stringstream strstr;
    strstr << "init/" << boundary << "/prob_" << std::fixed << std::setprecision(6) << prob_interactions;
    strstr << "/X_" << X << "_Y_" << Y;
    strstr << "/seed_" << seed;
    strstr << "/error_class_" << error_class;
    strstr << "/" << type << "/" << type << ".txt";

    return strstr.str();
}

std::vector<signed char> get_lattice_with_pre_run_result(float prob, int seed, int x, int y, std::vector<int> h_start, std::vector<int> h_end, int num_intervals, int num_walkers_total, int num_walkers_per_interval, char error_class, int boundary_type)
{
    std::string boundary = (boundary_type == 0) ? "periodic" : "open";
    namespace fs = std::filesystem;
    std::ostringstream oss;
    oss << "init/" << boundary << "/prob_" << std::fixed << std::setprecision(6) << prob;
    oss << "/X_" << x << "_Y_" << y;
    oss << "/seed_" << seed;
    oss << "/error_class_" << error_class;
    oss << "/lattice";

    std::string lattice_path = oss.str();
    std::vector<signed char> lattice_over_all_walkers;
    for (int interval_iterator = 0; interval_iterator < num_intervals; interval_iterator++)
    {
        std::cout << interval_iterator << " ";
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
                            //std::cout << "Processing file: " << entry.path() << " with energy: " << number << " for interval [" << h_start[interval_iterator] << ", " << h_end[interval_iterator] << "]" << std::endl;
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

// End test

// End test

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

    d_energy[tid] = energy;

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

    d_energy[tid] = energy;

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
                d_iter[tid] += 1;

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

            d_energy[tid] += energy_diff;
            d_lattice[d_offset_lattice[tid] + i * ny + j] *= -1;
        }
    }

    return;
}

__global__ void check_energy_ranges(int *d_energy, int *d_start, int *d_end, int total_walker)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= total_walker)
        return;

    int check = 1;

    if (d_energy[tid] > d_end[blockIdx.x] || d_energy[tid] < d_start[blockIdx.x])
    {
        check = 0;
    }

    assert(check);

    return;
}

__device__ void fisher_yates(int *d_shuffle, int seed, unsigned long long *d_offset_iter)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    int offset = blockDim.x * blockIdx.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_offset_iter[tid], &st);

    for (int i = blockDim.x - 1; i > 0; i--)
    {
        double randval = curand_uniform(&st);
        randval *= (i + 0.999999);
        int random_index = (int)trunc(randval);
        d_offset_iter[tid] += 1;

        int temp = d_shuffle[offset + i];
        d_shuffle[offset + i] = d_shuffle[offset + random_index];
        d_shuffle[offset + random_index] = temp;
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

__global__ void replica_exchange(
    int *d_offset_lattice, int *d_energy, int *d_start, int *d_end, int *d_indices,
    double *d_logG, int *d_offset_histogram, bool even, int seed,
    unsigned long long *d_offset_iter, const int num_intervals,
    const int walker_per_interactions, int *d_cond_interaction)
{

    // if last block in interaction return
    if (blockIdx.x % num_intervals == (num_intervals - 1))
        return;

    // if even only even blocks if odd only odd blocks
    if ((even && (blockIdx.x % 2 != 0)) || (!even && (blockIdx.x % 2 == 0)))
        return;

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    const int int_id = tid / walker_per_interactions;

    if (d_cond_interaction[int_id] == -1)
        return;

    long long cid = static_cast<long long>(blockDim.x) * (blockIdx.x + 1);

    if (threadIdx.x == 0)
    {
        fisher_yates(d_indices, seed, d_offset_iter);
    }

    __syncthreads();

    cid += d_indices[tid];

    if (d_energy[tid] > d_end[blockIdx.x + 1] || d_energy[tid] < d_start[blockIdx.x + 1])
        return;
    if (d_energy[cid] > d_end[blockIdx.x] || d_energy[cid] < d_start[blockIdx.x])
        return;

    double prob = min(1.0, exp(d_logG[d_offset_histogram[tid] + d_energy[tid] - d_start[blockIdx.x]] - d_logG[d_offset_histogram[tid] + d_energy[cid] - d_start[blockIdx.x]]) * exp(d_logG[d_offset_histogram[cid] + d_energy[cid] - d_start[blockIdx.x + 1]] - d_logG[d_offset_histogram[cid] + d_energy[tid] - d_start[blockIdx.x + 1]]));

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_offset_iter[tid], &st);

    if (curand_uniform(&st) < prob)
    {

        int temp_off = d_offset_lattice[tid];
        int temp_energy = d_energy[tid];

        d_offset_lattice[tid] = d_offset_lattice[cid];
        d_energy[tid] = d_energy[cid];

        d_offset_lattice[cid] = temp_off;
        d_energy[cid] = temp_energy;
    }

    d_offset_iter[tid] += 1;

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

        // printf("Walker %d in interval %d with min %lld alpha*average %2f and factor %.10f and d_cond %d \n", threadIdx.x, blockIdx.x, min, alpha * average, d_factor[tid], d_cond[blockId]);

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
    int *d_cond_interaction)
{

    // 1 block and threads as many as len_histogram_over_all_walkers
    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

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

    if (d_cond_interaction[int_id] == -1)
        return;

    // Index inside histogram of the int_id interaction
    int tid_int = tid - d_offset_histogram[int_id * num_intervals_per_interaction * num_walker_per_interval];
    int len_first_interval = (d_end[int_id * num_intervals_per_interaction] - d_start[int_id * num_intervals_per_interaction] + 1);
    int intervalId = (tid_int / (len_first_interval * num_walker_per_interval) < num_intervals_per_interaction) ? tid_int / (len_first_interval * num_walker_per_interval) : num_intervals_per_interaction - 1;

    int interval_over_interaction = int_id * num_intervals_per_interaction + intervalId;
    if (d_cond[interval_over_interaction] == 1 && tid_int < d_len_histograms[int_id])
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

        if (d_expected_energy_spectrum[d_offset_energy_spectrum[int_id] + d_start[interval_over_interaction] + energyId - d_start[int_id * num_intervals_per_interaction]] == 1)
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
    int *d_cond_interaction)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

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

    if (d_cond_interaction[int_id] == -1)
        return;

    int tid_int = tid - d_offset_histogram[int_id * num_intervals_per_interaction * num_walker_per_interval];

    if (tid_int < d_len_histograms[int_id])
    {

        int len_first_interval = (d_end[int_id * num_intervals_per_interaction] - d_start[int_id * num_intervals_per_interaction] + 1);
        int intervalId = (tid_int / (len_first_interval * num_walker_per_interval) < num_intervals_per_interaction) ? tid_int / (len_first_interval * num_walker_per_interval) : num_intervals_per_interaction - 1;
        int interval_over_interaction = int_id * num_intervals_per_interaction + intervalId;

        if (d_cond[interval_over_interaction] == 1)
        {
            int len_interval = d_end[interval_over_interaction] - d_start[interval_over_interaction] + 1;
            int walkerId;
            int energyId;

            if (intervalId != 0)
            {
                walkerId = (tid_int % (len_first_interval * num_walker_per_interval * intervalId)) / len_interval;
                energyId = (tid_int % (len_first_interval * num_walker_per_interval * intervalId)) % len_interval;
            }
            else
            {
                walkerId = tid_int / len_interval;
                energyId = tid_int % len_interval;
            }

            int linearised_walker_idx = intervalId * num_walker_per_interval + walkerId;

            if (energyId == 0)
            { // for each walker in finished interval a single thread sets the new factor and resets the condition array to update histogram again
                if (d_factor[int_id * num_intervals_per_interaction * num_walker_per_interval + linearised_walker_idx] > exp(beta))
                { // if not already in last factor iteration reset the cond array to update in next round log g and hist
                    d_cond[interval_over_interaction] = 0;
                }
            }

            d_log_G[tid] = d_shared_logG[d_offset_shared_logG[interval_over_interaction] + energyId];
        }
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

    int d_new_energy = d_energy[tid] + energy_diff;

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

    int d_new_energy = d_energy[tid] + energy_diff;

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
                    d_offset_iter[tid] += 1;
                }
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

void calc_energy(
    int blocks, int threads, const int boundary_type, signed char *lattice,
    signed char *interactions, int *d_energy, int *d_offset_lattice,
    const int nx, const int ny, const int total_walker, const int walker_per_interactions)
{

    switch (boundary_type)
    {
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

    std::string boundary = (options.boundary_type == 0) ? "periodic" : "open";

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
    int *d_offset_intervals, int num_intervals, int num_interactions,
    void *d_temp_storage, size_t &temp_storage_bytes)
{

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
