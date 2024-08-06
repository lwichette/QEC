#include "wlutils.cuh"
#include "cudamacro.h"

void parse_args(int argc, char *argv[], Options *options)
{
    // overlap decimal is more like the reciprocal non overlap parameter here, i.e. 0 as overlap_decimal is full overlap of intervals.

    int opt;

    while (1){
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
            {0, 0, 0, 0}};
        
        opt = getopt_long(argc, argv, "x:y:n:p:a:b:i:w:o:", long_options, &option_index);
        
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
        default:
            fprintf(stderr, "Usage: %s [-i num_intervals] [-m E_min] [-M E_max] [-w walker_per_interval] [-o overlap_decimal] [-r num_iterations]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
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

void writeToFile(const std::string& filename, const signed char* data, int nx_w, int ny) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < nx_w; i++) {
            for (int j = 0; j < ny; j++) {
                file << static_cast<int>(data[i * ny + j]) << " ";
            }
            file << std::endl;
        }
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
    file.close();
}

void write(
    signed char* array, const std::string& filename, long nx, long ny, 
    int num_lattices, bool lattice, const std::vector<int>& energies
) {
    std::cout << "Writing to " << filename << " ..." << std::endl;

    int nx_w = (lattice) ? nx : 2 * nx;
    std::vector<signed char> array_host(nx_w * ny * num_lattices);

    CHECK_CUDA(cudaMemcpy(array_host.data(), array, nx_w * ny * num_lattices * sizeof(*array), cudaMemcpyDeviceToHost));

    if (num_lattices == 1) {
        writeToFile(filename + ".txt", array_host.data(), nx_w, ny);
    } else {
        for (int l = 0; l < num_lattices; l++) {
            int offset = l * nx_w * ny;
            
            if (energies.empty() == false){
                if (energies[l] == 0 && array_host[offset] == 0) {
                    continue;
                }
            }

            std::string file_suffix = (energies.empty()) ? std::to_string(l) : std::to_string(energies[l]);
            writeToFile(filename + "_" + file_suffix + ".txt", array_host.data() + offset, nx_w, ny);
        }
    }
}

void create_directory(std::string path){
    if (!std::filesystem::exists(path)) {
        // Create directory
        if (std::filesystem::create_directories(path)) {
            std::cout << "Successfully created directory: " << path << std::endl; 
        } else {
            std::cerr << "Failed to create directory: " << path << std::endl;
        }
    } else {
        std::cout << "Directory already exists: " << path << std::endl;
    }
}

void write_histograms(unsigned long long *d_H, std::string path_histograms, int len_histogram, int seed, int E_min){
    
    printf("Writing to %s ...\n", path_histograms.c_str());

    std::vector<unsigned long long> h_histogram(len_histogram);

    CHECK_CUDA(cudaMemcpy(h_histogram.data(), d_H, len_histogram*sizeof(*d_H), cudaMemcpyDeviceToHost));

    std::ofstream f;
    f.open(std::string(path_histograms + "/histogram.txt"));
    
    if (f.is_open()) {
        for (int i=0; i < len_histogram; i++){     
            int energy = E_min + i;
            if (h_histogram[i] > 0){
                f << energy << " " << 1 << std::endl;   
            }
            else{
                f << energy << " " << 0 << std::endl;
            }
        }
    }
}

int read_histogram(const char *filename, std::vector<int> &h_expected_energy_spectrum, int *E_min, int *E_max){
    std::cout << filename;
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Failed to open file");
        return -1;
    }

    *E_min = INT_MAX;
    *E_max = INT_MIN;
    int value, count;
    h_expected_energy_spectrum.clear();

    int start_writing_zeros = 0;
    while (fscanf(file, "%d %d", &value, &count) != EOF)
    {
        if (count > 0)
        {
            if (value < *E_min)
                *E_min = value;
            if (value > *E_max)
                *E_max = value;

            h_expected_energy_spectrum.push_back(1);
            start_writing_zeros = 1;
        }
        else if (start_writing_zeros != 0)
        {
            h_expected_energy_spectrum.push_back(0);
        }
    }
    fclose(file);
    return 0;
}

void read(std::vector<signed char> &lattice, std::string filename)
{

    std::ifstream inputFile(filename);

    if (!inputFile){
        std::cerr << "Unable to open file " << filename << std::endl;
        return;
    }

    int spin = 0;

    while (inputFile >> spin){
        lattice.push_back(static_cast<signed char>(spin));
    }
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
}

char *constructFilePath(float prob_interactions, int X, int Y, int seed, std::string type)
{
    std::stringstream strstr;
    strstr << "init/prob_" << std::fixed << std::setprecision(6) << prob_interactions;
    strstr << "/X_" << X << "_Y_" << Y;
    strstr << "/seed_" << seed << "/" << type << "/" << type << ".txt";


    // Convert the stringstream to a string
    std::string filePathStr = strstr.str();

    // Allocate memory for the char* result and copy the string data to it
    char *filePathCStr = new char[filePathStr.length() + 1];
    std::strcpy(filePathCStr, filePathStr.c_str());

    return filePathCStr;
}

std::vector<signed char> get_lattice_with_pre_run_result(float prob, int seed, int x, int y, std::vector<int> h_start, std::vector<int> h_end, int num_intervals, int num_walkers_total, int num_walkers_per_interval){
    namespace fs = std::filesystem;
    std::ostringstream oss;
    oss << "init/prob_" << std::fixed << std::setprecision(6) << prob;
    oss << "/X_" << x << "_Y_" << y;
    oss << "/seed_" << seed;
    oss << "/lattice";

    std::string lattice_path = oss.str();
    std::vector<signed char> lattice_over_all_walkers;
    for(int interval_iterator = 0 ; interval_iterator < num_intervals; interval_iterator++){
        std::cout << interval_iterator << " ";
        try {
            for (const auto& entry : fs::directory_iterator(lattice_path)) {
                // Check if the entry is a regular file and has a .txt extension
                if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                    // Extract the number from the filename
                    std::string filename = entry.path().stem().string(); // Get the filename without extension
                    std::regex regex("lattice_(-?\\d+)");
                    std::smatch match;
                    if (std::regex_search(filename, match, regex)) {
                        int number = std::stoi(match[1]);
                        // Check if the number is between interval boundaries
                        if (number >= h_start[interval_iterator] && number <= h_end[interval_iterator]) {
                            std::cout << "Processing file: " << entry.path() << " with energy: " << number << " for interval [" << h_start[interval_iterator] << ", " << h_end[interval_iterator] << "]" << std::endl;
                            for(int walker_per_interval_iterator = 0; walker_per_interval_iterator < num_walkers_per_interval;  walker_per_interval_iterator++){
                                read(lattice_over_all_walkers, entry.path().string());
                            }
                            break;
                        } 
                    } else {
                        std::cerr << "Unable to open file: " << entry.path() << std::endl;
                    }
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Filesystem error: " << e.what() << std::endl;
        }
    }
    return lattice_over_all_walkers;
}

__device__ float atomicCAS_f32(float *p, float cmp, float val) {
	return __int_as_float(atomicCAS((int *) p, __float_as_int(cmp), __float_as_int(val)));
}

__global__ void init_lattice(signed char* lattice, float *d_probs, const int nx, const int ny, const int num_lattices, const int seed){

    const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= nx*ny*num_lattices) return;

    long long lattice_id = tid / (nx*ny);

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st); 
 
    atomicCAS_f32(&d_probs[lattice_id], 0.0f, curand_uniform(&st));
    
    __syncthreads();
    
    double randval = curand_uniform(&st);
    signed char val = (randval < d_probs[lattice_id]) ? -1 : 1;

    lattice[tid] = val;
}

__global__ void init_interactions(signed char* interactions, const int nx, const int ny, const int num_lattices, const int seed, const double prob){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    if (tid >= nx*ny*2*num_lattices) return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);

    double randval = curand_uniform(&st);
    signed char val = (randval < prob) ? -1 : 1;
    
    interactions[tid] = val;
}

__global__ void calc_energy_pre_run(signed char* lattice, signed char* interactions, int* d_energy, const int nx, const int ny, const int num_lattices){

    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    int energy = 0; 

    int offset_lattice = tid*nx*ny;

    for (int l = 0; l < nx*ny; l++){
        
        int i = l/ny;
        int j = l%ny;

        int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
        int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;  

        energy += lattice[offset_lattice + i*ny +j]*(lattice[offset_lattice + inn*ny + j]*interactions[nx*ny + inn*ny + j] + lattice[offset_lattice + i*ny + jnn]*interactions[i*ny + jnn]);
    }

    d_energy[tid] = energy;
}

__global__ void calc_energy(signed char *lattice, signed char *interactions, int *d_energy, int *d_offset_lattice, const int nx, const int ny, const int num_lattices)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    int energy = 0;

    for (int l = 0; l < nx * ny; l++)
    {

        int i = l / ny;
        int j = l % ny;

        int inn = (i - 1 >= 0) ? i - 1 : nx - 1;
        int jnn = (j - 1 >= 0) ? j - 1 : ny - 1;

        energy += lattice[d_offset_lattice[tid] + i * ny + j] * (lattice[d_offset_lattice[tid] + inn * ny + j] * interactions[nx * ny + inn * ny + j] + lattice[d_offset_lattice[tid] + i * ny + jnn] * interactions[i * ny + jnn]);
    }

    d_energy[tid] = energy;
}

__global__ void wang_landau_pre_run(
    signed char *d_lattice, signed char *d_interactions, int *d_energy, unsigned long long *d_H, unsigned long long* d_iter, int *d_found_interval,
    signed char *d_store_lattice, const int E_min, const int E_max, const int num_iterations, const int nx, const int ny, 
    const int seed, const int len_interval, const int found_interval
    ){
    
    long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

    const int offset_lattice = tid*nx*ny;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_iter[tid], &st);
    
    for (int it = 0; it < num_iterations; it++){

        // Generate random int --> is that actually uniformly?
        double randval = curand_uniform(&st);
        randval *= (nx*ny - 1 + 0.999999);
        int random_index = (int)trunc(randval);

        d_iter[tid] += 1;

        int i = random_index/ny;
        int j = random_index % ny;

        // Set up periodic boundary conditions
        int ipp = (i + 1 < nx) ? i + 1 : 0;
        int inn = (i - 1 >= 0) ? i - 1: nx - 1;
        int jpp = (j + 1 < ny) ? j + 1 : 0;
        int jnn = (j - 1 >= 0) ? j - 1: ny - 1; 

        // Nochmal checken
        signed char energy_diff = -2 * d_lattice[offset_lattice + i*ny +j]*(d_lattice[offset_lattice + inn*ny + j]*d_interactions[nx*ny + inn*ny + j] + d_lattice[offset_lattice + i*ny + jnn]*d_interactions[i*ny + jnn]
                                                                    + d_lattice[offset_lattice + ipp*ny + j]*d_interactions[nx*ny + i*ny + j] + d_lattice[offset_lattice + i*ny + jpp]*d_interactions[i*ny + j]);

        int d_new_energy = d_energy[tid] + energy_diff; 

        int index_old = d_energy[tid] - E_min;
        
        if (d_new_energy > E_max || d_new_energy < E_min){
            printf("Iterator %d \n", it);
            printf("Thread Id %lld \n", tid);
            printf("Randval %f \n", randval);
            printf("Energy out of range %d \n", d_new_energy);
            printf("Old energy %d \n", d_energy[tid]);
            assert(0);
            return;
        }
        else{
            
            int index_new = d_new_energy - E_min;

            double prob = exp(static_cast<double>(d_H[index_old]) - static_cast<double>(d_H[index_new]));

            if(curand_uniform(&st) < prob){
                d_lattice[offset_lattice + i*ny +j] *= -1;
                d_energy[tid] = d_new_energy;
                d_iter[tid] += 1;

                atomicAdd(&d_H[index_new], 1);

                if (found_interval == 0){
                    store_lattice(d_lattice, d_energy, d_found_interval, d_store_lattice, E_min, nx, ny, tid, len_interval);
                }
            }
            else{
                atomicAdd(&d_H[index_old], 1);
            }
        }
    }
}

__global__ void find_spin_config_in_energy_range(signed char *d_lattice, signed char *d_interactions, const int nx, const int ny, const int num_lattices, const int seed, int *d_start, int *d_end, int *d_energy, int *d_offset_lattice)
{
    const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    
    int blockId = blockIdx.x;

    if (tid >= num_lattices) return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);
    
    int accept_spin_config = 0;
    
    while (accept_spin_config == 0){
        if (d_energy[tid] <= d_end[blockId] && d_energy[tid] >= d_start[blockId]){
            // TO DO d_H and d_G update
            accept_spin_config = 1;
        }
        else{
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
}

__global__ void check_energy_ranges(int *d_energy, int *d_start, int *d_end)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    int check = 1;

    if (d_energy[tid] > d_end[blockIdx.x] || d_energy[tid] < d_start[blockIdx.x])
    {
        check = 0;
    }

    assert(check);
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
}

__device__ void store_lattice(
    signed char *d_lattice, int *d_energy, int* d_found_interval, signed char* d_store_lattice,
    const int E_min, const int nx, const int ny, const long long tid, const int len_interval
    ){
    
    int interval_index = (d_energy[tid] - E_min)/(len_interval);

    if (atomicCAS(&d_found_interval[interval_index], 0, 1) != 0) return;

    for (int i=0; i < nx; i++){
        for (int j=0; j < ny; j++){
            d_store_lattice[interval_index*nx*ny + i*ny + j] = d_lattice[tid*nx*ny + i*ny +j];
        }
    }

    return;
}

__global__ void init_indices(int *d_indices)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    d_indices[tid] = threadIdx.x;
}

__global__ void init_offsets_histogramm(int *d_offset_histogramm, int *d_start, int *d_end)
{

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    // Length of interval equal except last one --> length of array is given by num_threads_per_block * (length of interval + length of last interval)
    if (blockIdx.x == gridDim.x - 1)
    {
        d_offset_histogramm[tid] = (gridDim.x - 1) * blockDim.x * (d_end[0] - d_start[0] + 1) + threadIdx.x * (d_end[gridDim.x - 1] - d_start[gridDim.x - 1] + 1);
    }
    else
    {
        d_offset_histogramm[tid] = tid * (d_end[0] - d_start[0] + 1);
    }
}

__global__ void init_offsets_lattice(int *d_offset_lattice, int nx, int ny)
{
    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    d_offset_lattice[tid] = tid * nx * ny;
}

__global__ void replica_exchange(
    int *d_offset_lattice, int *d_energy, int *d_start, int *d_end, int *d_indices,
    double *d_logG, int *d_offset_histogram, bool even, int seed, unsigned long long *d_offset_iter)
{

    if (blockIdx.x == (gridDim.x - 1)) return;
    
    if ((even && (blockIdx.x % 2 != 0)) || (!even && (blockIdx.x % 2 == 0))) return;

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    long long cid = static_cast<long long>(blockDim.x) * (blockIdx.x + 1);

    if (threadIdx.x == 0){
        fisher_yates(d_indices, seed, d_offset_iter);
    }

    __syncthreads();

    cid += d_indices[tid];

    // Check energy ranges
    if (d_energy[tid] > d_end[blockIdx.x + 1] || d_energy[tid] < d_start[blockIdx.x + 1]) return;
    if (d_energy[cid] > d_end[blockIdx.x] || d_energy[tid] < d_start[blockIdx.x]) return;

    double prob = min(1.0, exp(d_logG[d_offset_histogram[tid] + d_energy[tid] - d_start[blockIdx.x]] - d_logG[d_offset_histogram[tid] + d_energy[cid] - d_start[blockIdx.x]]) * exp(d_logG[d_offset_histogram[cid] + d_energy[cid] - d_start[blockIdx.x+1]] - d_logG[d_offset_histogram[cid] + d_energy[tid] - d_start[blockIdx.x+1]]));

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_offset_iter[tid], &st);

    if (curand_uniform(&st) < prob){

        int temp_off = d_offset_lattice[tid];
        int temp_energy = d_energy[tid];

        d_offset_lattice[tid] = d_offset_lattice[cid];
        d_energy[tid] = d_energy[cid];

        d_offset_lattice[cid] = temp_off;
        d_energy[cid] = temp_energy;
    }
    
    d_offset_iter[tid] += 1;
}

__global__ void check_histogram(unsigned long long *d_H, double *d_log_G, double *d_shared_logG, int *d_offset_histogramm, int *d_end, int *d_start, double *d_factor, int nx, int ny, double alpha, double beta, int *d_expected_energy_spectrum, int len_energy_spectrum, int num_walker_total, signed char* d_cond){

    
    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    int blockId = blockIdx.x;

    __shared__ int walkers_finished;

    if (threadIdx.x == 0){
        walkers_finished = 0;
    }

    __syncthreads();

    const int len_interval = d_end[blockId] - d_start[blockId] + 1;

    if (tid < num_walker_total){
        int min = INT_MAX;
        double average = 0;
        int len_reduced_energy_spectrum = 0;

        // Here is average and min calculation over all bins in histogram which correspond to values in expected energy spectrum
        for (int i = 0; i < (d_end[blockId] - d_start[blockId] + 1); i++){
            if (d_expected_energy_spectrum[d_start[blockId] + i - d_start[0]] == 1){
                if (d_H[d_offset_histogramm[tid] + i] < min){
                    min = d_H[d_offset_histogramm[tid] + i];
                }
                average += d_H[d_offset_histogramm[tid] + i];
                len_reduced_energy_spectrum += 1;
            }
        }

        __syncthreads();
        

        if (len_reduced_energy_spectrum > 0){
        
            average = average / len_reduced_energy_spectrum;

            if (min >= alpha * average){
                atomicAdd(&walkers_finished, 1);
            }
        }
        else{
            printf("Error histogram has no sufficient length to check for flatness on walker %lld. \n", tid);
        }

        __syncthreads();

        if (walkers_finished == blockDim.x){
            d_cond[blockId] = 1;
        }
    }
}


__global__ void calc_average_log_g(int num_intervals, long long len_histogram_over_all_walkers, int num_walker_per_interval,  double *d_log_G, double *d_shared_logG, int *d_end, int *d_start, int *d_expected_energy_spectrum, int *d_cond){

    // 1 block and threads as many as len_histogram_over_all_walkers
    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid < len_histogram_over_all_walkers){ 
        int len_first_interval = (d_end[0] - d_start[0] + 1);
        long long intervalId = (tid/(len_first_interval*num_walker_per_interval) < num_intervals) ? tid/(len_first_interval*num_walker_per_interval) : num_intervals - 1;
        int len_interval = d_end[intervalId] - d_start[intervalId] + 1;

        long long walkerId = (tid%(len_interval*num_walker_per_interval))/len_interval;
        long long energyId = (tid%(len_interval*num_walker_per_interval))%len_interval;    

        long long linearised_walker_idx = intervalId*num_walker_per_interval+walkerId;

        if (d_cond[intervalId] == 1 && d_expected_energy_spectrum[d_start[intervalId] + energyId - d_start[0]] == 1){        
            atomicAdd(&d_shared_logG[intervalId*len_first_interval + energyId], d_log_G[tid]/num_walker_per_interval);   
        }  
    }
}

__global__ void redistribute_g_values(int num_intervals, long long len_histogram_over_all_walkers, int num_walker_per_interval,  double *d_log_G, double *d_shared_logG, int *d_end, int *d_start, double *d_factor, double beta, int *d_expected_energy_spectrum, int *d_cond){

    // 1 block and threads as many as len_histogram_over_all_walkers
    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid < len_histogram_over_all_walkers){ 


        int len_first_interval = (d_end[0] - d_start[0] + 1);
        long long intervalId = (tid/(len_first_interval*num_walker_per_interval) < num_intervals) ? tid/(len_first_interval*num_walker_per_interval) : num_intervals - 1;
        int len_interval = d_end[intervalId] - d_start[intervalId] + 1;
        long long walkerId = (tid%(len_interval*num_walker_per_interval))/len_interval;
        long long energyId = (tid%(len_interval*num_walker_per_interval))%len_interval;    
        long long linearised_walker_idx = intervalId*num_walker_per_interval+walkerId;

        if (d_cond[intervalId] == 1){
            if(energyId == 0){ // for each walker in finished interval a single thread sets the new factor and resets the condition array to update histogram again
                if(d_factor[linearised_walker_idx]>exp(beta)){ // if not already in last factor iteration reset the cond array to update in next round log g and hist 
                    d_cond[linearised_walker_idx] = 0;
                }
                d_factor[linearised_walker_idx] = sqrt(d_factor[linearised_walker_idx]);
            }
            if (d_expected_energy_spectrum[d_start[intervalId] + energyId - d_start[0]] == 1){
                d_log_G[tid] = d_shared_logG[intervalId*len_first_interval + energyId];
            }
        }  
    }
}

__device__ RBIM random_bond_ising(
    signed char *d_lattice, signed char *d_interactions, int *d_energy, int *d_offset_lattice, unsigned long long *d_offset_iter, 
    curandStatePhilox4_32_10_t *st, const long long tid, const int nx, const int ny
    ){
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

        signed char energy_diff = -2 * d_lattice[d_offset_lattice[tid] + i * ny + j] * (d_lattice[d_offset_lattice[tid] + inn * ny + j] * d_interactions[nx * ny + inn * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jnn] * d_interactions[i * ny + jnn] + d_lattice[d_offset_lattice[tid] + ipp * ny + j] * d_interactions[nx * ny + i * ny + j] + d_lattice[d_offset_lattice[tid] + i * ny + jpp] * d_interactions[i * ny + j]);

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
    const int seed, double *factor, unsigned long long *d_offset_iter, int *d_expected_energy_spectrum, int *d_newEnergies, int *foundFlag, 
    const int num_lattices, const double beta, signed char* d_cond
    ){

    long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    int blockId = blockIdx.x;

    if (tid >= num_lattices || factor[tid] <= exp(beta)) return;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, d_offset_iter[tid], &st);

    if (d_cond[tid] == 0){
        for (int it = 0; it < num_iterations; it++){

            RBIM result = random_bond_ising(d_lattice, d_interactions, d_energy, d_offset_lattice, d_offset_iter, &st, tid, nx, ny);

            // If no new energy is found, set it to 0, else to tid + 1
            foundFlag[tid] = (d_expected_energy_spectrum[result.new_energy - d_start[0]] == 1) ? 0 : tid + 1;

            if (foundFlag[tid] != 0){
                printf("new_energy %d index in spectrum %d \n", result.new_energy, result.new_energy - d_start[0]);
                d_newEnergies[tid] = result.new_energy;
                return;
            }

            int index_old = d_offset_histogramm[tid] + d_energy[tid] - d_start[blockId];

            if (result.new_energy > d_end[blockId] || result.new_energy < d_start[blockId]){
                d_H[index_old] += 1;
                d_logG[index_old] += log(factor[tid]);
            }
            else{

                int index_new = d_offset_histogramm[tid] + result.new_energy - d_start[blockId];

                double prob = min(1.0, exp(d_logG[index_old] - d_logG[index_new]));

                if (curand_uniform(&st) < prob){
                    d_lattice[d_offset_lattice[tid] + result.i * ny + result.j] *= -1;
                    d_H[index_new] += 1;
                    d_logG[index_new] += log(factor[tid]);
                    d_energy[tid] = result.new_energy;
                    d_offset_iter[tid] += 1;
                }
                else{
                    d_H[index_old] += 1;
                    d_logG[index_old] += log(factor[tid]);
                }
            }
        }
    }
    else{
        for (int it = 0; it < num_iterations; it++){

            RBIM result = random_bond_ising(d_lattice, d_interactions, d_energy, d_offset_lattice, d_offset_iter, &st, tid, nx, ny);

            // If no new energy is found, set it to 0, else to tid + 1
            foundFlag[tid] = (d_expected_energy_spectrum[result.new_energy - d_start[0]] == 1) ? 0 : tid + 1;

            if (foundFlag[tid] != 0){
                printf("new_energy %d index in spectrum %d \n", result.new_energy, result.new_energy - d_start[0]);
                d_newEnergies[tid] = result.new_energy;
                return;
            }

            if (result.new_energy <= d_end[blockId] || result.new_energy >= d_start[blockId]){
                int index_old = d_offset_histogramm[tid] + d_energy[tid] - d_start[blockId];
                int index_new = d_offset_histogramm[tid] + result.new_energy - d_start[blockId];

                double prob = min(1.0, exp(d_logG[index_old] - d_logG[index_new]));
                
                if (curand_uniform(&st) < prob){
                    d_lattice[d_offset_lattice[tid] + result.i * ny + result.j] *= -1;
                    d_energy[tid] = result.new_energy;
                    d_offset_iter[tid] += 1;
                }
            }
        }
    }
}

__global__ void print_finished_walker_ratio(double *d_factor, int num_walker_total, const double exp_beta, double *d_finished_walkers_ratio){
    extern __shared__ int shared_count[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int threadId = threadIdx.x;
    if (threadId == 0) {
        shared_count[0] = 0;
    }
    __syncthreads();
    if (tid < num_walker_total) {
        if (d_factor[tid] <= exp_beta) {
            atomicAdd(&shared_count[0], 1);
        }
    }
    __syncthreads();
    if (threadId == 0) {
        double ratio_of_finished_walkers = (double)shared_count[0] / num_walker_total;
        // printf("ratio of finished walkers: %f\n", ratio_of_finished_walkers);
        d_finished_walkers_ratio[0] = ratio_of_finished_walkers;
    }
}
