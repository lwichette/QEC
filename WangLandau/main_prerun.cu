#include "./header/cudamacro.h"
#include "./header/wlutils.cuh"
#include <thrust/extrema.h> //addition needed for my (Linnea's) version of thrust -- comment out if this causes issues

int THREADS = 128;

using namespace std;


int main(int argc, char **argv){
    // questions: num walker is total number of walkers here and hence should be always larger than intervals, right? This should be catched
    
    int X, Y;
    
    float prob_interactions;

    int num_wl_loops, num_iterations, num_walker;

    int seed;

    int num_intervals;
    
    char logical_error_type = 'I';

    char boundary_type = 'p';

    int och;

    while (1) {
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
            {0, 0, 0, 0}
        };

        och = getopt_long(argc, argv, "x:y:p:n:l:w:s:i:e:b:", long_options, &option_index);
        
        if (och == -1)
            break;
        switch (och) {
			case 0:// handles long opts with non-NULL flag field
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
                num_walker = atoi(optarg);
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
                boundary_type = *optarg;
                break;
			case '?':
				exit(EXIT_FAILURE);

			default:
				fprintf(stderr, "unknown option: %c\n", och);
				exit(EXIT_FAILURE);
        }
    }

    // // function map works only on host function - hence have to wrap wang landau kernels inside host function
    // std::map<char, std::function<void(signed char*, signed char*, int*, int*, const int, const int, const int)>> updateMap;
    // updateMap['o'] = &calc_energy_open_boundary;
    // updateMap['p'] = &calc_energy_periodic_boundary;

    double factor = std::exp(1);

    const int E_min = -2*X*Y;
    const int E_max = -E_min;

    IntervalResult interval_result = generate_intervals(E_min, E_max, num_intervals, 1, 1.0f);

    for (int i=0; i< num_intervals; i++){
        std::cout << interval_result.h_start[i] << " " << interval_result.h_end[i] << std::endl;
    }

    long long len_histogram = E_max - E_min + 1;
    
    unsigned long long *d_H; 
    CHECK_CUDA(cudaMalloc(&d_H, len_histogram * sizeof(*d_H)));
    CHECK_CUDA(cudaMemset(d_H, 0, len_histogram*sizeof(*d_H)));

    unsigned long long *d_iter;
    CHECK_CUDA(cudaMalloc(&d_iter, num_walker*sizeof(*d_iter)));
    CHECK_CUDA(cudaMemset(d_iter, 0, num_walker*sizeof(*d_iter)));

    // lattice & interactions and energies
    signed char *d_lattice;
    CHECK_CUDA(cudaMalloc(&d_lattice, num_walker * X * Y * sizeof(*d_lattice)));

    float *d_probs;
    CHECK_CUDA(cudaMalloc(&d_probs, num_intervals * sizeof(*d_probs)));
    CHECK_CUDA(cudaMemset(d_probs, 0, num_intervals*sizeof(*d_probs)));
    
    signed char* d_interactions;
    CHECK_CUDA(cudaMalloc(&d_interactions, X * Y * 2 * sizeof(*d_interactions)));

    int* d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, num_walker * sizeof(*d_energy)));

    signed char* d_store_lattice;
    CHECK_CUDA(cudaMalloc(&d_store_lattice, num_intervals*X*Y*sizeof(*d_store_lattice)));

    int *d_found_interval;
    CHECK_CUDA(cudaMalloc(&d_found_interval, num_intervals*sizeof(*d_found_interval)));
    CHECK_CUDA(cudaMemset(d_found_interval, 0, num_intervals*sizeof(*d_found_interval)));

    int *d_interval_energies;
    CHECK_CUDA(cudaMalloc(&d_interval_energies, num_intervals*sizeof(*d_interval_energies)));
    CHECK_CUDA(cudaMemset(d_interval_energies, 0, num_intervals*sizeof(*d_interval_energies)));

    int *d_offset_lattice_per_walker, *d_offset_lattice_per_interval;
    CHECK_CUDA(cudaMalloc(&d_offset_lattice_per_walker, num_walker * sizeof(*d_offset_lattice_per_walker)));
    CHECK_CUDA(cudaMalloc(&d_offset_lattice_per_interval, num_intervals * sizeof(*d_offset_lattice_per_interval)));

    int BLOCKS_INIT = (num_walker*X*Y*2 + THREADS - 1)/THREADS;
    int BLOCKS_ENERGY = (num_walker + THREADS - 1)/THREADS;
    int BLOCKS_INTERVAL = (num_intervals + THREADS - 1)/THREADS;

    init_lattice<<<BLOCKS_INIT, THREADS>>>(d_lattice, d_probs, X, Y, num_walker, seed);

    init_offsets_lattice<<<BLOCKS_ENERGY, THREADS>>>(d_offset_lattice_per_walker, X, Y, num_walker); 
    init_offsets_lattice<<<BLOCKS_INTERVAL, THREADS>>>(d_offset_lattice_per_interval, X, Y, num_intervals);

    init_interactions<<<BLOCKS_INIT, THREADS>>>(d_interactions, X, Y, 1, seed + 1, prob_interactions, logical_error_type);
    
    cudaDeviceSynchronize();

    write(d_lattice, "Test.txt", X, Y, num_walker, true);

    switch (boundary_type) {
        case 'o': // Open boundary
            calc_energy_open_boundary<<<BLOCKS_ENERGY, THREADS>>>(d_lattice, d_interactions, d_energy, d_offset_lattice_per_walker, X, Y, num_walker);
            break;

        case 'p': // Periodic boundary
            calc_energy_periodic_boundary<<<BLOCKS_ENERGY, THREADS>>>(d_lattice, d_interactions, d_energy, d_offset_lattice_per_walker, X, Y, num_walker);
            break;

        default:
            printf("Invalid boundary type!\n");
            break;
    }
    cudaDeviceSynchronize();
    return;


    int found_interval = 0;

    for (int i=0; i < num_wl_loops; i++){
        
        if (i % 100 == 0) printf("Num wl loop: %d \n", i);

        wang_landau_pre_run<<<BLOCKS_ENERGY, THREADS>>>(d_lattice, d_interactions, d_energy, d_H, d_iter, d_found_interval, d_store_lattice, E_min, E_max, num_iterations, X, Y, seed + 2, interval_result.len_interval, found_interval, num_walker, num_intervals);
        cudaDeviceSynchronize();
        
        if (found_interval == 0){
            thrust::device_ptr<int> d_found_interval_ptr(d_found_interval);
            thrust::device_ptr<int> min_found_interval_ptr = thrust::min_element(d_found_interval_ptr, d_found_interval_ptr + num_intervals);
            found_interval = *min_found_interval_ptr;
        }
    }

    switch (boundary_type) {
        case 'o': // Open boundary
            calc_energy_open_boundary<<<BLOCKS_INTERVAL, THREADS>>>(d_store_lattice, d_interactions, d_interval_energies, d_offset_lattice_per_interval, X, Y, num_intervals);
            break;

        case 'p': // Periodic boundary
            calc_energy_periodic_boundary<<<BLOCKS_INTERVAL, THREADS>>>(d_store_lattice, d_interactions, d_interval_energies, d_offset_lattice_per_interval, X, Y, num_intervals);
            break;

        default:
            printf("Invalid boundary type!\n");
            break;
    }

    std::vector<int> h_interval_energies(num_intervals);
    CHECK_CUDA(cudaMemcpy(h_interval_energies.data(), d_interval_energies, num_intervals*sizeof(*d_interval_energies), cudaMemcpyDeviceToHost));

    for (int i=0; i < num_intervals; i++){
        std::cout << h_interval_energies[i] << std::endl;
    }

    std::string path = "init/prob_" + std::to_string(prob_interactions) + "/X_" + std::to_string(X) + "_Y_" + std::to_string(Y) +  "/boundary_" +  boundary_type + "/seed_" + std::to_string(seed) + "/error_class_" + logical_error_type;

    create_directory(path + "/interactions");
    create_directory(path + "/lattice");
    create_directory(path + "/histogram");

    write(d_interactions, path + "/interactions/interactions", X, Y, 1, false);
    write(d_store_lattice, path + "/lattice/lattice", X, Y, num_intervals, true, h_interval_energies);
    write_histograms(d_H, path + "/histogram/", len_histogram, seed, E_min);

    return 0;
}