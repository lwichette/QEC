#include "./header/cudamacro.h"
#include "./header/wlutils.cuh"
#include <thrust/extrema.h> //addition needed for my (Linnea's) version of thrust -- comment out if this causes issues

int main(int argc, char **argv){
    /*
    IMPORTANT: 
    X, Y sepcifies grid of qubits not Ising spins.
    X is qubits per row and Y rows of grid such that Y must be even!
    This is motivated by grid example down below. The Y is not equal to qubits per column but Y/2 is! To have equal amount of qubits per column, this must be even.
    */

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming device 0
    int max_threads_per_block = prop.maxThreadsPerBlock;

    int X, Y;
    
    float prob_i_err = 0;
    float prob_x_err = 0;
    float prob_y_err = 0;
    float prob_z_err = 0;

    int num_wl_loops = 0; 
    int num_iterations = 0;
    int walker_per_interaction = 1;

    int seed = 42;

    int num_intervals_per_interaction = 1;
    
    char logical_error_type = 'I';

    int boundary_type = 0;

    int och;

    int num_interactions = 1;

    while (1) {
        int option_index = 0;
        static struct option long_options[] = {
            {"X", required_argument, 0, 'x'},
            {"Y", required_argument, 0, 'y'},
            {"prob_x", required_argument, 0, 'f'},
            {"prob_y", required_argument, 0, 'g'},
            {"prob_z", required_argument, 0, 'h'},
            {"nit", required_argument, 0, 'n'},
            {"nl", required_argument, 0, 'l'},
            {"num_walker_total", required_argument, 0, 'w'},
            {"seed", required_argument, 0, 's'},
            {"num_intervals", required_argument, 0, 'i'},
            {"logical_error", required_argument, 0, 'e'},
            {"boundary", required_argument, 0, 'b'},
            {"replicas", required_argument, 0, 'r'},
            {0, 0, 0, 0}
        };

        och = getopt_long(argc, argv, "x:y:f:g:h:n:l:w:s:i:e:b:r:", long_options, &option_index);
        
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
            case 'f':
                prob_x_err = atof(optarg);
                break;
            case 'g':
                prob_y_err = atof(optarg);
                break;
            case 'h':
                prob_z_err = atof(optarg);
                break;
			case 'n':
                num_iterations = atoi(optarg);
                break;
			case 'l':
                num_wl_loops = atoi(optarg);
                break;
			case 'w':
                walker_per_interaction = atoi(optarg);
                break;
			case 's':
				seed = atoi(optarg);
                break;
            case 'i':
			    num_intervals_per_interaction = atoi(optarg);
			    break;
            case 'e':
                logical_error_type = *optarg;
                break;
            case 'b':
                boundary_type = atoi(optarg);
                if(boundary_type != 0 && boundary_type !=1){
                    fprintf(stderr, "Error: Invalid value for boundary type. Must be 0 (periodic) or 1 (open).\n");
                    exit(EXIT_FAILURE);
                }
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

    if(prob_x_err+prob_y_err+prob_z_err>1 || prob_x_err*prob_y_err*prob_z_err == 0){
        fprintf(stderr, "Error: Invalid value for error probabilities. Must sum to less then 1 and not be 0.\n");
        exit(EXIT_FAILURE);
    }
    else{ 
        prob_i_err = 1-(prob_x_err+prob_y_err+prob_z_err);
    }

    if(Y%2!=0){
        fprintf(stderr, "Error: Invalid value for Y. Must be even.\n");
        exit(EXIT_FAILURE);
    }

    unsigned long long num_qubits = X * Y; // which is equivalent to amount of interactions, i.e. add further Ising spins for open boundary but dimensionality of physical system remains unchanged

    int total_walker = num_interactions*walker_per_interaction;
    int total_intervals = num_interactions*num_intervals_per_interaction;

    //Coupling strength from Nishimori condition in https://arxiv.org/pdf/1809.10704 eq 15 with beta = 1.
    double J_I = std::log(prob_i_err*prob_x_err*prob_y_err*prob_z_err)/4;
    double J_X = std::log((prob_i_err*prob_x_err)/(prob_y_err*prob_z_err))/4;
    double J_Y = std::log((prob_i_err*prob_z_err)/(prob_x_err*prob_y_err))/4;
    double J_Z = std::log((prob_i_err*prob_y_err)/(prob_x_err*prob_z_err))/4;

    std::cout << "J params for the run:" << std::endl;
    std::cout << "J_I = " << J_I << "J_X = " << J_X << "J_Y = " << J_Y << "J_Z = " << J_Z << std::endl;


    // declaration of Pauli error over grid of qubits
    int *d_pauli_errors;
    CHECK_CUDA(cudaMalloc(&d_pauli_errors, num_interactions * num_qubits * sizeof(*d_pauli_errors)));

    // declaration of interactions stemming from different commutator terms in https://arxiv.org/pdf/1809.10704 eq 14.
    double *d_interactions_x, *d_interactions_y, *d_interactions_z;
    CHECK_CUDA(cudaMalloc(&d_interactions_x, num_interactions * num_qubits * sizeof(*d_interactions_x)));
    CHECK_CUDA(cudaMalloc(&d_interactions_y, num_interactions * num_qubits * sizeof(*d_interactions_y)));
    CHECK_CUDA(cudaMalloc(&d_interactions_z, num_interactions * num_qubits * sizeof(*d_interactions_z)));

    /* 
    declaration of interaction arrays which should be inititalized in same ordering as done in pure bit flip implementation.
    o are qubits and their respictive error stored in d_pauli.
    b are ising spins with closed interactions determined by |[X,E]|
    r are ising spins with closed interactions determined by |[Z,E]|
    Four body terms are not depicted.
    right interaction of b (1,0) ising spin is d_interactions_x(2,1) (here may be at boundary periodically closed to get interaction from first in row)
    right interaction of r (1,0) ising spin is d_interactions_z(3,0)
    down interaction of b (1,0) ising spin is d_interactions_x(3,0)
    down interaction of r (1,0) ising spin is d_interactions_z(4,0) (here may be at boundary periodically closed to get interaction from first in column)
    (b and r are used as lattices are colored blue or red here)

    Example X=3, Y=6:
            X (measured in o's)

        o-b-o-b-o-b 
        | | | | | |
        r-o-r-o-r-o
        | | | | | |
        o-b-o-b-o-b
    Y   | | | | | |
        r-o-r-o-r-o 
        | | | | | |
        o-b-o-b-o-b
        | | | | | |
        r-o-r-o-r-o

    The four body interactions are imagined to be rooted on the b lattice. 
    for each b ising spin exist two types of four body interactions. 
    A right four body term:      and a down four body term:

                r                           b                              
                |                           |                                  
              b-o-b                       r-o-r                             
                |                           |                           
                r                           b                                   
    */
    double *d_interactions_r, *d_interactions_b, *d_interactions_down_four_body, *d_interactions_right_four_body; // single set of interaction arrays for all walkers to share
    CHECK_CUDA(cudaMalloc(&d_interactions_r, num_interactions * X * Y * sizeof(*d_interactions_r)));
    CHECK_CUDA(cudaMalloc(&d_interactions_b, num_interactions * X * Y * sizeof(*d_interactions_b)));
    CHECK_CUDA(cudaMalloc(&d_interactions_down_four_body, num_interactions* X * Y/2 * sizeof(*d_interactions_down_four_body)));
    CHECK_CUDA(cudaMalloc(&d_interactions_right_four_body, num_interactions * X * Y/2 * sizeof(*d_interactions_right_four_body)));

    // declare b and r lattice
    signed char *d_lattice_r, *d_lattice_b;
    CHECK_CUDA(cudaMalloc(&d_lattice_b, total_walker * num_qubits / 2 * sizeof(*d_lattice_b)));
    CHECK_CUDA(cudaMalloc(&d_lattice_r, total_walker * num_qubits / 2 * sizeof(*d_lattice_r)));
    // // init lattices for testing with spin up
    // CHECK_CUDA(cudaMemset(d_lattice_b, 1, num_walker * X * Y/2 * sizeof(*d_lattice_b)));
    // CHECK_CUDA(cudaMemset(d_lattice_r, 1, num_walker * X * Y/2 * sizeof(*d_lattice_r)));


    double factor = std::exp(1);
    
    const int E_min = -3*X*Y; // derived from 2 decoupled Ising lattices with dim (X, Y/2) -> 2*(-2)*(X*Y/2) and additionally two four body interactions rooted on spins of one lattice: -2*(X*Y/2)
    const int E_max = -E_min;

    
    IntervalResult interval_result = generate_intervals(E_min, E_max, num_intervals_per_interaction, 1, 1.0f);
    
    std::cout << "Intervals for the run" << std::endl;
  
    for (int i=0; i< num_intervals_per_interaction; i++){
        std::cout << "[" << interval_result.h_start[i] << ", " << interval_result.h_end[i] << "]" << std::endl;
    }
    
    long long len_histogram = E_max - E_min + 1; // len of histogram per interaction 
    long long len_total_histogram = num_interactions*len_histogram; // len of histogram over all interactions
    
    unsigned long long *d_H; 
    CHECK_CUDA(cudaMalloc(&d_H, len_total_histogram * sizeof(*d_H)));
    CHECK_CUDA(cudaMemset(d_H, 0, len_total_histogram * sizeof(*d_H)));
    
    unsigned long long *d_iter;
    CHECK_CUDA(cudaMalloc(&d_iter, total_walker * sizeof(*d_iter)));
    CHECK_CUDA(cudaMemset(d_iter, 0, total_walker * sizeof(*d_iter)));
    
    float *d_probs; // for lattice init
    CHECK_CUDA(cudaMalloc(&d_probs, total_walker * sizeof(*d_probs)));
    CHECK_CUDA(cudaMemset(d_probs, 0, total_walker * sizeof(*d_probs)));
    
    // for testing only single lattice
    double *d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, total_walker * sizeof(*d_energy)));

    signed char *d_store_lattice_b, *d_store_lattice_r; // to store found configs for given energy range
    CHECK_CUDA(cudaMalloc(&d_store_lattice_b, total_intervals * X * Y/2 * sizeof(*d_store_lattice_b)));
    CHECK_CUDA(cudaMalloc(&d_store_lattice_r, total_intervals * X * Y/2 * sizeof(*d_store_lattice_r)));

    int *d_found_interval; // signaler to identify intervals where configs where found inside
    CHECK_CUDA(cudaMalloc(&d_found_interval, total_intervals * sizeof(*d_found_interval)));
    CHECK_CUDA(cudaMemset(d_found_interval, 0, total_intervals * sizeof(*d_found_interval)));

    double *d_interval_energies;
    CHECK_CUDA(cudaMalloc(&d_interval_energies, total_intervals * sizeof(*d_interval_energies)));
    CHECK_CUDA(cudaMemset(d_interval_energies, 0, total_intervals * sizeof(*d_interval_energies)));

    int *d_offset_lattice_per_walker, *d_offset_lattice_per_interval; // holds for both b and r lattice
    CHECK_CUDA(cudaMalloc(&d_offset_lattice_per_walker, total_walker * sizeof(*d_offset_lattice_per_walker)));
    CHECK_CUDA(cudaMalloc(&d_offset_lattice_per_interval, total_intervals * sizeof(*d_offset_lattice_per_interval)));
 
    int blocks_qubit_x_thread = (num_interactions * num_qubits + max_threads_per_block - 1)/max_threads_per_block;
    int blocks_total_walker_x_thread = (total_walker + max_threads_per_block - 1)/max_threads_per_block;
    int blocks_total_intervals_x_thread = (total_intervals + max_threads_per_block - 1)/max_threads_per_block;

    // General question about when to use offset in curand_init

    generate_pauli_errors<<<blocks_qubit_x_thread, max_threads_per_block>>>(d_pauli_errors, num_qubits, num_interactions, seed, prob_i_err, prob_x_err, prob_y_err, prob_z_err);
    cudaDeviceSynchronize();

    get_interaction_from_commutator<<<blocks_qubit_x_thread, max_threads_per_block>>>(d_pauli_errors, d_interactions_x, d_interactions_y, d_interactions_z, num_qubits, num_interactions, J_X, J_Y, J_Z);
    cudaDeviceSynchronize();

    init_interactions_eight_vertex<<<blocks_qubit_x_thread, max_threads_per_block>>>(d_interactions_x, d_interactions_y, d_interactions_z, num_qubits, num_interactions,  X, Y, d_interactions_r, d_interactions_b, d_interactions_down_four_body, d_interactions_right_four_body);
    cudaDeviceSynchronize();

    // calc_energy_eight_vertex<<<num_blocks, max_threads_per_block>>>(d_energy, d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body , d_interactions_down_four_body, num_qubits, X, Y, );
    // cudaDeviceSynchronize();

    return 0;    

}
