#include "./header/cudamacro.h"
#include "./header/wlutils.cuh"
#include <thrust/extrema.h> //addition needed for my (Linnea's) version of thrust -- comment out if this causes issues

int main(int argc, char **argv){
    /*
    X, Y sepcifies grid of qubits not Ising spins - have to be careful with boundary conditions and dimensions here!
    */


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming device 0
    int max_threads_per_block = prop.maxThreadsPerBlock;

    int X, Y;
    
    float prob_i_err, prob_x_err, prob_y_err, prob_z_err;

    int num_wl_loops, num_iterations, num_walker;

    int seed;

    int num_intervals;
    
    char logical_error_type = 'I';

    int boundary_type = 0;

    int och;

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
            {0, 0, 0, 0}
        };

        och = getopt_long(argc, argv, "x:y:f:g:h:n:l:w:s:i:e:b:", long_options, &option_index);
        
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
                boundary_type = atoi(optarg);
                if(boundary_type != 0 && boundary_type !=1){
                    fprintf(stderr, "Error: Invalid value for boundary type. Must be 0 (periodic) or 1 (open).\n");
                    exit(EXIT_FAILURE);
                }
                break;
			case '?':
				exit(EXIT_FAILURE);

			default:
				fprintf(stderr, "unknown option: %c\n", och);
				exit(EXIT_FAILURE);
        }
    }

    if(prob_x_err+prob_y_err+prob_z_err>1){
        fprintf(stderr, "Error: Invalid value for error probabilities. Must sum to less then 1.\n");
        exit(EXIT_FAILURE);
    }
    else{ 
        prob_i_err = 1-(prob_x_err+prob_y_err+prob_z_err);
    }

    unsigned long long num_qubits = X*Y; // which is equivalent to amount of interactions, i.e. add further Ising spins for open boundary but dimensionality of physical system remains unchanged

    unsigned long long ising_x, ising_y;

    ising_x = (boundary_type == 0) ? X : X+1;
    ising_y = (boundary_type == 0) ? Y : Y+1;     

    
    //Coupling strength from Nishimori condition in https://arxiv.org/pdf/1809.10704 eq 15 with beta = 1.
    double J_I = std::log(prob_i_err*prob_x_err*prob_y_err*prob_z_err)/4;
    double J_X = std::log((prob_i_err*prob_x_err)/(prob_y_err*prob_z_err))/4;
    double J_Y = std::log((prob_i_err*prob_z_err)/(prob_x_err*prob_y_err))/4;
    double J_Z = std::log((prob_i_err*prob_y_err)/(prob_x_err*prob_z_err))/4;

    // declaration of interactions stemming from different commutator terms in https://arxiv.org/pdf/1809.10704 eq 14.
    signed char *d_interactions_x, *d_interactions_y, *d_interactions_z;
    CHECK_CUDA(cudaMalloc(&d_interactions_x, ising_x * ising_y * 2 * sizeof(*d_interactions_x)));
    CHECK_CUDA(cudaMalloc(&d_interactions_y, ising_x * ising_y * 2 * sizeof(*d_interactions_y)));
    CHECK_CUDA(cudaMalloc(&d_interactions_z, ising_x * ising_y * 2 * sizeof(*d_interactions_z)));
    
    return 0;

}
