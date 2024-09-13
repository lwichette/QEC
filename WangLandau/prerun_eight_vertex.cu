#include "./header/cudamacro.h"
#include "./header/wlutils.cuh"
#include <thrust/extrema.h> //addition needed for my (Linnea's) version of thrust -- comment out if this causes issues

int main(int argc, char **argv)
{
    /*
    IMPORTANT:
    X, Y sepcifies grid of qubits not Ising spins.
    X is qubits per row and Y rows of grid such that Y must be even!
    This is motivated by grid example down below. The Y is not equal to qubits per column but Y/2 is! To have equal amount of qubits per column, this must be even.
    */

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming device 0
    int max_threads_per_block = 128;

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

    // char logical_error_type = 'I';

    int boundary_type = 0;

    int och;

    int num_interactions = 1;

    int histogram_scale = 1;

    while (1)
    {
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
            {"boundary", required_argument, 0, 'b'},
            {"replicas", required_argument, 0, 'r'},
            {"hist_scale", required_argument, 0, 'q'},
            {0, 0, 0, 0}};

        och = getopt_long(argc, argv, "x:y:f:g:h:n:l:w:s:i:b:r:q:", long_options, &option_index);

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
        case 'b':
            boundary_type = atoi(optarg);
            if (boundary_type != 0 && boundary_type != 1)
            {
                fprintf(stderr, "Error: Invalid value for boundary type. Must be 0 (periodic) or 1 (open).\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'r':
            num_interactions = atoi(optarg);
            break;
        case 'q':
            histogram_scale = atoi(optarg);
            break;
        case '?':
            exit(EXIT_FAILURE);

        default:
            fprintf(stderr, "unknown option: %c\n", och);
            exit(EXIT_FAILURE);
        }
    }

    if (prob_x_err + prob_y_err + prob_z_err > 1 || prob_x_err * prob_y_err * prob_z_err == 0)
    {
        fprintf(stderr, "Error: Invalid value for error probabilities. Must sum to less then 1 and not be 0.\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        prob_i_err = 1 - (prob_x_err + prob_y_err + prob_z_err);
    }

    if (Y % 2 != 0)
    {
        fprintf(stderr, "Error: Invalid value for Y. Must be even.\n");
        exit(EXIT_FAILURE);
    }

    unsigned long long num_qubits = X * Y; // which is equivalent to amount of interactions, i.e. add further Ising spins for open boundary but dimensionality of physical system remains unchanged

    int total_walker = num_interactions * walker_per_interaction;
    int total_intervals = num_interactions * num_intervals_per_interaction;

    // Coupling strength from Nishimori condition in https://arxiv.org/pdf/1809.10704 eq 15 with beta = 1.
    double J_I = std::log(prob_i_err * prob_x_err * prob_y_err * prob_z_err) / 4;
    double J_X = std::log((prob_i_err * prob_x_err) / (prob_y_err * prob_z_err)) / 4;
    double J_Y = std::log((prob_i_err * prob_z_err) / (prob_x_err * prob_y_err)) / 4;
    double J_Z = std::log((prob_i_err * prob_y_err) / (prob_x_err * prob_z_err)) / 4;

    // Find the maximum absolute value of J_I, J_X, J_Y, J_Z to bound the energy range
    double max_J = std::max({std::abs(J_X), std::abs(J_Y), std::abs(J_Z)});

    // Rescale the values
    J_I *= (histogram_scale / max_J);
    J_X *= (histogram_scale / max_J);
    J_Y *= (histogram_scale / max_J);
    J_Z *= (histogram_scale / max_J);

    std::cout << "J params rescaled by hist_scale/ absolute max of J_i = " << histogram_scale / max_J << ":" << std::endl;
    std::cout << "J_I = " << J_I << " J_X = " << J_X << " J_Y = " << J_Y << " J_Z = " << J_Z << std::endl;

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
    CHECK_CUDA(cudaMalloc(&d_interactions_down_four_body, num_interactions * X * Y / 2 * sizeof(*d_interactions_down_four_body)));
    CHECK_CUDA(cudaMalloc(&d_interactions_right_four_body, num_interactions * X * Y / 2 * sizeof(*d_interactions_right_four_body)));

    // declare b and r lattice
    signed char *d_lattice_r, *d_lattice_b;
    CHECK_CUDA(cudaMalloc(&d_lattice_b, total_walker * num_qubits / 2 * sizeof(*d_lattice_b)));
    CHECK_CUDA(cudaMalloc(&d_lattice_r, total_walker * num_qubits / 2 * sizeof(*d_lattice_r)));
    // // init lattices for testing with spin up
    // CHECK_CUDA(cudaMemset(d_lattice_b, 1, total_walker * X * Y / 2 * sizeof(*d_lattice_b)));
    // CHECK_CUDA(cudaMemset(d_lattice_r, 1, total_walker * X * Y / 2 * sizeof(*d_lattice_r)));

    double factor = std::exp(1);

    const int E_min = -3 * histogram_scale * X * Y; // derived from 2 decoupled Ising lattices with dim (X, Y/2) -> 2*(-2)*(X*Y/2) and additionally two four body interactions rooted on spins of one lattice: -2*(X*Y/2)
    const int E_max = -E_min;

    IntervalResult interval_result = generate_intervals(E_min, E_max, num_intervals_per_interaction, 1, 1.0f);

    std::cout << "Intervals for the run" << std::endl;

    for (int i = 0; i < num_intervals_per_interaction; i++)
    {
        std::cout << "[" << interval_result.h_start[i] << ", " << interval_result.h_end[i] << "]" << std::endl;
    }

    long long len_histogram = E_max - E_min + 1;                      // len of histogram per interaction
    long long len_total_histogram = num_interactions * len_histogram; // len of histogram over all interactions

    unsigned long long *d_H;
    CHECK_CUDA(cudaMalloc(&d_H, len_total_histogram * sizeof(*d_H)));
    CHECK_CUDA(cudaMemset(d_H, 0, len_total_histogram * sizeof(*d_H)));

    unsigned long long *d_iter;
    CHECK_CUDA(cudaMalloc(&d_iter, total_walker * sizeof(*d_iter)));
    CHECK_CUDA(cudaMemset(d_iter, 0, total_walker * sizeof(*d_iter)));

    float *d_probs; // for lattice init
    CHECK_CUDA(cudaMalloc(&d_probs, total_walker * sizeof(*d_probs)));
    CHECK_CUDA(cudaMemset(d_probs, 0, total_walker * sizeof(*d_probs)));

    double *d_energy;
    CHECK_CUDA(cudaMalloc(&d_energy, total_walker * sizeof(*d_energy)));

    signed char *d_store_lattice_b, *d_store_lattice_r; // to store found configs for given energy range
    CHECK_CUDA(cudaMalloc(&d_store_lattice_b, total_intervals * X * Y / 2 * sizeof(*d_store_lattice_b)));
    CHECK_CUDA(cudaMalloc(&d_store_lattice_r, total_intervals * X * Y / 2 * sizeof(*d_store_lattice_r)));

    int *d_found_interval; // signaler to identify intervals where configs were found at
    CHECK_CUDA(cudaMalloc(&d_found_interval, total_intervals * sizeof(*d_found_interval)));
    CHECK_CUDA(cudaMemset(d_found_interval, 0, total_intervals * sizeof(*d_found_interval)));

    double *d_interval_energies;
    CHECK_CUDA(cudaMalloc(&d_interval_energies, total_intervals * sizeof(*d_interval_energies)));
    CHECK_CUDA(cudaMemset(d_interval_energies, 0, total_intervals * sizeof(*d_interval_energies)));

    int *d_offset_lattice_per_walker, *d_offset_lattice_per_interval; // holds for both b and r lattice
    CHECK_CUDA(cudaMalloc(&d_offset_lattice_per_walker, total_walker * sizeof(*d_offset_lattice_per_walker)));
    CHECK_CUDA(cudaMalloc(&d_offset_lattice_per_interval, total_intervals * sizeof(*d_offset_lattice_per_interval)));

    int blocks_qubit_x_thread = (num_interactions * num_qubits + max_threads_per_block - 1) / max_threads_per_block;
    int blocks_spins_single_color_x_thread = (total_walker * num_qubits / 2 + max_threads_per_block - 1) / max_threads_per_block;
    int blocks_total_walker_x_thread = (total_walker + max_threads_per_block - 1) / max_threads_per_block;
    int blocks_total_intervals_x_thread = (total_intervals + max_threads_per_block - 1) / max_threads_per_block;

    // General question about when to use offset in curand_init

    generate_pauli_errors<<<blocks_qubit_x_thread, max_threads_per_block>>>(d_pauli_errors, num_qubits, num_interactions, seed, prob_i_err, prob_x_err, prob_y_err, prob_z_err);
    cudaDeviceSynchronize();

    get_interaction_from_commutator<<<blocks_qubit_x_thread, max_threads_per_block>>>(d_pauli_errors, d_interactions_x, d_interactions_y, d_interactions_z, num_qubits, num_interactions, J_X, J_Y, J_Z);
    cudaDeviceSynchronize();

    init_interactions_eight_vertex<<<blocks_qubit_x_thread, max_threads_per_block>>>(d_interactions_x, d_interactions_y, d_interactions_z, num_qubits, num_interactions, X, Y, d_interactions_r, d_interactions_b, d_interactions_down_four_body, d_interactions_right_four_body);
    cudaDeviceSynchronize();

    init_lattice<<<blocks_spins_single_color_x_thread, max_threads_per_block>>>(d_lattice_b, d_probs, X, Y / 2, total_walker, seed - 2);
    init_lattice<<<blocks_spins_single_color_x_thread, max_threads_per_block>>>(d_lattice_r, d_probs, X, Y / 2, total_walker, seed - 1);
    init_offsets_lattice<<<blocks_total_walker_x_thread, max_threads_per_block>>>(d_offset_lattice_per_walker, X, Y / 2, total_walker);
    init_offsets_lattice<<<blocks_total_walker_x_thread, max_threads_per_block>>>(d_offset_lattice_per_interval, X, Y / 2, total_intervals);
    cudaDeviceSynchronize();

    calc_energy_eight_vertex<<<blocks_total_walker_x_thread, max_threads_per_block>>>(d_energy, d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, num_qubits, X, Y, total_walker, walker_per_interaction);
    cudaDeviceSynchronize();

    //-------------------------
    // TEST BLOCK
    std::vector<double> test_energies(total_walker);
    std::vector<double> test_energies_wl(total_walker);
    // std::vector<double> test_interactions_b(X * Y * num_interactions);
    // std::vector<double> test_interactions_r(X * Y * num_interactions);
    // std::vector<double> test_interactions_four_body_right(X * Y / 2 * num_interactions);
    // std::vector<double> test_interactions_four_body_down(X * Y / 2 * num_interactions);
    // std::vector<signed char> test_lattice_b(X * Y / 2 * total_walker);
    // std::vector<signed char> test_lattice_r(X * Y / 2 * total_walker);

    // CHECK_CUDA(cudaMemcpy(test_energies.data(), d_energy, total_walker * sizeof(*d_energy), cudaMemcpyDeviceToHost));
    // CHECK_CUDA(cudaMemcpy(test_interactions_b.data(), d_interactions_b, X * Y * num_interactions * sizeof(*d_interactions_b), cudaMemcpyDeviceToHost));
    // CHECK_CUDA(cudaMemcpy(test_interactions_r.data(), d_interactions_r, X * Y * num_interactions * sizeof(*d_interactions_r), cudaMemcpyDeviceToHost));
    // CHECK_CUDA(cudaMemcpy(test_interactions_four_body_right.data(), d_interactions_right_four_body, X * Y / 2 * num_interactions * sizeof(*d_interactions_right_four_body), cudaMemcpyDeviceToHost));
    // CHECK_CUDA(cudaMemcpy(test_interactions_four_body_down.data(), d_interactions_down_four_body, X * Y / 2 * num_interactions * sizeof(*d_interactions_down_four_body), cudaMemcpyDeviceToHost));
    // CHECK_CUDA(cudaMemcpy(test_lattice_b.data(), d_lattice_b, X * Y / 2 * total_walker * sizeof(*d_lattice_b), cudaMemcpyDeviceToHost));
    // CHECK_CUDA(cudaMemcpy(test_lattice_r.data(), d_lattice_r, X * Y / 2 * total_walker * sizeof(*d_lattice_r), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < num_interactions; i++)
    // {
    //     std::string path = "test/eight_vertex/periodic/prob_X_" + std::to_string(prob_x_err) + "__prob_Y_" + std::to_string(prob_y_err) + "__prob_Z_" + std::to_string(prob_z_err) + "/X_" + std::to_string(X) + "_Y_" + std::to_string(Y) + "/seed_" + std::to_string(seed + i) + "/error_class_I";

    //     int offset_interactions = i * X * Y;               // for interactions closed on a single colored sublattice
    //     int offset_four_body_interactions = i * X * Y / 2; // for interactions closed on a single colored sublattice
    //     int offset_lattice = i * walker_per_interaction * X * Y / 2;
    //     int offset_energies = i * walker_per_interaction;

    //     create_directory(path + "/interactions");
    //     create_directory(path + "/lattice");
    //     create_directory(path + "/histogram");

    //     write(test_interactions_b.data() + offset_interactions, path + "/interactions/interactions_b", Y, X, 1, false);
    //     write(test_interactions_r.data() + offset_interactions, path + "/interactions/interactions_r", Y, X, 1, false);
    //     write(test_interactions_four_body_right.data() + offset_four_body_interactions, path + "/interactions/interactions_four_body_right", Y / 2, X, 1, false);
    //     write(test_interactions_four_body_down.data() + offset_four_body_interactions, path + "/interactions/interactions_four_body_down", Y / 2, X, 1, false);
    //     write(test_lattice_b.data() + offset_lattice, path + "/lattice/lattice_b", Y / 2, X, walker_per_interaction, true, test_energies.data() + offset_energies);
    //     write(test_lattice_r.data() + offset_lattice, path + "/lattice/lattice_r", Y / 2, X, walker_per_interaction, true, test_energies.data() + offset_energies);
    // }
    // test_eight_vertex_periodic_wl_step<<<blocks_total_walker_x_thread, max_threads_per_block>>>(d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, d_energy, d_iter, num_qubits, X, Y, total_walker, walker_per_interaction);
    // cudaDeviceSynchronize();
    // CHECK_CUDA(cudaMemcpy(test_energies.data(), d_energy, total_walker * sizeof(*d_energy), cudaMemcpyDeviceToHost));
    // std::cout << "Before iterations:" << std::endl;
    // for (int idx = 0; idx < total_walker; idx++)
    // {
    //     std::cout << " walker idx: " << idx << " calc energy: " << test_energies[idx] << std::endl;
    // }
    //-------------------------

    int found_interval = 0;

    for (int i = 0; i < num_wl_loops; i++)
    {
        wang_landau_pre_run_eight_vertex<<<blocks_total_walker_x_thread, max_threads_per_block>>>(d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, d_energy, d_H, d_iter, d_found_interval, d_store_lattice_b, d_store_lattice_r, E_min, E_max, num_iterations, num_qubits, X, Y, seed, interval_result.len_interval, found_interval, total_walker, num_intervals_per_interaction, boundary_type, walker_per_interaction);
        cudaDeviceSynchronize();

        // TEST BLOCK
        //-----------
        CHECK_CUDA(cudaMemcpy(test_energies_wl.data(), d_energy, total_walker * sizeof(*d_energy), cudaMemcpyDeviceToHost)); // get energies from wl step with energy diff calc
        calc_energy_eight_vertex<<<blocks_total_walker_x_thread, max_threads_per_block>>>(d_energy, d_lattice_b, d_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, num_qubits, X, Y, total_walker, walker_per_interaction);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaMemcpy(test_energies.data(), d_energy, total_walker * sizeof(*d_energy), cudaMemcpyDeviceToHost)); // get energies from calc energy function
        assert(test_energies.size() == total_walker);
        assert(test_energies_wl.size() == total_walker);
        for (int idx = 0; idx < total_walker; idx++)
        {
            // printf("result_condition: %d \n ", std::abs(test_energies_wl[idx] - test_energies[idx]) > 0.0001);
            // printf("result_condition: %d \n ", std::abs(test_energies[idx] - test_energies[idx]) > 0.0001);
            if (std::abs(test_energies_wl[idx] - test_energies[idx]) > 0.000001)
            {
                std::cerr << "Assertion failed for iteration: " << i << " walker idx: " << idx << " calc energy: " << test_energies[idx] << " wl calc energy: " << test_energies_wl[idx] << " Diff: " << std::abs(test_energies_wl[idx] - test_energies[idx]) << std::endl;
            }
            // // assert(std::abs(test_energies_wl[idx] - test_energies[idx]) > 0.01);
        }
        //-----------

        if (found_interval == 0)
        {
            thrust::device_ptr<int> d_found_interval_ptr(d_found_interval);
            thrust::device_ptr<int> min_found_interval_ptr = thrust::min_element(d_found_interval_ptr, d_found_interval_ptr + total_intervals);
            found_interval = *min_found_interval_ptr;
        }
    }

    calc_energy_eight_vertex<<<blocks_total_intervals_x_thread, max_threads_per_block>>>(d_interval_energies, d_store_lattice_b, d_store_lattice_r, d_interactions_b, d_interactions_r, d_interactions_right_four_body, d_interactions_down_four_body, num_qubits, X, Y, total_intervals, num_intervals_per_interaction);
    cudaDeviceSynchronize();

    std::vector<double> h_interval_energies(total_intervals);
    std::vector<double> h_interactions_b(X * Y * num_interactions);
    std::vector<double> h_interactions_r(X * Y * num_interactions);
    std::vector<double> h_interactions_four_body_right(X * Y / 2 * num_interactions);
    std::vector<double> h_interactions_four_body_down(X * Y / 2 * num_interactions);
    std::vector<signed char> h_store_lattice_b(X * Y / 2 * total_intervals);
    std::vector<signed char> h_store_lattice_r(X * Y / 2 * total_intervals);
    std::vector<unsigned long long> h_H(len_total_histogram);

    CHECK_CUDA(cudaMemcpy(h_interval_energies.data(), d_interval_energies, total_intervals * sizeof(*d_energy), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_interactions_b.data(), d_interactions_b, X * Y * num_interactions * sizeof(*d_interactions_b), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_interactions_r.data(), d_interactions_r, X * Y * num_interactions * sizeof(*d_interactions_r), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_interactions_four_body_right.data(), d_interactions_right_four_body, X * Y / 2 * num_interactions * sizeof(*d_interactions_right_four_body), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_interactions_four_body_down.data(), d_interactions_down_four_body, X * Y / 2 * num_interactions * sizeof(*d_interactions_down_four_body), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_store_lattice_b.data(), d_store_lattice_b, X * Y / 2 * total_intervals * sizeof(*d_store_lattice_b), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_store_lattice_r.data(), d_store_lattice_r, X * Y / 2 * total_intervals * sizeof(*d_store_lattice_r), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_H.data(), d_H, len_total_histogram * sizeof(*d_H), cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_interactions; i++)
    {
        std::string path = "init/eight_vertex/periodic/prob_X_" + std::to_string(prob_x_err) + "__prob_Y_" + std::to_string(prob_y_err) + "__prob_Z_" + std::to_string(prob_z_err) + "/X_" + std::to_string(X) + "_Y_" + std::to_string(Y) + "/seed_" + std::to_string(seed + i) + "/error_class_I";

        int offset_interactions = i * X * Y;               // for interactions closed on a single colored sublattice
        int offset_four_body_interactions = i * X * Y / 2; // for interactions closed on a single colored sublattice
        int offset_lattice = i * num_intervals_per_interaction * X * Y / 2;
        int offset_energies = i * num_intervals_per_interaction;

        create_directory(path + "/interactions");
        create_directory(path + "/lattice");
        create_directory(path + "/histogram");

        write(h_interactions_b.data() + offset_interactions, path + "/interactions/interactions_b", Y, X, 1, false);
        write(h_interactions_r.data() + offset_interactions, path + "/interactions/interactions_r", Y, X, 1, false);
        write(h_interactions_four_body_right.data() + offset_four_body_interactions, path + "/interactions/interactions_four_body_right", Y / 2, X, 1, false);
        write(h_interactions_four_body_down.data() + offset_four_body_interactions, path + "/interactions/interactions_four_body_down", Y / 2, X, 1, false);
        write(h_store_lattice_b.data() + offset_lattice, path + "/lattice/lattice_b", Y / 2, X, num_intervals_per_interaction, true, h_interval_energies.data() + offset_energies);
        write(h_store_lattice_r.data() + offset_lattice, path + "/lattice/lattice_r", Y / 2, X, num_intervals_per_interaction, true, h_interval_energies.data() + offset_energies);
        write_histograms(h_H.data() + i * len_histogram, path + "/histogram/", (E_max - E_min + 1), seed, E_min);
    }

    // printf("Finished prerun for Lattice %d x %d, boundary condition %s, probability %f, error type %c and %d interactions \n", X, Y, boundary.c_str(), prob_interactions, logical_error_type, num_interactions);

    return 0;
}
