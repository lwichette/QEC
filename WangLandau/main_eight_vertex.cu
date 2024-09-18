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
    int histogram_scale = 1;

    float prob_i_err = 0;
    float prob_x_err = 0;
    float prob_z_err = 0;
    float overlap = 0;

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
            {"qubit_specific_noise", required_argument, 0, 'k'},
            {"replica_exchange_offsets", 1, 0, 'l'},
            {"num_intervals", required_argument, 0, 'm'},
            {"num_iterations", required_argument, 0, 'n'},
            {"overlap_decimal", 1, 0, 'o'},
            {"seed_histogram", 1, 0, 'p'},
            {"seed_run", 1, 0, 'q'},
            {"hist_scale", required_argument, 0, 'r'},
            {"num_interactions", required_argument, 0, 's'},
            {"walker_per_interval", required_argument, 0, 'w'},
            {"X", required_argument, 0, 'x'},
            {"Y", required_argument, 0, 'y'},
            {0, 0, 0, 0}};
    }