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