#include <cstdlib>
#include <fstream>
#include <iostream>
#include <time.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/complex.h>
#include <cmath>
#include <math.h>
#include <vector>
#include <string>
#include <sys/stat.h>

#include "defines.h"

using namespace std;

int main(void){
    
    cout << log(20);
    
    /*
    int L = 12;
    int seed = 0;

    // Setup cuRAND generator
    curandGenerator_t lattice_rng;
    curandCreateGenerator(&lattice_rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    float *lattice_randvals;
    cudaMalloc(&lattice_randvals, L * L/2 * sizeof(*lattice_randvals));

    for (int i=0; i<27; i++){
        for (int j=0; j<100;j++){
            curandSetPseudoRandomGeneratorSeed(lattice_rng, seed);
            seed = seed+1;
            
            curandGenerateUniform(lattice_rng, lattice_randvals, L*L/2);
            if (i == 26){
                if (j>86){
                    std::vector<float> h_randvals(L*L/2);
                    cudaMemcpy(h_randvals.data(), lattice_randvals, L*L/2*sizeof(float), cudaMemcpyDeviceToHost);
                    printf("s %u \n", j);
                    for (int i=0; i < L*L/2; i++){
                        cout << h_randvals[i] << endl;
                    }
                }
                if (j>91){
                    return;
                }
            }
        }
    }
    */
}