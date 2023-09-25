#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

using namespace std;

__global__ void setup_kernel(curandState *state, const int nx, const int ny, const int seed){
    long long tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    while (tid < nx*ny){
        curand_init(seed, tid, 0, &state[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void init_spins(signed char* lattice, curandState *state, const long long nx, const long long ny, const float p){
    
    long long tid = static_cast<long long>(threadIdx.x + blockIdx.x * blockDim.x);
    
    curandState local_state = state[tid];
    
    while (tid < nx*ny){
        float randval = curand_uniform(&local_state); 
        lattice[tid] = (randval<p)? -1 : 1;

        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char **argv){
    
    const int nx = 10;
    const int ny = 10;
    const int seed = 42;

    const float p = 0.5;

    std::vector<signed char> h_lattice(nx*ny);

    signed char *d_lattice;
    cudaMalloc(&d_lattice, nx*ny*sizeof(signed char));

    curandState *devStates;
    cudaMalloc((void **)&devStates, nx*ny*sizeof(curandStatenx));

    setup_kernel<<<1,24>>>(devStates, nx, ny, seed);

    init_spins<<<1,24>>>(d_lattice, devStates, nx, ny, p);

    cudaMemcpy(h_lattice.data(), d_lattice, nx*ny*sizeof(signed char), cudaMemcpyDeviceToHost);

    for (int i=0; i<nx; i++){
        for (int j=0; j<ny; j++){
            cout << int(h_lattice[i*nx+j]);
        }
        cout << endl;
    }
}