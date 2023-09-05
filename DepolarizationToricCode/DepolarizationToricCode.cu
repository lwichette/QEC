#include <chrono>       // date and time utility
#include <fstream>      // file manipulation utility
#include <getopt.h>     // command line parser
#include <iostream>     // reading and writing utility
#include <string>       // string data type
#include <math.h>       // math stuff -- SHOCKING!!

#include <cuda_fp16.h>  // half precision intrinsic functions: probably for computing exponential on the device
#include <curand.h>     // efficient generation of high-quality pseudorandom and quasirandom numbers
#include <cublas_v2.h>  // allows the user to access the computational resources of NVIDIA Graphics Processing Unit for Cuda Basic Linear Algebra Subprograms

#define THREADS 128
#define TCRIT  31.2003246f

// Initialize lattice spins
__global__ void init_spins(signed char* lattice,
                        const float* __restrict__ randvals,
                        const long long nx,
                        const long long ny){

    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    float randval = randvals[tid];
    signed char val = (randval<0.5)? -1:1;
    lattice[tid] = val;
}

// Write sublattice to file (Sanity Check)
void write_randvals(float *sublattice_d, std::string filename, long long nx, long long ny){
    printf("Writing sublattice to %s ... \n", filename.c_str());
    signed char *sublattice_h;

    sublattice_h = (signed char*)malloc(nx*ny/4*sizeof(*sublattice_h));

    cudaMemcpy(sublattice_h, sublattice_d, nx*ny/4*sizeof(*sublattice_d), cudaMemcpyDeviceToHost);

    std::ofstream f;
    f.open(filename);
    if (f.is_open()){
        for (int i = 0; i < nx/2; i++){
            for (int j = 0; j < ny/2; j++){
                f << (int)sublattice_h[i*ny/2 + j] << " ";
            }
            f << std::endl;
        }
    }
    f.close();



}

// Write sublattice to file (Sanity Check)
void write_sublattice(signed char *sublattice_d, std::string filename, long long nx, long long ny){
    printf("Writing sublattice to %s ... \n", filename.c_str());
    signed char *sublattice_h;

    sublattice_h = (signed char*)malloc(nx*ny/4*sizeof(*sublattice_h));

    cudaMemcpy(sublattice_h, sublattice_d, nx*ny/4*sizeof(*sublattice_d), cudaMemcpyDeviceToHost);

    std::ofstream f;
    f.open(filename);
    if (f.is_open()){
        for (int i = 0; i < nx/2; i++){
            for (int j = 0; j < ny/2; j++){
                f << (int)sublattice_h[i*ny/2 + j] << " ";
            }
            f << std::endl;
        }
    }
    f.close();



}

// Kernel for updating the spins on the device
template<bool is_Z, bool is_black>
__global__ void update_lattice(signed char *lattice, // lattice to change
                                const signed char* __restrict__ same_lattice_op, // lattice of same type (X,Z) but opposite color (color now corresponds to rows of full lattice)
                                const signed char* __restrict__ op_lattice_same, // lattice of opposite type but same color
                                const signed char* __restrict__ op_lattice_op,   // lattice of opposite type and opposite color   
                                const float* __restrict__ randvals,
                                const float inv_temp,
                                const long long nx, // for now I am choosing that nx and ny are the sizes of the sublattices i.e. nx/2 and ny/2
                                const long long ny){ 
    
    // Idnetifying the unique number that corresponds to the location of the parallel process
    const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    int i = tid/ny;
    int j = tid%ny;

    if (i >= nx || j >= ny) return;

    // Setting up periodic boundary conditions
    int ipp = (i + 1 < nx) ? i + 1 : 0;
    int inn = (i - 1 >= 0) ? i - 1: nx - 1;
    int jpp = (j + 1 < ny) ? j + 1 : 0;
    int jnn = (j - 1 >= 0) ? j - 1: ny - 1;
    signed char interactions_sum;

    // Interaction terms
    if (is_black){
        if (is_Z){
             interactions_sum = same_lattice_op[i*ny + j] + same_lattice_op[i*ny + jpp]           // 2-body contribution
                              + same_lattice_op[ipp*ny + j] + same_lattice_op[ipp*ny + jpp]       // 2-body contribution
                              +(same_lattice_op[i*ny + jpp]*op_lattice_op[i*ny + j]  
                              + same_lattice_op[ipp*ny + jpp]*op_lattice_op[ipp*ny + j])*op_lattice_same[i*ny + jpp]   // 4-body contribution
                              +(same_lattice_op[ipp*ny + j]*op_lattice_op[ipp*ny + j]
                              + same_lattice_op[i*ny + j]*op_lattice_op[i*ny + j])*op_lattice_same[i*ny + j];          // 4-body contribution
                                        
        }else{
             interactions_sum = same_lattice_op[i*ny + j] + same_lattice_op[i*ny + jnn]           // 2-body contribution
                              + same_lattice_op[ipp*ny + jnn] + same_lattice_op[ipp*ny + jpp]     // 2-body contribution
                              +(same_lattice_op[i*ny + j]*op_lattice_op[i*ny + j] 
                              + same_lattice_op[ipp*ny + jpp]*op_lattice_op[ipp*ny + j])*op_lattice_same[i*ny + j]     // 4-body contribution
                              +(same_lattice_op[i*ny + jnn]*op_lattice_op[i*ny + j] 
                              + same_lattice_op[ipp*ny + jnn]*op_lattice_op[ipp*ny + j])*op_lattice_same[i*ny + jnn];  // 4-body contribution
        }
    }else{
        if (is_Z){
             interactions_sum = same_lattice_op[i*ny + j] + same_lattice_op[inn*ny + j]           // 2-body contribution
                              + same_lattice_op[inn*ny + jnn] + same_lattice_op[i*ny + jnn]       // 2-body contribution
                              +(same_lattice_op[i*ny + j]*op_lattice_op[i*ny + j] 
                              + same_lattice_op[inn*ny + j]*op_lattice_op[inn*ny + j])*op_lattice_same[i*ny + j]       // 4-body contribution
                              +(same_lattice_op[inn*ny + jnn]*op_lattice_op[inn*ny + j] 
                              + same_lattice_op[i*ny + jnn]*op_lattice_op[i*ny + j])*op_lattice_same[i*ny + jnn];      // 4-body contribution
        }else{
             interactions_sum = same_lattice_op[i*ny + j] + same_lattice_op[inn*ny + j]           // 2-body contribution
                              + same_lattice_op[inn*ny + jpp] + same_lattice_op[i*ny + jpp]       // 2-body contribution
                              +(same_lattice_op[i*ny + j]*op_lattice_op[i*ny + j] 
                              + same_lattice_op[inn*ny + j]*op_lattice_op[inn*ny + j])*op_lattice_same[i*ny + j]       // 4-body contribution
                              +(same_lattice_op[inn*ny + jpp]*op_lattice_op[inn*ny + j] 
                              + same_lattice_op[i*ny + jpp]*op_lattice_op[i*ny + j])*op_lattice_same[i*ny + jpp];      // 4-body contribution
        }
    }

    // Determine whether to flip spin
    signed char lij = lattice[i * ny + j];
    float acceptance_ratio = exp(-2.0f * inv_temp * interactions_sum * lij);
    if (randvals[i*ny + j] < acceptance_ratio) {
        lattice[i * ny + j] = -lij;
    }
}

// Update function that calls the update lattice kernel from the host
void update(signed char *Z_lattice_w, signed char *Z_lattice_b, signed char *X_lattice_w, signed char *X_lattice_b,
             float* randvals, curandGenerator_t rng, 
             float inv_temp,
             long long nx,
             long long ny){
            
    // Cuda launch configuration
    int blocks = (nx * ny/4 + THREADS - 1) / THREADS;

    // Update white Z lattice
    curandGenerateUniform(rng, randvals, nx*ny/4);
    update_lattice<true, false><<<blocks, THREADS>>>(Z_lattice_w, Z_lattice_b, X_lattice_w, X_lattice_b, randvals, inv_temp, nx/2, ny/2);

    // Update white X lattice
    curandGenerateUniform(rng, randvals, nx*ny/4);
    update_lattice<false, false><<<blocks, THREADS>>>(X_lattice_w, X_lattice_b, Z_lattice_w, Z_lattice_b, randvals, inv_temp, nx/2, ny/2);

    // Update black Z lattice
    curandGenerateUniform(rng, randvals, nx*ny/4);
    update_lattice<true, true><<<blocks, THREADS>>>(Z_lattice_b, Z_lattice_w, X_lattice_w, X_lattice_b, randvals, inv_temp, nx/2, ny/2);
    
    // Update black X lattice
    curandGenerateUniform(rng, randvals, nx*ny/4);
    update_lattice<false, true><<<blocks, THREADS>>>(X_lattice_b, X_lattice_w, Z_lattice_w, Z_lattice_b, randvals, inv_temp, nx/2, ny/2);

    
}


// Write lattice to configuration file
void write_lattice(signed char* X_lattice_w, signed char* X_lattice_b, signed char* Z_lattice_w, signed char* Z_lattice_b,
                    std::string filename, long long nx, long long ny){
    printf("Writing lattice to %s ...\n", filename.c_str());
    signed char *lattice_h, *X_lattice_w_h, *X_lattice_b_h, *Z_lattice_w_h, *Z_lattice_b_h;

    // Allocate memory on the host
    lattice_h = (signed char*)malloc(nx*ny*sizeof(*lattice_h));
    X_lattice_w_h = (signed char*)malloc(nx*ny/4*sizeof(*X_lattice_w_h));
    X_lattice_b_h = (signed char*)malloc(nx*ny/4*sizeof(*X_lattice_b_h));
    Z_lattice_w_h = (signed char*)malloc(nx*ny/4*sizeof(*Z_lattice_w_h));
    Z_lattice_b_h = (signed char*)malloc(nx*ny/4*sizeof(*Z_lattice_b_h));

    // Copy the lattices from device to host
    cudaMemcpy(X_lattice_w_h, X_lattice_w, nx*ny/4*sizeof(*X_lattice_w), cudaMemcpyDeviceToHost);
    cudaMemcpy(X_lattice_b_h, X_lattice_b, nx*ny/4*sizeof(*X_lattice_b), cudaMemcpyDeviceToHost);
    cudaMemcpy(Z_lattice_w_h, Z_lattice_w, nx*ny/4*sizeof(*Z_lattice_w), cudaMemcpyDeviceToHost);
    cudaMemcpy(Z_lattice_b_h, Z_lattice_b, nx*ny/4*sizeof(*Z_lattice_b), cudaMemcpyDeviceToHost);

    // Fill host lattice
    for (int i = 0; i < nx/2; i++){
        for (int j = 0; j < ny/2; j++){
            lattice_h[2*i*ny + 2*j] = Z_lattice_w_h[i*ny/2 + j];
            lattice_h[2*i*ny + 2*j+1] = X_lattice_w_h[i*ny/2 + j];
            lattice_h[(2*i+1)*ny + 2*j] = X_lattice_b_h[i*ny/2 + j];
            lattice_h[(2*i+1)*ny + 2*j+1] = Z_lattice_b_h[i*ny/2 + j];
        }
    }
    

    std::ofstream f;
    f.open(filename);
    if (f.is_open()){
        for (int i = 0; i < nx; i++){
            for (int j = 0; j < ny; j++){
                f << (int)lattice_h[i*ny + j] << " ";
            }
            f << std::endl;
        }
    }
    f.close();

    free(lattice_h);
    free(X_lattice_b_h);
    free(X_lattice_w_h);
    free(Z_lattice_b_h);
    free(Z_lattice_w_h);
    cudaFree(X_lattice_w);
    cudaFree(X_lattice_b);
    cudaFree(Z_lattice_b);
    cudaFree(Z_lattice_w);


}

int main (int argc, char **argv){
    long nx = 200;
    long ny = 200;
    int nwarmup = 500;
    float alpha = 0.1f;
    float inv_temp = 1.0f / (alpha*TCRIT);
    unsigned long long seed = 1234ULL;

    int blocks = (nx*ny/4 + THREADS -1)/THREADS;
    

    // Setup cuRAND generator
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(rng,seed);
    float *randvals;

    // Allocate memory on device
    cudaMalloc(&randvals, nx*ny/4*sizeof(*randvals));

    // Setup X,Z black and white sublattices
    signed char *X_lattice_w, *X_lattice_b, *Z_lattice_b, *Z_lattice_w;
    cudaMalloc(&X_lattice_w, nx*ny/4*sizeof(*X_lattice_w));
    cudaMalloc(&X_lattice_b, nx*ny/4*sizeof(*X_lattice_b));
    cudaMalloc(&Z_lattice_w, nx*ny/4*sizeof(*Z_lattice_w));
    cudaMalloc(&Z_lattice_b, nx*ny/4*sizeof(*Z_lattice_b));

    // Intitialize all sublattices on the device
    curandGenerateUniform(rng, randvals, nx*ny/4);
    init_spins<<< blocks, THREADS>>>(X_lattice_b, randvals, nx/2, ny/2);

    curandGenerateUniform(rng, randvals, nx*ny/4);
    init_spins<<< blocks, THREADS>>>(X_lattice_w, randvals, nx/2, ny/2);

    curandGenerateUniform(rng, randvals, nx*ny/4);
    init_spins<<< blocks, THREADS>>>(Z_lattice_b, randvals, nx/2, ny/2);

    curandGenerateUniform(rng, randvals, nx*ny/4);
    init_spins<<< blocks, THREADS>>>(Z_lattice_w, randvals, nx/2, ny/2);

    // Synchronize devices
    cudaDeviceSynchronize();


    auto t0 = std::chrono::high_resolution_clock::now();
    // Warmup iterations
    printf("Starting warmup...\n");
    for (int i = 0; i < nwarmup; i++) {
        update(Z_lattice_w, Z_lattice_b, X_lattice_w, X_lattice_b, randvals, rng, inv_temp, nx, ny);
    }

    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    printf("\telapsed time: %f sec\n", duration * 1e-6);

    // Writing sublattices (Sanity check)
    write_sublattice(X_lattice_b, "X_lattice_b.txt",nx, ny);

    write_sublattice(X_lattice_w, "X_lattice_w.txt",nx, ny);

    write_sublattice(Z_lattice_w, "Z_lattice_w.txt",nx, ny);

    write_sublattice(Z_lattice_b, "Z_lattice_b.txt",nx, ny);


    // Writing full lattice (Sanity check)
    write_lattice(X_lattice_w, X_lattice_b, Z_lattice_w, Z_lattice_b, "lattice.txt", nx, ny);

    return 0;
}
