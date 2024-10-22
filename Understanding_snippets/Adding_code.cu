#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <getopt.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_kernel.h>
#include "cudamacro.h"
#include "utils.h"
#include <iostream>
#include <sys/stat.h>
#include <thrust/complex.h>
#include <fstream>
#include <iomanip>


using namespace std;

#define THREADS 128

#define BLOCK_X (2)
#define BLOCK_Y (8)

#define BMULT_X (1)
#define BMULT_Y (1)

#define MAX_GPU (256)


// forceinline is for the compiler to replace the function by its expression when called, uint2 is a CUDA data structure that has some memory access benefits
__device__ __forceinline__ uint2 __mymake_int2(const unsigned int x, const unsigned int y){
    return make_uint2(x,y);
}

/* The kernels are separated into three parts, preparation, manipulation, and finally storage. This is to minimize global memory access */


//Lattice initialization kernel
template<int BDIM_X, int BDIM_Y, int LOOP_X, int LOOP_Y, int BITXSP, int COLOR, typename INT_T, typename INT2_T>
__global__ void latticeInit_k(const int devid, const long long seed,const int it, const long long, begY, const long long dimX, INT2_T *__restrict__ vDst, bool up){

    // Identify i and j indices:
    __i = blockDim.y*BDIM_Y*LOOP_Y + threadIdx.y;
    __j = blockDim.x*BDIM_X*LOOP_X + threadIdx.x;

    // Calculate the number of spins per word
    int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

    //threadidx
    const long long tid = ((devid*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x)*BDIM_X*BDIM_Y +
    threadIdx.y*BDIM_X + threadIdx.x; 

    // Random number generator
    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid,static_cast<long long>(2*SPIN_X_WORD)*LOOP_X*LOOP_Y*(2*it+COLOR), &st);

    /*temporary 2D array of type unsigned long long and size (1x2), this is also called the type INT2_T for our code. In this array, we will
    all the initial vlaues that this thread is going to be initialize. In our picture, each thread goes over LOOP_X*LOOP_Y*/
    INT2_T __tmp[LOOP_Y][LOOP_X];


    // Initializing temporary memory all up
    if (up){
        #pragma unroll
        for (int i = 0; i < LOOP_Y; i++){
            #pragma unroll
            for (int j = 0; j < LOOP_X; j++){
                __tmp[i][j] = __mymake_int2(INT_T(0x1111111111111111), INT_T(0x1111111111111111));
            }
        }
    }
    
    // Initialize temporary memory to 0
    else{
        #pragma unroll
        for (int i = 0; i < LOOP_Y; i++){
            #pragma unroll
            for (int j = 0; j < LOOP_X; j++){
                __tmp[i][j] = __mymake_int2(INT_T(0), INT_T(0));
            }
        }

        #pragma unroll
        for (int i = 0; i< LOOP_Y; i++){
            #pragma unroll
            for (j = 0; j < LOOP_X; j++){
                #pragma unroll
                for (k = 0; k < 8*sizeof(INT_T); k += BIT_X_SPIN){
                    if (curand_init(&st) < 0.5f){
                        __tmp[i][j].x |= INT_T(1) < k;
                    }

                    if (curand_init(&st) < 0.5f){
                        __tmp[i][j].y |= INT_T(1) < k;
                    }
                }
                
            }
        }

    }

     // write to global memory

     #pragma unroll
     for (int i = 0; i < LOOP_Y; i++){
         #pragma unroll
         for (j = 0; j < LOOP_X; j++){
             vDst[(begY + __i + i*BDIM_Y)*dimX + __j + j*BDIM_X] = __tmp[i][j];
         }
     }
     return;
    
}

// Initializing the interaction terms for the Random Bond Ising Model from the perspective of the black lattice
template<int BDIM_X, int BDIM_Y, int LOOP_X, int LOOP_Y, int BITXSP, int COLOR, typename INT_T, typename INT2_T>
__global__ void hamiltInitB_k(const int devid, const float tgtProb, const long long seed, const long long begY, const long long dimX, INT2_T *__restrict__ hamB){

    // Linearized indices
    const int __i = threadIdx.y + blockDim.y*BDIM_Y*LOOP_Y;
    const int __j = threadIdx.x + blockDim.x*BDIM_X*LOOP_X;

    // Spins per word
    const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

    const long long tid = ((devid*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x)*BDIM_X*BDIM_Y + threadIdx.y*BDIM_X + threadIdx.x;

    //RNG
    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);


    // Initialize memory to 0s
    INT2_T __tmp[LOOP_Y][LOOP_X];
    #pragma unroll 
    for (int i = 0; i < LOOP_Y; i++){
        #pragma unroll
        for (int j = 0; j < LOOP_X; j++){
            __tmp[i][j] = __mymake_int2(INT_T(0), INT_T(0));
        }
    }

    // Fill interaction terms
    #pragma unroll
    for (int i = 0; i < LOOP_Y; i++){
        #pragma unroll
        for (int j = 0; j < LOOP_X; j++){
            #pragma unroll
            for (int k = 0; k < 8*sizeof(INT_T); k +=BITXSP){
                #pragma unroll
                for (int l = 0; l < BITXSP; l++){
                    if (curand_uniform(&st) < tgtProb){
                        __tmp[i][j].x |= INT_T(1) << k + l;
                    }

                    if (curand_uniform(&st) < tgtProb){
                        __tmp[i][j].y |= INT_T(1) << k + l;
                    }
                }
            }
        }
    }

    // Write to global memory
    #pragma unroll
    for (int i = 0; i < LOOP_Y; i++){
        #pragma unroll
        for (int j = 0; j < LOOP_X; j++){
        hamB[(begY + __i + i*BDIM_Y)*dimX + j*BDIM_X + __j] = __tmp[i][j];    
        }
    }
    return;    
}

// Loading the tile around the two spin words that are being updated
template<int BDIM_X, int BDIM_Y, int TILE_X, int TILE_Y, int FRAME_X, int FRAME_Y, typename INT2_T>
__device__ void loadTile(const int slX, const int slY, const long long begY, const long long dimX, const INT2_T *__restrict__ v, INT2_T tile[][TILE_X+2*FRAME_X]){

    const int blkx = blockDim.x;
    const int blky = blockDim.y;

    const tidx = threadIdx.x;
    const tidy = threadIdx.y;

    const startX = blkx*TILE_X;
    const startY = begY + blky*TILE_Y;

    #pragma unroll
    for (int j = 0; j < TILE_Y; j+= BDIM_Y){
        
        int yoff = startY + j + tidy;

        #pragma unroll 
        for (int i = 0; i < TILE_X; i+= BDIM_X){
            int xoff = startX + i + tidx;

            tile[FRAME_Y + j + tidy][FRAME_X + i + tidx] = v[yoff*dimX + xoff];

        }
    }

    if (tidy == 0){
        int yoff = (startY % slY ) == 0 ? startY + slY -1 : startY - 1;

        #pragma unroll
        for (int i = 0; i < TILE_X; i += BDIM_X){
            const int xoff = startX + i + tidx;
            tile[0][FRAME_X + i + tidx] = v[yoff*dimX + xoff]; // up neighbours
        }

        yoff = ((startY+TILE_Y % slY)  % slY) == 0 ? startY+TILE_Y - slY : startY+TILE_Y;

        #pragma unroll
        for(int i = 0; i < TILE_X; j += BDIM_X){
            const int xoff = startX + i + tidx;
            tile[FRAME_Y + TILE_Y][FRAME_X + i + tidx] = v[yoff*dimX + xoff]; // Down neighbours
        }

        // come back to this later with a specific lattice size
    }
}


