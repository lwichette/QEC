#include <chrono>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <math.h> 

#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>

#include <cub/cub.cuh>
#define CUB_CHUNK_SIZE ((1ll<<31) - (1ll<<28))

#define THREADS 128
//2.26918531421f


// Initialize lattice spins
__global__ void init_spins(signed char* lattice,
    const float* __restrict__ randvals,
    const long long nx,
    const long long ny) {

const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
if (tid >= nx * ny) return;

float randval = randvals[tid];
signed char val = (randval < 0.5f) ? -1 : 1;
lattice[tid] = val;
}

// Initialize random bond signs
__global__ void init_randombond(signed char* interactions,
                                const float* __restrict__ interaction_randvals,
                                const long long nx,
                                const long long ny,
                                const float p){
  const long long tid = static_cast<long long>(threadIdx.x + blockIdx.x * blockDim.x);
  if (tid >= 2*nx*ny) return;
  
  float bondrandval = interaction_randvals[tid];
  signed char bondval = (bondrandval<p)? -1 : 1;
  interactions[tid] = bondval;                                  
}

// Device kernel for updating the spins with a RBIM Hamiltonian
template<bool is_black>
__global__ void update_lattice(signed char* lattice, signed char* __restrict__ op_lattice, const float* __restrict__ randvals, signed char* interactions,
                               const float inv_temp,
                               const long long nx,
                               const long long ny,
                               const float coupling_constant) {

  const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;
  
  const int i = tid/ny;
  const int j = tid%ny;

  if (i>=nx || j >= ny) return;

  // Set up periodic boundary conditions
  int ipp = (i + 1 < nx) ? i + 1 : 0;
  int inn = (i - 1 >= 0) ? i - 1: nx - 1;
  int jpp = (j + 1 < ny) ? j + 1 : 0;
  int jnn = (j - 1 >= 0) ? j - 1: ny - 1;

int joff;
int jcouplingoff;
int icouplingpp;
int icouplingnn;

if (is_black) {
    icouplingpp = 2*(nx-1)*ny + 2*(ny*(i+1) + j) + (i+1)%2;
    icouplingnn = 2*(nx-1)*ny + 2*(ny*(inn+1) + j) + (i+1)%2;
    joff = (i % 2) ? jnn : jpp;

    if (i % 2) {
        jcouplingoff = 2 * (i * ny + joff) + 1;
    } else {
        if (j + 1 >= ny) {
            jcouplingoff = 2 * (i * ny + j + 1) - 1;
        } else {
            jcouplingoff = 2 * (i * ny + joff) - 1;
        }
    }
} else {
    icouplingpp = 2*(nx-1)*ny + 2*(ny*(i+1) + j) + i%2;
    icouplingnn = 2*(nx-1)*ny + 2*(ny*(inn+1) + j) + i%2;
    joff = (i % 2) ? jpp : jnn;

    if (i % 2) {
        if (j+1 >= ny) {
            jcouplingoff = 2 * (i * ny + j + 1) - 1;
        } else {
            jcouplingoff = 2 * (i * ny + joff) - 1;
        }
    } else {
        jcouplingoff = 2 * (i * ny + joff) + 1;
    }
}

  // Compute sum of nearest neighbor spins times the coupling
  signed char nn_sum = op_lattice[inn * ny + j]*interactions[icouplingnn] + op_lattice[i * ny + j]*interactions[2*(i*ny + j)] 
                     + op_lattice[ipp * ny + j]*interactions[icouplingpp] + op_lattice[i * ny + joff]*interactions[jcouplingoff];

  // Compute sum of nearest neighbor spins
  //signed char nn_sum = op_lattice[inn * ny + j] + op_lattice[i * ny + j] + op_lattice[ipp * ny + j] + op_lattice[i * ny + joff];

  // Determine whether to flip spin
  signed char lij = lattice[i * ny + j];
  float acceptance_ratio = exp(-2 * coupling_constant * nn_sum * lij);
  if (randvals[i*ny + j] < acceptance_ratio) {
    lattice[i * ny + j] = -lij;
  }  
}


void update(signed char *lattice_b, signed char *lattice_w, float* randvals, curandGenerator_t rng, signed char* interactions, float inv_temp, long long nx, long long ny, float coupling_constant) {
 
    // Setup CUDA launch configuration
    int blocks = (nx * ny/2 + THREADS - 1) / THREADS;
  
    // Update black
    curandGenerateUniform(rng, randvals, nx*ny/2);
    update_lattice<true><<<blocks, THREADS>>>(lattice_b, lattice_w, randvals,interactions, inv_temp, nx, ny/2,coupling_constant);
  
    // Update white
    curandGenerateUniform(rng, randvals, nx*ny/2);
    update_lattice<false><<<blocks, THREADS>>>(lattice_w, lattice_b, randvals,interactions, inv_temp, nx, ny/2, coupling_constant);
}


 // Write lattice configuration to file
 void write_lattice(signed char *lattice_b, signed char *lattice_w, std::string filename, long long nx, long long ny) {
  printf("Writing lattice to %s...\n", filename.c_str());
  signed char *lattice_h, *lattice_b_h, *lattice_w_h;
  lattice_h = (signed char*) malloc(nx * ny * sizeof(*lattice_h));
  lattice_b_h = (signed char*) malloc(nx * ny/2 * sizeof(*lattice_b_h));
  lattice_w_h = (signed char*) malloc(nx * ny/2 * sizeof(*lattice_w_h));

  cudaMemcpy(lattice_b_h, lattice_b, nx * ny/2 * sizeof(*lattice_b), cudaMemcpyDeviceToHost);
  cudaMemcpy(lattice_w_h, lattice_w, nx * ny/2 * sizeof(*lattice_w), cudaMemcpyDeviceToHost);

  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny/2; j++) {
      if (i % 2) {
        lattice_h[i*ny + 2*j+1] = lattice_w_h[i*ny/2 + j];
        lattice_h[i*ny + 2*j] = lattice_b_h[i*ny/2 + j];
      } else {
        lattice_h[i*ny + 2*j] = lattice_w_h[i*ny/2 + j];
        lattice_h[i*ny + 2*j+1] = lattice_b_h[i*ny/2 + j];
      }
    }
  }

  std::ofstream f;
  f.open(filename);
  if (f.is_open()) {
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
         f << (int)lattice_h[i * ny + j] << " ";
      }
      f << std::endl;
    }
  }
  f.close();

  free(lattice_h);
  free(lattice_b_h);
  free(lattice_w_h);
}


// Write interaction bonds to file
void write_bonds(signed char* interactions, std::string filename, long long nx, long long ny){
    printf("Writing bonds to %s ...\n", filename.c_str());
    signed char *interactions_host;
    interactions_host = (signed char*)malloc(2*nx*ny*sizeof(*interactions_host));
    cudaMemcpy(interactions_host,interactions, 2*nx*ny*sizeof(*interactions), cudaMemcpyDeviceToHost);
        
      std::ofstream f;
      f.open(filename);
      if (f.is_open()) {
        for (int i = 0; i < 2*nx; i++) {
          for (int j = 0; j < ny; j++) {
             f << (int)interactions_host[i * ny + j] << " ";
          }
          f << std::endl;
        }
      }
      f.close();
      cudaFree(interactions);
      free(interactions_host);
}



static void usage(const char *pname) {

    const char *bname = rindex(pname, '/');
    if (!bname) {bname = pname;}
    else        {bname++;}
  
    fprintf(stdout,
            "Usage: %s [options]\n"
            "options:\n"
            "\t-x|--lattice-n <LATTICE_N>\n"
            "\t\tnumber of lattice rows\n"
            "\n"
            "\t-y|--lattice_m <LATTICE_M>\n"
            "\t\tnumber of lattice columns\n"
            "\n"
            "\t-w|--nwarmup <NWARMUP>\n"
            "\t\tnumber of warmup iterations\n"
            "\n"
            "\t-n|--niters <NITERS>\n"
            "\t\tnumber of trial iterations\n"
            "\n"
            "\t-a|--alpha <ALPHA>\n"
            "\t\tcoefficient of critical temperature\n"
            "\n"
            "\t-s|--seed <SEED>\n"
            "\t\tseed for random number generation\n"
            "\n"
            "\t-o|--write-lattice\n"
            "\t\twrite final lattice configuration to file\n\n",
            bname);
    exit(EXIT_SUCCESS);
  }


  int main(int argc, char **argv) {
    long nx = 12;
    long ny = 12;
    int niters = 1000;
    float alpha = 1.0f;
    int nwarmup = 100;
    float TCRIT = 8.0f;
    float inv_temp = 1.0f / (alpha*TCRIT);
    bool write = true;
    unsigned long long seed = 0ULL;
    const float p = 0.031091730001f;
    const float coupling_constant = 0.5*TCRIT*log((1-p)/p);
    // 0.5*log((1-p)/p);2.0f;
    //alpha = 0.5f

    printf("Hallo");
    
    while (1) {
        static struct option long_options[] = {
            {     "lattice-n", required_argument, 0, 'x'},
            {     "lattice-m", required_argument, 0, 'y'},
            {         "alpha", required_argument, 0, 'y'},
            {          "seed", required_argument, 0, 's'},
            {       "nwarmup", required_argument, 0, 'w'},
            {        "niters", required_argument, 0, 'n'},
            { "write-lattice",       no_argument, 0, 'o'},
            {          "help",       no_argument, 0, 'h'},
            {               0,                 0, 0,   0}
        };
    
        int option_index = 0;
        int ch = getopt_long(argc, argv, "x:y:a:s:w:n:oh", long_options, &option_index);
        if (ch == -1) break;
    
        switch(ch) {
          case 0:
            break;
          case 'x':
            nx = atoll(optarg); break;
          case 'y':
            ny = atoll(optarg); break;
          case 'a':
            alpha = atof(optarg); break;
          case 's':
            seed = atoll(optarg); break;
          case 'w':
            nwarmup = atoi(optarg); break;
          case 'n':
            niters = atoi(optarg); break;
          case 'o':
            write = true; break;
          case 'h':
            usage(argv[0]); break;
          case '?':
            exit(EXIT_FAILURE);
          default:
            fprintf(stderr, "unknown option: %c\n", ch);
            exit(EXIT_FAILURE);
        }
      }
    
      // Check arguments
      if (nx % 2 != 0 || ny % 2 != 0) {
        fprintf(stderr, "ERROR: Lattice dimensions must be even values.\n");
        exit(EXIT_FAILURE);
      }

    int blocks = (nx*ny*2 + THREADS -1)/THREADS;

    // Setup cuRAND generator
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(rng, seed);
    float *randvals;
    cudaMalloc(&randvals, nx * ny/2 * sizeof(*randvals));

    // Setup black and white lattice arrays on device
    signed char *lattice_b, *lattice_w;
    cudaMalloc(&lattice_b, nx * ny/2 * sizeof(*lattice_b));
    cudaMalloc(&lattice_w, nx * ny/2 * sizeof(*lattice_w));

    //Initialize the arrays for white and black lattice
    curandGenerateUniform(rng, randvals, nx*ny/2);
    init_spins<<<blocks, THREADS>>>(lattice_b, randvals, nx, ny/2);
    curandGenerateUniform(rng, randvals, nx*ny/2);
    init_spins<<<blocks, THREADS>>>(lattice_w, randvals, nx, ny/2);

    //Setup cuRAND generator for the random bond sign
    curandGenerator_t interaction_rng;
    curandCreateGenerator(&interaction_rng,CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(interaction_rng,seed);
    float *interaction_randvals;
    cudaMalloc(&interaction_randvals,nx*ny*2*sizeof(*interaction_randvals));

    //Setup interaction lattice on device
    signed char *interactions;
    cudaMalloc(&interactions, nx*ny*2*sizeof(interactions));

    curandGenerateUniform(interaction_rng,interaction_randvals,nx*ny*2);
    init_randombond<<<blocks, THREADS>>>(interactions, interaction_randvals,nx,ny,p);

    //Synchronize devices
    cudaDeviceSynchronize();
    
    // Warmup iterations
    printf("Starting warmup...\n");

    for (int i = 0; i < nwarmup; i++) {
      update(lattice_b, lattice_w, randvals, rng, interactions, inv_temp, nx, ny, coupling_constant);
    }

    //Synchronize devices
    cudaDeviceSynchronize();

    printf("Starting trial iterations...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < niters; i++) {
      update(lattice_b, lattice_w, randvals, rng, interactions, inv_temp, nx, ny,coupling_constant);
      if (i % 1000 == 0) printf("Completed %d/%d iterations...\n", i+1, niters);
    }
  
    cudaDeviceSynchronize();

    if (write) write_lattice(lattice_b, lattice_w, "final.txt", nx, ny);
    write_bonds(interactions, "final_bonds.txt" ,nx, ny);
    auto t1 = std::chrono::high_resolution_clock::now();
  
    double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    printf("REPORT:\n");
    printf("\tnGPUs: %d\n", 1);
    printf("\ttemperature: %f * %f\n", alpha, TCRIT);
    printf("\tseed: %llu\n", seed);
    printf("\twarmup iterations: %d\n", nwarmup);
    printf("\ttrial iterations: %d\n", niters);
    printf("\tlattice dimensions: %lld x %lld\n", nx, ny);
    printf("\telapsed time: %f sec\n", duration * 1e-6);
    printf("\tupdates per ns: %f\n", (double) (nx * ny) * niters / duration * 1e-3);
  
    // Reduce
    double* devsum;
    int nchunks = (nx * ny/2 + CUB_CHUNK_SIZE - 1)/ CUB_CHUNK_SIZE;
    
    cudaMalloc(&devsum, 2 * nchunks * sizeof(*devsum));
    size_t cub_workspace_bytes = 0;
    void* workspace = NULL;
    
    cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, lattice_b, devsum, CUB_CHUNK_SIZE);
    cudaMalloc(&workspace, cub_workspace_bytes);
    
    for (int i = 0; i < nchunks; i++) {
      
      cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, &lattice_b[i*CUB_CHUNK_SIZE], devsum + 2*i,
                             std::min((long long) CUB_CHUNK_SIZE, nx * ny/2 - i * CUB_CHUNK_SIZE));


      cub::DeviceReduce::Sum(workspace, cub_workspace_bytes, &lattice_w[i*CUB_CHUNK_SIZE], devsum + 2*i + 1,
                             std::min((long long) CUB_CHUNK_SIZE, nx * ny/2 - i * CUB_CHUNK_SIZE));
    }
  
    double* hostsum;
    hostsum = (double*)malloc(2 * nchunks * sizeof(*hostsum));
    cudaMemcpy(hostsum, devsum, 2 * nchunks * sizeof(*devsum), cudaMemcpyDeviceToHost);
    
    double fullsum = 0.0;
    
    for (int i = 0; i < 2 * nchunks; i++) {
      fullsum += hostsum[i];
    }
    
    std::cout << "\taverage magnetism (absolute): " << abs(fullsum / (nx * ny)) << std::endl;
  

  


    //cudaDeviceSynchronize();
    //write_lattice(lattice_b, lattice_w, "final.txt", nx, ny);
    //cudaDeviceSynchronize();
    //write_bonds(interactions, "final_bonds.txt", nx, ny);
    
    // Cleanup
    //cudaFree(randvals);
    //cudaFree(interaction_randvals);
    //cudaFree(interactions);
    //cudaFree(lattice_b);
    //cudaFree(lattice_w);
    //curandDestroyGenerator(interaction_rng);
    //curandDestroyGenerator(rng); 
    return 0;
}
