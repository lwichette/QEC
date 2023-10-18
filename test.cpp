#include <iostream>
#include <getopt.h>

using namespace std;

int main(int argc, char* argv[]){

  long long nx, ny;
  int nwarmup = 100;
  int niters = 1000;

  while (1) {
    static struct option long_options[] = {
      {     "lattice-n", required_argument, 0, 'x'},
      {     "lattice-m", required_argument, 0, 'y'},
      {       "nwarmup", required_argument, 0, 'w'},
      {        "niters", required_argument, 0, 'n'},
    };
    
    int option_index = 0;
    int ch = getopt_long(argc, argv, "x:y:w:n", long_options, &option_index);
    if (ch == -1) break;

    switch(ch) {
      case 0:
        break;
      case 'x':
        nx = atoll(optarg); break;
      case 'y':
        ny = atoll(optarg); break;
      case 'w':
        nwarmup = atoi(optarg); break;
      case 'n':
        niters = atoi(optarg); break;
      case '?':
        exit(EXIT_FAILURE);
      default:
        fprintf(stderr, "unknown option: %c\n", ch);
        exit(EXIT_FAILURE);
    }
  }

  printf("%lli \n", nx);
}