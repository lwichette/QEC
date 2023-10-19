#include <boost/program_options.hpp>
#include <iostream>
#include <vector>

using namespace std;

namespace po = boost::program_options;

int main(int argc, const char *argv[]){
  
  int nx;
  vector<int> L;

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("nx", po::value<int>(), "set lattice size")
    ("L", po::value<std::vector<int>>()->multitoken(), "Lattice")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("nx")) {
    nx = vm["nx"].as<int>();
  }
  
  if (vm.count("L")){
    L = vm["L"].as<vector<int>>();
  }

  cout << nx << endl;

  for (int l=0; l<L.size(); l++){
    cout << L[l] << endl;
  }
}