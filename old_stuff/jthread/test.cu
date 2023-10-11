#include <chrono>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>
#include <future>

using namespace std::literals;


void add(int x, int y, std::promise<int> && p){
  //printf("Added numbers: %u \n", x+y);
  p.set_value(x+y);
}

int main(int argc, char **argv) {

    std::vector<std::jthread> v;
    std::vector<std::promise<int>> prom(3);
    std::vector<std::future<int>> future(3);

    for(int i=0; i<3; i++){
      future[i] = prom[i].get_future();
    }

    for(int i=0; i<3; i++){
        v.emplace_back(add, i, 2*i, std::move(prom[i]));
    }

    for(int i=0; i<3; i++){
      printf("%u \n", future[i].get());
    }
}
