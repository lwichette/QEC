#include <chrono>
#include <iostream>
#include <thread>
#include <utility>
 
using namespace std::literals;

void multiply(int n){
    printf("%u", 2*n);
}

int main(){
    for (int i=0; i<10;i++){
        std::jthread multiply(i);
    }
}