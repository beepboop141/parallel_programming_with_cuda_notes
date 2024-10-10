#include <stdio.h> 
#include <iostream>
#include <stdlib.h>


__global__ void hello_world() {
    printf("Hello World from the GPU!\n");
    // try printf("Hello from thread %d\n", threadIdx.x); after you are done
}

int main() {
    printf("Hello World from the CPU!\n"); 

    //Launch the kernel with 1 block and 10 threads
    hello_world<<<1, 10>>>(); 

    //Wait for the GPU to finish before continuing on the CPU
    cudaDeviceSynchronize(); 
    return 0; 
}