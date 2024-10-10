#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Size of array (multiple of 256 for simplicity)
#define THREADS_PER_BLOCK 256  // Number of threads per block

__global__ void reductionKernel(int *input, int *output, int size) {
    __shared__ int sharedMem[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int globalIndex = blockIdx.x * blockDim.x + tid;
    
    // Load data into shared memory (assuming all globalIndex < size)
    if (globalIndex < size)
        sharedMem[tid] = input[globalIndex];
    else
        sharedMem[tid] = 0;

    // Ensure all threads have loaded data into shared memory
    __syncthreads();

    // Perform reduction within each block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();  // Make sure all threads have completed this step
    }

    // Write the result of this block to global memory using atomicAdd
    if (tid == 0) {
        atomicAdd(output, sharedMem[0]);  // Atomic operation to avoid race condition
    }
}

int main() {
    // Size of array
    int size = N * sizeof(int);
    
    // Allocate host memory
    int h_input[N], h_output = 0;
    
    // Initialize input array
    for (int i = 0; i < N; i++) {
        h_input[i] = 1;  // For simplicity, let's set every element to 1
    }

    // Allocate device memory
    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &h_output, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with (N/THREADS_PER_BLOCK) blocks and THREADS_PER_BLOCK threads
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    reductionKernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);

    // Copy result from device to host
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Output the result
    printf("Sum of array elements: %d\n", h_output);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
