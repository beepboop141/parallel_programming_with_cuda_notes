#include <iostream>
#include <cuda_runtime.h>

// Define the kernel for vector addition
__global__ void vec_add(float* A, float* B, float* C, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int inputLength = 1024 * 1024;  // Length of the vectors
    int streamCount = 4;            // Number of CUDA streams
    int chunkSize = inputLength / streamCount;  // Size of each chunk
    
    float *hostInput1, *hostInput2, *hostOutput;
    float *deviceInput1[streamCount], *deviceInput2[streamCount], *deviceOutput[streamCount];
    cudaStream_t streams[streamCount];

    // Allocate pinned memory on the host
    cudaHostAlloc((void**)&hostInput1, inputLength * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&hostInput2, inputLength * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&hostOutput, inputLength * sizeof(float), cudaHostAllocDefault);

    // Initialize input vectors with random values
    for (int i = 0; i < inputLength; ++i) {
        hostInput1[i] = static_cast<float>(rand()) / RAND_MAX;
        hostInput2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on the device and create streams
    for (int i = 0; i < streamCount; ++i) {
        cudaMalloc((void**)&deviceInput1[i], chunkSize * sizeof(float));
        cudaMalloc((void**)&deviceInput2[i], chunkSize * sizeof(float));
        cudaMalloc((void**)&deviceOutput[i], chunkSize * sizeof(float));
        cudaStreamCreate(&streams[i]);  // Create CUDA streams
    }

    // Set up execution configuration
    dim3 blockDim(256);
    dim3 gridDim((chunkSize + blockDim.x - 1) / blockDim.x);

    // Process each chunk in a separate stream
    for (int i = 0; i < streamCount; ++i) {
        // Asynchronously copy data from host to device for each stream
        cudaMemcpyAsync(deviceInput1[i], hostInput1 + i * chunkSize, chunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(deviceInput2[i], hostInput2 + i * chunkSize, chunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[i]);

        // Launch kernel in each stream
        vec_add<<<gridDim, blockDim, 0, streams[i]>>>(deviceInput1[i], deviceInput2[i], deviceOutput[i], chunkSize);

        // Asynchronously copy results back from device to host for each stream
        cudaMemcpyAsync(hostOutput + i * chunkSize, deviceOutput[i], chunkSize * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize all streams to make sure all operations are completed
    for (int i = 0; i < streamCount; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // Validate the result
    bool resultCorrect = true;
    for (int i = 0; i < inputLength; ++i) {
        if (fabs(hostOutput[i] - (hostInput1[i] + hostInput2[i])) > 1e-5) {
            resultCorrect = false;
            break;
        }
    }

    if (resultCorrect) {
        std::cout << "Results are correct!" << std::endl;
    } else {
        std::cout << "Results are incorrect!" << std::endl;
    }

    // Free memory
    for (int i = 0; i < streamCount; ++i) {
        cudaFree(deviceInput1[i]);
        cudaFree(deviceInput2[i]);
        cudaFree(deviceOutput[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFreeHost(hostInput1);
    cudaFreeHost(hostInput2);
    cudaFreeHost(hostOutput);

    return 0;
}
