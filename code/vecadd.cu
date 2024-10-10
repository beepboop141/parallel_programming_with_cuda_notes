#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Define the kernel for vector addition
__global__ void vec_add(float* A, float* B, float* C, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int inputLength = 1024;  
    float *hostInput1, *hostInput2, *hostOutput, *expectedOutput;
    float *deviceInput1, *deviceInput2, *deviceOutput;

    // Allocate host memory
    hostInput1 = new float[inputLength];
    hostInput2 = new float[inputLength];
    hostOutput = new float[inputLength];
    expectedOutput = new float[inputLength];

    // Initialize the input data 
    for (int i = 0; i < inputLength; ++i) {
        hostInput1[i] = static_cast<float>(rand()) / RAND_MAX;  // Random values between 0 and 1
        hostInput2[i] = static_cast<float>(rand()) / RAND_MAX;  // Random values between 0 and 1
        expectedOutput[i] = hostInput1[i] + hostInput2[i];      // Expected result for verification
    }

    // Allocate device memory
    cudaMalloc((void **)&deviceInput1, inputLength * sizeof(float));
    cudaMalloc((void **)&deviceInput2, inputLength * sizeof(float));
    cudaMalloc((void **)&deviceOutput, inputLength * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);

    // Set up execution configuration
    dim3 blockDim(32);
    dim3 gridDim((inputLength + blockDim.x - 1) / blockDim.x);

    // Launch the kernel
    vec_add<<<gridDim, blockDim>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results with expected output
    bool resultCorrect = true;
    for (int i = 0; i < inputLength; ++i) {
        if (std::abs(hostOutput[i] - expectedOutput[i]) > 1e-5) {
            resultCorrect = false;
            break;
        }
    }

    if (resultCorrect) {
        std::cout << "Results match the expected output." << std::endl;
    } else {
        std::cout << "Results do not match the expected output." << std::endl;
    }

    // Free GPU memory
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    // Free host memory
    delete[] hostInput1;
    delete[] hostInput2;
    delete[] hostOutput;
    delete[] expectedOutput;

    return 0;
}
