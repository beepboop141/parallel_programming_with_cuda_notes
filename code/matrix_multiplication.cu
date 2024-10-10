#include <iostream>
#include <cuda_runtime.h>

__global__ void matrix_multiplication(float *input1, float *input2, float *output, int input1_rows, int input1_cols, int input2_cols) {
    // Calculate the global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check that the thread is within the bounds of the output matrix 
    if (row < input1_rows && col < input2_cols) {
        float sum = 0.0f;

        // Compute the dot product for the row of input1 and the column of input2
        for (int i = 0; i < input1_cols; ++i) {
            sum += input1[row * input1_cols + i] * input2[i * input2_cols + col]; 
        }

        // Store the result in the output matrix
        output[row * input2_cols + col] = sum; 
    }
}

int main() {
    // Matrix dimensions
    int input1_rows = 1024; // Example dimensions
    int input1_cols = 512;  
    int input2_cols = 256;  

    // Allocate host memory
    float *hostInput1 = new float[input1_rows * input1_cols];
    float *hostInput2 = new float[input1_cols * input2_cols];
    float *hostOutput = new float[input1_rows * input2_cols];

    // Initialize input matrices with random values (for testing)
    for (int i = 0; i < input1_rows * input1_cols; i++) {
        hostInput1[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < input1_cols * input2_cols; i++) {
        hostInput2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *deviceInput1, *deviceInput2, *deviceOutput;
    cudaMalloc((void **)&deviceInput1, input1_rows * input1_cols * sizeof(float));
    cudaMalloc((void **)&deviceInput2, input1_cols * input2_cols * sizeof(float));
    cudaMalloc((void **)&deviceOutput, input1_rows * input2_cols * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(deviceInput1, hostInput1, input1_rows * input1_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, input1_cols * input2_cols * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16); // Example: 16x16 threads per block
    dim3 gridDim((input2_cols + blockDim.x - 1) / blockDim.x, (input1_rows + blockDim.y - 1) / blockDim.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the GPU Kernel
    matrix_multiplication<<<gridDim, blockDim>>>(deviceInput1, deviceInput2, deviceOutput, input1_rows, input1_cols, input2_cols);
    cudaDeviceSynchronize(); // Ensure the kernel is complete

    // Record the end event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // Wait for the stop event to complete

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken by GPU kernel: " << milliseconds << " ms" << std::endl;

    // Copy the output matrix from device to host
    cudaMemcpy(hostOutput, deviceOutput, input1_rows * input2_cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Optionally: Print the output (for debugging purposes)
    // Note: Printing large matrices can be overwhelming. Comment this out if not needed.
    /*
    for (int i = 0; i < input1_rows; ++i) {
        for (int j = 0; j < input2_cols; ++j) {
            std::cout << hostOutput[i * input2_cols + j] << " ";
        }
        std::cout << std::endl;
    }
    */

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device memory
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    // Free host memory
    delete[] hostInput1;
    delete[] hostInput2;
    delete[] hostOutput;

    return 0;
}
