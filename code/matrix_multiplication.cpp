#include <iostream>
#include <vector>
#include <cstdlib> 
#include <ctime>   
#include <chrono>  

// Function to multiply two matrices
void matrix_multiply(const std::vector<std::vector<float>>& input1, 
                     const std::vector<std::vector<float>>& input2, 
                     std::vector<std::vector<float>>& output) {
    int rows1 = input1.size();
    int cols1 = input1[0].size();
    int cols2 = input2[0].size();

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            output[i][j] = 0; // Initialize the output element
            for (int k = 0; k < cols1; ++k) {
                output[i][j] += input1[i][k] * input2[k][j]; // Dot product
            }
        }
    }
}

int main() {
    // Matrix dimensions
    int input1_rows = 1024; // Example dimensions
    int input1_cols = 512;  
    int input2_cols = 256;  

    // Initialize matrices
    std::vector<std::vector<float>> hostInput1(input1_rows, std::vector<float>(input1_cols));
    std::vector<std::vector<float>> hostInput2(input1_cols, std::vector<float>(input2_cols));
    std::vector<std::vector<float>> hostOutput(input1_rows, std::vector<float>(input2_cols));

    // Initialize random seed
    srand(static_cast<unsigned int>(time(0)));

    // Fill the input matrices with random values
    for (int i = 0; i < input1_rows; i++) {
        for (int j = 0; j < input1_cols; j++) {
            hostInput1[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    for (int i = 0; i < input1_cols; i++) {
        for (int j = 0; j < input2_cols; j++) {
            hostInput2[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Measure the start time
    auto start = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication
    matrix_multiply(hostInput1, hostInput2, hostOutput);

    // Measure the end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Time taken for matrix multiplication: " << elapsed.count() << " ms" << std::endl;

    // Optionally: Print the output (for debugging purposes)
    /*
    for (int i = 0; i < input1_rows; ++i) {
        for (int j = 0; j < input2_cols; ++j) {
            std::cout << hostOutput[i][j] << " ";
        }
        std::cout << std::endl;
    }
    */

    return 0;
}
