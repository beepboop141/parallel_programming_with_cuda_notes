#include <stdio.h>
#include <cuda.h>

#define BLOCK_SIZE 512
#define BQ_CAPACITY 2048 // Maximum number of elements that can be inserted into a block queue

// Global queuing kernel
__global__ void gpu_global_queuing_kernel(
    int *nodePtrs, int *nodeNeighbors, int *nodeVisited,
    int *currLevelNodes, int *nextLevelNodes,
    const unsigned int numCurrLevelNodes, int *numNextLevelNodes) {

  for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < numCurrLevelNodes; tid += gridDim.x * blockDim.x) {
    int node = currLevelNodes[tid]; 
    for (int neighbor = nodePtrs[node]; neighbor < nodePtrs[node+1]; ++neighbor) {
      int n = nodeNeighbors[neighbor];
      int visited = atomicExch(&(nodeVisited[n]), 1);  // Check and mark as visited atomically
      if (visited == 0) {
        int queue = atomicAdd(numNextLevelNodes, 1);   // Add to global queue atomically
        nextLevelNodes[queue] = n;
      }
    }
  }
}

// Block queuing kernel
__global__ void gpu_block_queuing_kernel(
    int *nodePtrs, int *nodeNeighbors, int *nodeVisited,
    int *currLevelNodes, int *nextLevelNodes,
    const unsigned int numCurrLevelNodes, int *numNextLevelNodes) {

   int tid = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ int block_queue[BQ_CAPACITY];
  __shared__ int block_queue_number, global_queue_i; 

  if (threadIdx.x == 0)
    block_queue_number = 0; 
  __syncthreads(); 

  for (tid = tid; tid < numCurrLevelNodes; tid += (gridDim.x * blockDim.x)) {
    int node = currLevelNodes[tid]; 
    for (int neighbor = nodePtrs[node]; neighbor < nodePtrs[node+1]; ++neighbor) {
      int n = nodeNeighbors[neighbor];
      int visited = atomicExch(&(nodeVisited[n]), 1);
      if (visited == 0) {
        int block_queue_id = atomicAdd(&block_queue_number, 1);
        if (block_queue_id < BQ_CAPACITY) {
          block_queue[block_queue_id] = n;
        } else {
          block_queue_number = BQ_CAPACITY;
          int queue = atomicAdd(numNextLevelNodes, 1);
          nextLevelNodes[queue] = n;
        }
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    global_queue_i = atomicAdd(numNextLevelNodes, block_queue_number);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < block_queue_number; i += blockDim.x) {
    nextLevelNodes[global_queue_i + i] = block_queue[i];
  }
}

// Host function for global queuing invocation
void gpu_global_queuing(int *nodePtrs, int *nodeNeighbors,
                        int *nodeVisited, int *currLevelNodes,
                        int *nextLevelNodes,
                        unsigned int numCurrLevelNodes,
                        int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(
      nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
      numCurrLevelNodes, numNextLevelNodes);
  cudaDeviceSynchronize(); // Ensure all threads finish execution
}

// Host function for block queuing invocation
void gpu_block_queuing(int *nodePtrs, int *nodeNeighbors, int *nodeVisited,
                       int *currLevelNodes, int *nextLevelNodes,
                       unsigned int numCurrLevelNodes,
                       int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(
      nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
      numCurrLevelNodes, numNextLevelNodes);
  cudaDeviceSynchronize(); // Ensure all threads finish execution
}

int main(int argc, char *argv[]) {
    // Example graph:
    // Node 0 -> Node 1, Node 2
    // Node 1 -> Node 0, Node 2
    // Node 2 -> Node 1

    // Variables
    int numNodes = 3;  // Total number of nodes in the graph
    int numTotalNeighbors_h = 5;  // Total number of edges (neighbors)

    // Adjacency list representation
    // nodePtrs_h indicates the start of the neighbors list for each node
    int nodePtrs_h[] = {0, 2, 4, 5};  // nodePtrs: node 0 starts at 0, node 1 starts at 2, node 2 starts at 4
    int nodeNeighbors_h[] = {1, 2, 0, 2, 1};  // nodeNeighbors: Neighbors of node 0, 1, and 2
    int nodeVisited_h[] = {0, 0, 0};  // Visited array: all nodes initially unvisited

    // Initialize BFS: Starting from node 0
    int currLevelNodes_h[] = {0};  // BFS starting from node 0
    int numCurrLevelNodes = 1;  // Only one node in the current level (root node 0)

    // Next level nodes: Initially empty
    int numNextLevelNodes_h = 0;  // Initially, no nodes are in the next level
    int nextLevelNodes_h[numNodes];  // Array to hold nodes for the next BFS level

    // Device variables
    int *nodePtrs_d, *nodeNeighbors_d, *nodeVisited_d, *currLevelNodes_d;
    int *nextLevelNodes_d, *numNextLevelNodes_d;

    // Allocate memory on the device
    cudaMalloc((void **)&nodePtrs_d, (numNodes + 1) * sizeof(int));
    cudaMalloc((void **)&nodeVisited_d, numNodes * sizeof(int));
    cudaMalloc((void **)&nodeNeighbors_d, numTotalNeighbors_h * sizeof(int));
    cudaMalloc((void **)&currLevelNodes_d, numCurrLevelNodes * sizeof(int));
    cudaMalloc((void **)&nextLevelNodes_d, numNodes * sizeof(int));
    cudaMalloc((void **)&numNextLevelNodes_d, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(nodePtrs_d, nodePtrs_h, (numNodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nodeVisited_d, nodeVisited_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nodeNeighbors_d, nodeNeighbors_h, numTotalNeighbors_h * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(currLevelNodes_d, currLevelNodes_h, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(numNextLevelNodes_d, 0, sizeof(int));

    // Kernel launch (use global or block queuing)
    gpu_global_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d, currLevelNodes_d, nextLevelNodes_d, numCurrLevelNodes, numNextLevelNodes_d);
    // Alternatively:
    // gpu_block_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d, currLevelNodes_d, nextLevelNodes_d, numCurrLevelNodes, numNextLevelNodes_d);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy results back to the host
    cudaMemcpy(&numNextLevelNodes_h, numNextLevelNodes_d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nextLevelNodes_h, nextLevelNodes_d, numNodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nodeVisited_h, nodeVisited_d, numNodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the result (Visited nodes)
    printf("Visited nodes: ");
    for (int i = 0; i < numNodes; i++) {
        printf("%d ", nodeVisited_h[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(nodePtrs_d);
    cudaFree(nodeVisited_d);
    cudaFree(nodeNeighbors_d);
    cudaFree(currLevelNodes_d);
    cudaFree(nextLevelNodes_d);
    cudaFree(numNextLevelNodes_d);

    return 0;
}
