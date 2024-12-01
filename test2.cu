#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>
#include <cfloat>
#include <opencv2/opencv.hpp>  // OpenCV header for visualization

#define CHECK(call)                                 \
{                                                   \
    const cudaError_t error = call;                 \
    if (error != cudaSuccess)                       \
    {                                               \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                    \
    }                                               \
}

// Matrix multiplication kernel with 16x16 tiles
__global__ void matrixMulKernel(float *A, float *B, float *C, int N, int offset) {
    __shared__ float A_tile[16][16];
    __shared__ float B_tile[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;

    for (int k = 0; k < (N + 15) / 16; k++) {
        if (k * 16 + threadIdx.x < N && row < N)
            A_tile[threadIdx.y][threadIdx.x] = A[row * N + k * 16 + threadIdx.x];
        else
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;

        if (k * 16 + threadIdx.y < N && col < N)
            B_tile[threadIdx.y][threadIdx.x] = B[(k * 16 + threadIdx.y) * N + col];
        else
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int e = 0; e < 16; e++) {
            value += A_tile[threadIdx.y][e] * B_tile[e][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// Kernel for finding two minimums in 16x16 tiles
__global__ void findTwoMinsKernel(float *C, float *minVals, int *minIndices, int N, int offset) {
    __shared__ float sharedMins[32]; 
    __shared__ int sharedIndices[32]; 

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid + offset;

    sharedMins[2 * tid] = FLT_MAX;
    sharedMins[2 * tid + 1] = FLT_MAX;
    sharedIndices[2 * tid] = -1;
    sharedIndices[2 * tid + 1] = -1;

    if (index < N * N) {
        float value = C[index];
        sharedMins[2 * tid] = value;
        sharedIndices[2 * tid] = index;
    }
    __syncthreads();

    // Parallel reduction to find two minimums
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            int firstIdx = 2 * tid;
            int secondIdx = 2 * (tid + stride);

            if (sharedMins[secondIdx] < sharedMins[firstIdx]) {
                sharedMins[firstIdx + 1] = sharedMins[firstIdx];
                sharedIndices[firstIdx + 1] = sharedIndices[firstIdx];
                sharedMins[firstIdx] = sharedMins[secondIdx];
                sharedIndices[firstIdx] = sharedIndices[secondIdx];
            } else if (sharedMins[secondIdx] < sharedMins[firstIdx + 1]) {
                sharedMins[firstIdx + 1] = sharedMins[secondIdx];
                sharedIndices[firstIdx + 1] = sharedIndices[secondIdx];
            }
        }
        __syncthreads();
    }

    // Store results for this block
    if (tid == 0) {
        minVals[2 * blockIdx.x] = sharedMins[0];
        minIndices[2 * blockIdx.x] = sharedIndices[0];
        minVals[2 * blockIdx.x + 1] = sharedMins[1];
        minIndices[2 * blockIdx.x + 1] = sharedIndices[1];
    }
}

// Dijkstra's algorithm kernel using 16x16 tiles
__global__ void dijkstraKernel(float *C, float *dist, bool *visited, int *predecessor, int N, int source, int target, int offset) {
    __shared__ float localDist[1024];
    __shared__ bool localVisited[1024];
    __shared__ int localPredecessor[1024];  // Stores the predecessor of each node
    int tid = threadIdx.x;

    int index = tid + offset;

    if (index < N) {
        localDist[index] = FLT_MAX;
        localVisited[index] = false;
        localPredecessor[index] = -1;  // Initialize predecessor
    }
    __syncthreads();

    if (tid == source) {
        localDist[tid] = 0;
    }
    __syncthreads();

    for (int i = 0; i < N; i++) {
        int minNode = -1;
        float minDist = FLT_MAX;

        if (!localVisited[tid] && localDist[tid] < minDist) {
            minDist = localDist[tid];
            minNode = tid;
        }

        if (minNode != -1) {
            localVisited[minNode] = true;

            // Relax the edges (use edge weight 1 for simplicity)
            for (int neighbor = 0; neighbor < N; neighbor++) {
                if (C[minNode * N + neighbor] != -1 && !localVisited[neighbor]) {
                    float newDist = localDist[minNode] + 1;  // Use 1 as the edge weight
                    if (newDist < localDist[neighbor]) {
                        localDist[neighbor] = newDist;
                        localPredecessor[neighbor] = minNode;  // Update predecessor
                    }
                }
            }
        }
        __syncthreads();
    }

    if (tid < N) {
        dist[tid] = localDist[tid];
        visited[tid] = localVisited[tid];
        predecessor[tid] = localPredecessor[tid];  // Store the predecessor for the node
    }
}

std::vector<int> reconstructPath(int *predecessor, int N, int target) {
    std::vector<int> path;
    int node = target;

    // Reconstruct the path by backtracking through the predecessors
    while (node != -1) {
        path.push_back(node);
        node = predecessor[node];
    }

    // Reverse the path to get it from source to target
    std::reverse(path.begin(), path.end());

    return path;
}

// Function to display the matrix using OpenCV
void visualizeMatrix(const std::vector<float>& matrix, int N) {
    // Convert the matrix to a grayscale image
    cv::Mat img(N, N, CV_32F, const_cast<float*>(matrix.data())); // Create a 2D matrix
    img.convertTo(img, CV_8U, 255.0); // Convert to 8-bit for display
    
    // Display the matrix using OpenCV
    cv::imshow("Matrix Visualization", img); // Display the matrix
    cv::waitKey(0); // Wait until a key is pressed to close the window
}

int main() {
    int N;
    std::cout << "Enter matrix size (N): ";
    std::cin >> N;

    if (N <= 0) {
        std::cerr << "Error: N must be greater than zero." << std::endl;
        return -1;
    }

    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate memory for GPU 0
    float *d_A, *d_B, *d_C;
    CHECK(cudaSetDevice(0));
    CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
    CHECK(cudaMalloc(&d_B, N * N * sizeof(float)));
    CHECK(cudaMalloc(&d_C, N * N * sizeof(float)));

    CHECK(cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);
    int offset = 0;
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N, offset);
    CHECK(cudaDeviceSynchronize());

    // Allocate memory for GPU 1
    float *d_A2, *d_B2, *d_C2;
    CHECK(cudaSetDevice(1));
    CHECK(cudaMalloc(&d_A2, N * N * sizeof(float)));
    CHECK(cudaMalloc(&d_B2, N * N * sizeof(float)));
    CHECK(cudaMalloc(&d_C2, N * N * sizeof(float)));

    CHECK(cudaMemcpy(d_A2, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B2, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

    offset = N / 2; // Split the workload
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A2, d_B2, d_C2, N, offset);
    CHECK(cudaDeviceSynchronize());

    // Perform minimum finding on GPU 0
    float *d_minVals, *d_minVals2;
    int *d_minIndices, *d_minIndices2;
    CHECK(cudaSetDevice(0));
    CHECK(cudaMalloc(&d_minVals, numBlocks.x * numBlocks.y * 2 * sizeof(float)));
    CHECK(cudaMalloc(&d_minIndices, numBlocks.x * numBlocks.y * 2 * sizeof(int)));

    findTwoMinsKernel<<<numBlocks, threadsPerBlock>>>(d_C, d_minVals, d_minIndices, N, offset);
    CHECK(cudaDeviceSynchronize());

    // Perform minimum finding on GPU 1
    CHECK(cudaSetDevice(1));
    CHECK(cudaMalloc(&d_minVals2, numBlocks.x * numBlocks.y * 2 * sizeof(float)));
    CHECK(cudaMalloc(&d_minIndices2, numBlocks.x * numBlocks.y * 2 * sizeof(int)));

    findTwoMinsKernel<<<numBlocks, threadsPerBlock>>>(d_C2, d_minVals2, d_minIndices2, N, offset);
    CHECK(cudaDeviceSynchronize());

    // Copy results from both GPUs
    std::vector<float> h_minVals(numBlocks.x * numBlocks.y * 2);
    std::vector<int> h_minIndices(numBlocks.x * numBlocks.y * 2);

    CHECK(cudaMemcpy(h_minVals.data(), d_minVals, numBlocks.x * numBlocks.y * 2 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_minIndices.data(), d_minIndices, numBlocks.x * numBlocks.y * 2 * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<float> h_minVals2(numBlocks.x * numBlocks.y * 2);
    std::vector<int> h_minIndices2(numBlocks.x * numBlocks.y * 2);

    CHECK(cudaMemcpy(h_minVals2.data(), d_minVals2, numBlocks.x * numBlocks.y * 2 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_minIndices2.data(), d_minIndices2, numBlocks.x * numBlocks.y * 2 * sizeof(int), cudaMemcpyDeviceToHost));

    // Merge the results and compute shortest path using Dijkstra
    float min1 = FLT_MAX, min2 = FLT_MAX;
    int index1 = -1, index2 = -1;

    for (int i = 0; i < numBlocks.x * numBlocks.y * 2; i += 2) {
        if (h_minVals[i] < min1) {
            min2 = min1;
            index2 = index1;
            min1 = h_minVals[i];
            index1 = h_minIndices[i];
        } else if (h_minVals[i] < min2 && h_minIndices[i] != index1) {
            min2 = h_minVals[i];
            index2 = h_minIndices[i];
        }

        if (h_minVals[i + 1] < min1) {
            min2 = min1;
            index2 = index1;
            min1 = h_minVals[i + 1];
            index1 = h_minIndices[i + 1];
        } else if (h_minVals[i + 1] < min2 && h_minIndices[i + 1] != index1) {
            min2 = h_minVals[i + 1];
            index2 = h_minIndices[i + 1];
        }
    }

    float *d_dist, *d_dist2;
    bool *d_visited, *d_visited2;

    CHECK(cudaSetDevice(0));
    CHECK(cudaMalloc(&d_dist, N * sizeof(float)));
    CHECK(cudaMalloc(&d_visited, N * sizeof(bool)));
    int *d_predecessor, *d_predecessor2;
    CHECK(cudaMalloc(&d_predecessor, N * sizeof(int)));
    CHECK(cudaSetDevice(0));
    dijkstraKernel<<<numBlocks, threadsPerBlock>>>(d_C, d_dist, d_visited, d_predecessor, N, index1, index2, 0);
    CHECK(cudaSetDevice(1));
    CHECK(cudaMalloc(&d_dist2, N * sizeof(float)));
    CHECK(cudaMalloc(&d_visited2, N * sizeof(bool)));
    CHECK(cudaMalloc(&d_predecessor2, N * sizeof(int)));
    dijkstraKernel<<<numBlocks, threadsPerBlock>>>(d_C2, d_dist2, d_visited2, d_predecessor2, N, index1, index2, N / 2);
    
    CHECK(cudaDeviceSynchronize());

    std::vector<float> h_dist(N);
    CHECK(cudaMemcpy(h_dist.data(), d_dist, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> h_dist2(N);
    CHECK(cudaMemcpy(h_dist2.data(), d_dist2, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Shortest distance from " << index1 << " to " << index2 << " is: " << h_dist[index2] << std::endl;

    // Cleanup memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_A2));
    CHECK(cudaFree(d_B2));
    CHECK(cudaFree(d_C2));
    CHECK(cudaFree(d_minVals));
    CHECK(cudaFree(d_minIndices));
    CHECK(cudaFree(d_minVals2));
    CHECK(cudaFree(d_minIndices2));
    CHECK(cudaFree(d_dist));
    CHECK(cudaFree(d_dist2));
    CHECK(cudaFree(d_visited));
    CHECK(cudaFree(d_visited2));

    return 0;
}
